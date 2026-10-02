"""QuizForge's Flask application: local quiz UI, JSON API and fixture parity.

The app intentionally has no account system, analytics, CDN dependency, remote
font, external image, or network call from the browser.  It is an instrumented
local target for QuizEngine, not a public quiz-hosting service.  The default CLI
binds to 127.0.0.1; binding beyond loopback requires an explicit opt-in.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, abort, jsonify, make_response, render_template, request, session

from .catalog import (
    CHAOS_MODES,
    QUESTIONS,
    THEMES,
    VARIANTS,
    ZOOM_LEVELS,
    Screen,
    answer_key,
    build_results_screen,
    build_screen,
    describe_variants,
)


@dataclass
class QuizState:
    """Per-browser session state; no answer key is sent until explicitly exported."""

    questions: List[Tuple[str, Tuple[str, ...], int]]
    variants: List[str]
    theme: str = "light"
    zoom: float = 1.0
    chaos: List[str] = field(default_factory=list)
    index: int = 0
    selected: Optional[int] = None
    answered: bool = False
    correct_count: int = 0
    answers: List[Dict[str, Any]] = field(default_factory=list)
    overlay_dismissed: bool = False
    started_at: float = field(default_factory=time.time)
    order_seed: int = 0
    notice: str = ""
    # Explicit user-selected layout preference; None means follow the fixture sequence.
    layout_override: Optional[str] = None
    dismissed_questions: set[int] = field(default_factory=set)
    settled_chaos: set[Tuple[int, str]] = field(default_factory=set)
    navigation_lock: Any = field(default_factory=threading.RLock, repr=False, compare=False)
    session_epoch: Optional[str] = field(default=None, repr=False, compare=False)

    @property
    def done(self) -> bool:
        return self.index >= len(self.questions)

    def current(self) -> Screen:
        if self.done:
            return build_results_screen(
                total=len(self.questions), correct=self.correct_count,
                theme=self.theme, zoom=self.zoom,
            )
        question, original_options, original_correct = self.questions[self.index]
        variant_key = self.layout_override or self.variants[self.index % len(self.variants)]
        saved_answer = next(
            (item for item in self.answers if int(item.get("question_index", -1)) == self.index),
            None,
        )
        options, correct = _options_for_variant(
            variant_key,
            original_options,
            original_correct,
            selected_text=saved_answer.get("selected_text") if saved_answer and self.answered else None,
        )
        overlays = [
            mode for mode in self.chaos
            if mode not in {"layout_shift", "stale"} or (self.index, mode) not in self.settled_chaos
        ]
        dismissed = self.index in self.dismissed_questions or self.overlay_dismissed
        if "popup" in overlays and dismissed:
            overlays.remove("popup")
        order = None
        if "reorder" in overlays:
            # Stable permutation per question; key and selected marker are remapped.
            import random
            order = list(range(len(options)))
            random.Random(self.index + 17).shuffle(order)
        selected_text = saved_answer.get("selected_text") if saved_answer and self.answered else None
        selected_position = options.index(selected_text) if selected_text in options else None
        if selected_position is not None and order is not None:
            selected_position = order.index(selected_position)
        return build_screen(
            variant_key,
            question,
            options,
            correct=correct,
            selected=selected_position,
            index=self.index,
            total=len(self.questions),
            theme=self.theme,
            zoom=self.zoom,
            chaos=overlays,
            final=self.index == len(self.questions) - 1,
            order=order,
            overlay_dismissed=dismissed,
        )

    def public_state(self) -> Dict[str, Any]:
        screen = self.current()
        answer = next(
            (item for item in self.answers if int(item.get("question_index", -1)) == self.index),
            None,
        )
        selected_index = next((option.index for option in screen.options if option.selected), None)
        feedback = None
        if self.answered and answer is not None and screen.correct_index is not None and selected_index is not None:
            feedback = {
                "correct": bool(answer["correct"]),
                "selected_letter": chr(ord("A") + int(selected_index)),
                "correct_letter": chr(ord("A") + int(screen.correct_index)),
                "correct_text": screen.options[int(screen.correct_index)].text,
            }
        navigation_state = "COMPLETED" if self.done else ("READY" if self.answered else "NOT_READY")
        question_id = None
        if not self.done:
            question_id = _question_id(self.index, self.questions[self.index][0])
            screen_data = screen.to_dict()
            screen_data["question"]["id"] = question_id
        else:
            screen_data = screen.to_dict()
        question_state = {
            "question_id": question_id,
            "current_question": self.index,
            "selected_option": selected_index,
            "answer_submitted": self.answered,
            "answer_correct": feedback["correct"] if feedback else None,
            "can_navigate": bool(self.answered and not self.done),
            "navigation_state": navigation_state,
        }
        return {
            "screen": screen_data,
            "session_epoch": self.session_epoch,
            "session_recovery": {
                "dismissed_questions": sorted(self.dismissed_questions),
                "settled_chaos": [[index, mode] for index, mode in sorted(self.settled_chaos)],
            },
            "selected": selected_index,
            "answered": self.answered,
            "feedback": feedback,
            "question_state": question_state,
            "navigation_state": navigation_state,
            "correct_count": self.correct_count,
            "question_count": len(self.questions),
            "done": self.done,
            "score_percent": round(100 * self.correct_count / max(1, len(self.answers))) if self.answers else None,
            "notice": self.notice,
            "settings": {
                "theme": self.theme,
                "zoom": self.zoom,
                "layout": self.layout_override,
                "chaos": list(self.chaos),
                "variants": list(self.variants),
            },
            "controls": {
                "themes": list(THEMES),
                "zoom_levels": list(ZOOM_LEVELS),
                "chaos_modes": list(CHAOS_MODES),
                "variants": list(VARIANTS),
            },
        }

    def reset(self) -> None:
        """Reset question/session data; preserve independent UI preferences."""
        self.index = 0
        self.selected = None
        self.answered = False
        self.correct_count = 0
        self.answers.clear()
        self.dismissed_questions.clear()
        self.settled_chaos.clear()
        self.overlay_dismissed = False
        self.notice = ""
        self.started_at = time.time()


class StateStore:
    """Bounded thread-safe memory store; session ids are random and cookie-bound.

    Expired and least-recently-used sessions are evicted so a long-lived local
    server cannot accumulate unbounded per-browser state. The same-tab UI keeps
    a validated sessionStorage recovery snapshot so a server restart or eviction
    does not silently rewind an active quiz; no server-side answer data is written
    to disk or an external service.
    """

    def __init__(self, *, max_sessions: int = 512, ttl_seconds: float = 12 * 60 * 60) -> None:
        self._states: Dict[str, QuizState] = {}
        self._last_seen: Dict[str, float] = {}
        self._max_sessions = max(1, int(max_sessions))
        self._ttl_seconds = max(60.0, float(ttl_seconds))
        self._lock = threading.RLock()

    def _prune(self, now: float, *, reserve: int = 0) -> None:
        expired = [key for key, touched in self._last_seen.items() if now - touched > self._ttl_seconds]
        for key in expired:
            self._states.pop(key, None)
            self._last_seen.pop(key, None)
        while len(self._states) + reserve > self._max_sessions:
            oldest = min(self._last_seen, key=self._last_seen.get) if self._last_seen else None
            if oldest is None:
                break
            self._states.pop(oldest, None)
            self._last_seen.pop(oldest, None)

    def get_or_create(self, key: str) -> QuizState:
        now = time.monotonic()
        with self._lock:
            self._prune(now, reserve=1 if key not in self._states else 0)
            if key not in self._states:
                self._states[key] = _new_state()
            self._last_seen[key] = now
            return self._states[key]

    def replace(self, key: str) -> QuizState:
        now = time.monotonic()
        with self._lock:
            self._prune(now, reserve=1 if key not in self._states else 0)
            self._states[key] = _new_state()
            self._last_seen[key] = now
            return self._states[key]


def _options_for_variant(
    variant_key: str,
    options: Tuple[str, ...],
    correct_index: int,
    *,
    selected_text: Optional[str] = None,
) -> Tuple[List[str], int]:
    """Make advertised two/six-choice variants truthful without losing answer identity."""
    values = list(options)
    correct_text = values[correct_index]
    if variant_key == "two_options":
        # Keep the correct answer and (when revisiting) the operator's recorded
        # choice. Otherwise pair the correct answer with the first distractor.
        selected_index = values.index(selected_text) if selected_text in values else None
        picks = {correct_index}
        if selected_index is not None:
            picks.add(selected_index)
        if len(picks) < 2:
            picks.add(next(index for index in range(len(values)) if index != correct_index))
        chosen = sorted(picks)
        reduced = [values[index] for index in chosen]
        return reduced, reduced.index(correct_text)
    if variant_key == "six_options_long" and len(values) < 6:
        extras = [f"Additional practice choice {number}" for number in range(1, 7)]
        for choice in extras:
            if len(values) >= 6:
                break
            if choice not in values:
                values.append(choice)
    return values, values.index(correct_text)


def _question_id(index: int, question: str) -> str:
    """Stable unique identity for one question slot in a quiz sequence.

    Include the zero-based slot as well as content: identical question text may
    legitimately appear twice, but navigation still has to distinguish them.
    """
    digest = hashlib.sha256(f"{index}\0{question}".encode("utf-8")).hexdigest()[:12]
    return f"q-{index + 1:04d}-{digest}"


def _navigation_contract(state_payload: Dict[str, Any], from_question_id: Optional[str]) -> Dict[str, Any]:
    """Add an explicit identity/content contract while preserving the state schema."""
    done = bool(state_payload.get("done"))
    screen = state_payload["screen"]
    question = screen.get("question", {})
    identity = state_payload["question_state"].get("question_id")
    state_payload.update({
        "success": True,
        "from_question_id": from_question_id,
        "question_id": identity,
        "question_number": None if done else int(question["index"]) + 1,
        "total_questions": int(state_payload["question_count"]),
        "question": None if done else question["text"],
        "options": [] if done else screen["options"],
        "selected_index": state_payload.get("selected"),
        "navigation_state": "COMPLETED" if done else state_payload["navigation_state"],
    })
    return state_payload


def _restore_state_snapshot(current: QuizState, snapshot: Any) -> bool:
    """Restore a same-tab, fixture-validated snapshot after a server restart.

    The browser snapshot contains only the visible quiz position, preferences,
    and choices the user already made. Question text/options/correctness are
    reconstructed from the local fixture bank; answer keys are never trusted
    from or copied into the snapshot.
    """
    if not isinstance(snapshot, dict) or snapshot.get("session_epoch") == current.session_epoch:
        return False
    public = snapshot.get("state")
    ledger = snapshot.get("answers", [])
    if not isinstance(public, dict) or not isinstance(ledger, list):
        raise ValueError("session recovery data is malformed")
    recovery_screen = public.get("screen")
    recovery_question = recovery_screen.get("question") if isinstance(recovery_screen, dict) else None
    if not isinstance(recovery_question, dict):
        raise ValueError("session recovery question is malformed")
    settings = public.get("settings")
    if not isinstance(settings, dict):
        raise ValueError("session recovery settings are missing")
    try:
        total = int(public.get("question_count"))
        index = int(recovery_question["index"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("session recovery position is invalid") from None
    if total not in {12, 18} or total > len(QUESTIONS) or index < 0 or index > total:
        raise ValueError("session recovery position is outside the quiz")
    variants = settings.get("variants")
    if (not isinstance(variants, list) or len(variants) != total or
            any(not isinstance(key, str) or key not in VARIANTS or key == "results_screen" for key in variants)):
        raise ValueError("session recovery layout sequence is invalid")
    theme = settings.get("theme")
    try:
        zoom = float(settings.get("zoom"))
    except (TypeError, ValueError):
        raise ValueError("session recovery zoom is invalid") from None
    chaos = settings.get("chaos", [])
    layout = settings.get("layout")
    if theme not in THEMES or zoom not in ZOOM_LEVELS:
        raise ValueError("session recovery preferences are invalid")
    if not isinstance(chaos, list) or any(mode not in CHAOS_MODES for mode in chaos):
        raise ValueError("session recovery conditions are invalid")
    if layout is not None and (not isinstance(layout, str) or layout not in VARIANTS or layout == "results_screen"):
        raise ValueError("session recovery layout override is invalid")
    if index < total:
        expected_id = _question_id(index, QUESTIONS[index][0])
        question_state = public.get("question_state")
        if (not isinstance(question_state, dict) or question_state.get("question_id") != expected_id or
                recovery_question.get("text") != QUESTIONS[index][0]):
            raise ValueError("session recovery question identity does not match the fixture")
    elif not public.get("done"):
        raise ValueError("session recovery completion state is inconsistent")

    current.questions = list(QUESTIONS[:total])
    current.variants = list(variants)
    current.index = index
    current.theme = theme
    current.zoom = zoom
    current.chaos = list(dict.fromkeys(chaos))
    current.layout_override = layout
    current.answers = []
    current.correct_count = 0
    dismissed = snapshot.get("dismissed_questions", [])
    settled = snapshot.get("settled_chaos", [])
    if not isinstance(dismissed, list) or not isinstance(settled, list):
        raise ValueError("session recovery markers are malformed")
    current.dismissed_questions = {int(i) for i in dismissed if str(i).isdigit() and 0 <= int(i) < total}
    current.settled_chaos = {
        (int(row[0]), row[1]) for row in settled
        if isinstance(row, (list, tuple)) and len(row) == 2 and str(row[0]).isdigit()
        and isinstance(row[1], str) and 0 <= int(row[0]) < total
        and row[1] in {"layout_shift", "stale"}
    }
    seen: set[int] = set()
    for item in ledger:
        if not isinstance(item, dict):
            continue
        try:
            qindex = int(item.get("question_index"))
        except (TypeError, ValueError):
            continue
        selected_text = item.get("selected_text")
        if qindex < 0 or qindex >= total or qindex in seen or not isinstance(selected_text, str):
            continue
        question, original_options, original_correct = current.questions[qindex]
        variant_key = current.layout_override or current.variants[qindex % len(current.variants)]
        options, correct = _options_for_variant(variant_key, original_options, original_correct, selected_text=selected_text)
        order = None
        if "reorder" in current.chaos:
            import random
            order = list(range(len(options)))
            random.Random(qindex + 17).shuffle(order)
        screen = build_screen(
            variant_key, question, options, correct=correct, index=qindex, total=total,
            theme=current.theme, zoom=current.zoom, chaos=[], final=qindex == total - 1,
            order=order, overlay_dismissed=True,
        )
        selected_index = next((option.index for option in screen.options if option.text == selected_text), None)
        if selected_index is None or screen.correct_index is None:
            continue
        correct_text = screen.options[screen.correct_index].text
        record = {
            "question_index": qindex, "question": question, "selected": selected_index,
            "selected_text": selected_text, "correct_index": screen.correct_index,
            "correct_text": correct_text, "correct": selected_index == screen.correct_index,
        }
        current.answers.append(record)
        seen.add(qindex)
    current.answers.sort(key=lambda row: int(row["question_index"]))
    current.correct_count = sum(bool(row["correct"]) for row in current.answers)
    current.selected = None
    current.answered = False
    if index < total:
        current_record = next((row for row in current.answers if int(row["question_index"]) == index), None)
        if current_record:
            current.selected = int(current_record["selected"])
            current.answered = True
    current.overlay_dismissed = index in current.dismissed_questions
    current.notice = str(public.get("notice") or "")
    return True


def _new_state() -> QuizState:
    # A representative but short default quiz.  The variants page and API can
    # still start a full 19-screen layout sweep explicitly.
    return QuizState(
        questions=list(QUESTIONS[:12]),
        variants=list(VARIANTS[:12]),
    )


def create_app(config: Optional[Dict[str, Any]] = None, *, state_store: Optional[StateStore] = None) -> Flask:
    """Create QuizForge; injectable config/store make all routes testable offline."""
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.update(
        SECRET_KEY=os.environ.get("QUIZFORGE_SECRET", secrets.token_hex(32)),
        JSON_SORT_KEYS=False,
        MAX_CONTENT_LENGTH=32 * 1024,
        QUIZFORGE_ALLOW_REMOTE=os.environ.get("QUIZFORGE_ALLOW_REMOTE", "0") == "1",
    )
    if config:
        app.config.update(config)
    store = state_store or StateStore()
    app.extensions["quizforge_state_store"] = store

    def state() -> QuizState:
        key = session.get("quizforge_session")
        if not key:
            key = secrets.token_urlsafe(24)
            session["quizforge_session"] = key
        current = store.get_or_create(key)
        if current.session_epoch is None:
            # A new token per in-memory state detects both process restarts and
            # LRU/TTL eviction while a browser tab still holds an older screen.
            current.session_epoch = secrets.token_urlsafe(12)
        return current

    def json_error(message: str, status: int = 400):
        return jsonify({"error": message}), status

    def serialized_session(handler):
        """Serialize state reads/mutations per browser session, including other tabs."""
        @wraps(handler)
        def wrapped(*args, **kwargs):
            current = state()
            with current.navigation_lock:
                payload = request.get_json(silent=True) or {}
                if request.path != "/api/session/resume":
                    try:
                        _restore_state_snapshot(current, payload.get("client_snapshot"))
                    except ValueError as error:
                        return json_error(str(error), 409)
                    expected_id = payload.get("expected_question_id")
                    if expected_id is not None:
                        actual_id = current.public_state()["question_state"]["question_id"]
                        if expected_id != actual_id:
                            return json_error("question state changed in another tab; resync before continuing", 409)
                return handler(*args, **kwargs)
        return wrapped

    @app.after_request
    def disable_api_caching(response):
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    @app.before_request
    def local_only_guard():
        # Flask's test client has no remote address. Real server operation is
        # loopback-only by default; the CLI independently refuses non-loopback
        # binds without QUIZFORGE_ALLOW_REMOTE=1.
        remote = request.remote_addr
        if remote and remote not in {"127.0.0.1", "::1", "localhost", "testclient"}:
            if not app.config.get("QUIZFORGE_ALLOW_REMOTE", False):
                abort(403, description="QuizEngine UI is local-only; set QUIZFORGE_ALLOW_REMOTE=1 to expose it")
        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            origin = request.headers.get("Origin")
            host = request.host.split(":", 1)[0].strip("[]")
            if origin:
                from urllib.parse import urlparse
                origin_host = (urlparse(origin).hostname or "").lower()
                if origin_host and origin_host not in {host.lower(), "localhost", "127.0.0.1", "::1"}:
                    abort(403, description="cross-origin mutation refused")

    @app.get("/")
    @serialized_session
    def index():
        current = state()
        quiz_variants = [row for row in describe_variants() if row["screen_role"] != "end_state"]
        return render_template("index.html", initial=current.public_state(), variants=quiz_variants)

    @app.get("/variants")
    def variants_page():
        return render_template("variants.html", variants=describe_variants())

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "service": "quizengine-ui", "variants": len(VARIANTS)})

    @app.post("/api/session/resume")
    @serialized_session
    def resume_session():
        payload = request.get_json(silent=True) or {}
        try:
            _restore_state_snapshot(state(), payload.get("snapshot"))
        except ValueError as error:
            return json_error(str(error), 409)
        return jsonify(state().public_state())

    @app.get("/api/session")
    @serialized_session
    def get_session():
        return jsonify(state().public_state())

    @app.post("/api/answer")
    @serialized_session
    def answer():
        current = state()
        if current.done:
            return json_error("quiz is already complete", 409)
        if current.answered:
            return json_error("this question has already been answered", 409)
        if any(overlay.get("kind") == "modal" for overlay in current.current().overlays):
            return json_error("dismiss the blocking notice before answering", 409)
        payload = request.get_json(silent=True) or {}
        try:
            selected = int(payload.get("option"))
        except (TypeError, ValueError):
            return json_error("option must be an integer index")
        question, _options, _correct = current.questions[current.index]
        screen = current.current()
        if selected < 0 or selected >= len(screen.options):
            return json_error(f"option index must be from 0 to {len(screen.options) - 1}")
        shown_correct = screen.correct_index if screen.correct_index is not None else correct
        selected_text = screen.options[selected].text
        correct_text = screen.options[shown_correct].text
        was_correct = selected == shown_correct
        current.selected = selected
        current.answered = True
        current.correct_count += int(was_correct)
        record = {
            "question_index": current.index,
            "question": question,
            "selected": selected,
            "selected_text": selected_text,
            "correct_index": shown_correct,
            "correct_text": correct_text,
            "correct": was_correct,
        }
        current.answers = [item for item in current.answers if int(item.get("question_index", -1)) != current.index]
        current.answers.append(record)
        current.answers.sort(key=lambda item: int(item["question_index"]))
        current.correct_count = sum(bool(item["correct"]) for item in current.answers)
        current.notice = "Correct — nice work." if was_correct else "Not quite. The correct answer is shown below."
        return jsonify(current.public_state())

    @app.post("/api/next")
    @serialized_session
    def next_question():
        current = state()
        if current.done:
            return json_error("quiz is already complete", 409)
        if not current.answered:
            return json_error("choose an answer before continuing", 409)
        from_question_id = current.public_state()["question_state"]["question_id"]
        if current.overlay_dismissed:
            current.dismissed_questions.add(current.index)
        current.index += 1
        saved_answer = next(
            (item for item in current.answers if int(item.get("question_index", -1)) == current.index),
            None,
        )
        current.selected = int(saved_answer["selected"]) if saved_answer else None
        current.answered = saved_answer is not None
        current.overlay_dismissed = current.index in current.dismissed_questions
        current.notice = ""
        return jsonify(_navigation_contract(current.public_state(), from_question_id))

    @app.post("/api/previous")
    @serialized_session
    def previous_question():
        current = state()
        if current.index <= 0:
            return json_error("already at the first question", 409)
        from_question_id = current.public_state()["question_state"]["question_id"]
        if current.overlay_dismissed:
            current.dismissed_questions.add(current.index)
        current.index -= 1
        previous_answer = next(
            (item for item in current.answers if int(item.get("question_index", -1)) == current.index),
            None,
        )
        current.selected = int(previous_answer["selected"]) if previous_answer else None
        current.answered = previous_answer is not None
        current.overlay_dismissed = current.index in current.dismissed_questions
        current.notice = "Returned to your previous answer." if previous_answer else "Returned to the previous question."
        return jsonify(_navigation_contract(current.public_state(), from_question_id))

    @app.post("/api/restart")
    @serialized_session
    def restart():
        key = session.get("quizforge_session")
        current = state() if not key else store.get_or_create(key)
        current.reset()
        return jsonify(current.public_state())

    @app.post("/api/settings")
    @serialized_session
    def settings():
        current = state()
        payload = request.get_json(silent=True) or {}

        # Validate the whole preference update first; a malformed setting must
        # not partially mutate theme/zoom/layout before returning HTTP 400.
        theme = payload.get("theme", current.theme)
        if theme not in THEMES:
            return json_error(f"theme must be one of {', '.join(THEMES)}")
        zoom = current.zoom
        if "zoom" in payload:
            try:
                zoom = float(payload["zoom"])
            except (TypeError, ValueError):
                return json_error("zoom must be a number")
            if zoom not in ZOOM_LEVELS:
                return json_error(f"zoom must be one of {', '.join(map(str, ZOOM_LEVELS))}")
        chaos = current.chaos
        if "chaos" in payload:
            requested_chaos = payload["chaos"]
            if not isinstance(requested_chaos, list) or any(mode not in CHAOS_MODES for mode in requested_chaos):
                return json_error("chaos must be a list of supported chaos mode names")
            chaos = list(dict.fromkeys(requested_chaos))
        variant_key = None
        if "variant" in payload:
            variant_key = payload["variant"]
            if variant_key not in VARIANTS or variant_key == "results_screen":
                choices = [name for name in VARIANTS if name != "results_screen"]
                return json_error(f"choose an answerable variant: {', '.join(choices)}")

        current.theme = theme
        current.zoom = zoom
        if "chaos" in payload:
            current.chaos = chaos
            current.overlay_dismissed = False
            current.dismissed_questions.clear()
            current.settled_chaos.clear()
        if variant_key is not None:
            current.variants[current.index % len(current.variants)] = variant_key
            current.layout_override = variant_key
        if payload.get("full_matrix") is True:
            # The nineteenth catalog entry is terminal results, not an answerable
            # layout. Walk the 18 quiz layouts, then show the results summary.
            current.variants = list(VARIANTS[:-1])
            current.questions = list(QUESTIONS[:len(current.variants)])
            current.layout_override = None
            current.index = 0
            current.selected = None
            current.answered = False
            current.answers.clear()
            current.correct_count = 0
            current.overlay_dismissed = False
            current.dismissed_questions.clear()
            current.settled_chaos.clear()
            current.notice = ""
        return jsonify(current.public_state())

    @app.post("/api/overlay/dismiss")
    @serialized_session
    def dismiss_overlay():
        current = state()
        screen = current.current()
        if not screen.overlays:
            return json_error("there is no blocking overlay to dismiss", 409)
        current.overlay_dismissed = True
        current.dismissed_questions.add(current.index)
        current.notice = "Notice dismissed."
        return jsonify(current.public_state())

    @app.post("/api/chaos/advance")
    @serialized_session
    def chaos_advance():
        current = state()
        screen_chaos = current.current().chaos
        if screen_chaos.get("layout_shift"):
            current.settled_chaos.add((current.index, "layout_shift"))
            current.notice = "The layout has settled after its shift."
        elif screen_chaos.get("stale"):
            current.settled_chaos.add((current.index, "stale"))
            current.notice = "A fresh frame is available."
        elif "delay" in current.chaos:
            time.sleep(1.0)
        return jsonify(current.public_state())

    @app.get("/api/answer-key")
    @serialized_session
    def export_answer_key():
        # Deliberately separate from /api/session. A caller has to explicitly
        # request this route (UI button labelled "Export answer key").
        current = state()
        screens: List[Screen] = []
        for index, (question, original_options, original_correct) in enumerate(current.questions):
            variant_key = current.layout_override or current.variants[index % len(current.variants)]
            options, correct = _options_for_variant(variant_key, original_options, original_correct)
            order = None
            if "reorder" in current.chaos:
                import random
                order = list(range(len(options)))
                random.Random(index + 17).shuffle(order)
            screens.append(build_screen(
                variant_key, question, options, correct=correct, index=index,
                total=len(current.questions), theme=current.theme, zoom=current.zoom,
                chaos=[], final=index == len(current.questions) - 1,
                order=order, overlay_dismissed=True,
            ))
        key = answer_key(screens)
        correct_texts = {
            screen.question_text: screen.options[screen.correct_index].text
            for screen in screens
            if screen.correct_index is not None and screen.options and screen.question_text
        }
        fmt = request.args.get("format", "json").lower()
        if fmt == "csv":
            from .catalog import answer_key_csv
            response = make_response(answer_key_csv(key, correct_texts))
            response.headers["Content-Type"] = "text/csv; charset=utf-8"
            response.headers["Content-Disposition"] = "attachment; filename=quizengine-answer-key.csv"
            return response
        if fmt != "json":
            return json_error("format must be json or csv")
        response = jsonify({"answer_key": key, "count": len(key)})
        response.headers["Content-Disposition"] = "attachment; filename=quizengine-answer-key.json"
        return response

    @app.get("/api/variants")
    def get_variants():
        return jsonify({"variants": describe_variants(), "themes": list(THEMES), "zoom_levels": list(ZOOM_LEVELS), "chaos_modes": list(CHAOS_MODES)})

    @app.get("/api/parity/<variant_key>")
    def parity(variant_key: str):
        if variant_key not in VARIANTS:
            return json_error(f"unknown variant {variant_key!r}", 404)
        from quizengine.fixtures import LAYOUT_VARIANTS, build_scene

        question, options, correct = QUESTIONS[0]
        if LAYOUT_VARIANTS[variant_key].screen_role == "end_state":
            question, options, correct = "", (), None
        screen = build_screen(variant_key, question, options, correct=correct, total=1)
        scene = build_scene(variant_key, question, options, correct=correct, progress=(1, 1))
        annotation = scene.annotation()
        expected = {
            "variant": variant_key,
            "layout_type": annotation["layout_type"],
            "style": scene.options[0].style if scene.options else LAYOUT_VARIANTS[variant_key].style,
            "question": annotation["question_text"],
            "option_texts": [o["text"] for o in annotation["options"]],
            "option_count": len(annotation["options"]),
            "screen_role": annotation["screen_role"],
        }
        actual = {
            "variant": screen.variant,
            "layout_type": screen.layout_type,
            "style": screen.style,
            "question": screen.question_text,
            "option_texts": [o.text for o in screen.options],
            "option_count": len(screen.options),
            "screen_role": screen.screen_role,
        }
        mismatches = [name for name in expected if expected[name] != actual[name]]
        return jsonify({"variant": variant_key, "parity": not mismatches, "mismatches": mismatches, "quizengine_ui": actual, "quizengine_fixture": expected})

    @app.errorhandler(413)
    def too_large(_error):
        return jsonify({"error": "request body is too large"}), 413

    return app


__all__ = ["QuizState", "StateStore", "create_app"]
