"""Deterministic offline provider -- the default when no model is configured.

Purpose: the whole engine (loop, replay harness, test suite) must run with zero
API keys and zero network.  This provider is **not** a stand-in for a real model
and never pretends to be one:

* answers come from an explicitly primed answer key, or from transparent local
  heuristics (arithmetic evaluation, lexical overlap);
* heuristic answers carry deliberately *low* confidence, so the uncertainty
  policy (FR-7.5.2) -- not this provider -- decides what happens next;
* Tier-2 perception requests are answered by echoing the Tier-1 hints in the
  required schema, flagged ``offline_echo`` so reconciliation can penalize it.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from ..arith import find_expression, numbers_match, safe_eval
from ..contracts import ModelRequest, ModelResponse
from .provider import ModelProvider, extract_json

_ARITHMETIC = re.compile(r"[-+]?\d+(?:\.\d+)?\s*(?:[-+*/%^]\s*[-+]?\d+(?:\.\d+)?)+")
_STOPWORDS = {
    "the", "a", "an", "of", "is", "are", "was", "were", "what", "which", "who", "whom", "when",
    "where", "why", "how", "to", "in", "on", "for", "with", "and", "or", "not", "it", "this",
    "that", "these", "those", "be", "been", "as", "at", "by", "from", "does", "do", "did",
}


def payload_from_request(request: ModelRequest) -> Dict[str, Any]:
    """Extract the structured payload the caller embedded in the last user message."""
    for message in reversed(request.messages):
        if message.role != "user":
            continue
        parsed = extract_json(message.content)
        if parsed is not None:
            return parsed
    return {}


def normalize_text(text: str) -> str:
    """Case/punctuation/whitespace-insensitive key used for answer-key lookups."""
    lowered = (text or "").lower()
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return " ".join(lowered.split())


class OfflineProvider(ModelProvider):
    name = "offline"
    kind = "offline"
    supports_images = False

    #: Confidence reported when the answer came from an explicit key.
    KEYED_CONFIDENCE = 0.93
    #: Confidence reported for heuristic answers -- intentionally below
    #: ``confidence.low_conf`` in many cases so the policy layer engages.
    HEURISTIC_CONFIDENCE = 0.46

    def __init__(
        self,
        answer_key: Optional[Dict[str, Any]] = None,
        *,
        latency_ms: float = 0.0,
        tier2_confidence: float = 0.60,
        seed: int = 0,
    ) -> None:
        self.answer_key: Dict[str, Any] = {}
        self.latency_ms = float(latency_ms)
        self.tier2_confidence = float(tier2_confidence)
        self.seed = int(seed)
        self.calls: List[Dict[str, Any]] = []
        if answer_key:
            self.prime(answer_key)

    # -- priming ----------------------------------------------------------- #
    def prime(self, answer_key: Dict[str, Any]) -> None:
        """Load ``{question text: answer}`` where answer is a letter, index or option text."""
        for question, answer in (answer_key or {}).items():
            self.answer_key[normalize_text(str(question))] = answer

    @classmethod
    def from_file(cls, path: str) -> "OfflineProvider":
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        key = data.get("answer_key") if isinstance(data, dict) else data
        return cls(answer_key=key if isinstance(key, dict) else {})

    # -- ModelProvider ------------------------------------------------------ #
    def complete(self, request: ModelRequest) -> ModelResponse:
        payload = payload_from_request(request)
        handler = {
            "solve": self._solve,
            "perceive": self._perceive,
            "transcribe_math": self._transcribe_math,
            "describe_image": self._describe_image,
            "classify_popup": self._classify_popup,
        }.get(request.task)
        if handler is None:
            parsed: Dict[str, Any] = {"error": f"unsupported task {request.task!r}"}
        else:
            parsed = handler(payload, request)
        self.calls.append({"task": request.task, "correlation_id": request.correlation_id, "parsed": parsed})
        return ModelResponse(
            text=json.dumps(parsed, ensure_ascii=False),
            parsed=parsed,
            provider=self.name,
            model="deterministic-offline",
            latency_ms=self.latency_ms,
            finish_reason="stop",
        )

    # -- tasks ------------------------------------------------------------- #
    def _solve(self, payload: Dict[str, Any], request: ModelRequest) -> Dict[str, Any]:
        question = str(payload.get("question", ""))
        options = [str(o) for o in payload.get("options", []) or []]
        if not options:
            return {"answer": None, "confidence": 0.0, "rationale": "no options supplied"}

        keyed = self._lookup_key(question, options)
        if keyed is not None:
            letter, rationale = keyed
            return {
                "answer": letter,
                "confidence": self.KEYED_CONFIDENCE,
                "rationale": rationale,
                "source": "offline_answer_key",
            }

        arithmetic = self._solve_arithmetic(question, options)
        if arithmetic is not None:
            letter, value = arithmetic
            return {
                "answer": letter,
                "confidence": 0.88,
                "rationale": f"evaluated the expression in the question: {value}",
                "source": "offline_arithmetic",
            }

        overlap = self._solve_overlap(question, options)
        letter, score = overlap
        return {
            "answer": letter,
            "confidence": round(min(self.HEURISTIC_CONFIDENCE, 0.25 + score), 3),
            "rationale": (
                "offline lexical-overlap heuristic (no model configured); "
                "confidence is intentionally low so the uncertainty policy decides"
            ),
            "source": "offline_heuristic",
        }

    def _lookup_key(self, question: str, options: List[str]) -> Optional[Tuple[str, str]]:
        normalized = normalize_text(question)
        if not normalized:
            return None
        answer = self.answer_key.get(normalized)
        if answer is None:
            for key, value in self.answer_key.items():
                if key and (key in normalized or normalized in key):
                    answer = value
                    break
        if answer is None:
            return None
        index = _answer_to_index(answer, options)
        if index is None:
            return None
        return (chr(ord("A") + index), f"answer key match for {normalized[:48]!r}")

    @staticmethod
    def _solve_arithmetic(question: str, options: List[str]) -> Optional[Tuple[str, Any]]:
        expression = find_expression(question.replace("^", "**")) or (
            _ARITHMETIC.search(question.replace("^", "**")).group(0)
            if _ARITHMETIC.search(question.replace("^", "**"))
            else None
        )
        if expression is None:
            return None
        value = safe_eval(expression)
        if value is None:
            return None
        for index, option in enumerate(options):
            if numbers_match(value, option):
                return (chr(ord("A") + index), value)
        return None

    @staticmethod
    def _solve_overlap(question: str, options: List[str]) -> Tuple[str, float]:
        question_tokens = {t for t in normalize_text(question).split() if t not in _STOPWORDS}
        best_index, best_score = 0, 0.0
        for index, option in enumerate(options):
            option_tokens = {t for t in normalize_text(option).split() if t not in _STOPWORDS}
            if not option_tokens or not question_tokens:
                continue
            score = len(option_tokens & question_tokens) / float(len(option_tokens | question_tokens))
            if score > best_score:
                best_index, best_score = index, score
        return (chr(ord("A") + best_index), round(best_score, 3))

    def _perceive(self, payload: Dict[str, Any], request: ModelRequest) -> Dict[str, Any]:
        """Echo Tier-1 hints in the Tier-2 schema (FR-7.2.5 hint channel)."""
        hints = payload.get("tier1_hints") or {}
        options = []
        for option in hints.get("options", []) or []:
            options.append(
                {
                    "index": option.get("index", len(options)),
                    "handle": option.get("handle", f"opt_{len(options)}"),
                    "text": option.get("text", ""),
                    "hit_box": option.get("hit_box") or option.get("box"),
                    "text_box": option.get("text_box"),
                    "text_conf": option.get("text_conf", 0.9),
                    "selected_marker": option.get("selected_marker", "none"),
                }
            )
        return {
            "layout_type": hints.get("layout_type", "unknown"),
            "question_region": hints.get("question_region"),
            "question_text": hints.get("question_text", ""),
            "question_text_verbatim": hints.get("question_text", ""),
            "options": options,
            "navigation": hints.get("navigation") or {"next_btn": None, "prev_btn": None, "progress_text": None},
            "overlays": hints.get("overlays", []),
            "confidence": self.tier2_confidence,
            "source": "offline_echo",
            "notes": "offline provider: Tier-1 hints echoed, no vision performed",
        }

    def _transcribe_math(self, payload: Dict[str, Any], request: ModelRequest) -> Dict[str, Any]:
        text = str(payload.get("ocr_text", ""))
        return {
            "latex": _ascii_math_to_latex(text),
            "confidence": 0.35,
            "source": "offline_transcription",
            "notes": "offline provider cannot read images; OCR text passed through",
        }

    def _describe_image(self, payload: Dict[str, Any], request: ModelRequest) -> Dict[str, Any]:
        return {
            "description": str(payload.get("ocr_text", "")) or "no description available (offline provider)",
            "values": {},
            "confidence": 0.2,
            "source": "offline_caption",
        }

    def _classify_popup(self, payload: Dict[str, Any], request: ModelRequest) -> Dict[str, Any]:
        text = str(payload.get("text", "")).lower()
        kind = "unknown"
        for keyword, candidate in (
            ("captcha", "captcha"),
            ("not a robot", "human_verification"),
            ("verify you are human", "human_verification"),
            ("are you a human", "human_verification"),
            ("cookie", "cookie_banner"),
            ("loading", "loading"),
            ("saved", "toast"),
        ):
            if keyword in text:
                kind = candidate
                break
        dismissible = None if kind == "unknown" else kind in {"toast", "cookie_banner", "loading"}
        return {"kind": kind, "dismissible": dismissible, "confidence": 0.4 if kind == "unknown" else 0.75}


def _answer_to_index(answer: Any, options: List[str]) -> Optional[int]:
    """Answer keys may hold a letter, an index or the option text."""
    if isinstance(answer, bool):
        return None
    if isinstance(answer, int):
        return answer if 0 <= answer < len(options) else None
    if isinstance(answer, str):
        text = answer.strip()
        if len(text) == 1 and text.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            index = ord(text.upper()) - ord("A")
            return index if index < len(options) else None
        if text.isdigit():
            index = int(text)
            return index if 0 <= index < len(options) else None
        normalized = normalize_text(text)
        for index, option in enumerate(options):
            if normalize_text(option) == normalized:
                return index
        for index, option in enumerate(options):
            if normalized and (normalized in normalize_text(option) or normalize_text(option) in normalized):
                return index
    return None


def _ascii_math_to_latex(text: str) -> str:
    """Tiny ASCII -> LaTeX pass so the offline path still returns the schema shape."""
    replacements = [
        ("sqrt(", "\\sqrt{"),
        ("*", " \\cdot "),
        ("^", "^"),
        ("pi", "\\pi"),
        ("alpha", "\\alpha"),
        ("beta", "\\beta"),
        ("theta", "\\theta"),
    ]
    out = text
    for source, target in replacements:
        out = out.replace(source, target)
    return out.strip()


__all__ = ["OfflineProvider", "normalize_text", "payload_from_request"]
