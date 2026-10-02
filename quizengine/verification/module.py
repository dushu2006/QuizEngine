"""Verification Module (PRD section 7.9).

Every state-changing action must produce a :class:`VerificationRecord`
(**AC-14.3**, **L2**).  Verification is *element aware* (FR-7.9.2): a global
pixel diff is never sufficient evidence that "option B is now selected", so the
primary check is the semantic state of the target element, with the pixel diff
in the target region as corroboration.

Deadlines (FR-7.9.1): first evidence within ``verify_timeout_ms``, one more
look inside ``verify_recheck_ms`` for UIs that animate late.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..config import EngineConfig
from ..contracts import (
    ExpectedEffectType,
    Frame,
    Intent,
    PerceptionResult,
    RunEventName,
    ScreenTransition,
    SelectedMarker,
    State,
    VerificationRecord,
)
from ..geometry import Box, box_center, box_clip, box_intersection

SnapshotFn = Callable[[], "VerificationSnapshot"]


@dataclass
class VerificationSnapshot:
    """A frame plus its perception -- the two things verification compares."""

    frame: Frame
    perception: PerceptionResult

    @property
    def seq(self) -> int:
        return self.frame.seq


def expected_marker_for_style(style: Optional[str]) -> Optional[SelectedMarker]:
    """Marker a style is supposed to show when selected (fixture/QuizForge)."""
    if not style:
        return None
    normalized = str(style).lower()
    if normalized == "radio":
        return SelectedMarker.DOT
    if normalized == "checkbox":
        return SelectedMarker.CHECK
    if normalized in {"card", "tile", "button", "text_only", "highlight"}:
        return SelectedMarker.HIGHLIGHT
    return None


class VerificationModule:
    def __init__(
        self,
        config: EngineConfig,
        *,
        telemetry: Any = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.verify_config = config.verification
        self.telemetry = telemetry
        self._clock = clock
        self._sleep = sleep
        self.records: List[VerificationRecord] = []
        self.stats: Dict[str, Any] = {
            "verifications": 0,
            "passed": 0,
            "failed": 0,
            "polls": 0,
            "by_effect": {},
        }

    # -- polling loop ------------------------------------------------------- #
    def verify(
        self,
        intent: Intent,
        pre: VerificationSnapshot,
        snapshot_fn: SnapshotFn,
        *,
        timeout_ms: Optional[float] = None,
        recheck_ms: Optional[float] = None,
    ) -> VerificationRecord:
        first_deadline_ms = timeout_ms if timeout_ms is not None else self.verify_config.verify_timeout_ms
        recheck_deadline_ms = recheck_ms if recheck_ms is not None else self.verify_config.verify_recheck_ms
        max_frames = max(1, int(self.verify_config.max_post_action_frames))
        poll_interval = max(0.0, (first_deadline_ms / 1000.0) / max_frames)

        started = self._clock()
        first_limit = started + first_deadline_ms / 1000.0
        final_limit = started + max(first_deadline_ms, recheck_deadline_ms) / 1000.0

        record = self.compare(intent, pre, pre)
        attempts = 0
        while attempts < max_frames:
            post = snapshot_fn()
            attempts += 1
            self.stats["polls"] += 1
            record = self.compare(intent, pre, post, attempts=attempts)
            record.elapsed_ms = round((self._clock() - started) * 1000.0, 2)
            if record.passed:
                break
            now = self._clock()
            if now >= final_limit:
                break
            self._sleep(poll_interval if now < first_limit else poll_interval)

        record.attempts = max(1, attempts)
        record.elapsed_ms = round((self._clock() - started) * 1000.0, 2)
        self._record(intent, record)
        return record

    # -- pure comparison ---------------------------------------------------- #
    def compare(
        self,
        intent: Intent,
        pre: VerificationSnapshot,
        post: VerificationSnapshot,
        *,
        attempts: int = 1,
    ) -> VerificationRecord:
        effect = intent.expected_effect.type
        region = intent.expected_effect.region or intent.target_box
        change_pct = self.region_change_pct(pre.frame, post.frame, region)
        evidence: List[str] = [f"region change {change_pct:.1f}% (min {self._min_change(effect):.1f}%)"]
        marker_before = self._marker(intent, pre.perception)
        marker_after = self._marker(intent, post.perception)
        transition = self._transition(pre.perception, post.perception)
        passed = False

        if effect is ExpectedEffectType.NONE:
            passed = True
            evidence.append("no effect declared (non state-changing action)")
        elif effect is ExpectedEffectType.SELECTION_CHANGED:
            passed, extra = self._check_selection(intent, pre, post, change_pct, marker_before, marker_after)
            evidence.extend(extra)
        elif effect is ExpectedEffectType.NAVIGATION:
            passed, extra = self._check_navigation(intent, pre, post, change_pct, transition)
            evidence.extend(extra)
        elif effect is ExpectedEffectType.CONTENT_SHIFT:
            threshold = self._min_change(effect)
            passed = change_pct >= threshold
            evidence.append(
                f"content shift {'confirmed' if passed else 'not observed'} in {region or 'the full frame'}"
            )
        elif effect is ExpectedEffectType.DISMISS_OVERLAY:
            handle = intent.expected_effect.target_handle or intent.handle
            before = [o for o in pre.perception.overlays if handle in (o.handle, (o.close_btn.handle if o.close_btn else None))]
            after = [o for o in post.perception.overlays if handle in (o.handle, (o.close_btn.handle if o.close_btn else None))]
            still_open = any(o.kind.value not in {"toast"} for o in post.perception.overlays)
            passed = bool(before) and not after and not still_open
            evidence.append(
                f"overlay {handle!r}: {len(before)} before -> {len(after)} after"
                + (f", {len(post.perception.overlays)} overlay(s) still on screen" if still_open else "")
            )
        else:  # pragma: no cover - enum is exhaustive
            evidence.append(f"unknown expected effect {effect!r}")

        return VerificationRecord(
            intent_id=intent.intent_id,
            passed=bool(passed),
            evidence=evidence,
            pre_frame_seq=pre.frame.seq,
            post_frame_seq=post.frame.seq,
            region_change_pct=round(float(change_pct), 3),
            marker_before=marker_before,
            marker_after=marker_after,
            elapsed_ms=0.0,
            attempts=attempts,
            expected_effect_type=effect,
            transition=transition,
        )

    # -- effect checks ------------------------------------------------------ #
    def _check_selection(
        self,
        intent: Intent,
        pre: VerificationSnapshot,
        post: VerificationSnapshot,
        change_pct: float,
        marker_before: Optional[SelectedMarker],
        marker_after: Optional[SelectedMarker],
    ) -> Tuple[bool, List[str]]:
        evidence: List[str] = []
        handle = intent.expected_effect.target_handle or intent.handle or ""
        expected = intent.expected_effect.expected_marker
        option_after = self._option(post.perception, handle)
        option_before = self._option(pre.perception, handle)

        # Auto-advancing platforms (FR-7.8.4) consume the click and immediately
        # show the *next* question, so the selected marker is never observable.
        # The screen transition is then the evidence: the option we bound to was
        # clicked and the platform moved on.  Binding already proved the click
        # landed inside that option's hit box in the acting frame (AC-14.2), so
        # this cannot be confused with a stray click on a navigation control.
        if self._transition(pre.perception, post.perception) is ScreenTransition.NEW_QUESTION:
            evidence.append(
                f"platform auto-advanced to a new question after clicking {handle!r}; "
                "the transition itself verifies the selection"
            )
            return True, evidence

        if option_after is None:
            evidence.append(f"target {handle!r} is not present in the post-action perception")
            if change_pct >= self._min_change(ExpectedEffectType.SELECTION_CHANGED):
                evidence.append("falling back to pixel evidence only (element not re-found)")
                return True, evidence
            return False, evidence

        marker_changed = (marker_before or SelectedMarker.NONE) != (marker_after or SelectedMarker.NONE)
        selected_now = marker_after is not None and marker_after != SelectedMarker.NONE
        expected_ok = expected is None or marker_after == expected

        # radio exclusivity: no other option may still be selected
        others = [
            o
            for o in post.perception.options
            if o.handle != option_after.handle and o.selected_marker != SelectedMarker.NONE
        ]
        exclusive_ok = not others or (marker_after == SelectedMarker.CHECK)

        evidence.append(
            f"element {handle!r}: marker {marker_before.value if marker_before else 'none'} -> "
            f"{marker_after.value if marker_after else 'none'}"
            + (f" (expected {expected.value})" if expected is not None else "")
        )
        if others and marker_after == SelectedMarker.DOT:
            evidence.append(f"radio exclusivity violated: {len(others)} other option(s) still marked selected")
        if option_before is not None and option_before.selected_marker == marker_after and marker_after != SelectedMarker.NONE:
            evidence.append("option was already selected before the action (idempotent click)")

        passed = selected_now and expected_ok and (marker_changed or _already_selected(evidence)) and exclusive_ok
        if self.verify_config.element_aware is False:
            passed = change_pct >= self._min_change(ExpectedEffectType.SELECTION_CHANGED)
            evidence.append("element_aware=false: pixel change is the only accepted evidence")
        return passed, evidence

    def _check_navigation(
        self,
        intent: Intent,
        pre: VerificationSnapshot,
        post: VerificationSnapshot,
        change_pct: float,
        transition: Optional[ScreenTransition],
    ) -> Tuple[bool, List[str]]:
        evidence: List[str] = []
        before_text = (pre.perception.question_text or "").strip()
        after_text = (post.perception.question_text or "").strip()
        progress_before = pre.perception.navigation.progress_current
        progress_after = post.perception.navigation.progress_current
        end_state = bool(post.perception.end_state_evidence)

        if before_text and after_text and before_text != after_text:
            evidence.append(f"question text changed ({before_text[:32]!r} -> {after_text[:32]!r})")
            return True, evidence
        if progress_before is not None and progress_after is not None and progress_after != progress_before:
            evidence.append(f"progress advanced {progress_before} -> {progress_after}")
            return True, evidence
        if end_state:
            evidence.append(f"end-state evidence appeared: {post.perception.end_state_evidence[0][:60]!r}")
            return True, evidence
        if not after_text and before_text:
            evidence.append("question region disappeared after navigation")
            return True, evidence
        threshold = self._min_change(ExpectedEffectType.NAVIGATION)
        if change_pct >= threshold:
            evidence.append(f"question region repainted ({change_pct:.1f}% >= {threshold:.1f}%)")
            return True, evidence
        evidence.append("navigation had no observable effect (same question, same progress, no repaint)")
        return False, evidence

    # -- pixel evidence ----------------------------------------------------- #
    def region_change_pct(self, pre: Frame, post: Frame, region: Optional[Box] = None) -> float:
        """Percentage of pixels in ``region`` that changed beyond the threshold."""
        if pre.pixels is None or post.pixels is None:
            return 0.0
        try:
            import numpy as np
        except ImportError:  # pragma: no cover
            return 0.0
        a = np.asarray(pre.pixels)
        b = np.asarray(post.pixels)
        if a.ndim == 2:
            a = a[:, :, None]
        if b.ndim == 2:
            b = b[:, :, None]
        height = min(a.shape[0], b.shape[0])
        width = min(a.shape[1], b.shape[1])
        if height <= 0 or width <= 0:
            return 0.0
        a = a[:height, :width, :3].astype("int16")
        b = b[:height, :width, :3].astype("int16")
        if region is not None:
            left, top, w, h = box_clip(region, width, height)
            if w <= 0 or h <= 0:
                return 0.0
            a = a[top : top + h, left : left + w]
            b = b[top : top + h, left : left + w]
        if a.size == 0:
            return 0.0
        diff = np.abs(a - b).max(axis=2)
        changed = int((diff > int(self.verify_config.pixel_diff_threshold)).sum())
        return 100.0 * changed / float(diff.size)

    def frame_diff_pct(self, pre: Frame, post: Frame) -> float:
        return self.region_change_pct(pre, post, None)

    # -- helpers ------------------------------------------------------------ #
    def _min_change(self, effect: ExpectedEffectType) -> float:
        base = float(self.verify_config.min_region_change_pct)
        if effect is ExpectedEffectType.SELECTION_CHANGED:
            return max(0.5, base / 5.0)  # a marker glyph changes few pixels
        return base

    @staticmethod
    def _option(perception: PerceptionResult, handle: str) -> Optional[Any]:
        if not handle:
            return None
        option = perception.option_by_handle(handle)
        if option is not None:
            return option
        from ..binding.module import handle_index

        index = handle_index(handle)
        return perception.option_by_index(index) if index is not None else None

    def _marker(self, intent: Intent, perception: PerceptionResult) -> Optional[SelectedMarker]:
        handle = intent.expected_effect.target_handle or intent.handle or ""
        option = self._option(perception, handle)
        return option.selected_marker if option is not None else None

    @staticmethod
    def _transition(pre: PerceptionResult, post: PerceptionResult) -> Optional[ScreenTransition]:
        if post.end_state_evidence and not pre.end_state_evidence:
            return ScreenTransition.END_STATE
        if post.overlays and not pre.overlays:
            return ScreenTransition.POPUP
        if (pre.question_text or "").strip() and not (post.question_text or "").strip():
            return ScreenTransition.UNKNOWN
        if (pre.question_text or "").strip() != (post.question_text or "").strip():
            return ScreenTransition.NEW_QUESTION
        return ScreenTransition.SAME_QUESTION

    def _record(self, intent: Intent, record: VerificationRecord) -> None:
        self.records.append(record)
        self.stats["verifications"] += 1
        self.stats["passed" if record.passed else "failed"] += 1
        by_effect = self.stats["by_effect"]
        key = record.expected_effect_type.value
        bucket = by_effect.setdefault(key, {"passed": 0, "failed": 0})
        bucket["passed" if record.passed else "failed"] += 1
        if self.telemetry is not None:
            self.telemetry.event(
                RunEventName.ACTION_VERIFIED if record.passed else RunEventName.ACTION_UNVERIFIED,
                state=State.VERIFYING,
                module="verification",
                latency_ms=record.elapsed_ms,
                correlation_id=intent.correlation_id,
                intent_id=intent.intent_id,
                passed=record.passed,
                effect=record.expected_effect_type.value,
                attempts=record.attempts,
                region_change_pct=record.region_change_pct,
                marker_before=record.marker_before.value if record.marker_before else None,
                marker_after=record.marker_after.value if record.marker_after else None,
                evidence=record.evidence[:4],
            )
            self.telemetry.metrics.inc("verifications_total", label="passed" if record.passed else "failed")

    def describe(self) -> Dict[str, Any]:
        return {"stats": dict(self.stats), "config": self.verify_config.model_dump(mode="json")}


def _already_selected(evidence: Sequence[str]) -> bool:
    return any("already selected before the action" in line for line in evidence)


__all__ = ["VerificationModule", "VerificationSnapshot", "expected_marker_for_style"]
