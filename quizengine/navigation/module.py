"""Navigation Module (PRD section 7.8).

Decides *how* to get to the next question and proves it worked:

* FR-7.8.1 strategy cascade: ``next_button`` -> ``auto_advance`` ->
  ``scroll_reveal`` -> ``keyboard`` (keyboard is opt-in, FR-7.8.1.d)
* FR-7.8.2 scroll-exhausted detection (no more motion -> stop scrolling)
* FR-7.8.3 stuck detection (same question after navigation)
* FR-7.8.4 end-state confirmation over N consecutive frames

Blocking overlays are dismissed before any navigation attempt: a modal that is
never closed makes every later check meaningless.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..config import EngineConfig
from ..contracts import (
    ActionType,
    ExpectedEffect,
    ExpectedEffectType,
    Intent,
    OverlayKind,
    PerceptionResult,
    ScreenTransition,
)
from ..geometry import Box

_BLOCKING = {OverlayKind.MODAL, OverlayKind.COOKIE_BANNER, OverlayKind.UNKNOWN}


@dataclass
class NavigationPlan:
    strategy: str
    reason: str
    intent: Optional[Intent] = None
    end_state: bool = False
    evidence: List[str] = field(default_factory=list)
    wait_ms: int = 0

    def describe(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "reason": self.reason,
            "end_state": self.end_state,
            "wait_ms": self.wait_ms,
            "intent": (
                {
                    "intent_id": self.intent.intent_id,
                    "action": self.intent.action.value,
                    "handle": self.intent.handle,
                    "target_box": list(self.intent.target_box) if self.intent.target_box else None,
                    "effect": self.intent.expected_effect.type.value,
                }
                if self.intent is not None
                else None
            ),
            "evidence": list(self.evidence),
        }


class NavigationModule:
    def __init__(
        self,
        config: EngineConfig,
        *,
        telemetry: Any = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.config = config
        self.nav_config = config.navigation
        self.telemetry = telemetry
        self._clock = clock
        self.stats: Dict[str, Any] = {
            "plans": 0,
            "by_strategy": {},
            "end_states": 0,
            "stuck": 0,
            "scrolls": 0,
            "auto_advances": 0,
        }

    # -- planning ----------------------------------------------------------- #
    def plan(
        self,
        perception: PerceptionResult,
        *,
        frame_seq: int,
        frame_size: Optional[Box] = None,
        correlation_id: Optional[str] = None,
        scroll_steps: int = 0,
        selection_pending: bool = False,
    ) -> NavigationPlan:
        self.stats["plans"] += 1
        evidence = self.end_state_evidence(perception)
        if self.end_state_candidate(perception):
            self.stats["end_states"] += 1
            return self._tally(
                NavigationPlan(
                    strategy="end_state",
                    reason="no question region and end-state evidence on screen",
                    intent=None,
                    end_state=True,
                    evidence=evidence,
                )
            )

        overlay_intent = self.dismiss_overlay_intent(perception, frame_seq=frame_seq, correlation_id=correlation_id)
        if overlay_intent is not None:
            return self._tally(
                NavigationPlan(
                    strategy="dismiss_overlay",
                    reason="a blocking overlay must be dismissed before navigating",
                    intent=overlay_intent,
                    evidence=evidence,
                )
            )

        for strategy in self.nav_config.strategy_cascade:
            if strategy == "next_button":
                intent = self._next_button_intent(perception, frame_seq=frame_seq, correlation_id=correlation_id)
                if intent is not None:
                    return self._tally(
                        NavigationPlan(
                            strategy="next_button",
                            reason="enabled Next/Submit control found in this frame",
                            intent=intent,
                            evidence=evidence,
                        )
                    )
            elif strategy == "auto_advance":
                if self._auto_advance_likely(perception, selection_pending):
                    self.stats["auto_advances"] += 1
                    return self._tally(
                        NavigationPlan(
                            strategy="auto_advance",
                            reason="no navigation control: the platform advances on selection, waiting and re-perceiving",
                            intent=None,
                            wait_ms=int(self.nav_config.auto_advance_wait_ms),
                            evidence=evidence,
                        )
                    )
            elif strategy == "scroll_reveal":
                intent = self._scroll_intent(
                    perception,
                    frame_seq=frame_seq,
                    frame_size=frame_size,
                    correlation_id=correlation_id,
                    scroll_steps=scroll_steps,
                )
                if intent is not None:
                    self.stats["scrolls"] += 1
                    return self._tally(
                        NavigationPlan(
                            strategy="scroll_reveal",
                            reason="next control or remaining options are below the fold",
                            intent=intent,
                            evidence=evidence,
                        )
                    )
            elif strategy == "keyboard":
                if not self.nav_config.keyboard_nav_enabled:
                    continue
                keys = list(self.nav_config.keyboard_keys) or ["enter"]
                key = keys[min(scroll_steps, len(keys) - 1)]
                return self._tally(
                    NavigationPlan(
                        strategy="keyboard",
                        reason=f"opt-in keyboard navigation (FR-7.8.1.d): pressing {key!r}",
                        intent=self._intent(
                            ActionType.KEY,
                            handle=f"key_{key}",
                            target_box=None,
                            effect=ExpectedEffectType.NAVIGATION,
                            frame_seq=frame_seq,
                            correlation_id=correlation_id,
                            key_name=key,
                        ),
                        evidence=evidence,
                    )
                )

        return self._tally(
            NavigationPlan(
                strategy="none",
                reason="no navigation strategy applies to this screen",
                intent=None,
                evidence=evidence,
            )
        )

    # -- strategies --------------------------------------------------------- #
    def _next_button_intent(
        self, perception: PerceptionResult, *, frame_seq: int, correlation_id: Optional[str]
    ) -> Optional[Intent]:
        for button in (perception.navigation.next_btn, perception.navigation.submit_btn):
            if button is None or not button.enabled:
                continue
            return self._intent(
                ActionType.CLICK,
                handle=button.handle,
                target_box=button.box,
                effect=ExpectedEffectType.NAVIGATION,
                frame_seq=frame_seq,
                correlation_id=correlation_id,
            )
        return None

    @staticmethod
    def _auto_advance_likely(perception: PerceptionResult, selection_pending: bool) -> bool:
        """No nav control at all -> the platform advances by itself."""
        nav = perception.navigation
        return nav.next_btn is None and nav.submit_btn is None and selection_pending

    def _scroll_intent(
        self,
        perception: PerceptionResult,
        *,
        frame_seq: int,
        frame_size: Optional[Box],
        correlation_id: Optional[str],
        scroll_steps: int,
    ) -> Optional[Intent]:
        if scroll_steps >= self.nav_config.scroll_max_steps:
            return None
        height = frame_size[3] if frame_size else None
        needs_scroll = False
        if height is not None:
            for option in perception.options:
                bottom = option.hit_box[1] + option.hit_box[3]
                if bottom >= height - 4:
                    needs_scroll = True
                    break
            for button in (perception.navigation.next_btn, perception.navigation.submit_btn):
                if button is not None and not button.enabled:
                    needs_scroll = True
        if not needs_scroll:
            return None
        region = _options_region(perception, frame_size)
        return self._intent(
            ActionType.SCROLL,
            handle="scroll_reveal",
            target_box=region,
            effect=ExpectedEffectType.CONTENT_SHIFT,
            frame_seq=frame_seq,
            correlation_id=correlation_id,
            scroll_delta=-int(self.nav_config.scroll_increment_px),
        )

    def dismiss_overlay_intent(
        self, perception: PerceptionResult, *, frame_seq: int, correlation_id: Optional[str]
    ) -> Optional[Intent]:
        for overlay in perception.overlays:
            if overlay.kind not in _BLOCKING:
                continue
            if overlay.close_btn is not None:
                return self._intent(
                    ActionType.CLICK,
                    handle=overlay.close_btn.handle or "overlay_close",
                    target_box=overlay.close_btn.box,
                    effect=ExpectedEffectType.DISMISS_OVERLAY,
                    frame_seq=frame_seq,
                    correlation_id=correlation_id,
                )
        return None

    # -- end state (FR-7.8.4) ---------------------------------------------- #
    def end_state_evidence(self, perception: PerceptionResult) -> List[str]:
        evidence: List[str] = list(perception.end_state_evidence)
        haystack = " ".join(
            [perception.question_text or ""] + [b.text for b in perception.text_blocks]
        ).lower()
        for keyword in self.nav_config.end_state_keywords:
            if keyword.lower() in haystack:
                evidence.append(f"keyword {keyword!r} present")
        progress = perception.navigation
        if (
            progress.progress_current is not None
            and progress.progress_total is not None
            and progress.progress_current >= progress.progress_total
        ):
            evidence.append(f"progress {progress.progress_current}/{progress.progress_total} complete")
        if not perception.has_question_like_region() and not perception.options:
            evidence.append("no question-like region and no options on screen")
        # de-duplicate while preserving order
        seen: set = set()
        unique: List[str] = []
        for item in evidence:
            if item not in seen:
                seen.add(item)
                unique.append(item)
        return unique

    def end_state_candidate(self, perception: PerceptionResult) -> bool:
        """Does this screen look terminal?

        A results page often still contains text rows that Tier 1 reads as
        "options" (a score breakdown, a review list).  What makes a screen
        terminal is that there is **no way to answer** -- no enabled Next/Submit
        control -- plus explicit end evidence (keyword, completed progress, or a
        genuinely empty question region).
        """
        evidence = self.end_state_evidence(perception)
        if not evidence:
            return False
        if not perception.has_question_like_region():
            return True
        actionable = (
            perception.navigation.next_btn is not None or perception.navigation.submit_btn is not None
        )
        # Tier 1 phrases these as "end-state keyword 'x' present" / "progress 6/6 complete".
        strong = any(("keyword" in item or "progress" in item) for item in evidence)
        return bool(strong and not actionable)

    def is_end_state(self, perception: PerceptionResult, streak: int) -> bool:
        """FR-7.8.4: N consecutive frames agreeing the quiz is over."""
        if streak < max(1, int(self.nav_config.end_state_required_frames)):
            return False
        return self.end_state_candidate(perception)

    # -- stuck / auto-advance detection ------------------------------------ #
    def detect_stuck(
        self,
        previous: Optional[PerceptionResult],
        current: PerceptionResult,
        *,
        navigations: int,
        same_question: bool,
    ) -> bool:
        """FR-7.8.3: navigation happened but the screen did not."""
        if navigations <= 0:
            return False
        if not same_question:
            return False
        if previous is not None and previous.frame_seq == current.frame_seq:
            return True
        return True

    def detect_auto_advance(self, pre: PerceptionResult, post: PerceptionResult) -> bool:
        """The platform moved on without any navigation intent."""
        if post.end_state_evidence and not pre.end_state_evidence:
            return True
        return (pre.question_text or "").strip() != (post.question_text or "").strip()

    def scroll_exhausted(self, change_pct: float) -> bool:
        """FR-7.8.2: scrolling no longer moves content."""
        return change_pct < float(self.nav_config.min_scroll_motion_pct)

    def transition(self, pre: PerceptionResult, post: PerceptionResult) -> ScreenTransition:
        if post.end_state_evidence and not pre.end_state_evidence:
            return ScreenTransition.END_STATE
        if post.overlays and not pre.overlays:
            return ScreenTransition.POPUP
        if (pre.question_text or "").strip() == (post.question_text or "").strip():
            return ScreenTransition.SAME_QUESTION
        return ScreenTransition.NEW_QUESTION

    # -- helpers ------------------------------------------------------------ #
    def _intent(
        self,
        action: ActionType,
        *,
        handle: str,
        target_box: Optional[Box],
        effect: ExpectedEffectType,
        frame_seq: int,
        correlation_id: Optional[str],
        scroll_delta: Optional[int] = None,
        key_name: Optional[str] = None,
    ) -> Intent:
        return Intent(
            intent_id=f"intent-{uuid.uuid4().hex[:10]}",
            action=action,
            handle=handle,
            target_box=target_box,
            expected_effect=ExpectedEffect(
                type=effect,
                target_handle=handle,
                region=target_box,
                expected_transition=(
                    ScreenTransition.NEW_QUESTION
                    if effect is ExpectedEffectType.NAVIGATION
                    else None
                ),
            ),
            max_wait_ms=int(self.config.verification.verify_recheck_ms),
            verify="standard",
            frame_seq=frame_seq,
            correlation_id=correlation_id,
            scroll_delta=scroll_delta,
            key_name=key_name,
        )

    def _tally(self, plan: NavigationPlan) -> NavigationPlan:
        bucket = self.stats["by_strategy"].setdefault(plan.strategy, 0)
        self.stats["by_strategy"][plan.strategy] = bucket + 1
        return plan

    def describe(self) -> Dict[str, Any]:
        return {"stats": dict(self.stats), "config": self.nav_config.model_dump(mode="json")}


def _options_region(perception: PerceptionResult, frame_size: Optional[Box]) -> Optional[Box]:
    from ..geometry import box_union_all

    boxes = [o.hit_box for o in perception.options]
    union = box_union_all(boxes) if boxes else None
    if union is None:
        return frame_size
    return union


__all__ = ["NavigationModule", "NavigationPlan"]
