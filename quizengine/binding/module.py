"""Element Binding & Re-resolution Module (PRD section 7.7).

**FR-7.7.1**: actions reference *semantic handles* (``opt_2``, ``nav_next``,
``overlay_close``), never coordinates carried across frames.  Coordinates are
re-derived from the perception of the frame that is about to be acted on.

**FR-7.7.2**: immediately before an action the handle is re-resolved.  When the
screen moved, the element is re-found by template match / phase correlation; if
that is not confident enough the resolver escalates to a full re-perception
instead of clicking a stale box (**L1**, AC-14.2).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..capabilities import has
from ..config import EngineConfig
from ..contracts import (
    ElementResolution,
    Frame,
    Intent,
    Overlay,
    PerceptionResult,
)
from ..geometry import Box, box_center, box_contains_point, box_iou, box_union_all

OPTION_HANDLE_PREFIX = "opt_"
NAV_HANDLES = ("nav_next", "nav_prev", "nav_submit", "nav_other")
OVERLAY_CLOSE_HANDLE = "overlay_close"


def option_handle(index: int) -> str:
    return f"{OPTION_HANDLE_PREFIX}{int(index)}"


def handle_index(handle: str) -> Optional[int]:
    """``opt_3`` -> 3 (semantic index binding, FR-7.7.1)."""
    if handle and handle.startswith(OPTION_HANDLE_PREFIX):
        suffix = handle[len(OPTION_HANDLE_PREFIX) :]
        if suffix.isdigit():
            return int(suffix)
    return None


@dataclass
class BindingResult:
    resolution: ElementResolution
    box: Box
    stale: bool = False
    reasons: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.reasons is None:
            self.reasons = []


class ElementResolver:
    def __init__(
        self,
        config: EngineConfig,
        *,
        telemetry: Any = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.config = config
        self.action_config = config.action
        self.telemetry = telemetry
        self._clock = clock
        self.stats: Dict[str, int] = {
            "resolutions": 0,
            "fresh": 0,
            "template_matched": 0,
            "phase_correlated": 0,
            "escalated": 0,
            "unbound": 0,
        }

    # -- handle -> box (FR-7.7.1) ------------------------------------------ #
    def bind_handle(self, handle: Optional[str], perception: PerceptionResult) -> Optional[Box]:
        """Look a semantic handle up in a perception result."""
        if not handle:
            return None
        option = perception.option_by_handle(handle)
        if option is not None:
            return option.hit_box
        index = handle_index(handle)
        if index is not None:
            option = perception.option_by_index(index)
            if option is not None:
                return option.hit_box
        for button in (perception.navigation.next_btn, perception.navigation.prev_btn, perception.navigation.submit_btn):
            if button is not None and button.handle == handle:
                return button.box
        for overlay in perception.overlays:
            if overlay.handle == handle:
                return overlay.box
            if overlay.close_btn is not None and overlay.close_btn.handle == handle:
                return overlay.close_btn.box
            if handle == OVERLAY_CLOSE_HANDLE and overlay.close_btn is not None:
                return overlay.close_btn.box
        return None

    def marker_for(self, handle: Optional[str], perception: PerceptionResult) -> Optional[Any]:
        """Selected-marker state of an option handle (used by verification)."""
        if not handle:
            return None
        option = perception.option_by_handle(handle) or perception.option_by_index(handle_index(handle) or -1)
        return option.selected_marker if option is not None else None

    # -- re-resolution (FR-7.7.2) ------------------------------------------ #
    def resolve(
        self,
        intent: Intent,
        *,
        frame: Frame,
        perception: PerceptionResult,
        template: Optional[Any] = None,
        expected_box: Optional[Box] = None,
    ) -> BindingResult:
        self.stats["resolutions"] += 1
        reasons: List[str] = []
        expected = expected_box if expected_box is not None else intent.target_box
        bound = self.bind_handle(intent.handle, perception)

        fresh = intent.frame_seq < 0 or intent.frame_seq == frame.seq
        if bound is not None and fresh:
            self.stats["fresh"] += 1
            displacement = _displacement(bound, expected)
            reasons.append(f"handle {intent.handle!r} bound in frame {frame.seq} (fresh perception)")
            return BindingResult(
                resolution=ElementResolution(
                    handle=intent.handle or "",
                    resolved_box=bound,
                    match_confidence=1.0,
                    displacement_px=displacement,
                    method="relative_geometry",
                    frame_seq=frame.seq,
                    escalated=False,
                ),
                box=bound,
                stale=False,
                reasons=reasons,
            )

        if bound is not None and not fresh:
            # Handle re-bound in a newer frame: acceptable, but record the drift.
            displacement = _displacement(bound, expected)
            self.stats["fresh"] += 1
            reasons.append(
                f"handle {intent.handle!r} re-bound in frame {frame.seq} "
                f"(intent referenced frame {intent.frame_seq}, drift {displacement:.1f}px)"
            )
            escalated = displacement > self.action_config.rematch_max_displacement_px
            if escalated:
                self.stats["escalated"] += 1
                reasons.append(f"drift exceeds {self.action_config.rematch_max_displacement_px}px -> full re-perception")
            return BindingResult(
                resolution=ElementResolution(
                    handle=intent.handle or "",
                    resolved_box=bound,
                    match_confidence=1.0 if not escalated else 0.5,
                    displacement_px=displacement,
                    method="full_reperception" if escalated else "relative_geometry",
                    frame_seq=frame.seq,
                    escalated=escalated,
                ),
                box=bound,
                stale=escalated,
                reasons=reasons,
            )

        # Handle not present any more: try to re-find it visually.
        if bound is None and expected is not None:
            reasons.append(f"handle {intent.handle!r} absent from frame {frame.seq}")
            match = self._template_match(frame, template, expected)
            if match is not None:
                box, score = match
                displacement = _displacement(box, expected)
                ok_score = score >= self.action_config.rematch_min_confidence
                ok_shift = displacement <= self.action_config.rematch_max_displacement_px
                self.stats["template_matched"] += 1
                reasons.append(f"template match score {score:.2f}, displacement {displacement:.1f}px")
                if ok_score and ok_shift:
                    return BindingResult(
                        resolution=ElementResolution(
                            handle=intent.handle or "",
                            resolved_box=box,
                            match_confidence=round(float(score), 4),
                            displacement_px=displacement,
                            method="template_match",
                            frame_seq=frame.seq,
                            escalated=False,
                        ),
                        box=box,
                        stale=False,
                        reasons=reasons,
                    )
                self.stats["escalated"] += 1
                reasons.append("template match below thresholds -> full re-perception")
                return BindingResult(
                    resolution=ElementResolution(
                        handle=intent.handle or "",
                        resolved_box=box,
                        match_confidence=round(float(score), 4),
                        displacement_px=displacement,
                        method="full_reperception",
                        frame_seq=frame.seq,
                        escalated=True,
                    ),
                    box=box,
                    stale=True,
                    reasons=reasons,
                )

            shift = self._phase_shift(frame, template, expected)
            if shift is not None:
                box, confidence = shift
                self.stats["phase_correlated"] += 1
                reasons.append(f"phase correlation shift, confidence {confidence:.2f}")
                return BindingResult(
                    resolution=ElementResolution(
                        handle=intent.handle or "",
                        resolved_box=box,
                        match_confidence=round(float(confidence), 4),
                        displacement_px=_displacement(box, expected),
                        method="phase_correlation",
                        frame_seq=frame.seq,
                        escalated=confidence < self.action_config.rematch_min_confidence,
                    ),
                    box=box,
                    stale=confidence < self.action_config.rematch_min_confidence,
                    reasons=reasons,
                )

        self.stats["unbound"] += 1
        self.stats["escalated"] += 1
        fallback = expected if expected is not None else _frame_box(frame)
        reasons.append("no visual re-match possible -> full re-perception required")
        return BindingResult(
            resolution=ElementResolution(
                handle=intent.handle or "",
                resolved_box=fallback,
                match_confidence=0.0,
                displacement_px=0.0,
                method="full_reperception",
                frame_seq=frame.seq,
                escalated=True,
            ),
            box=fallback,
            stale=True,
            reasons=reasons,
        )

    # -- visual re-match helpers ------------------------------------------- #
    def _template_match(self, frame: Frame, template: Optional[Any], expected: Box) -> Optional[Tuple[Box, float]]:
        if template is None or frame.pixels is None or not has("opencv"):
            return None
        try:
            import cv2
            import numpy as np

            screen = np.asarray(frame.pixels)
            if screen.ndim == 3:
                screen_gray = cv2.cvtColor(screen[:, :, :3], cv2.COLOR_RGB2GRAY)
            else:
                screen_gray = screen
            tpl = np.asarray(template)
            if tpl.ndim == 3:
                tpl = cv2.cvtColor(tpl[:, :, :3], cv2.COLOR_RGB2GRAY)
            if tpl.shape[0] < 8 or tpl.shape[1] < 8:
                return None
            if tpl.shape[0] > screen_gray.shape[0] or tpl.shape[1] > screen_gray.shape[1]:
                return None
            result = cv2.matchTemplate(screen_gray, tpl, cv2.TM_CCOEFF_NORMED)
            _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(result)
            if max_val <= 0:
                return None
            x, y = int(max_loc[0]), int(max_loc[1])
            box = (x, y, int(tpl.shape[1]), int(tpl.shape[0]))
            return box, float(max(0.0, min(1.0, max_val)))
        except Exception:
            return None

    def _phase_shift(self, frame: Frame, template: Optional[Any], expected: Box) -> Optional[Tuple[Box, float]]:
        """Global shift estimate between the validated frame and the live one."""
        if template is None or frame.pixels is None or not has("opencv"):
            return None
        try:
            import cv2
            import numpy as np

            tpl = np.asarray(template)
            screen = np.asarray(frame.pixels)
            if tpl.ndim == 3:
                tpl = cv2.cvtColor(tpl[:, :, :3], cv2.COLOR_RGB2GRAY)
            if screen.ndim == 3:
                screen = cv2.cvtColor(screen[:, :, :3], cv2.COLOR_RGB2GRAY)
            h = min(tpl.shape[0], screen.shape[0])
            w = min(tpl.shape[1], screen.shape[1])
            if h < 16 or w < 16:
                return None
            window = screen[:h, :w].astype("float32")
            patch = tpl[:h, :w].astype("float32")
            (dx, dy), response = cv2.phaseCorrelate(patch, window)
            if response <= 0:
                return None
            scale = float(self.action_config.rematch_scale_search or 0.2)
            if abs(dx) > w * scale or abs(dy) > h * scale:
                return None
            x, y, bw, bh = expected
            box = (int(x + dx), int(y + dy), bw, bh)
            return box, float(max(0.0, min(1.0, response)))
        except Exception:
            return None

    def describe(self) -> Dict[str, Any]:
        return {"stats": dict(self.stats), "config": self.action_config.model_dump(mode="json")}


def _displacement(box: Optional[Box], expected: Optional[Box]) -> float:
    if box is None or expected is None:
        return 0.0
    cx, cy = box_center(box)
    ex, ey = box_center(expected)
    return float(((cx - ex) ** 2 + (cy - ey) ** 2) ** 0.5)


def _frame_box(frame: Frame) -> Box:
    width, height = frame.size_px
    return (0, 0, int(width), int(height))


__all__ = [
    "NAV_HANDLES",
    "OPTION_HANDLE_PREFIX",
    "OVERLAY_CLOSE_HANDLE",
    "BindingResult",
    "ElementResolver",
    "handle_index",
    "option_handle",
]
