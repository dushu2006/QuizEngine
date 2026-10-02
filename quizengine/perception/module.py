"""Perception Module (PRD section 7.2) -- the two-tier pipeline.

Tier 1 runs on every frame; Tier 2 runs when Tier 1 is ambiguous, when the layout
changes, and at least once per question (``perception.tier2_trigger``).  Results
are reconciled per FR-7.2.7 and the outcome carries everything the confidence
module needs (section 7.5).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..config import EngineConfig
from ..contracts import (
    FailureCode,
    Frame,
    LayoutType,
    PerceptionResult,
    RunEventName,
    State,
)
from ..failures import FailureSignal
from ..geometry import Box, box_union_all
from ..models.provider import ModelProvider
from .ocr.base import NullOCR, OCREngine
from .ocr import build_ocr_engine
from .reconcile import Reconciler, ReconciliationResult
from .tier1 import Tier1Perception, Tier1Result
from .tier2 import Tier2Outcome, Tier2Perception


@dataclass
class PerceptionOutcome:
    perception: PerceptionResult
    tier1: Tier1Result
    tier2: Optional[Tier2Outcome]
    reconciliation: ReconciliationResult
    ambiguity: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    agreement: float = 1.0
    tier1_budget_exceeded: bool = False
    zoomed: bool = False

    @property
    def option_count(self) -> int:
        return len(self.perception.options)

    @property
    def confidence(self) -> float:
        """Perception-level confidence (content x structure x agreement)."""
        base = 0.55 * self.perception.tier1_confidence + 0.45 * self.agreement
        if self.tier2 is not None and self.perception.tier2_confidence is not None:
            base = 0.45 * base + 0.35 * float(self.perception.tier2_confidence) + 0.20 * self.agreement
        return float(max(0.0, min(1.0, base)))

    def describe(self) -> Dict[str, Any]:
        return {
            "option_count": self.option_count,
            "layout_type": self.perception.layout_type.value,
            "tier1_confidence": round(self.perception.tier1_confidence, 3),
            "tier2_used": self.perception.tier2_used,
            "agreement": round(self.agreement, 3),
            "confidence": round(self.confidence, 3),
            "ambiguity": list(self.ambiguity),
            "flags": list(self.perception.reconciliation_flags),
            "latency_ms": round(self.latency_ms, 2),
            "zoomed": self.zoomed,
        }


class PerceptionModule:
    def __init__(
        self,
        config: EngineConfig,
        *,
        ocr: Optional[OCREngine] = None,
        tier2_provider: Optional[ModelProvider] = None,
        capture: Any = None,
        telemetry: Any = None,
        clock: Callable[[], float] = time.perf_counter,
        end_state_keywords: Optional[Sequence[str]] = None,
    ) -> None:
        self.config = config
        self.telemetry = telemetry
        self._clock = clock
        self.capture = capture
        notes: List[str] = []
        self.ocr = ocr if ocr is not None else _build_ocr(config, notes)
        self.ocr_notes = notes
        self.tier1 = Tier1Perception(
            config.perception,
            self.ocr,
            end_state_keywords=end_state_keywords if end_state_keywords is not None else config.navigation.end_state_keywords,
            clock=clock,
        )
        self.tier2: Optional[Tier2Perception] = (
            Tier2Perception(tier2_provider, config.perception, telemetry=telemetry, clock=clock)
            if tier2_provider is not None
            else None
        )
        self.reconciler = Reconciler(config.perception)
        self._last_layout: Optional[LayoutType] = None
        self._last_perception: Optional[PerceptionResult] = None
        self._questions_since_tier2 = 0
        self.stats: Dict[str, Any] = {
            "frames": 0,
            "tier2_runs": 0,
            "tier1_budget_exceeded": 0,
            "low_confidence": 0,
            "zoomed_reperceptions": 0,
        }

    # -- main entry --------------------------------------------------------- #
    def perceive(
        self,
        frame: Frame,
        *,
        force_tier2: bool = False,
        correlation_id: Optional[str] = None,
        new_question: bool = False,
    ) -> PerceptionOutcome:
        started = self._clock()
        self.stats["frames"] += 1
        if new_question:
            self._questions_since_tier2 += 1

        tier1 = self.tier1.analyze(frame)
        if tier1.latency_ms > self.config.perception.tier1_budget_ms:
            self.stats["tier1_budget_exceeded"] += 1
            self._warn(
                f"Tier-1 perception took {tier1.latency_ms:.0f}ms "
                f"(budget {self.config.perception.tier1_budget_ms:.0f}ms)",
                correlation_id=correlation_id,
            )
        if self.telemetry is not None:
            self.telemetry.observe_latency("tier1_latency_ms", tier1.latency_ms)

        layout_changed = self._last_layout is not None and tier1.perception.layout_type != self._last_layout
        run_tier2 = force_tier2 or self._should_run_tier2(tier1.ambiguity, layout_changed)

        tier2_outcome: Optional[Tier2Outcome] = None
        tier2_analysis = None
        if run_tier2:
            if self.tier2 is None:
                tier1.ambiguity.append("tier2 unavailable (no model provider configured)")
            else:
                try:
                    tier2_outcome = self.tier2.analyze(
                        frame, tier1.perception, ambiguity=tier1.ambiguity, correlation_id=correlation_id
                    )
                    tier2_analysis = tier2_outcome.analysis
                    self.stats["tier2_runs"] += 1
                    self._questions_since_tier2 = 0
                    if self.telemetry is not None:
                        self.telemetry.metrics.inc("tier2_invocations")
                except FailureSignal as signal:
                    # FR-7.2.6: schema retries exhausted -> PERCEPTION_LOW_CONFIDENCE.
                    if signal.code == FailureCode.MODEL_SCHEMA_FAILURE and self.telemetry is not None:
                        self.telemetry.metrics.inc("tier2_schema_failures")
                    tier1.ambiguity.append(f"tier2 failed: {signal.code.value}: {signal.message[:120]}")
                    raise FailureSignal(
                        FailureCode.PERCEPTION_LOW_CONFIDENCE,
                        f"Tier-2 perception failed: {signal.message}",
                        origin_state=State.PERCEIVING,
                        detail={"cause": signal.code.value, "ambiguity": list(tier1.ambiguity)},
                    ) from signal

        reconciliation = self.reconciler.reconcile(tier1.perception, tier2_analysis)
        self._last_layout = reconciliation.perception.layout_type
        self._last_perception = reconciliation.perception

        ambiguity = list(tier1.ambiguity)
        ambiguity.extend(flag for flag in reconciliation.flags if flag.startswith("disagree"))
        if ambiguity:
            self.stats["low_confidence"] += 1

        latency_ms = (self._clock() - started) * 1000.0
        outcome = PerceptionOutcome(
            perception=reconciliation.perception,
            tier1=tier1,
            tier2=tier2_outcome,
            reconciliation=reconciliation,
            ambiguity=ambiguity,
            latency_ms=latency_ms,
            agreement=reconciliation.agreement,
            tier1_budget_exceeded=tier1.latency_ms > self.config.perception.tier1_budget_ms,
        )
        if self.telemetry is not None:
            self.telemetry.event(
                RunEventName.PERCEPTION_DONE,
                state=State.PERCEIVING,
                module="perception",
                latency_ms=latency_ms,
                correlation_id=correlation_id,
                confidence=outcome.confidence,
                options=outcome.option_count,
                layout=outcome.perception.layout_type.value,
                tier2=outcome.perception.tier2_used,
                ambiguity=len(ambiguity),
            )
        return outcome

    # -- zoomed re-perception (recovery for PERCEPTION_LOW_CONFIDENCE) ------- #
    def perceive_zoomed(
        self,
        frame: Frame,
        *,
        correlation_id: Optional[str] = None,
        region: Optional[Box] = None,
    ) -> PerceptionOutcome:
        """Re-perceive using a x2 zoomed crop of the question/option region.

        This is the section 11 recovery action for ``PERCEPTION_LOW_CONF``:
        small or low-contrast text becomes legible, then results are mapped back
        into the original frame's coordinate space.
        """
        if self.capture is None:
            raise FailureSignal(
                FailureCode.PERCEPTION_LOW_CONFIDENCE,
                "zoomed re-perception needs a capture module",
                origin_state=State.PERCEIVING,
            )
        self.stats["zoomed_reperceptions"] += 1
        target = region or self._content_region(frame)
        zoom_frame = self.capture.capture_zoom_window(target, correlation_id=correlation_id)
        scale = float(zoom_frame.backend_meta.get("scale", self.config.capture.zoom_crop_scale))
        origin = zoom_frame.backend_meta.get("source_box") or list(target)

        tier1 = self.tier1.analyze(zoom_frame)
        _translate(tier1.perception, origin, scale)
        tier1.ambiguity.append("zoomed re-perception")

        tier2_outcome: Optional[Tier2Outcome] = None
        tier2_analysis = None
        if self.tier2 is not None:
            try:
                tier2_outcome = self.tier2.analyze(
                    zoom_frame, tier1.perception, ambiguity=tier1.ambiguity, correlation_id=correlation_id
                )
                tier2_analysis = tier2_outcome.analysis
                _translate_tier2(tier2_analysis, origin, scale)
            except FailureSignal:
                tier2_analysis = None

        reconciliation = self.reconciler.reconcile(tier1.perception, tier2_analysis)
        outcome = PerceptionOutcome(
            perception=reconciliation.perception,
            tier1=tier1,
            tier2=tier2_outcome,
            reconciliation=reconciliation,
            ambiguity=list(tier1.ambiguity),
            latency_ms=0.0,
            agreement=reconciliation.agreement,
            zoomed=True,
        )
        if self.telemetry is not None:
            self.telemetry.event(
                RunEventName.PERCEPTION_DONE,
                state=State.PERCEIVING,
                module="perception",
                correlation_id=correlation_id,
                confidence=outcome.confidence,
                zoomed=True,
                options=outcome.option_count,
            )
        return outcome

    def _content_region(self, frame: Frame) -> Box:
        """Best guess at where the content is, for the zoom crop."""
        previous = self._last_perception
        boxes: List[Box] = []
        if previous is not None:
            if previous.question_region:
                boxes.append(previous.question_region)
            boxes.extend(o.hit_box for o in previous.options)
        if not boxes:
            width, height = frame.width, frame.height
            return (int(width * 0.05), int(height * 0.1), int(width * 0.9), int(height * 0.7))
        return box_union_all(boxes) or frame.full_box

    # -- tier-2 trigger ------------------------------------------------------ #
    def _should_run_tier2(self, ambiguity: Sequence[str], layout_changed: bool) -> bool:
        if self.tier2 is None:
            return False
        return self.tier2.should_run(
            tier1_ambiguity=ambiguity,
            layout_changed=layout_changed,
            questions_since_last=self._questions_since_tier2,
        )

    def note_question_seen(self) -> None:
        """Called by the orchestrator when a new validated question starts."""
        self._questions_since_tier2 += 1

    def reset_question_counter(self) -> None:
        self._questions_since_tier2 = 0

    # -- misc ---------------------------------------------------------------- #
    @property
    def last_layout(self) -> Optional[LayoutType]:
        return self._last_layout

    def describe(self) -> Dict[str, Any]:
        return {
            "ocr": self.ocr.describe(),
            "ocr_resolution": list(self.ocr_notes),
            "tier2": self.tier2.describe() if self.tier2 else None,
            "stats": dict(self.stats),
            "last_layout": self._last_layout.value if self._last_layout else None,
        }

    def _warn(self, message: str, *, correlation_id: Optional[str] = None) -> None:
        if self.telemetry is None:
            return
        self.telemetry.event(
            RunEventName.PERCEPTION_LOW_CONFIDENCE,
            state=State.PERCEIVING,
            module="perception",
            correlation_id=correlation_id,
            message=message,
        )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _build_ocr(config: EngineConfig, notes: List[str]) -> OCREngine:
    """Resolve the OCR engine, degrading loudly rather than silently."""
    try:
        engine, trail = build_ocr_engine(config.perception, config.capture.backend, notes=notes)
        return engine
    except Exception as exc:
        notes.append(f"ocr resolution failed: {exc}")
        # Never crash at construction: Tier 2 can still carry structure, and the
        # orchestrator will surface PERCEPTION_LOW_CONFIDENCE on the first frame.
        return NullOCR()


def _translate(perception: PerceptionResult, origin: Sequence[float], scale: float) -> None:
    """Map boxes from zoom-crop space back into the original frame space."""
    dx, dy = float(origin[0]), float(origin[1])
    divisor = scale if scale else 1.0

    def map_box(box: Optional[Box]) -> Optional[Box]:
        if box is None:
            return None
        return (
            int(round(dx + box[0] / divisor)),
            int(round(dy + box[1] / divisor)),
            int(round(box[2] / divisor)),
            int(round(box[3] / divisor)),
        )

    perception.question_region = map_box(perception.question_region)
    for option in perception.options:
        option.hit_box = map_box(option.hit_box) or option.hit_box
        option.text_box = map_box(option.text_box)
    for block in perception.text_blocks:
        block.box = map_box(block.box) or block.box
    for proposal in perception.regions:
        proposal.box = map_box(proposal.box) or proposal.box
    for button in (perception.navigation.next_btn, perception.navigation.prev_btn, perception.navigation.submit_btn):
        if button is not None:
            button.box = map_box(button.box) or button.box
    for overlay in perception.overlays:
        overlay.box = map_box(overlay.box) or overlay.box
        if overlay.close_btn is not None:
            overlay.close_btn.box = map_box(overlay.close_btn.box) or overlay.close_btn.box


def _translate_tier2(analysis: Any, origin: Sequence[float], scale: float) -> None:
    """Same mapping for a Tier-2 analysis (boxes already in crop space)."""
    dx, dy = float(origin[0]), float(origin[1])
    divisor = scale if scale else 1.0

    def shift(box: Optional[Sequence[float]]) -> Optional[List[float]]:
        if box is None:
            return None
        return [dx + box[0] / divisor, dy + box[1] / divisor, box[2] / divisor, box[3] / divisor]

    analysis.question_region = shift(analysis.question_region)
    for option in analysis.options:
        option.hit_box = shift(option.hit_box) or option.hit_box
        option.text_box = shift(option.text_box)
    for button in (analysis.navigation.next_btn, analysis.navigation.prev_btn, analysis.navigation.submit_btn):
        if button is not None:
            button.box = shift(button.box) or button.box
    for overlay in analysis.overlays:
        overlay.box = shift(overlay.box) or overlay.box


__all__ = ["PerceptionModule", "PerceptionOutcome"]
