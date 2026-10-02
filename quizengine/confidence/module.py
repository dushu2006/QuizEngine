"""Confidence & Uncertainty Module (PRD section 7.5).

FR-7.5.1 composite::

    composite = w_ocr * ocr + w_agree * agreement + w_solver * solver - penalties

FR-7.5.2 policy:

* ``composite >= high_conf``          -> ACT
* ``low_conf <= composite < high_conf`` -> configured policy (default VERIFY_AGAIN)
* ``composite < low_conf``            -> PAUSE FOR HUMAN (mandatory, **L7**)

The module only consumes section 8 contract types plus two scalars the
orchestrator lifts out of the perception outcome, so ``confidence/`` never
imports ``perception/`` internals (**L5**).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..config import EngineConfig
from ..contracts import (
    ConfidenceBreakdown,
    Decision,
    Overlay,
    OverlayKind,
    PerceptionResult,
    Question,
    UncertaintyPolicy,
)

_BLOCKING = {OverlayKind.MODAL, OverlayKind.CAPTCHA, OverlayKind.HUMAN_VERIFICATION, OverlayKind.UNKNOWN}


@dataclass
class ConfidenceOutcome:
    """What the orchestrator is allowed to do with a decision."""

    breakdown: ConfidenceBreakdown
    policy: UncertaintyPolicy
    act: bool
    reasons: List[str] = field(default_factory=list)
    #: Mid band, first pass: ask for one more perception+solver cycle (FR-7.5.2).
    escalate: bool = False
    #: Composite that a re-verification pass must reach before acting.
    verify_target: float = 0.85
    attempt: int = 1

    @property
    def tier(self) -> str:
        return self.breakdown.tier

    @property
    def composite(self) -> float:
        return self.breakdown.composite

    def describe(self) -> Dict[str, Any]:
        return {
            "composite": round(self.composite, 4),
            "tier": self.tier,
            "policy": self.policy.value,
            "act": self.act,
            "escalate": self.escalate,
            "attempt": self.attempt,
            "verify_target": self.verify_target,
            "reasons": list(self.reasons),
            "components": {
                "ocr": round(self.breakdown.ocr_confidence, 4),
                "agreement": round(self.breakdown.perception_agreement, 4),
                "solver": round(self.breakdown.solver_confidence, 4),
                "penalty": round(self.breakdown.penalty, 4),
            },
        }


class ConfidenceModule:
    def __init__(self, config: EngineConfig, *, telemetry: Any = None) -> None:
        self.config = config
        self.confidence_config = config.confidence
        self.telemetry = telemetry
        self.stats: Dict[str, int] = {"high": 0, "mid": 0, "low": 0, "act": 0, "escalate": 0, "pause": 0}

    # -- main --------------------------------------------------------------- #
    def assess(
        self,
        *,
        question: Question,
        decision: Decision,
        perception: PerceptionResult,
        agreement: float = 1.0,
        reconciliation_penalty: float = 0.0,
        attempt: int = 1,
        consecutive_failures: int = 0,
        image_description: Optional[str] = None,
        overlays: Optional[Sequence[Overlay]] = None,
    ) -> ConfidenceOutcome:
        weights = self.confidence_config.weights
        reasons: List[str] = []

        ocr_confidence = self._ocr_component(question, perception)
        agreement_component = self._agreement_component(agreement, perception)
        solver_confidence = float(decision.confidence)

        weight_sum = weights.ocr + weights.perception_agreement + weights.solver
        raw = (
            weights.ocr * ocr_confidence
            + weights.perception_agreement * agreement_component
            + weights.solver * solver_confidence
        ) / weight_sum if weight_sum > 0 else 0.0

        penalty = float(max(0.0, min(1.0, reconciliation_penalty)))
        if penalty > 0:
            reasons.append(f"tier reconciliation disagreement (-{penalty:.2f})")

        blocking = [o for o in (overlays or perception.overlays) if o.kind in _BLOCKING]
        if blocking:
            penalty = min(1.0, penalty + weights.penalty_overlay)
            reasons.append(f"blocking overlay present ({blocking[0].kind.value}) (-{weights.penalty_overlay:.2f})")
        if ocr_confidence < 0.60:
            penalty = min(1.0, penalty + weights.penalty_low_ocr)
            reasons.append(f"low OCR confidence ({ocr_confidence:.2f}) (-{weights.penalty_low_ocr:.2f})")
        if question.flags.has_image and not image_description:
            penalty = min(1.0, penalty + weights.penalty_low_ocr)
            reasons.append("question contains an image with no visual description (-%.2f)" % weights.penalty_low_ocr)
        if question.flags.low_contrast:
            penalty = min(1.0, penalty + weights.penalty_reconciliation / 2.0)
            reasons.append("low-contrast rendering detected")
        if not question.options:
            penalty = 1.0
            reasons.append("no options bound to the question")

        composite = float(max(0.0, min(1.0, raw - penalty)))
        tier = self._tier(composite)
        self.stats[tier] = self.stats.get(tier, 0) + 1

        # -- hard gates that outrank the band ------------------------------ #
        if decision.option_index >= len(question.options):
            reasons.append(
                f"decision points at option index {decision.option_index} but only "
                f"{len(question.options)} option(s) exist -- binding error, refusing to act"
            )
            return self._build(
                composite, tier, ocr_confidence, agreement_component, solver_confidence, penalty, reasons,
                policy=UncertaintyPolicy.PAUSE_HUMAN, act=False, escalate=False, attempt=attempt,
            )
        if consecutive_failures >= self.config.budgets.max_consecutive_failures:
            reasons.append(
                f"{consecutive_failures} consecutive failures (limit "
                f"{self.config.budgets.max_consecutive_failures}) -- pausing for the operator"
            )
            return self._build(
                composite, tier, ocr_confidence, agreement_component, solver_confidence, penalty, reasons,
                policy=UncertaintyPolicy.PAUSE_HUMAN, act=False, escalate=False, attempt=attempt,
            )

        policy, act, escalate = self._policy(tier, composite, attempt, reasons)
        outcome = self._build(
            composite, tier, ocr_confidence, agreement_component, solver_confidence, penalty, reasons,
            policy=policy, act=act, escalate=escalate, attempt=attempt,
        )
        if policy is UncertaintyPolicy.PAUSE_HUMAN:
            self.stats["pause"] += 1
        elif act:
            self.stats["act"] += 1
        if escalate:
            self.stats["escalate"] += 1
        return outcome

    # -- components --------------------------------------------------------- #
    @staticmethod
    def _ocr_component(question: Question, perception: PerceptionResult) -> float:
        option_conf = [o.text_conf for o in question.options if o.text_conf > 0]
        mean_option = sum(option_conf) / len(option_conf) if option_conf else 0.0
        extraction = float(question.extraction_confidence or 0.0)
        if mean_option <= 0:
            return extraction
        return float(0.6 * extraction + 0.4 * mean_option)

    @staticmethod
    def _agreement_component(agreement: float, perception: PerceptionResult) -> float:
        """Tier-1/Tier-2 agreement blended with Tier-1 structural confidence.

        When Tier 2 never ran there is nothing to disagree with, so ``agreement``
        is 1.0 by construction; blending with ``tier1_confidence`` keeps a
        Tier-1-only run from being treated as perfectly corroborated.
        """
        tier1 = float(perception.tier1_confidence or 0.0)
        return float(max(0.0, min(1.0, 0.5 * float(agreement) + 0.5 * tier1)))

    def _tier(self, composite: float) -> str:
        if composite >= self.confidence_config.high_conf:
            return "high"
        if composite < self.confidence_config.low_conf:
            return "low"
        return "mid"

    def _policy(self, tier: str, composite: float, attempt: int, reasons: List[str]):
        target = self.confidence_config.verify_again_target
        configured = self.confidence_config.uncertainty_policy
        if tier == "high":
            reasons.append(f"composite {composite:.2f} >= high_conf {self.confidence_config.high_conf:.2f}")
            return UncertaintyPolicy.ACT, True, False
        if tier == "low":
            reasons.append(
                f"composite {composite:.2f} < low_conf {self.confidence_config.low_conf:.2f} -> pause for human (L7)"
            )
            if self.confidence_config.allow_low_conf_act:
                reasons.append("confidence.allow_low_conf_act is set (test-only override); acting anyway")
                return UncertaintyPolicy.ACT, True, False
            return UncertaintyPolicy.PAUSE_HUMAN, False, False

        # mid band
        if attempt <= 1 and configured is UncertaintyPolicy.VERIFY_AGAIN:
            reasons.append(
                f"composite {composite:.2f} in the mid band -> re-verify once "
                f"(target {target:.2f}, FR-7.5.2)"
            )
            return UncertaintyPolicy.VERIFY_AGAIN, False, True
        if configured is UncertaintyPolicy.ACT and attempt <= 1:
            reasons.append(f"composite {composite:.2f} in the mid band; configured policy is 'act'")
            return UncertaintyPolicy.ACT, True, False
        if composite >= target:
            reasons.append(f"re-verified composite {composite:.2f} >= target {target:.2f}")
            return UncertaintyPolicy.ACT, True, False
        reasons.append(
            f"re-verification did not reach {target:.2f} (composite {composite:.2f}, attempt {attempt}) "
            "-> pause for human"
        )
        return UncertaintyPolicy.PAUSE_HUMAN, False, False

    def _build(
        self,
        composite: float,
        tier: str,
        ocr_confidence: float,
        agreement: float,
        solver_confidence: float,
        penalty: float,
        reasons: List[str],
        *,
        policy: UncertaintyPolicy,
        act: bool,
        escalate: bool,
        attempt: int,
    ) -> ConfidenceOutcome:
        weights = self.confidence_config.weights
        breakdown = ConfidenceBreakdown(
            ocr_confidence=round(float(ocr_confidence), 4),
            perception_agreement=round(float(agreement), 4),
            solver_confidence=round(float(solver_confidence), 4),
            penalty=round(float(penalty), 4),
            weights={
                "ocr": weights.ocr,
                "perception_agreement": weights.perception_agreement,
                "solver": weights.solver,
            },
            composite=round(float(composite), 4),
            tier=tier,  # type: ignore[arg-type]
            policy=policy,
            reasons=list(reasons),
        )
        return ConfidenceOutcome(
            breakdown=breakdown,
            policy=policy,
            act=act,
            reasons=list(reasons),
            escalate=escalate,
            verify_target=self.confidence_config.verify_again_target,
            attempt=attempt,
        )

    # -- explicit outcomes -------------------------------------------------- #
    def human_decision(self, *, question: Question, decision: Decision) -> ConfidenceOutcome:
        """An operator answer is authoritative: FR-7.5.2 PAUSE_HUMAN resolved.

        The composite stays 1.0 because the human *is* the escalation target; no
        amount of machine evidence outranks it.
        """
        reasons = [f"operator supplied the answer for question {question.ordinal} ({decision.letter})"]
        self.stats["act"] = self.stats.get("act", 0) + 1
        return self._build(
            1.0, "high", 1.0, 1.0, float(decision.confidence), 0.0, reasons,
            policy=UncertaintyPolicy.ACT, act=True, escalate=False, attempt=1,
        )

    def refuse(self, message: str, *, attempt: int = 1) -> ConfidenceOutcome:
        """No usable answer at all -> mandatory pause for the operator (L7)."""
        reasons = [message, "no decision to act on -> pause for human"]
        self.stats["pause"] = self.stats.get("pause", 0) + 1
        return self._build(
            0.0, "low", 0.0, 0.0, 0.0, 0.0, reasons,
            policy=UncertaintyPolicy.PAUSE_HUMAN, act=False, escalate=False, attempt=attempt,
        )

    def describe(self) -> Dict[str, Any]:
        return {"stats": dict(self.stats), "config": self.confidence_config.model_dump(mode="json")}


__all__ = ["ConfidenceModule", "ConfidenceOutcome"]
