"""Recovery & Self-Healing Module (PRD section 7.11).

Phase 1 is always *re-observe* (FR-7.11.1): no recovery action is taken on the
frame that failed.  Phase 2 picks the handler named by the section 11 matrix.
Phase 3 accounts for the attempt against a bounded budget (FR-7.11.2, **L3**)
and phase 4 either writes an artifact bundle (FR-7.11.3) and escalates, or hands
control to the uncertainty policy.

This module decides *what* to do; the orchestrator owns the handler dispatch
table so ``recovery/`` never imports capture, perception or action (**L5**).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..config import EngineConfig
from ..contracts import (
    BudgetCounters,
    DecisionTrace,
    FailureBundleRef,
    FailureCode,
    Frame,
    RunEventName,
    State,
)
from ..failures import FailureRecord, FailureSignal, RecoveryBudgetTracker, spec_for

HANDLERS = (
    "recapture",
    "reperceive_question_region",
    "reperceive_zoomed",
    "reobserve_and_react",
    "navigation_cascade",
    "retry_model_call",
    "refocus_window",
    "uncertainty_policy",
    "safe_stop",
)


@dataclass
class RecoveryContext:
    """What the recovery decision may look at."""

    state: State = State.RECOVERING
    scope_key: str = "run"
    frames: Sequence[Frame] = field(default_factory=tuple)
    perception: Any = None
    decision_trace: Optional[DecisionTrace] = None
    question_hash: Optional[str] = None
    counters: Optional[BudgetCounters] = None
    correlation_id: Optional[str] = None


@dataclass
class RecoveryPlan:
    code: FailureCode
    handler: str
    allowed: bool
    attempt: int
    budget: int
    next_state: State
    reobserve: bool
    policy_driven: bool
    message: str
    reasons: List[str] = field(default_factory=list)
    bundle: Optional[FailureBundleRef] = None
    record: Optional[FailureRecord] = None
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def halt(self) -> bool:
        return self.next_state in {State.RUN_HALTED, State.RUN_ABORTED, State.FAILED_SAFE}

    def describe(self) -> Dict[str, Any]:
        return {
            "code": self.code.value,
            "handler": self.handler,
            "allowed": self.allowed,
            "attempt": self.attempt,
            "budget": self.budget,
            "next_state": self.next_state.value,
            "reobserve": self.reobserve,
            "policy_driven": self.policy_driven,
            "reasons": list(self.reasons),
            "bundle": self.bundle.path if self.bundle else None,
            "message": self.message[:200],
        }


class RecoveryModule:
    def __init__(
        self,
        config: EngineConfig,
        *,
        tracker: Optional[RecoveryBudgetTracker] = None,
        telemetry: Any = None,
        artifacts: Any = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.config = config
        self.telemetry = telemetry
        self.artifacts = artifacts
        self._clock = clock
        self.tracker = tracker or RecoveryBudgetTracker(config.budgets.max_consecutive_failures)
        self.stats: Dict[str, Any] = {"plans": 0, "allowed": 0, "exhausted": 0, "halted": 0, "bundles": 0}

    # -- main --------------------------------------------------------------- #
    def plan(self, signal: FailureSignal, context: Optional[RecoveryContext] = None) -> RecoveryPlan:
        context = context or RecoveryContext()
        spec = signal.spec if signal.spec.code == signal.code else spec_for(signal.code)
        scope_key = context.scope_key or "run"
        record = self.tracker.consume(
            signal.code, scope_key, context.state, signal.message, dict(signal.detail or {})
        )
        self.stats["plans"] += 1
        reasons: List[str] = [f"matrix: {spec.trigger} -> {spec.recovery}"]

        allowed = bool(spec.recoverable and record.attempt <= spec.budget and not spec.policy_driven)
        if spec.policy_driven:
            reasons.append("policy-driven failure: the uncertainty policy decides, not a retry")
            next_state = spec.escalate_to
        elif not spec.recoverable:
            reasons.append(f"budget for {signal.code.value} is 0 -- no recovery permitted (L3/L4)")
            next_state = spec.escalate_to
        elif record.attempt > spec.budget:
            reasons.append(f"attempt {record.attempt} exceeds the budget of {spec.budget} for {signal.code.value}")
            next_state = spec.escalate_to
        else:
            reasons.append(f"attempt {record.attempt}/{spec.budget} for {signal.code.value}")
            next_state = State.RECOVERING

        if self.tracker.consecutive_budget_exceeded() and next_state is State.RECOVERING:
            allowed = False
            next_state = State.RUN_ABORTED
            reasons.append(
                f"{self.tracker.consecutive_failures} consecutive unrecovered failures "
                f"(limit {self.tracker.max_consecutive_failures}) -> RUN_ABORTED"
            )

        bundle: Optional[FailureBundleRef] = None
        if spec.artifact_bundle and (not allowed or next_state is not State.RECOVERING):
            bundle = self._bundle(signal, context, record)
            if bundle is not None:
                record.bundle_path = bundle.path
                reasons.append(f"artifact bundle written to {bundle.path}")

        if allowed:
            self.stats["allowed"] += 1
        else:
            self.stats["exhausted"] += 1
        if next_state in {State.RUN_HALTED, State.RUN_ABORTED, State.FAILED_SAFE}:
            self.stats["halted"] += 1

        plan = RecoveryPlan(
            code=signal.code,
            handler=spec.handler or ("safe_stop" if not allowed else "safe_stop"),
            allowed=allowed,
            attempt=record.attempt,
            budget=spec.budget,
            next_state=next_state,
            reobserve=spec.reobserve_first,
            policy_driven=spec.policy_driven,
            message=signal.message,
            reasons=reasons,
            bundle=bundle,
            record=record,
            detail=dict(signal.detail or {}),
        )
        self._emit(signal, plan, context)
        return plan

    def resolve(self, plan: RecoveryPlan, *, success: bool) -> None:
        """Close the loop: a successful recovery resets the consecutive counter."""
        if plan.record is None:
            return
        if success:
            self.tracker.mark_recovered(plan.record)
        if self.telemetry is not None:
            self.telemetry.event(
                RunEventName.RECOVERY_SUCCESS if success else RunEventName.RECOVERY_EXHAUSTED,
                state=State.RECOVERING,
                module="recovery",
                code=plan.code,
                handler=plan.handler,
                attempt=plan.attempt,
                budget=plan.budget,
                success=success,
                correlation_id=None,
            )
            self.telemetry.metrics.record_recovery(plan.code.value, success)

    def reset_scope(self, scope_key: str) -> None:
        """A new question gets a fresh budget (FR-7.11.2)."""
        self.tracker.reset_scope(scope_key)

    # -- helpers ------------------------------------------------------------ #
    def _bundle(self, signal: FailureSignal, context: RecoveryContext, record: FailureRecord) -> Optional[FailureBundleRef]:
        if self.artifacts is None:
            return None
        try:
            return self.artifacts.failure_bundle(
                signal.code,
                context.state,
                frames=list(context.frames),
                perception=context.perception,
                decision_trace=context.decision_trace,
                detail={
                    "attempt": record.attempt,
                    "budget": record.budget,
                    "scope_key": context.scope_key,
                    "question_hash": context.question_hash,
                    "correlation_id": context.correlation_id,
                    "consecutive_failures": self.tracker.consecutive_failures,
                },
                message=signal.message,
            )
        except Exception:
            return None

    def _emit(self, signal: FailureSignal, plan: RecoveryPlan, context: RecoveryContext) -> None:
        if self.telemetry is None:
            return
        self.telemetry.event(
            RunEventName.RECOVERY_STARTED if plan.allowed else RunEventName.RECOVERY_EXHAUSTED,
            state=context.state,
            module="recovery",
            code=signal.code,
            handler=plan.handler,
            allowed=plan.allowed,
            attempt=plan.attempt,
            budget=plan.budget,
            next_state=plan.next_state.value,
            correlation_id=context.correlation_id,
            reason=plan.reasons[-1] if plan.reasons else "",
        )
        self.telemetry.metrics.inc("failures_total", label=signal.code.value)
        if not plan.allowed:
            self.telemetry.event(
                RunEventName.HUMAN_REQUIRED if plan.next_state is State.AWAITING_HUMAN else RunEventName.RUN_HALTED,
                state=plan.next_state,
                module="recovery",
                code=signal.code,
                correlation_id=context.correlation_id,
                reason=signal.message[:200],
            )

    def describe(self) -> Dict[str, Any]:
        return {
            "stats": dict(self.stats),
            "by_code": self.tracker.summary(),
            "recovery_stats": self.tracker.recovery_stats(),
            "consecutive_failures": self.tracker.consecutive_failures,
            "handlers": list(HANDLERS),
        }


__all__ = ["HANDLERS", "RecoveryContext", "RecoveryModule", "RecoveryPlan"]
