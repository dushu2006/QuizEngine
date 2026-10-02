"""Failure catalog, recovery matrix and bounded budgets (PRD section 11).

Design laws enforced here:

* **[L3] bounded everything** -- every failure class has an explicit attempt
  budget.  A budget of ``0`` means *no recovery is permitted*: the agent stops.
* **[L4] fail safe, never fail blind** -- exhaustion escalates to a halt state
  with an artifact bundle, never to a guessed action.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .contracts import FailureCode, State


class QuizEngineError(Exception):
    """Base class for every engine-raised error."""

    code: FailureCode = FailureCode.BUDGET_EXCEEDED

    def __init__(self, message: str, *, detail: Optional[dict] = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail: dict = detail or {}


class ConfigError(QuizEngineError):
    """Invalid configuration -- refuse to run with actionable errors (NFR-17.4)."""

    code = FailureCode.CONFIG_INVALID


class AttestationError(QuizEngineError):
    """No recorded operator authorization (FR-7.14.1)."""

    code = FailureCode.ATTESTATION_MISSING


class RestrictedEnvironmentError(QuizEngineError):
    """A restricted environment was detected; the run must halt (section 3.3)."""

    code = FailureCode.RESTRICTED_ENVIRONMENT


class CapabilityError(QuizEngineError):
    """A required optional dependency / backend is unavailable."""

    code = FailureCode.CONFIG_INVALID


class FailureSignal(QuizEngineError):
    """Raised by any module to hand control to the recovery framework.

    Modules never implement their own retry loops for cross-cutting failures:
    they classify and signal, the orchestrator decides (FR-7.11.1).
    """

    def __init__(
        self,
        code: FailureCode,
        message: str,
        *,
        origin_state: Optional[State] = None,
        detail: Optional[dict] = None,
        recoverable: Optional[bool] = None,
    ) -> None:
        super().__init__(message, detail=detail)
        self.code = code
        self.origin_state = origin_state
        self.recoverable = recoverable

    @property
    def spec(self) -> "RecoverySpec":
        return RECOVERY_MATRIX.get(self.code, _DEFAULT_SPEC)


class BudgetExceeded(QuizEngineError):
    """A global run budget was exhausted (FR-7.10.2)."""

    code = FailureCode.BUDGET_EXCEEDED


class IllegalTransition(QuizEngineError):
    """FSM bug: fail loudly in dev, halt safely in prod (FR-7.10.1)."""

    code = FailureCode.ILLEGAL_TRANSITION


class StaleCoordinateError(QuizEngineError):
    """An action referenced geometry not validated in the live frame (L1)."""

    code = FailureCode.STALE_COORDINATES


@dataclass(frozen=True)
class RecoverySpec:
    """One row of the section 11 matrix."""

    code: FailureCode
    trigger: str
    detection: str
    recovery: str
    budget: int
    #: State to escalate to when the budget is exhausted.
    escalate_to: State = State.RUN_HALTED
    #: When true the uncertainty policy decides instead of a fixed escalation.
    policy_driven: bool = False
    #: Always write a failure artifact bundle (FR-7.11.3).
    artifact_bundle: bool = True
    #: Recovery must re-observe before doing anything (FR-7.11.1 phase 1).
    reobserve_first: bool = True
    handler: Optional[str] = None

    @property
    def recoverable(self) -> bool:
        return self.budget > 0


_DEFAULT_SPEC = RecoverySpec(
    code=FailureCode.BUDGET_EXCEEDED,
    trigger="unclassified failure",
    detection="orchestrator",
    recovery="safe stop",
    budget=0,
)

#: The recovery matrix, verbatim from PRD section 11 (plus additive codes).
RECOVERY_MATRIX: Dict[FailureCode, RecoverySpec] = {
    FailureCode.CAPTURE_FAILURE: RecoverySpec(
        code=FailureCode.CAPTURE_FAILURE,
        trigger="Blank/stale/failed frame",
        detection="Frame validation (FR-7.1.2)",
        recovery="Re-capture x3 with 250ms backoff",
        budget=3,
        handler="recapture",
    ),
    FailureCode.EXTRACTION_FAILURE: RecoverySpec(
        code=FailureCode.EXTRACTION_FAILURE,
        trigger="Invalid Question",
        detection="Extraction gates (FR-7.3.2)",
        recovery="Re-crop + re-OCR, then VLM re-perceive",
        budget=2,
        handler="reperceive_question_region",
    ),
    FailureCode.ACTION_UNVERIFIED: RecoverySpec(
        code=FailureCode.ACTION_UNVERIFIED,
        trigger="No expected effect post-click",
        detection="Verification controller (FR-7.9)",
        recovery="Re-observe -> re-resolve coords -> single re-click, only if state confirms unselected",
        budget=2,
        handler="reobserve_and_react",
    ),
    FailureCode.NAVIGATION_STUCK: RecoverySpec(
        code=FailureCode.NAVIGATION_STUCK,
        trigger="Scroll/button ineffective",
        detection="Navigation discipline (FR-7.8.2)",
        recovery="Alternate strategy cascade; keyboard nav if enabled",
        budget=2,
        handler="navigation_cascade",
    ),
    FailureCode.PERCEPTION_LOW_CONFIDENCE: RecoverySpec(
        code=FailureCode.PERCEPTION_LOW_CONFIDENCE,
        trigger="Tier disagreements / schema validation failures",
        detection="Perception (FR-7.2.6/7.2.7)",
        recovery="VLM re-perceive with zoomed crops",
        budget=2,
        handler="reperceive_zoomed",
    ),
    FailureCode.MODEL_TIMEOUT: RecoverySpec(
        code=FailureCode.MODEL_TIMEOUT,
        trigger="Solver/VLM call exceeded timeout",
        detection="Model provider wrapper (FR-6.2)",
        recovery="Retry with backoff, then fallback provider",
        budget=2,
        handler="retry_model_call",
    ),
    FailureCode.MODEL_SCHEMA_FAILURE: RecoverySpec(
        code=FailureCode.MODEL_SCHEMA_FAILURE,
        trigger="Model output failed schema validation",
        detection="Perception/solver schema gate (FR-7.2.6)",
        recovery="2 retries with error-feedback re-prompt",
        budget=2,
        handler="retry_model_call",
    ),
    FailureCode.FOCUS_LOST: RecoverySpec(
        code=FailureCode.FOCUS_LOST,
        trigger="Target window not foreground",
        detection="Pre-action focus check (FR-7.6.4)",
        recovery="Re-focus by title match; verify",
        budget=2,
        handler="refocus_window",
    ),
    FailureCode.POPUP_UNKNOWN: RecoverySpec(
        code=FailureCode.POPUP_UNKNOWN,
        trigger="Unrecognized overlay",
        detection="Perception overlays (FR-7.2.5)",
        recovery="Safe stop + artifact bundle (never Escape-spam)",
        budget=0,
        handler="safe_stop",
    ),
    FailureCode.RESTRICTED_ENVIRONMENT: RecoverySpec(
        code=FailureCode.RESTRICTED_ENVIRONMENT,
        trigger="Gatekeeper signature matched",
        detection="Environment gatekeeper (FR-7.14.2)",
        recovery="Halt + report, no recovery",
        budget=0,
        handler="safe_stop",
    ),
    FailureCode.SOLVER_LOW_CONFIDENCE: RecoverySpec(
        code=FailureCode.SOLVER_LOW_CONFIDENCE,
        trigger="Composite confidence below low_conf",
        detection="Confidence module (FR-7.5.2)",
        recovery="Uncertainty policy (default: pause for human)",
        budget=0,
        policy_driven=True,
        escalate_to=State.AWAITING_HUMAN,
        handler="uncertainty_policy",
    ),
    FailureCode.SOLVER_TIMEOUT: RecoverySpec(
        code=FailureCode.SOLVER_TIMEOUT,
        trigger="Total solver budget per question exhausted (FR-7.4.4)",
        detection="Solver module",
        recovery="Uncertainty policy",
        budget=0,
        policy_driven=True,
        escalate_to=State.AWAITING_HUMAN,
        handler="uncertainty_policy",
    ),
    FailureCode.SOLVER_NO_ANSWER: RecoverySpec(
        code=FailureCode.SOLVER_NO_ANSWER,
        trigger="No strategy in the chain produced an answer",
        detection="Solver module (FR-7.4.1)",
        recovery="Uncertainty policy",
        budget=0,
        policy_driven=True,
        escalate_to=State.AWAITING_HUMAN,
        handler="uncertainty_policy",
    ),
    FailureCode.STALE_COORDINATES: RecoverySpec(
        code=FailureCode.STALE_COORDINATES,
        trigger="Action box not validated in the live frame",
        detection="Actuator pre-flight check (L1)",
        recovery="Re-resolve from a fresh frame; never click the stale box",
        budget=2,
        handler="reobserve_and_react",
    ),
    FailureCode.ATTESTATION_MISSING: RecoverySpec(
        code=FailureCode.ATTESTATION_MISSING,
        trigger="No --attest provided",
        detection="Pre-run authorization gate (FR-7.14.1)",
        recovery="Refuse to start",
        budget=0,
        handler="safe_stop",
    ),
    FailureCode.CONFIG_INVALID: RecoverySpec(
        code=FailureCode.CONFIG_INVALID,
        trigger="Config failed schema validation",
        detection="Startup (NFR-17.4)",
        recovery="Refuse to run with actionable errors",
        budget=0,
        handler="safe_stop",
    ),
    FailureCode.UNSUPPORTED_QUESTION_TYPE: RecoverySpec(
        code=FailureCode.UNSUPPORTED_QUESTION_TYPE,
        trigger="Multi-select / fill-in / drag-and-drop detected (section 3.2)",
        detection="Extraction module",
        recovery="Safe stop with diagnostic; out of scope for v1.0",
        budget=0,
        handler="safe_stop",
    ),
    FailureCode.BUDGET_EXCEEDED: RecoverySpec(
        code=FailureCode.BUDGET_EXCEEDED,
        trigger="Global run budget exhausted",
        detection="Orchestrator (FR-7.10.2)",
        recovery="RUN_ABORTED with report",
        budget=0,
        escalate_to=State.RUN_ABORTED,
        handler="safe_stop",
    ),
    FailureCode.ILLEGAL_TRANSITION: RecoverySpec(
        code=FailureCode.ILLEGAL_TRANSITION,
        trigger="FSM transition not in the table",
        detection="Orchestrator (FR-7.10.1)",
        recovery="Fail loudly in dev; halt safely in prod",
        budget=0,
        handler="safe_stop",
    ),
    FailureCode.OPERATOR_STOP: RecoverySpec(
        code=FailureCode.OPERATOR_STOP,
        trigger="Operator requested stop()",
        detection="Console control (FR-7.10.3)",
        recovery="Finish verification, then report",
        budget=0,
        handler="safe_stop",
    ),
}


def spec_for(code: FailureCode) -> RecoverySpec:
    return RECOVERY_MATRIX.get(code, _DEFAULT_SPEC)


def is_recoverable(code: FailureCode) -> bool:
    return spec_for(code).recoverable


def recovery_budget(code: FailureCode) -> int:
    return spec_for(code).budget


@dataclass
class FailureRecord:
    code: FailureCode
    message: str
    state: State
    attempt: int
    budget: int
    recovered: bool = False
    ts: float = 0.0
    detail: dict = field(default_factory=dict)
    bundle_path: Optional[str] = None


class RecoveryBudgetTracker:
    """Per-failure-class, per-question attempt accounting (FR-7.11.2).

    Budget is scoped to ``(code, scope_key)`` where ``scope_key`` is normally the
    question hash: a fresh question gets a fresh budget, but one question can
    never consume unbounded retries.
    """

    def __init__(self, max_consecutive_failures: int = 3) -> None:
        self._used: Dict[tuple, int] = {}
        self._consecutive = 0
        self.max_consecutive_failures = max_consecutive_failures
        self.records: List[FailureRecord] = []

    # -- accounting -------------------------------------------------------- #
    def attempts_used(self, code: FailureCode, scope_key: str) -> int:
        return self._used.get((code, scope_key), 0)

    def attempts_remaining(self, code: FailureCode, scope_key: str) -> int:
        return max(0, recovery_budget(code) - self.attempts_used(code, scope_key))

    def can_recover(self, code: FailureCode, scope_key: str) -> bool:
        spec = spec_for(code)
        if spec.budget <= 0:
            return False
        return self.attempts_remaining(code, scope_key) > 0

    def consume(self, code: FailureCode, scope_key: str, state: State, message: str, detail: dict | None = None) -> FailureRecord:
        """Register a failure and consume one attempt of its budget."""
        spec = spec_for(code)
        used = self._used.get((code, scope_key), 0) + 1
        self._used[(code, scope_key)] = used
        self._consecutive += 1
        record = FailureRecord(
            code=code,
            message=message,
            state=state,
            attempt=used,
            budget=spec.budget,
            detail=detail or {},
        )
        self.records.append(record)
        return record

    def mark_recovered(self, record: FailureRecord) -> None:
        record.recovered = True
        self._consecutive = 0

    def reset_scope(self, scope_key: str) -> None:
        for key in [k for k in self._used if k[1] == scope_key]:
            del self._used[key]

    def reset_consecutive(self) -> None:
        self._consecutive = 0

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive

    def consecutive_budget_exceeded(self) -> bool:
        """3 consecutive unrecovered failures -> RUN_ABORTED (FR-7.11.2)."""
        return self._consecutive >= self.max_consecutive_failures

    # -- reporting --------------------------------------------------------- #
    def summary(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for (code, _scope), count in self._used.items():
            out[code.value] = out.get(code.value, 0) + count
        return out

    def recovery_stats(self) -> Dict[str, Dict[str, int]]:
        stats: Dict[str, Dict[str, int]] = {}
        for record in self.records:
            entry = stats.setdefault(record.code.value, {"invoked": 0, "recovered": 0})
            entry["invoked"] += 1
            if record.recovered:
                entry["recovered"] += 1
        return stats


#: Handlers are looked up by name on the recovery controller (keeps the matrix
#: declarative and data-driven rather than a pile of lambdas).
HandlerFn = Callable[[FailureSignal], object]


def describe_matrix() -> str:
    """Human-readable dump of the matrix (used by ``quizengine doctor``)."""
    lines = [
        f"{'CODE':32} {'BUDGET':>6}  {'ESCALATE_TO':14} RECOVERY",
        "-" * 110,
    ]
    for code in FailureCode:
        spec = spec_for(code)
        budget = "-" if spec.policy_driven else str(spec.budget)
        lines.append(
            f"{code.value:32} {budget:>6}  {spec.escalate_to.value:14} {spec.recovery}"
        )
    return "\n".join(lines)
