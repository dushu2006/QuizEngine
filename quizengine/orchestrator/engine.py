"""Orchestrator & FSM (PRD section 7.10, section 10).

The orchestrator owns the cycle::

    CAPTURING -> PERCEIVING -> EXTRACTING -> DECIDING -> PRE_ACTION_RESOLVE
              -> ACTING -> VERIFYING -> NAVIGATING -> (CAPTURING | END_DETECTED)

and is the only component that

* moves the FSM (FR-7.10.1, section 10 table),
* enforces the global budgets (FR-7.10.2, **L3**),
* polls the operator for pause/stop (FR-7.10.3),
* emits the per-cycle correlation id and events (FR-7.10.4),
* dispatches recovery handlers by name (FR-7.11.1),
* builds the ``DecisionTrace`` audit record (FR-7.13.4) and the run report
  (FR-7.12.2).

Every other module stays pure and side-effect free (**L5**, **L6**).  A cycle
ends in one of five ways: ``answered``, ``skipped``, ``retry`` (recovered -- the
loop re-observes), ``end_state`` or ``failed``/``stop`` (terminal).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..action import ActionModule, ActionOutcome, build_backend
from ..binding import ElementResolver, option_handle
from ..capture import CaptureModule
from ..confidence import ConfidenceModule, ConfidenceOutcome
from ..config import EngineConfig
from ..contracts import (
    ActionType,
    AnsweredQuestion,
    BudgetCounters,
    Decision,
    DecisionTrace,
    ExpectedEffect,
    ExpectedEffectType,
    FailureCode,
    Frame,
    Intent,
    OverlayKind,
    PerceptionResult,
    Question,
    RunEventName,
    RunOutcome,
    ScreenTransition,
    SelectedMarker,
    SolverStrategy,
    State,
    VerificationRecord,
)
from ..extraction import ExtractionModule, ExtractionOutcome
from ..failures import FailureSignal
from ..models import ModelStack, build_model_stack
from ..navigation import NavigationModule
from ..perception import PerceptionModule, PerceptionOutcome
from ..persistence import RunArtifacts, SessionStore
from ..recovery import RecoveryContext, RecoveryModule, RecoveryPlan
from ..render import crop_b64
from ..safety import SafetyGatekeeper
from ..solver import SolverContext, SolverModule, SolverOutcome, bind_answer_to_index
from ..solver.base import StrategyResult
from ..telemetry import Telemetry
from ..telemetry.console import CommandKind
from ..verification import VerificationModule, VerificationSnapshot, expected_marker_for_style
from .fsm import StateMachine

_BLOCKING_OVERLAYS = {OverlayKind.MODAL, OverlayKind.UNKNOWN, OverlayKind.COOKIE_BANNER}
_TERMINAL_CYCLE_KINDS = {"end_state", "stop", "failed"}


@dataclass
class CycleResult:
    """Outcome of one question cycle."""

    kind: str  # answered | skipped | retry | end_state | stop | failed
    outcome: Optional[RunOutcome] = None
    code: Optional[FailureCode] = None
    detail: str = ""
    trace: Optional[DecisionTrace] = None
    question: Optional[Question] = None
    decision: Optional[Decision] = None
    verification: Optional[VerificationRecord] = None

    @property
    def stop(self) -> bool:
        return self.kind in _TERMINAL_CYCLE_KINDS


@dataclass
class HumanReply:
    """What AWAITING_HUMAN produced."""

    kind: str  # answer | skip | resume | abort
    detail: str = ""
    decision_outcome: Optional[SolverOutcome] = None
    assessment: Optional[ConfidenceOutcome] = None


@dataclass
class OrchestratorStats:
    cycles: int = 0
    answered: int = 0
    skipped_duplicate: int = 0
    skipped_unanswered: int = 0
    human_answers: int = 0
    verifications_passed: int = 0
    verifications_failed: int = 0
    recoveries: int = 0
    retries: int = 0
    navigations: int = 0
    scrolls: int = 0
    end_state_frames: int = 0
    verify_again_passes: int = 0
    stale_refusals: int = 0
    overlays_dismissed: int = 0
    latency_ms: Dict[str, List[float]] = field(default_factory=dict)
    iteration_latencies_ms: List[float] = field(default_factory=list)

    def observe(self, name: str, value_ms: float) -> None:
        self.latency_ms.setdefault(name, []).append(max(0.0, float(value_ms)))

    def summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {key: value for key, value in self.__dict__.items()
                                if key not in {"latency_ms", "iteration_latencies_ms"}}
        out["latency_ms"] = {
            key: _latency_summary(values)
            for key, values in self.latency_ms.items()
            if values
        }
        out["iteration_latency_ms"] = _latency_summary(self.iteration_latencies_ms)
        return out


def _latency_summary(values: Sequence[float]) -> Dict[str, float | int]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"n": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    def percentile(percent: float) -> float:
        position = (len(ordered) - 1) * percent / 100.0
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        fraction = position - low
        return ordered[low] * (1.0 - fraction) + ordered[high] * fraction
    return {
        "n": len(ordered),
        "mean": round(sum(ordered) / len(ordered), 2),
        "median": round(percentile(50), 2),
        "p95": round(percentile(95), 2),
        "max": round(ordered[-1], 2),
    }


class Orchestrator:
    """Closed-loop quiz automation agent."""

    def __init__(
        self,
        config: EngineConfig,
        *,
        telemetry: Optional[Telemetry] = None,
        capture: Optional[CaptureModule] = None,
        perception: Optional[PerceptionModule] = None,
        extraction: Optional[ExtractionModule] = None,
        solver: Optional[SolverModule] = None,
        confidence: Optional[ConfidenceModule] = None,
        action: Optional[ActionModule] = None,
        verification: Optional[VerificationModule] = None,
        navigation: Optional[NavigationModule] = None,
        recovery: Optional[RecoveryModule] = None,
        safety: Optional[SafetyGatekeeper] = None,
        binding: Optional[ElementResolver] = None,
        session: Optional[SessionStore] = None,
        artifacts: Optional[RunArtifacts] = None,
        models: Optional[ModelStack] = None,
        backend: Any = None,
        world: Any = None,
        answer_key: Optional[Dict[str, Any]] = None,
        operator: Any = None,
        clock: Callable[[], float] = time.time,
        perf: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self._clock = clock
        self._perf = perf
        self._sleep = sleep
        self.world = world

        # -- observability first: every module reports through it ----------- #
        self.run_id = config.run.run_id or Telemetry.new_run_id()
        self.telemetry = telemetry if telemetry is not None else Telemetry(config, self.run_id)
        if operator is not None:
            self.telemetry.operator = operator
        self.artifacts = (
            artifacts if artifacts is not None else RunArtifacts(config, self.telemetry)
        )
        self.session = (
            session
            if session is not None
            else SessionStore(config, run_id=self.run_id, telemetry=self.telemetry, clock=clock)
        )

        # -- models --------------------------------------------------------- #
        self.models = models
        if self.models is None:
            key = answer_key if answer_key is not None else self._answer_key_from(capture, world)
            self.models = build_model_stack(config.models, answer_key=key, on_event=self._model_event, sleep=sleep)
        self.answer_key: Dict[str, Any] = (
            answer_key if answer_key is not None else dict(self.models.offline_answer_key or {})
        )

        # -- modules -------------------------------------------------------- #
        self.safety = safety if safety is not None else SafetyGatekeeper(config, telemetry=self.telemetry, clock=clock)
        self.capture = (
            capture
            if capture is not None
            else CaptureModule(config, telemetry=self.telemetry, clock=clock, sleep=sleep, perf=perf)
        )
        self.perception = perception if perception is not None else PerceptionModule(
            config, tier2_provider=self.models.tier2, telemetry=self.telemetry, capture=self.capture, clock=perf
        )
        self.extraction = extraction if extraction is not None else ExtractionModule(
            config, telemetry=self.telemetry, solver_provider=self.models.solver, clock=perf
        )
        self.solver = solver if solver is not None else SolverModule(
            config, provider=self.models.solver, telemetry=self.telemetry, clock=perf
        )
        self.confidence = confidence if confidence is not None else ConfidenceModule(config, telemetry=self.telemetry)
        self.binding = binding if binding is not None else ElementResolver(config, telemetry=self.telemetry, clock=perf)
        self.navigation = navigation if navigation is not None else NavigationModule(config, telemetry=self.telemetry, clock=clock)
        self.verification = (
            verification
            if verification is not None
            else VerificationModule(config, telemetry=self.telemetry, clock=perf, sleep=sleep)
        )
        self.recovery = (
            recovery
            if recovery is not None
            else RecoveryModule(config, telemetry=self.telemetry, artifacts=self.artifacts, clock=clock)
        )
        if action is not None:
            self.action = action
        else:
            actuator_backend = backend
            if actuator_backend is None:
                if world is not None:
                    from ..action import SimulatedBackend

                    actuator_backend = SimulatedBackend(world)
                else:
                    actuator_backend = build_backend(
                        config.action.backend, failsafe_corner=config.action.failsafe_corner
                    )
            self.action = ActionModule(
                config,
                actuator_backend,
                telemetry=self.telemetry,
                safety=self.safety,
                clock=perf,
                wall_clock=clock,
                sleep=sleep,
            )

        self.sm = StateMachine(
            State.IDLE,
            strict=config.run.strict_transitions,
            dev_mode=config.run.dev_mode,
            telemetry=self.telemetry,
            clock=clock,
        )
        self.stats = OrchestratorStats()
        self.traces: List[DecisionTrace] = []
        self.counters = BudgetCounters()
        self._started_at: float = 0.0
        self._end_state_streak: int = 0
        self._navigation_attempts: int = 0
        self._scroll_steps: int = 0
        self._stop_requested: bool = False
        self._halted_code: Optional[FailureCode] = None
        self._halted_detail: str = ""
        self._last_question: Optional[Question] = None
        self._last_perception: Optional[PerceptionResult] = None
        self._current_trace: Optional[DecisionTrace] = None
        self._pending_result: Optional[CycleResult] = None

    # ------------------------------------------------------------------ #
    # public entry point
    # ------------------------------------------------------------------ #
    def request_stop(self) -> None:
        """Thread-safe best-effort cancellation request for the active run."""
        self._stop_requested = True
        if self.models is not None:
            cancel = getattr(self.models, "cancel_current", None)
            if callable(cancel):
                cancel()

    def run(self) -> Any:
        """Run to a terminal state and return the section 8 ``RunReport``."""
        self._started_at = self._clock()
        self.telemetry.set_state(State.IDLE)
        outcome = RunOutcome.COMPLETED
        try:
            self.sm.transition(State.GATE_CHECK, reason="run start")
            self._gate_check()
            self.sm.transition(State.CAPTURING, reason="authorization + environment gate passed")
            self.telemetry.event(
                RunEventName.RUN_STARTED, state=State.CAPTURING, module="orchestrator", run_id=self.run_id
            )
            result = self._loop()
            outcome = result.outcome or RunOutcome.COMPLETED
            if result.code is not None:
                self._halted_code = result.code
                self._halted_detail = result.detail
        except FailureSignal as signal:
            outcome = self._classify_signal_outcome(signal)
            self._halted_code = signal.code
            self._halted_detail = signal.message
            self.telemetry.event(
                RunEventName.RUN_HALTED,
                state=self.sm.state,
                module="orchestrator",
                code=signal.code,
                reason=signal.message[:200],
            )
        except Exception as exc:  # L4: never fail blind
            outcome = RunOutcome.FAILED_SAFE
            self._halted_code = FailureCode.ILLEGAL_TRANSITION
            self._halted_detail = f"{type(exc).__name__}: {exc}"
            self.sm.force(State.FAILED_SAFE, reason=self._halted_detail)
            self.telemetry.event(
                RunEventName.RUN_HALTED,
                state=State.FAILED_SAFE,
                module="orchestrator",
                code=FailureCode.ILLEGAL_TRANSITION,
                reason=self._halted_detail[:200],
            )
        return self._finish(outcome)

    # ------------------------------------------------------------------ #
    # gates (FR-7.14)
    # ------------------------------------------------------------------ #
    def _gate_check(self) -> None:
        attestation_text = self.config.run.attestation
        if not attestation_text:
            raise FailureSignal(
                FailureCode.ATTESTATION_MISSING,
                "no operator attestation: pass --i-am-authorized on the CLI or set run.attestation",
                origin_state=State.GATE_CHECK,
            )
        self.safety.attest(attestation_text, method="config")
        scan = self.safety.scan(state=State.GATE_CHECK)
        if scan.restricted:
            raise FailureSignal(
                FailureCode.RESTRICTED_ENVIRONMENT,
                f"pre-run environment scan failed: {scan.halt_reason}",
                origin_state=State.GATE_CHECK,
                detail={"indicators": [i.model_dump(mode="json") for i in scan.indicators]},
            )
        self.telemetry.banner(self.config.fingerprint())
        self.session.begin(platform_profile=self._platform_profile(scan), attestation_text=attestation_text)

    def _platform_profile(self, scan: Any) -> Dict[str, Any]:
        return {
            "platform": self.config.run.platform_profile,
            "capture_backend": self.capture.backend.name,
            "action_backend": self.action.backend.name,
            "ocr_engine": getattr(getattr(self.perception, "ocr", None), "name", "none"),
            "solver_provider": self.models.describe()["solver"]["primary"]["name"] if self.models else "none",
            "offline": bool(self.models and self.models.is_offline),
            "env_scan": scan.summary,
        }

    # ------------------------------------------------------------------ #
    # main loop
    # ------------------------------------------------------------------ #
    def _loop(self) -> CycleResult:
        while True:
            self._budget_check()
            self._poll_operator_control()
            if self._stop_requested:
                return CycleResult(
                    kind="stop", outcome=RunOutcome.STOPPED_BY_OPERATOR, detail="operator requested stop (FR-7.10.3)"
                )
            result = self._cycle()
            if result.stop:
                return result
            if result.kind == "retry":
                self.stats.retries += 1

    def _cycle(self) -> CycleResult:
        """One iteration with unconditional release of per-question model cache."""
        iteration_started = self._perf()
        try:
            return self._cycle_impl()
        finally:
            cleanup_started = self._perf()
            if self.models is not None:
                cleanup = getattr(self.models, "cleanup_iteration", None)
                if callable(cleanup):
                    cleanup()
            cleanup_ms = (self._perf() - cleanup_started) * 1000.0
            self.stats.observe("cleanup_ms", cleanup_ms)
            total_ms = (self._perf() - iteration_started) * 1000.0
            self.stats.observe("total_iteration_ms", total_ms)
            self.stats.iteration_latencies_ms.append(total_ms)

    def _cycle_impl(self) -> CycleResult:
        """One question: capture -> perceive -> extract -> decide -> act -> verify -> navigate."""
        self.stats.cycles += 1
        self.counters.cycles += 1
        self._pending_result = None
        cycle_started = self._perf()
        correlation_id = self.telemetry.new_correlation_id(self._next_ordinal())
        self._current_trace = None

        # -- CAPTURING ------------------------------------------------------ #
        frame = self._capture(correlation_id=correlation_id)

        # -- PERCEIVING ----------------------------------------------------- #
        outcome = self._perceive(frame, correlation_id=correlation_id, new_question=True)
        perception = outcome.perception
        end_state = self._end_state_check(perception, frame, correlation_id)
        if end_state is not None:
            return end_state

        overlay_result = self._handle_overlays(perception, frame, correlation_id)
        if overlay_result is not None:
            return overlay_result

        # -- EXTRACTING ----------------------------------------------------- #
        extraction = self._extract(perception, frame, outcome, correlation_id)
        if extraction is None:
            return self._pending_result or CycleResult(
                kind="failed",
                outcome=RunOutcome.HALTED,
                code=FailureCode.EXTRACTION_FAILURE,
                detail="extraction failed and the recovery budget is exhausted",
                trace=self._current_trace,
            )
        question = extraction.question
        if question is None:
            self.sm.transition(State.CAPTURING, reason="screen has no answerable question; re-observing")
            return CycleResult(kind="retry", detail="no answerable question on this screen")

        if extraction.transition is not None and extraction.transition.already_answered:
            self.stats.skipped_duplicate += 1
            self.telemetry.event(
                RunEventName.QUESTION_EXTRACTED,
                state=State.EXTRACTING,
                module="orchestrator",
                correlation_id=correlation_id,
                question_hash=question.hash,
                skipped="already answered this session (L9)",
            )
            nav = self._navigate_after(perception, frame, correlation_id, selection_pending=False)
            if nav is not None:
                return nav
            return CycleResult(kind="skipped", question=question, detail="already answered (L9)")

        self._current_trace = DecisionTrace(
            correlation_id=correlation_id,
            question_hash=question.hash,
            ordinal=question.ordinal,
            question_text=question.text[:300],
            option_texts=[o.text for o in question.options],
            frame_seq=frame.seq,
        )
        self.telemetry.event(
            RunEventName.QUESTION_EXTRACTED,
            state=State.EXTRACTING,
            module="orchestrator",
            correlation_id=correlation_id,
            question_hash=question.hash,
            ordinal=question.ordinal,
            options=len(question.options),
            layout=perception.layout_type.value,
            type=question.type.value,
            extraction_confidence=question.extraction_confidence,
        )

        # -- DECIDING (FR-7.4) with the FR-7.5.2 uncertainty policy --------- #
        attempt = 1
        resumes = 0
        decision_outcome: Optional[SolverOutcome] = None
        assessment: Optional[ConfidenceOutcome] = None
        while True:
            decision_outcome, assessment = self._decide(
                question, perception, outcome, frame, extraction, correlation_id, attempt=attempt
            )
            if decision_outcome is None or assessment is None:
                return self._pending_result or CycleResult(
                    kind="retry", detail="solver failure recovered; re-observing"
                )
            if assessment.act:
                break
            if assessment.escalate and attempt == 1:
                attempt += 1
                self.stats.verify_again_passes += 1
                self.sm.transition(State.PERCEIVING, reason="FR-7.5.2 verify_again")
                frame = self._capture(correlation_id=correlation_id)
                outcome = self._perceive(frame, correlation_id=correlation_id, force_tier2=True)
                perception = outcome.perception
                re_extracted = self._extract(
                    perception, frame, outcome, correlation_id, ordinal_hint=question.ordinal
                )
                if re_extracted is not None and re_extracted.question is not None:
                    question = re_extracted.question
                    extraction = re_extracted
                continue

            reply = self._await_human(question, decision_outcome, assessment, perception, frame, correlation_id)
            if reply.kind == "answer" and reply.decision_outcome is not None and reply.assessment is not None:
                decision_outcome, assessment = reply.decision_outcome, reply.assessment
                break
            if reply.kind == "skip":
                self.stats.skipped_unanswered += 1
                self._finish_trace(
                    question, None, None, outcome="skipped_by_operator", started=cycle_started, frame=frame
                )
                nav = self._navigate_after(perception, frame, correlation_id, selection_pending=False)
                return nav or CycleResult(kind="skipped", question=question, detail=reply.detail)
            if reply.kind == "resume" and resumes < 2:
                resumes += 1
                attempt = 1
                self.sm.transition(State.PERCEIVING, reason="operator asked to re-observe")
                frame = self._capture(correlation_id=correlation_id)
                outcome = self._perceive(frame, correlation_id=correlation_id)
                perception = outcome.perception
                continue
            return CycleResult(
                kind="stop",
                outcome=RunOutcome.STOPPED_BY_OPERATOR,
                code=FailureCode.OPERATOR_STOP,
                detail=reply.detail,
                trace=self._current_trace,
                question=question,
            )

        if self._stop_requested:
            return CycleResult(
                kind="stop",
                outcome=RunOutcome.STOPPED_BY_OPERATOR,
                code=FailureCode.OPERATOR_STOP,
                detail="operator requested stop before action; no click issued",
                trace=self._current_trace,
                question=question,
            )

        decision = decision_outcome.decision
        if decision is None:  # pragma: no cover - act implies a decision
            return CycleResult(
                kind="failed", outcome=RunOutcome.HALTED, code=FailureCode.SOLVER_NO_ANSWER, detail="no decision"
            )

        # -- PRE_ACTION_RESOLVE (FR-7.7) ------------------------------------ #
        intent = self._selection_intent(question, decision, frame, correlation_id)
        self.sm.transition(State.PRE_ACTION_RESOLVE, reason=f"binding {decision.letter} -> {intent.handle}")
        binding = self.binding.resolve(
            intent, frame=frame, perception=perception, template=self._template_crop(frame, intent)
        )
        if binding.stale or binding.resolution.escalated:
            self.sm.transition(State.PERCEIVING, reason="re-resolution escalated: fresh perception required (L1)")
            frame = self._capture(correlation_id=correlation_id)
            outcome = self._perceive(frame, correlation_id=correlation_id)
            perception = outcome.perception
            intent = self._rebase_intent(intent, frame)
            binding = self.binding.resolve(
                intent,
                frame=frame,
                perception=perception,
                template=self._template_crop(frame, intent),
                expected_box=intent.target_box,
            )
            if binding.stale:
                self._recover(
                    FailureSignal(
                        FailureCode.STALE_COORDINATES,
                        f"element {intent.handle!r} could not be re-resolved in frame {frame.seq}",
                        origin_state=State.PRE_ACTION_RESOLVE,
                        detail={"binding": binding.reasons},
                    ),
                    question=question,
                    perception=perception,
                    correlation_id=correlation_id,
                )
                return self._pending_result or CycleResult(kind="retry", detail="binding escalation recovered")

        # -- ACTING (FR-7.6) ------------------------------------------------ #
        pre_snapshot = VerificationSnapshot(frame=frame, perception=perception)
        self.sm.transition(State.ACTING, reason=f"clicking {intent.handle}")
        action_outcome = self._act(
            intent, frame=frame, resolution=binding.resolution, perception=perception, correlation_id=correlation_id
        )
        if action_outcome is None:
            return self._pending_result or CycleResult(kind="retry", detail="action refused or failed")

        # -- VERIFYING (FR-7.9, L2) ----------------------------------------- #
        self.sm.transition(State.VERIFYING, reason="closed-loop verification")
        record = self._verify(intent, pre_snapshot, correlation_id)
        if record.passed:
            self.stats.verifications_passed += 1
        else:
            self.stats.verifications_failed += 1
            retried = self._retry_unverified(intent, question, record, pre_snapshot, correlation_id)
            if retried is not None:
                record = retried
            if not record.passed:
                return self._pending_result or CycleResult(
                    kind="retry", detail="selection could not be verified; re-observing"
                )

        # -- ledger (L9) ---------------------------------------------------- #
        self.session.record_answer(
            question_hash=question.hash,
            ordinal=question.ordinal,
            decision=decision,
            content_hash=question.content_hash,
            verified=record.passed,
            decision_trace_id=correlation_id,
        )
        self.stats.answered += 1
        self.recovery.reset_scope(question.hash)
        self.recovery.tracker.reset_consecutive()
        self.counters.consecutive_failures = 0
        self._last_question = question
        self._finish_trace(
            question,
            decision,
            record,
            outcome="answered",
            started=cycle_started,
            frame=frame,
            assessment=assessment,
            attempts=decision_outcome.attempts,
        )

        # -- NAVIGATING (FR-7.8) -------------------------------------------- #
        post = self._snapshot(correlation_id=correlation_id)
        nav = self._navigate_after(
            post.perception, post.frame, correlation_id, selection_pending=True, answered=question
        )
        if nav is not None:
            return nav
        self.stats.observe("cycle_ms", (self._perf() - cycle_started) * 1000.0)
        return CycleResult(
            kind="answered", question=question, decision=decision, verification=record, trace=self.traces[-1] if self.traces else None
        )

    # ------------------------------------------------------------------ #
    # state handlers
    # ------------------------------------------------------------------ #
    def _capture(self, *, correlation_id: Optional[str] = None) -> Frame:
        self.sm.transition(State.CAPTURING, reason="acquiring a frame")
        started = self._perf()
        frame = self.capture.capture(correlation_id=correlation_id)
        self.stats.observe("capture_ms", (self._perf() - started) * 1000.0)
        return frame

    def _perceive(
        self,
        frame: Frame,
        *,
        correlation_id: Optional[str] = None,
        new_question: bool = False,
        force_tier2: bool = False,
    ) -> PerceptionOutcome:
        self.sm.transition(State.PERCEIVING, reason="tier-1 perception")
        outcome = self.perception.perceive(
            frame, force_tier2=force_tier2, correlation_id=correlation_id, new_question=new_question
        )
        self.stats.observe("perception_ms", outcome.latency_ms)
        if outcome.tier2 is not None:
            self.stats.observe("tier2_perception_ms", outcome.tier2.latency_ms)
        else:
            self.stats.observe("tier2_perception_ms", 0.0)
        self._last_perception = outcome.perception
        self.telemetry.event(
            RunEventName.PERCEPTION_DONE,
            state=State.PERCEIVING,
            module="orchestrator",
            latency_ms=outcome.latency_ms,
            correlation_id=correlation_id,
            frame_seq=frame.seq,
            layout=outcome.perception.layout_type.value,
            options=len(outcome.perception.options),
            agreement=round(outcome.agreement, 3),
            tier2=outcome.tier2 is not None,
            confidence=round(outcome.confidence, 3),
            ambiguity=len(outcome.ambiguity),
            trace_frame=frame if self.config.telemetry.trace_screenshots else None,
            trace_tag="perceiving",
        )
        return outcome

    def _snapshot(self, *, correlation_id: Optional[str] = None) -> VerificationSnapshot:
        """Re-observe without demanding a changed frame (polling is not a change)."""
        frame = self.capture.capture(correlation_id=correlation_id)
        outcome = self.perception.perceive(frame, correlation_id=correlation_id)
        self._last_perception = outcome.perception
        return VerificationSnapshot(frame=frame, perception=outcome.perception)

    def _end_state_check(
        self, perception: PerceptionResult, frame: Frame, correlation_id: Optional[str]
    ) -> Optional[CycleResult]:
        """FR-7.8.4: the end state must be confirmed over N consecutive frames.

        Returns ``None`` when this is an ordinary quiz screen, a ``retry`` result
        while the streak is still being established (so the screen is never
        mistaken for an answerable question), and an ``end_state`` result once it
        is confirmed.
        """
        evidence = self.navigation.end_state_evidence(perception)
        if not self.navigation.end_state_candidate(perception):
            self._end_state_streak = 0
            return None
        self._end_state_streak += 1
        self.stats.end_state_frames += 1
        if not self.navigation.is_end_state(perception, self._end_state_streak):
            self.telemetry.log(
                f"end-state candidate ({self._end_state_streak}/"
                f"{self.config.navigation.end_state_required_frames} frames): re-observing before concluding",
                module="orchestrator",
            )
            self.sm.transition(State.CAPTURING, reason="confirming the end state over N frames (FR-7.8.4)")
            return CycleResult(kind="retry", detail="end-state candidate, awaiting confirmation")
        self.sm.transition(State.END_DETECTED, reason="; ".join(evidence[:2])[:200])
        self.telemetry.event(
            RunEventName.END_STATE_DETECTED,
            state=State.END_DETECTED,
            module="orchestrator",
            correlation_id=correlation_id,
            streak=self._end_state_streak,
            evidence=evidence[:4],
            trace_frame=frame if self.config.telemetry.trace_screenshots else None,
            trace_tag="end_state",
        )
        return CycleResult(kind="end_state", outcome=RunOutcome.COMPLETED, detail="; ".join(evidence[:2]))

    def _handle_overlays(
        self, perception: PerceptionResult, frame: Frame, correlation_id: Optional[str]
    ) -> Optional[CycleResult]:
        """Dismiss a known overlay; safe-stop on an unknown one (FR-7.2.5, section 11)."""
        blocking = [o for o in perception.overlays if o.kind in _BLOCKING_OVERLAYS]
        if not blocking:
            return None
        plan = self.navigation.plan(perception, frame_seq=frame.seq, correlation_id=correlation_id)
        if plan.intent is None or plan.strategy != "dismiss_overlay":
            self._recover(
                FailureSignal(
                    FailureCode.POPUP_UNKNOWN,
                    f"unrecognized overlay blocks the quiz: {(blocking[0].text or '')[:80]!r} "
                    f"(kind={blocking[0].kind.value})",
                    origin_state=State.PERCEIVING,
                    detail={"overlays": [o.kind.value for o in blocking]},
                ),
                perception=perception,
                correlation_id=correlation_id,
            )
            return self._pending_result or CycleResult(
                kind="failed", outcome=RunOutcome.HALTED, code=FailureCode.POPUP_UNKNOWN, detail="unknown popup"
            )

        self.sm.transition(State.NAVIGATING, reason="dismissing a blocking overlay")
        pre = VerificationSnapshot(frame=frame, perception=perception)
        self.sm.transition(State.ACTING, reason=f"dismiss {plan.intent.handle}")
        executed = self._act(plan.intent, frame=frame, perception=perception, correlation_id=correlation_id)
        if executed is None:
            return self._pending_result or CycleResult(kind="retry", detail="overlay dismissal failed")
        self.sm.transition(State.VERIFYING, reason="verify overlay dismissal")
        record = self._verify(plan.intent, pre, correlation_id)
        if not record.passed:
            self._recover(
                FailureSignal(
                    FailureCode.POPUP_UNKNOWN,
                    "overlay dismissal could not be verified on screen",
                    origin_state=State.VERIFYING,
                    detail={"evidence": record.evidence},
                ),
                perception=perception,
                correlation_id=correlation_id,
            )
            return self._pending_result or CycleResult(kind="retry", detail="overlay still present")
        self.stats.overlays_dismissed += 1
        self.sm.transition(State.CAPTURING, reason="overlay dismissed; re-observing the question")
        return CycleResult(kind="retry", detail="overlay dismissed")

    def _extract(
        self,
        perception: PerceptionResult,
        frame: Frame,
        outcome: PerceptionOutcome,
        correlation_id: Optional[str],
        *,
        ordinal_hint: Optional[int] = None,
    ) -> Optional[ExtractionOutcome]:
        self.sm.transition(State.EXTRACTING, reason="validating the perceived question")
        hints = {"contrast_score": getattr(outcome.tier1, "contrast_score", None)}
        kwargs: Dict[str, Any] = dict(
            ordinal_hint=ordinal_hint if ordinal_hint is not None else self._next_ordinal(),
            previous=self._last_question,
            answered_hashes=self.session.answered_hashes(),
            answered_content_hashes=self.session.answered_content_hashes(),
            frame=frame,
            hints=hints,
            correlation_id=correlation_id,
        )
        extraction = self.extraction.extract(perception, **kwargs)
        if extraction.valid and extraction.question is not None:
            if extraction.transition is not None and extraction.transition.transition is ScreenTransition.END_STATE:
                self._end_state_streak = max(
                    self._end_state_streak, self.config.navigation.end_state_required_frames
                )
            return extraction
        if extraction.transition is not None and extraction.transition.transition is ScreenTransition.POPUP:
            return extraction

        signal = FailureSignal(
            FailureCode.EXTRACTION_FAILURE,
            "; ".join(extraction.errors) or "question failed the extraction gates (FR-7.3.2)",
            origin_state=State.EXTRACTING,
            detail={"errors": extraction.errors, "warnings": extraction.warnings, "hashes": extraction.hashes},
        )
        plan = self._recovery_plan(signal, question=None, perception=perception, correlation_id=correlation_id)
        if not plan.allowed:
            self._pending_result = self._escalate(signal, plan, question=None)
            return None

        self.stats.recoveries += 1
        self.counters.recoveries_invoked += 1
        self.sm.transition(State.RECOVERING, reason=f"EXTRACTION_FAILURE -> {plan.handler}")
        region = perception.question_region or (0, 0, frame.width, frame.height)
        zoomed = plan.handler == "reperceive_zoomed"
        recaptured = (
            self.capture.capture_zoom_window(region, correlation_id=correlation_id, source_frame=frame)
            if zoomed
            else self.capture.capture_region(region, correlation_id=correlation_id)
        )
        reperceived = self.perception.perceive(recaptured, force_tier2=zoomed, correlation_id=correlation_id)
        retry = self.extraction.extract(reperceived.perception, **{**kwargs, "frame": recaptured})
        self.recovery.resolve(plan, success=retry.valid)
        if retry.valid:
            self.counters.recoveries_succeeded += 1
            self.recovery.tracker.reset_consecutive()
            self.sm.transition(State.EXTRACTING, reason="extraction recovery succeeded")
            return retry
        self._pending_result = self._escalate(signal, plan, question=None)
        return None

    def _decide(
        self,
        question: Question,
        perception: PerceptionResult,
        outcome: PerceptionOutcome,
        frame: Frame,
        extraction: ExtractionOutcome,
        correlation_id: Optional[str],
        *,
        attempt: int = 1,
    ) -> Tuple[Optional[SolverOutcome], Optional[ConfidenceOutcome]]:
        """Solve + assess.  ``(None, None)`` means a failure was recovered."""
        self.sm.transition(State.DECIDING, reason=f"solver cascade (attempt {attempt})")
        region = question.question_region or perception.question_region
        context = SolverContext(
            frame=frame,
            perception=perception,
            crop_b64=crop_b64(frame, region) if (question.flags.has_image or question.flags.has_math) else None,
            correlation_id=correlation_id,
            extra=dict(extraction.solver_context),
        )
        started = self._perf()
        try:
            solver_outcome = self.solver.solve(question, context, correlation_id=correlation_id)
        except FailureSignal as signal:
            self.stats.observe("solver_ms", (self._perf() - started) * 1000.0)
            if signal.spec.policy_driven:
                assessment = self.confidence.refuse(signal.message, attempt=attempt)
                return SolverOutcome(decision=None, attempts=[], latency_ms=0.0), assessment
            self._recover(signal, question=question, perception=perception, correlation_id=correlation_id)
            return None, None
        self.stats.observe("solver_ms", (self._perf() - started) * 1000.0)

        decision = solver_outcome.decision
        if decision is None:  # pragma: no cover - solve() raises instead
            return solver_outcome, self.confidence.refuse("solver produced no decision", attempt=attempt)

        assessment = self.confidence.assess(
            question=question,
            decision=decision,
            perception=perception,
            agreement=outcome.agreement,
            reconciliation_penalty=outcome.reconciliation.penalty,
            attempt=attempt,
            consecutive_failures=self.recovery.tracker.consecutive_failures,
            image_description=extraction.image_description,
        )
        self.telemetry.status(
            question_no=question.ordinal,
            total=self.config.budgets.max_questions,
            confidence=assessment.composite,
            cycle=self.counters.cycles,
            message=(
                f"{decision.letter or ''} via {decision.strategy.value} "
                f"(solver {decision.confidence:.2f}, policy {assessment.policy.value})"
            ),
        )
        if not assessment.act:
            self.telemetry.event(
                RunEventName.PERCEPTION_LOW_CONFIDENCE,
                state=State.DECIDING,
                module="orchestrator",
                correlation_id=correlation_id,
                confidence=assessment.composite,
                tier=assessment.tier,
                policy=assessment.policy.value,
                escalate=assessment.escalate,
                reasons=assessment.reasons[:3],
            )
        return solver_outcome, assessment

    def _selection_intent(
        self, question: Question, decision: Decision, frame: Frame, correlation_id: Optional[str]
    ) -> Intent:
        option = question.options[decision.option_index]
        handle = option.handle or option_handle(option.index)
        marker = expected_marker_for_style(self._scene_style(decision.option_index))
        return Intent(
            intent_id=f"intent-{uuid.uuid4().hex[:10]}",
            action=ActionType.CLICK,
            handle=handle,
            target_box=option.hit_box,
            expected_effect=ExpectedEffect(
                type=ExpectedEffectType.SELECTION_CHANGED,
                target_handle=handle,
                region=option.hit_box,
                expected_marker=marker,
                min_region_change_pct=max(0.5, self.config.verification.min_region_change_pct / 5.0),
            ),
            max_wait_ms=int(self.config.verification.verify_recheck_ms),
            verify="standard",
            frame_seq=frame.seq,
            correlation_id=correlation_id,
            jitter_seed=(self.config.run.seed + question.ordinal) if self.config.run.deterministic else None,
        )

    def _rebase_intent(self, intent: Intent, frame: Frame) -> Intent:
        """Re-issue the same intent against a freshly captured frame (**L1**)."""
        return intent.model_copy(update={"frame_seq": frame.seq, "intent_id": f"{intent.intent_id}-r"})

    def _template_crop(self, frame: Frame, intent: Intent) -> Optional[Any]:
        if intent.target_box is None or frame.pixels is None:
            return None
        try:
            import numpy as np

            from ..render import crop

            pixels = crop(np.asarray(frame.pixels), intent.target_box)
            return pixels if pixels.size else None
        except Exception:
            return None

    def _act(
        self,
        intent: Intent,
        *,
        frame: Frame,
        resolution: Optional[Any] = None,
        perception: Optional[PerceptionResult] = None,
        correlation_id: Optional[str] = None,
    ) -> Optional[ActionOutcome]:
        started = self._perf()
        self.telemetry.event(
            RunEventName.INTENT_DECLARED,
            state=State.ACTING,
            module="orchestrator",
            correlation_id=correlation_id,
            intent_id=intent.intent_id,
            action=intent.action.value,
            handle=intent.handle,
            frame_seq=intent.frame_seq,
            effect=intent.expected_effect.type.value,
        )
        try:
            return self.action.execute(intent, frame=frame, resolution=resolution, perception=perception)
        except FailureSignal as signal:
            if signal.code is FailureCode.STALE_COORDINATES:
                self.stats.stale_refusals += 1
            self._recover(signal, perception=perception, correlation_id=correlation_id)
            return None
        finally:
            self.stats.observe("action_ms", (self._perf() - started) * 1000.0)

    def _verify(self, intent: Intent, pre: VerificationSnapshot, correlation_id: Optional[str]) -> VerificationRecord:
        def snapshot_fn() -> VerificationSnapshot:
            frame = self.capture.capture(correlation_id=correlation_id)
            perceived = self.perception.perceive(frame, correlation_id=correlation_id)
            return VerificationSnapshot(frame=frame, perception=perceived.perception)

        started = self._perf()
        record = self.verification.verify(intent, pre, snapshot_fn)
        self.stats.observe("verification_ms", (self._perf() - started) * 1000.0)
        return record

    def _retry_unverified(
        self,
        intent: Intent,
        question: Question,
        record: VerificationRecord,
        pre: VerificationSnapshot,
        correlation_id: Optional[str],
    ) -> Optional[VerificationRecord]:
        """FR-7.9 / section 11: re-observe, re-resolve, one single re-click."""
        signal = FailureSignal(
            FailureCode.ACTION_UNVERIFIED,
            f"expected effect {intent.expected_effect.type.value} not observed for {intent.handle!r}",
            origin_state=State.VERIFYING,
            detail={"evidence": record.evidence},
        )
        plan = self._recovery_plan(signal, question=question, perception=pre.perception, correlation_id=correlation_id)
        if not plan.allowed or plan.handler != "reobserve_and_react":
            self._pending_result = self._escalate(signal, plan, question=question)
            return None

        self.stats.recoveries += 1
        self.counters.recoveries_invoked += 1
        self.sm.transition(State.RECOVERING, reason="ACTION_UNVERIFIED: re-observe and re-resolve")
        post = self._snapshot(correlation_id=correlation_id)
        marker_now = self.binding.marker_for(intent.handle, post.perception)
        if marker_now is not None and marker_now != SelectedMarker.NONE:
            self.recovery.resolve(plan, success=True)
            self.counters.recoveries_succeeded += 1
            self.sm.transition(State.VERIFYING, reason="post-action state confirms the selection landed")
            return self.verification.compare(intent, pre, post, attempts=record.attempts + 1)

        self.sm.transition(State.ACTING, reason="single re-click after re-resolution (FR-7.9)")
        retried_intent = self._rebase_intent(intent, post.frame)
        rebound = self.binding.resolve(
            retried_intent,
            frame=post.frame,
            perception=post.perception,
            template=self._template_crop(post.frame, intent),
            expected_box=intent.target_box,
        )
        executed = self._act(
            retried_intent,
            frame=post.frame,
            resolution=rebound.resolution,
            perception=post.perception,
            correlation_id=correlation_id,
        )
        if executed is None:
            self.recovery.resolve(plan, success=False)
            return None
        new_record = self._verify(retried_intent, post, correlation_id)
        self.recovery.resolve(plan, success=new_record.passed)
        if new_record.passed:
            self.counters.recoveries_succeeded += 1
        else:
            self._pending_result = self._escalate(signal, plan, question=question)
        return new_record

    def _navigate_after(
        self,
        perception: PerceptionResult,
        frame: Frame,
        correlation_id: Optional[str],
        *,
        selection_pending: bool,
        answered: Optional[Question] = None,
    ) -> Optional[CycleResult]:
        started = self._perf()
        try:
            return self._navigate_after_impl(
                perception, frame, correlation_id,
                selection_pending=selection_pending, answered=answered,
            )
        finally:
            self.stats.observe("navigation_ms", (self._perf() - started) * 1000.0)

    def _navigate_after_impl(
        self,
        perception: PerceptionResult,
        frame: Frame,
        correlation_id: Optional[str],
        *,
        selection_pending: bool,
        answered: Optional[Question] = None,
    ) -> Optional[CycleResult]:
        """FR-7.8: get to the next question, or confirm the end state."""
        self.sm.transition(State.NAVIGATING, reason="navigation cascade")
        if self._already_on_next_question(perception, answered):
            # The platform moved on by itself (auto-advance, FR-7.8.4).  This
            # screen holds a question nobody has answered yet: clicking Next
            # again would silently skip it, so hand control back to the loop.
            self.telemetry.log(
                "platform auto-advanced; re-observing the new question instead of navigating",
                module="orchestrator",
                correlation_id=correlation_id,
            )
            self.sm.transition(State.CAPTURING, reason="auto-advance: fresh question on screen")
            return CycleResult(kind="retry", detail="platform auto-advanced to the next question")
        self.stats.navigations += 1
        plan = self.navigation.plan(
            perception,
            frame_seq=frame.seq,
            frame_size=(0, 0, frame.width, frame.height),
            correlation_id=correlation_id,
            scroll_steps=self._scroll_steps,
            selection_pending=selection_pending,
        )
        if plan.end_state:
            self._end_state_streak = max(self._end_state_streak, self.config.navigation.end_state_required_frames)
            if self.navigation.is_end_state(perception, self._end_state_streak):
                self.sm.transition(State.END_DETECTED, reason=plan.reason[:200])
                self.telemetry.event(
                    RunEventName.END_STATE_DETECTED,
                    state=State.END_DETECTED,
                    module="orchestrator",
                    correlation_id=correlation_id,
                    evidence=plan.evidence[:4],
                    trace_frame=frame if self.config.telemetry.trace_screenshots else None,
                    trace_tag="end_state",
                )
                return CycleResult(kind="end_state", outcome=RunOutcome.COMPLETED, detail=plan.reason)
            self.sm.transition(State.CAPTURING, reason="end state not yet confirmed over N frames")
            return None

        if plan.intent is None:
            if plan.wait_ms:
                self._sleep(plan.wait_ms / 1000.0)
            self._navigation_attempts += 1
            self.sm.transition(State.CAPTURING, reason=plan.reason[:200])
            return None

        pre = VerificationSnapshot(frame=frame, perception=perception)
        self.sm.transition(State.ACTING, reason=f"navigation: {plan.strategy}")
        executed = self._act(plan.intent, frame=frame, perception=perception, correlation_id=correlation_id)
        if executed is None:
            return self._pending_result or CycleResult(kind="retry", detail="navigation action failed")
        self.sm.transition(State.VERIFYING, reason="verify navigation")
        record = self._verify(plan.intent, pre, correlation_id)
        if plan.strategy == "scroll_reveal":
            self._scroll_steps += 1
            self.stats.scrolls += 1
            if self.navigation.scroll_exhausted(record.region_change_pct):
                self.telemetry.log("scroll exhausted: content no longer moves (FR-7.8.2)", module="orchestrator")
        else:
            self._scroll_steps = 0

        if not record.passed:
            self._navigation_attempts += 1
            signal = FailureSignal(
                FailureCode.NAVIGATION_STUCK,
                f"navigation strategy {plan.strategy!r} produced no observable change",
                origin_state=State.NAVIGATING,
                detail={"evidence": record.evidence, "attempts": self._navigation_attempts},
            )
            self._recover(signal, perception=perception, correlation_id=correlation_id)
            return self._pending_result or CycleResult(kind="retry", detail="navigation stuck; recovered")

        self._navigation_attempts = 0
        self.telemetry.event(
            RunEventName.NAVIGATION_SUCCESS,
            state=State.NAVIGATING,
            module="orchestrator",
            correlation_id=correlation_id,
            strategy=plan.strategy,
            change_pct=record.region_change_pct,
            transition=record.transition.value if record.transition else None,
        )
        self.sm.transition(State.CAPTURING, reason="navigation verified")
        return None

    # ------------------------------------------------------------------ #
    # human in the loop (FR-7.5.2, FR-7.10.3, UC-8)
    # ------------------------------------------------------------------ #
    def _already_on_next_question(
        self, perception: PerceptionResult, answered: Optional[Question]
    ) -> bool:
        """True when the screen shows a different question than the one just answered."""
        if answered is None or not perception.question_text:
            return False
        if perception.question_region is None or len(perception.options) < 2:
            return False
        from ..solver.base import normalize

        return normalize(perception.question_text) != normalize(answered.text)

    def _await_human(
        self,
        question: Question,
        solver_outcome: Optional[SolverOutcome],
        assessment: ConfidenceOutcome,
        perception: PerceptionResult,
        frame: Frame,
        correlation_id: Optional[str],
    ) -> HumanReply:
        self.sm.transition(State.AWAITING_HUMAN, reason=f"uncertainty policy {assessment.policy.value}")
        self.counters.human_interventions += 1
        self.telemetry.event(
            RunEventName.HUMAN_REQUIRED,
            state=State.AWAITING_HUMAN,
            module="orchestrator",
            correlation_id=correlation_id,
            question_hash=question.hash,
            composite=assessment.composite,
            tier=assessment.tier,
            policy=assessment.policy.value,
            reasons=assessment.reasons[:3],
            trace_frame=frame if self.config.telemetry.trace_screenshots else None,
            trace_tag="awaiting_human",
        )
        best = solver_outcome.decision if solver_outcome else None
        prompt = (
            f"[{question.ordinal}] {question.text}\n"
            + "\n".join(f"  {o.handle or option_handle(o.index)}: {o.text}" for o in question.options)
            + f"\ncomposite confidence {assessment.composite:.2f} ({assessment.tier})"
            + (f"; engine would pick {best.letter}" if best is not None else "")
            + "\n"
            + "; ".join(assessment.reasons[:2])
            + "\nanswer <letter> | skip | resume | abort > "
        )
        command = self.telemetry.operator.request_decision(
            prompt, assessment.policy, {"question": question.to_wire(), "assessment": assessment.describe()}
        )
        self.telemetry.event(
            RunEventName.HUMAN_RESPONSE,
            state=State.AWAITING_HUMAN,
            module="orchestrator",
            correlation_id=correlation_id,
            command=command.kind,
            payload=command.payload,
        )
        if command.kind is CommandKind.ANSWER and command.payload:
            index = bind_answer_to_index(command.payload, question)
            if index is None:
                return HumanReply(kind="abort", detail=f"operator answer {command.payload!r} binds to no option")
            decision = Decision(
                question_hash=question.hash,
                option_index=index,
                strategy=SolverStrategy.HUMAN,
                confidence=1.0,
                rationale=f"operator answered {command.payload!r}",
                letter=chr(ord("A") + index),
            )
            self.stats.human_answers += 1
            outcome = SolverOutcome(
                decision=decision,
                attempts=[],
                accepted=StrategyResult(
                    option_index=index,
                    confidence=1.0,
                    rationale=decision.rationale,
                    strategy=SolverStrategy.HUMAN,
                ),
                latency_ms=0.0,
            )
            reassessed = self.confidence.human_decision(question=question, decision=decision)
            self.sm.transition(State.DECIDING, reason="operator supplied the answer")
            return HumanReply(kind="answer", decision_outcome=outcome, assessment=reassessed)
        if command.kind is CommandKind.SKIP:
            return HumanReply(kind="skip", detail="operator skipped the question")
        if command.kind is CommandKind.RESUME:
            return HumanReply(kind="resume", detail="operator asked to re-observe")
        if command.is_terminal:
            return HumanReply(kind="abort", detail=f"operator requested {command.kind}")
        return HumanReply(kind="abort", detail="operator channel returned no usable command")

    def _poll_operator_control(self) -> None:
        command = self.telemetry.operator.poll_command()
        if command.kind in {CommandKind.STOP, CommandKind.ABORT}:
            self._stop_requested = True
        elif command.kind is CommandKind.PAUSE:
            self.telemetry.log("operator requested pause; waiting for resume", module="orchestrator")
            for _ in range(10_000):
                inner = self.telemetry.operator.poll_command()
                if inner.kind in {CommandKind.RESUME, CommandKind.STOP, CommandKind.ABORT}:
                    self._stop_requested = inner.kind in {CommandKind.STOP, CommandKind.ABORT}
                    break
                self._sleep(0.2)

    # ------------------------------------------------------------------ #
    # recovery (FR-7.11)
    # ------------------------------------------------------------------ #
    def _recovery_plan(
        self,
        signal: FailureSignal,
        *,
        question: Optional[Question] = None,
        perception: Optional[PerceptionResult] = None,
        correlation_id: Optional[str] = None,
    ) -> RecoveryPlan:
        context = RecoveryContext(
            state=self.sm.state,
            scope_key=question.hash if question is not None else "run",
            frames=self.capture.forensic_frames(self.config.telemetry.artifact_bundle_frames),
            perception=perception,
            decision_trace=self._current_trace,
            question_hash=question.hash if question else None,
            counters=self.counters,
            correlation_id=correlation_id,
        )
        return self.recovery.plan(signal, context)

    def _recover(
        self,
        signal: FailureSignal,
        *,
        question: Optional[Question] = None,
        perception: Optional[PerceptionResult] = None,
        correlation_id: Optional[str] = None,
    ) -> CycleResult:
        """Classify, budget, dispatch.  Returns the cycle result to propagate."""
        plan = self._recovery_plan(signal, question=question, perception=perception, correlation_id=correlation_id)
        self.counters.total_failures += 1
        if not plan.allowed:
            self._pending_result = self._escalate(signal, plan, question=question)
            return self._pending_result

        self.stats.recoveries += 1
        self.counters.recoveries_invoked += 1
        self.sm.transition(State.RECOVERING, reason=f"{signal.code.value} -> {plan.handler}")
        handled = self._dispatch_recovery(plan, signal, perception, correlation_id)
        self.recovery.resolve(plan, success=handled)
        if handled:
            self.counters.recoveries_succeeded += 1
            self.recovery.tracker.reset_consecutive()
            self.counters.consecutive_failures = 0
            self.sm.transition(State.CAPTURING, reason="recovery succeeded; re-observing from a fresh frame")
            return CycleResult(kind="retry", code=signal.code, detail=f"recovered from {signal.code.value}")
        self.counters.consecutive_failures = self.recovery.tracker.consecutive_failures
        self._pending_result = self._escalate(signal, plan, question=question)
        return self._pending_result

    def _escalate(self, signal: FailureSignal, plan: RecoveryPlan, *, question: Optional[Question]) -> CycleResult:
        """Budget exhausted or non-recoverable: halt safely with artifacts (L4)."""
        if plan.next_state is State.AWAITING_HUMAN:
            # Policy-driven: the caller's uncertainty policy asks the operator.
            self.sm.transition(State.AWAITING_HUMAN, reason=f"{signal.code.value} -> uncertainty policy")
            self._pending_result = CycleResult(
                kind="retry", code=signal.code, detail="uncertainty policy: awaiting the operator"
            )
            return self._pending_result
        target = plan.next_state if plan.next_state in {State.RUN_HALTED, State.RUN_ABORTED} else State.RUN_HALTED
        self.sm.transition(target, reason=f"{signal.code.value}: {plan.reasons[-1][:120]}")
        self._halted_code = signal.code
        self._halted_detail = signal.message
        self._finish_trace(question, None, None, outcome=f"halted:{signal.code.value}", started=self._perf())
        result = CycleResult(
            kind="failed",
            outcome=RunOutcome.HALTED if target is State.RUN_HALTED else RunOutcome.ABORTED,
            code=signal.code,
            detail=signal.message,
            trace=self.traces[-1] if self.traces else None,
            question=question,
        )
        self._pending_result = result
        return result

    def _dispatch_recovery(
        self,
        plan: RecoveryPlan,
        signal: FailureSignal,
        perception: Optional[PerceptionResult],
        correlation_id: Optional[str],
    ) -> bool:
        handler = plan.handler
        try:
            if handler == "recapture":
                self.sm.transition(State.CAPTURING, reason="recovery: recapture with backoff")
                self.capture.capture(correlation_id=correlation_id)
                return True
            if handler in {"reperceive_question_region", "reperceive_zoomed"}:
                self.sm.transition(State.PERCEIVING, reason=f"recovery: {handler}")
                frame = self.capture.capture(correlation_id=correlation_id)
                region = (perception.question_region if perception is not None else None) or (
                    0,
                    0,
                    frame.width,
                    frame.height,
                )
                if handler == "reperceive_zoomed":
                    frame = self.capture.capture_zoom_window(region, correlation_id=correlation_id, source_frame=frame)
                self.perception.perceive(frame, force_tier2=True, correlation_id=correlation_id)
                return True
            if handler == "reobserve_and_react":
                self.sm.transition(State.PERCEIVING, reason="recovery: re-observe before acting (FR-7.11.1)")
                self._snapshot(correlation_id=correlation_id)
                return True
            if handler == "navigation_cascade":
                self.sm.transition(State.NAVIGATING, reason="recovery: alternate navigation strategy")
                self._scroll_steps += 1
                return True
            if handler == "retry_model_call":
                self.sm.transition(State.DECIDING, reason="recovery: retry the model call")
                return True
            if handler == "refocus_window":
                self.sm.transition(State.ACTING, reason="recovery: re-focus the target window (FR-7.6.4)")
                return bool(self.action.backend.focus_ok(self.config.action.window_title_match))
            return False  # safe_stop / uncertainty_policy
        except FailureSignal:
            return False
        except Exception as exc:  # pragma: no cover - defensive
            self.telemetry.log(
                f"recovery handler {handler!r} raised {type(exc).__name__}: {exc}", module="orchestrator"
            )
            return False

    # ------------------------------------------------------------------ #
    # budgets (FR-7.10.2, L3)
    # ------------------------------------------------------------------ #
    def _budget_check(self) -> None:
        elapsed_s = self._clock() - self._started_at
        self.counters.elapsed_s = round(elapsed_s, 3)
        problems: List[str] = []
        if len(self.session.state.questions_answered) >= self.config.budgets.max_questions:
            problems.append(f"max_questions={self.config.budgets.max_questions} reached")
        if elapsed_s >= self.config.max_runtime_s:
            problems.append(f"max_runtime={self.config.budgets.max_runtime_min}min reached ({elapsed_s:.0f}s)")
        if self.counters.cycles >= self.config.budgets.max_cycles:
            problems.append(f"max_cycles={self.config.budgets.max_cycles} reached")
        if self.recovery.tracker.consecutive_failures >= self.config.budgets.max_consecutive_failures:
            problems.append(
                f"max_consecutive_failures={self.config.budgets.max_consecutive_failures} reached "
                f"({self.recovery.tracker.consecutive_failures})"
            )
        if not problems:
            return
        raise FailureSignal(
            FailureCode.BUDGET_EXCEEDED,
            "run budget exhausted: " + "; ".join(problems),
            origin_state=self.sm.state,
            detail={"problems": problems, "counters": self.counters.model_dump(mode="json")},
        )

    def _next_ordinal(self) -> int:
        if self._last_question is not None:
            return int(self._last_question.ordinal) + 1
        progress = getattr(self._last_perception, "navigation", None)
        if progress is not None and progress.progress_current is not None:
            return int(progress.progress_current)
        return len(self.session.state.questions_answered) + 1

    # ------------------------------------------------------------------ #
    # traces & report
    # ------------------------------------------------------------------ #
    def _finish_trace(
        self,
        question: Optional[Question],
        decision: Optional[Decision],
        verification: Optional[VerificationRecord],
        *,
        outcome: str,
        started: float,
        frame: Optional[Frame] = None,
        assessment: Optional[ConfidenceOutcome] = None,
        attempts: Optional[Sequence[Any]] = None,
    ) -> Optional[DecisionTrace]:
        trace = self._current_trace
        if trace is None:
            return None
        trace.decision = decision
        trace.verification = verification
        trace.confidence_breakdown = assessment.breakdown if assessment is not None else None
        trace.attempts = list(attempts or [])
        if decision is not None:
            for attempt in trace.attempts:
                attempt.accepted = (
                    attempt.strategy is decision.strategy and attempt.option_index == decision.option_index
                )
        trace.outcome = outcome
        trace.total_latency_ms = round(max(0.0, (self._perf() - started) * 1000.0), 2)
        if frame is not None:
            trace.frame_seq = frame.seq
        if question is not None:
            trace.question_text = question.text[:300]
            trace.option_texts = [o.text for o in question.options]
        self.traces.append(trace)
        self.telemetry.record_decision_trace(trace)
        self._current_trace = None
        return trace

    def _scene_style(self, option_index: int) -> Optional[str]:
        """Ground-truth option style when a fixture/simulated world is attached."""
        try:
            annotation = self.capture.annotation()
        except Exception:
            return None
        if not annotation:
            return None
        for option in annotation.get("options", []):
            if option.get("index") == option_index:
                return option.get("style")
        return None

    @staticmethod
    def _answer_key_from(capture: Optional[CaptureModule], world: Any) -> Dict[str, Any]:
        """Pull the QuizForge/fixture answer key when one is attached."""
        for candidate in (world, getattr(capture, "backend", None), getattr(getattr(capture, "backend", None), "world", None)):
            sequence = getattr(candidate, "sequence", None)
            key = getattr(sequence, "answer_key", None)
            if key:
                return dict(key)
        return {}

    def _accuracy(self) -> Tuple[Optional[float], Optional[int], Optional[int]]:
        if not self.answer_key:
            return None, None, None
        from ..solver.base import normalize

        by_hash: Dict[str, str] = {trace.question_hash: trace.question_text for trace in self.traces}
        correct = incorrect = 0
        for entry in self.session.state.questions_answered:
            text = by_hash.get(entry.hash)
            if text is None:
                continue
            expected = self.answer_key.get(text)
            if expected is None:
                expected = self.answer_key.get(normalize(text))
            if expected is None:
                for key, value in self.answer_key.items():
                    if normalize(str(key)) == normalize(text):
                        expected = value
                        break
            if expected is None:
                continue
            if int(expected) == int(entry.chosen_index):
                correct += 1
            else:
                incorrect += 1
        total = correct + incorrect
        if total == 0:
            return None, correct, incorrect
        return correct / float(total), correct, incorrect

    def _finish(self, outcome: RunOutcome) -> Any:
        if self._current_trace is not None:
            self._finish_trace(None, None, None, outcome="interrupted", started=self._perf())
        target_state = {
            RunOutcome.COMPLETED: State.DONE,
            RunOutcome.STOPPED_BY_OPERATOR: State.DONE,
            RunOutcome.HALTED: State.RUN_HALTED,
            RunOutcome.ABORTED: State.RUN_ABORTED,
            RunOutcome.FAILED_SAFE: State.FAILED_SAFE,
        }[outcome]
        if target_state in {State.RUN_HALTED, State.RUN_ABORTED} and self.sm.state is not target_state:
            self.sm.transition(target_state, reason=f"terminal outcome {outcome.value}")
        if self.sm.state is not State.REPORTING:
            self.sm.transition(State.REPORTING, reason="writing the run report")
        accuracy, correct, incorrect = self._accuracy()
        self.session.set_state(target_state)
        self.session.state.budget_counters = self.counters
        self.session.save()
        scan = self.safety.status.last_scan
        report = self.artifacts.write_report(
            run_id=self.run_id,
            outcome=outcome,
            started_at=self._started_at,
            session=self.session.state,
            traces=self.traces,
            env_scan_summary=scan.summary if scan is not None else "not scanned",
            attestation=self.safety.status.attestation,
            accuracy=accuracy,
            correct=correct,
            incorrect=incorrect,
            halted_code=self._halted_code,
            halted_detail=self._halted_detail,
            config_fingerprint=self.config.fingerprint(),
        )
        self.telemetry.event(
            RunEventName.RUN_COMPLETE,
            state=State.REPORTING,
            module="orchestrator",
            outcome=outcome.value,
            answered=len(self.session.state.questions_answered),
            accuracy=accuracy,
            cycles=self.counters.cycles,
            elapsed_s=report.total_time_s,
            halted_code=self._halted_code.value if self._halted_code else None,
        )
        final_state = (
            State.DONE if outcome in {RunOutcome.COMPLETED, RunOutcome.STOPPED_BY_OPERATOR} else State.FAILED_SAFE
        )
        if self.sm.state is not final_state:
            self.sm.transition(final_state, reason="report written")
        self.artifacts.prune()
        if self.models is not None:
            close = getattr(self.models, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:  # resource release must not hide the run report
                    self.telemetry.log(f"model client cleanup warning: {type(exc).__name__}", module="models")
        return report

    def _classify_signal_outcome(self, signal: FailureSignal) -> RunOutcome:
        if signal.code is FailureCode.RESTRICTED_ENVIRONMENT:
            self.sm.force(State.RUN_HALTED, reason="restricted environment (section 3.3)")
            return RunOutcome.HALTED
        if signal.code is FailureCode.BUDGET_EXCEEDED:
            self.sm.force(State.RUN_ABORTED, reason="budget exhausted (FR-7.10.2)")
            return RunOutcome.ABORTED
        if signal.code is FailureCode.OPERATOR_STOP:
            self.sm.force(State.RUN_ABORTED, reason="operator stop (FR-7.10.3)")
            return RunOutcome.STOPPED_BY_OPERATOR
        self.sm.force(State.RUN_HALTED, reason=signal.message[:120])
        return RunOutcome.HALTED

    def _model_event(self, name: str, detail: Dict[str, Any]) -> None:
        if name in {"MODEL_CALL", "MODEL_ROLE_CALL"}:
            self.counters.model_calls += 1
        event = {
            "MODEL_CALL": RunEventName.MODEL_CALL,
            "MODEL_ROLE_CALL": RunEventName.MODEL_CALL,
            "MODEL_EARLY_EXIT": RunEventName.MODEL_CALL,
            "MODEL_CONSENSUS": RunEventName.MODEL_CALL,
            "MODEL_FALLBACK": RunEventName.MODEL_CALL,
            "MODEL_TIMEOUT": RunEventName.MODEL_CALL,
            "MODEL_ERROR": RunEventName.MODEL_CALL,
        }.get(name)
        if event is None:
            return
        self.telemetry.event(event, module="models", record=name in {"MODEL_CALL", "MODEL_ROLE_CALL"}, model_event=name, **detail)

    # ------------------------------------------------------------------ #
    def describe(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "state": self.sm.state.value,
            "fsm": self.sm.describe(),
            "stats": self.stats.summary(),
            "counters": self.counters.model_dump(mode="json"),
            "session": self.session.snapshot(),
            "safety": self.safety.describe(),
            "solver": self.solver.describe(),
            "confidence": self.confidence.describe(),
            "recovery": self.recovery.describe(),
            "navigation": self.navigation.describe(),
            "verification": self.verification.describe(),
            "binding": self.binding.describe(),
            "action": self.action.describe(),
            "models": self.models.describe() if self.models else None,
            "traces": len(self.traces),
        }


__all__ = ["CycleResult", "HumanReply", "Orchestrator", "OrchestratorStats"]
