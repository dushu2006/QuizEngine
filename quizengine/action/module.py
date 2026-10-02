"""Action Module (PRD section 7.6).

Turns an :class:`~quizengine.contracts.Intent` into physical (or simulated)
input and records exactly what happened in an :class:`ActionRecord`.

Guards enforced here, in order:

1. safety gate -- attestation + restricted-environment re-scan (FR-7.14.1/.3)
2. **stale-coordinate refusal** -- ``intent.frame_seq`` must equal the live frame
   (**L1**, AC-14.2).  A refusal is *recorded* before it is raised, so the audit
   trail shows the attempt.
3. verification honesty -- a state-changing intent may not declare
   ``verify="none"`` (AC-14.3)
4. focus check (FR-7.6.4) and pyautogui failsafe corner (FR-7.6.5)
5. human-plausible timings/jitter from ``action.input_profile`` (FR-7.6.2)
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..config import EngineConfig
from ..contracts import (
    ActionRecord,
    ActionType,
    ElementResolution,
    FailureCode,
    Frame,
    Intent,
    PerceptionResult,
    RunEventName,
    State,
)
from ..failures import FailureSignal
from ..geometry import Box, box_center, box_clip, box_contains_point
from .backends import ActuatorBackend, BackendReport, Timings

Point = Tuple[int, int]


@dataclass
class ActionOutcome:
    record: ActionRecord
    point: Optional[Point] = None
    detail: Dict[str, Any] = field(default_factory=dict)
    timings: Optional[Timings] = None
    backend: str = "unknown"

    @property
    def executed(self) -> bool:
        return not self.record.refused

    def describe(self) -> Dict[str, Any]:
        return {
            "intent_id": self.record.intent_id,
            "action": self.record.action.value,
            "backend": self.record.backend,
            "point": list(self.point) if self.point else None,
            "executed": self.executed,
            "refused": self.record.refused,
            "refusal_reason": self.record.refusal_reason,
            "duration_ms": round((self.record.finished_ts - self.record.started_ts) * 1000.0, 2),
            "detail": self.detail,
        }


class ActionModule:
    def __init__(
        self,
        config: EngineConfig,
        backend: ActuatorBackend,
        *,
        telemetry: Any = None,
        safety: Any = None,
        rng: Optional[random.Random] = None,
        clock: Callable[[], float] = time.perf_counter,
        wall_clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.action_config = config.action
        self.backend = backend
        self.telemetry = telemetry
        self.safety = safety
        self._clock = clock
        self._wall = wall_clock
        self._sleep = sleep
        seed = config.run.seed if config.run.deterministic else None
        self.rng = rng if rng is not None else random.Random(seed)
        self.ledger: List[ActionRecord] = []
        self.stats: Dict[str, Any] = {
            "intents": 0,
            "executed": 0,
            "refused": 0,
            "stale_refusals": 0,
            "focus_failures": 0,
            "backend_errors": 0,
        }

    # -- point planning ----------------------------------------------------- #
    def plan_point(
        self,
        intent: Intent,
        frame: Optional[Frame],
        resolution: Optional[ElementResolution] = None,
    ) -> Optional[Point]:
        """Centre of the freshly resolved box plus bounded jitter (FR-7.6.2)."""
        if intent.action not in {ActionType.CLICK, ActionType.MOVE, ActionType.SCROLL}:
            return None
        box: Optional[Box] = resolution.resolved_box if resolution is not None else intent.target_box
        if box is None:
            return None
        cx, cy = box_center(box)
        point: Point = (int(round(cx)), int(round(cy)))
        jitter = int(self.action_config.input_profile.click_jitter_px or 0)
        if jitter > 0:
            if intent.jitter_seed is not None:
                local = random.Random(intent.jitter_seed)
            else:
                local = self.rng
            dx = local.randint(-jitter, jitter)
            dy = local.randint(-jitter, jitter)
            candidate = (point[0] + dx, point[1] + dy)
            if frame is not None:
                candidate = _clip_point_to_frame(candidate, frame)
            if box_contains_point(box, candidate):
                point = candidate
        return point

    # -- execution ---------------------------------------------------------- #
    def execute(
        self,
        intent: Intent,
        *,
        frame: Optional[Frame] = None,
        resolution: Optional[ElementResolution] = None,
        perception: Optional[PerceptionResult] = None,
    ) -> ActionOutcome:
        self.stats["intents"] += 1

        # 1. safety gate (FR-7.14.1/.3) ------------------------------------ #
        if self.safety is not None and intent.is_state_changing:
            self.safety.guard_intent(intent, perception=perception, state=State.ACTING)

        # 2. stale coordinates (L1 / AC-14.2) ------------------------------- #
        if frame is not None and intent.frame_seq >= 0 and frame.seq != intent.frame_seq:
            record = self._refuse(
                intent,
                frame,
                reason=(
                    f"stale frame_seq: intent references frame {intent.frame_seq} "
                    f"but the live frame is {frame.seq} (L1/AC-14.2)"
                ),
            )
            self.stats["stale_refusals"] += 1
            raise FailureSignal(
                FailureCode.STALE_COORDINATES,
                record.refusal_reason or "stale coordinates",
                origin_state=State.ACTING,
                detail={"record": record.to_wire(), "intent": intent.to_wire()},
            )

        # 3. verification honesty (AC-14.3) --------------------------------- #
        if intent.is_state_changing and intent.verify == "none":
            record = self._refuse(
                intent, frame, reason="state-changing intent declared verify='none'; AC-14.3 requires a VerificationRecord"
            )
            raise FailureSignal(
                FailureCode.ACTION_UNVERIFIED,
                record.refusal_reason or "unverifiable intent",
                origin_state=State.ACTING,
                detail={"record": record.to_wire(), "intent": intent.to_wire()},
            )

        # 4. focus + failsafe (FR-7.6.4/.5) --------------------------------- #
        if self.action_config.focus_check and not self.backend.focus_ok(self.action_config.window_title_match):
            self.stats["focus_failures"] += 1
            record = self._refuse(intent, frame, reason="target window is not in focus (FR-7.6.4)")
            raise FailureSignal(
                FailureCode.FOCUS_LOST,
                record.refusal_reason or "focus lost",
                origin_state=State.ACTING,
                detail={"record": record.to_wire()},
            )
        if self.action_config.failsafe_corner and self.backend.failsafe_triggered():
            record = self._refuse(intent, frame, reason="mouse is in the failsafe corner -- operator aborted (FR-7.6.5)")
            raise FailureSignal(
                FailureCode.OPERATOR_STOP,
                record.refusal_reason or "failsafe triggered",
                origin_state=State.ACTING,
                detail={"record": record.to_wire()},
            )

        if not self.backend.available():
            record = self._refuse(intent, frame, reason=f"actuator backend '{self.backend.name}' is not available")
            raise FailureSignal(
                FailureCode.ACTION_UNVERIFIED,
                record.refusal_reason or "backend unavailable",
                origin_state=State.ACTING,
                detail={"record": record.to_wire(), "backend": self.backend.describe()},
            )

        # 5. act ------------------------------------------------------------ #
        point = self.plan_point(intent, frame, resolution)
        box = resolution.resolved_box if resolution is not None else intent.target_box
        timings = Timings.sample(self.action_config.input_profile, self.rng)
        started = self._wall()
        perf_start = self._clock()
        report: BackendReport = self.backend.perform(intent, point, timings)
        finished = self._wall()
        elapsed_ms = (self._clock() - perf_start) * 1000.0

        record = ActionRecord(
            intent_id=intent.intent_id,
            action=intent.action,
            point=point,
            box=box,
            frame_seq_at_execution=frame.seq if frame is not None else intent.frame_seq,
            started_ts=started,
            finished_ts=finished,
            backend=self.backend.name,
            jitter_px=float(timings.jitter_px),
            move_duration_s=round(timings.move_duration_s, 4),
            refused=not report.ok,
            refusal_reason=report.error,
        )
        self.ledger.append(record)
        if not report.ok:
            self.stats["backend_errors"] += 1
        else:
            self.stats["executed"] += 1
        self._audit(intent, record, report, elapsed_ms)

        if not report.ok:
            raise FailureSignal(
                FailureCode.ACTION_UNVERIFIED,
                f"actuator backend '{self.backend.name}' failed: {report.error}",
                origin_state=State.ACTING,
                detail={"record": record.to_wire(), "backend_error": report.error},
            )
        return ActionOutcome(
            record=record, point=point, detail=dict(report.detail), timings=timings, backend=self.backend.name
        )

    # -- helpers ------------------------------------------------------------ #
    def _refuse(self, intent: Intent, frame: Optional[Frame], *, reason: str) -> ActionRecord:
        now = self._wall()
        record = ActionRecord(
            intent_id=intent.intent_id,
            action=intent.action,
            point=None,
            box=intent.target_box,
            frame_seq_at_execution=frame.seq if frame is not None else intent.frame_seq,
            started_ts=now,
            finished_ts=now,
            backend=self.backend.name,
            refused=True,
            refusal_reason=reason,
        )
        self.ledger.append(record)
        self.stats["refused"] += 1
        self._audit(intent, record, BackendReport(ok=False, error=reason), 0.0)
        return record

    def _audit(self, intent: Intent, record: ActionRecord, report: BackendReport, elapsed_ms: float) -> None:
        entry = {
            "intent_id": intent.intent_id,
            "action": intent.action.value,
            "handle": intent.handle,
            "expected_effect": intent.expected_effect.type.value,
            "target_box": list(intent.target_box) if intent.target_box else None,
            "point": list(record.point) if record.point else None,
            "frame_seq_intent": intent.frame_seq,
            "frame_seq_at_execution": record.frame_seq_at_execution,
            "backend": record.backend,
            "refused": record.refused,
            "refusal_reason": record.refusal_reason,
            "verify": intent.verify,
            "correlation_id": intent.correlation_id,
            "duration_ms": round(elapsed_ms, 2),
            "detail": dict(report.detail or {}),
        }
        if self.telemetry is not None:
            self.telemetry.record_intent(entry)
            self.telemetry.event(
                RunEventName.ACTION_EXECUTED,
                state=State.ACTING,
                module="action",
                latency_ms=elapsed_ms,
                correlation_id=intent.correlation_id,
                intent_id=intent.intent_id,
                action=intent.action.value,
                handle=intent.handle,
                point=list(record.point) if record.point else None,
                frame_seq=record.frame_seq_at_execution,
                backend=record.backend,
                refused=record.refused,
                refusal_reason=record.refusal_reason,
            )
            self.telemetry.metrics.inc("actions_total")
            if record.refused:
                self.telemetry.metrics.inc("actions_refused")
            if "stale" in (record.refusal_reason or "").lower():
                self.telemetry.metrics.inc("stale_coordinate_violations")

    def describe(self) -> Dict[str, Any]:
        return {
            "stats": dict(self.stats),
            "backend": self.backend.describe(),
            "ledger_size": len(self.ledger),
            "input_profile": self.action_config.input_profile.model_dump(mode="json"),
        }


def _clip_point_to_frame(point: Point, frame: Frame) -> Point:
    width, height = frame.size_px
    return (max(0, min(int(width) - 1, point[0])), max(0, min(int(height) - 1, point[1])))


__all__ = ["ActionModule", "ActionOutcome"]
