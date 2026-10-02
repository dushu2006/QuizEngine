"""The section 10 finite state machine (FR-7.10.1).

The transition table is the *only* place state changes are legal.  An attempt to
leave a state along an edge that is not in the table is a bug in the engine, not
a runtime condition, so it is handled by FR-7.10.1: raise loudly in dev mode,
halt safely otherwise.  ``FAILED_SAFE`` is reachable from every non-terminal
state (**L4**).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set

from ..contracts import RunEventName, State, TERMINAL_STATES
from ..failures import FailureSignal, IllegalTransition
from ..contracts import FailureCode

#: Escape hatches reachable from any non-terminal state.
_UNIVERSAL: FrozenSet[State] = frozenset(
    {State.FAILED_SAFE, State.RUN_HALTED, State.RUN_ABORTED, State.REPORTING}
)

LEGAL_TRANSITIONS: Dict[State, FrozenSet[State]] = {
    State.IDLE: frozenset({State.GATE_CHECK, State.DONE}),
    State.GATE_CHECK: frozenset({State.CAPTURING, State.AWAITING_HUMAN, State.IDLE}),
    State.CAPTURING: frozenset({State.PERCEIVING, State.RECOVERING, State.AWAITING_HUMAN, State.END_DETECTED}),
    State.PERCEIVING: frozenset(
        {
            State.EXTRACTING,
            State.RECOVERING,
            State.NAVIGATING,
            State.END_DETECTED,
            State.AWAITING_HUMAN,
            State.CAPTURING,
        }
    ),
    State.EXTRACTING: frozenset(
        {
            State.DECIDING,
            State.RECOVERING,
            State.NAVIGATING,
            State.END_DETECTED,
            State.AWAITING_HUMAN,
            State.CAPTURING,
            State.PERCEIVING,
        }
    ),
    State.DECIDING: frozenset(
        {
            State.PRE_ACTION_RESOLVE,
            State.PERCEIVING,  # FR-7.5.2 verify_again
            State.AWAITING_HUMAN,
            State.RECOVERING,
            State.NAVIGATING,
            State.END_DETECTED,
        }
    ),
    State.PRE_ACTION_RESOLVE: frozenset({State.ACTING, State.PERCEIVING, State.RECOVERING, State.AWAITING_HUMAN}),
    State.ACTING: frozenset({State.VERIFYING, State.RECOVERING, State.AWAITING_HUMAN}),
    State.VERIFYING: frozenset(
        {
            State.NAVIGATING,
            State.ACTING,  # FR-7.9 single re-click after re-resolution
            State.PERCEIVING,
            State.CAPTURING,  # verified: go observe the next screen
            State.RECOVERING,
            State.AWAITING_HUMAN,
            State.END_DETECTED,
        }
    ),
    State.NAVIGATING: frozenset(
        {
            State.CAPTURING,
            State.ACTING,  # execute the navigation intent
            State.VERIFYING,
            State.END_DETECTED,
            State.RECOVERING,
            State.AWAITING_HUMAN,
        }
    ),
    State.RECOVERING: frozenset(
        {
            State.CAPTURING,
            State.PERCEIVING,
            State.EXTRACTING,
            State.DECIDING,
            State.PRE_ACTION_RESOLVE,
            State.ACTING,
            State.VERIFYING,
            State.NAVIGATING,
            State.AWAITING_HUMAN,
            State.END_DETECTED,
        }
    ),
    State.AWAITING_HUMAN: frozenset(
        {
            State.CAPTURING,
            State.PERCEIVING,
            State.DECIDING,
            State.PRE_ACTION_RESOLVE,
            State.ACTING,
            State.NAVIGATING,
            State.RECOVERING,
            State.END_DETECTED,
            State.DONE,
        }
    ),
    State.END_DETECTED: frozenset({State.REPORTING, State.CAPTURING, State.AWAITING_HUMAN}),
    State.REPORTING: frozenset({State.DONE, State.FAILED_SAFE}),
    State.RUN_HALTED: frozenset({State.REPORTING, State.DONE}),
    State.RUN_ABORTED: frozenset({State.REPORTING, State.DONE}),
    State.DONE: frozenset(),
    # A fail-safe stop still owes the operator a report (L4 + FR-7.12.2).
    State.FAILED_SAFE: frozenset({State.REPORTING, State.DONE}),
}


def legal_targets(state: State) -> FrozenSet[State]:
    """Table entry plus the universal escapes (and a re-entry to self)."""
    base = set(LEGAL_TRANSITIONS.get(state, frozenset()))
    if state not in TERMINAL_STATES:
        base |= set(_UNIVERSAL)
    base.add(state)  # re-entry (polling loops) is not an illegal transition
    return frozenset(base)


def describe_table() -> str:
    lines = [f"{'STATE':22} -> LEGAL TARGETS", "-" * 100]
    for state in State:
        targets = sorted(t.value for t in LEGAL_TRANSITIONS.get(state, frozenset()))
        lines.append(f"{state.value:22} -> {', '.join(targets) if targets else '(terminal)'}")
    lines.append("-" * 100)
    lines.append(f"universal escapes from any non-terminal state: {', '.join(sorted(s.value for s in _UNIVERSAL))}")
    return "\n".join(lines)


@dataclass
class TransitionRecord:
    src: State
    dst: State
    reason: str = ""
    ts: float = 0.0
    illegal: bool = False


class StateMachine:
    """Guarded state holder with a full audit trail."""

    def __init__(
        self,
        initial: State = State.IDLE,
        *,
        strict: bool = True,
        dev_mode: bool = True,
        telemetry: Any = None,
        clock: Any = None,
    ) -> None:
        self._state = initial
        self.strict = strict
        self.dev_mode = dev_mode
        self.telemetry = telemetry
        self._clock = clock
        self.history: List[TransitionRecord] = []
        self.illegal_transitions: int = 0

    @property
    def state(self) -> State:
        return self._state

    @property
    def is_terminal(self) -> bool:
        return self._state in TERMINAL_STATES

    def can(self, target: State) -> bool:
        return target in legal_targets(self._state)

    def transition(self, target: State, *, reason: str = "") -> State:
        import time as _time

        now = _time.time() if self._clock is None else float(self._clock())
        if self._state == target:
            self.history.append(TransitionRecord(target, target, reason or "re-entry", now))
            return self._state
        if not self.can(target):
            self.illegal_transitions += 1
            record = TransitionRecord(self._state, target, reason, now, illegal=True)
            self.history.append(record)
            if self.telemetry is not None:
                self.telemetry.metrics.inc("illegal_transitions")
                self.telemetry.event(
                    RunEventName.ILLEGAL_TRANSITION,
                    state=self._state,
                    module="orchestrator",
                    code=FailureCode.ILLEGAL_TRANSITION,
                    src=self._state.value,
                    dst=target.value,
                    reason=reason[:200],
                )
            message = (
                f"illegal state transition {self._state.value} -> {target.value}"
                + (f" ({reason})" if reason else "")
            )
            if self.dev_mode:
                raise IllegalTransition(message, detail={"src": self._state.value, "dst": target.value, "reason": reason})
            # production: halt safely instead of continuing in an unknown state (L4)
            self._state = State.FAILED_SAFE
            self.history.append(TransitionRecord(target, State.FAILED_SAFE, "halted after illegal transition", now))
            raise FailureSignal(
                FailureCode.ILLEGAL_TRANSITION,
                message + " -- halted safely",
                origin_state=target,
                detail={"src": record.src.value, "dst": record.dst.value},
            )

        self.history.append(TransitionRecord(self._state, target, reason, now))
        previous = self._state
        self._state = target
        if self.telemetry is not None:
            self.telemetry.set_state(target)
            self.telemetry.event(
                RunEventName.STATE_TRANSITION,
                state=target,
                module="orchestrator",
                src=previous.value,
                dst=target.value,
                reason=reason[:200],
            )
            self.telemetry.metrics.inc("state_transitions", label=target.value)
        return self._state

    def force(self, target: State, *, reason: str) -> State:
        """Emergency transition used only on the way to a terminal state."""
        previous = self._state
        self._state = target
        self.history.append(TransitionRecord(previous, target, f"forced: {reason}", illegal=previous not in TERMINAL_STATES))
        if self.telemetry is not None:
            self.telemetry.set_state(target)
        return self._state

    def describe(self) -> Dict[str, Any]:
        return {
            "state": self._state.value,
            "transitions": len(self.history),
            "illegal": self.illegal_transitions,
            "path": [h.dst.value for h in self.history][-25:],
        }


__all__ = ["LEGAL_TRANSITIONS", "StateMachine", "TransitionRecord", "describe_table", "legal_targets"]
