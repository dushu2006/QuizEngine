"""Solver strategy interface (PRD section 7.4).

The solver is a **pure function**: ``Question (+ context) -> Decision``.  It has
no screen access, no capture import and no I/O of its own beyond the injected
model provider, which is why it is fully unit-testable on fixtures (NFR-17.2).
"""

from __future__ import annotations

import abc
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..config import SolverConfig
from ..contracts import (
    Decision,
    Frame,
    PerceptionResult,
    Question,
    SolverSample,
    SolverStrategy,
    StrategyAttempt,
)


@dataclass
class SolverContext:
    """Everything the solver may look at besides the Question itself."""

    frame: Optional[Frame] = None
    perception: Optional[PerceptionResult] = None
    #: Pre-computed crop of the question region (FR-7.4.1 step 4).  Prepared by
    #: the orchestrator so the solver never touches the capture module.
    crop_b64: Optional[str] = None
    correlation_id: Optional[str] = None
    deadline: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def remaining_s(self, clock: Callable[[], float]) -> float:
        if not self.deadline:
            return float("inf")
        return max(0.0, self.deadline - clock())

    def prompt_context(self) -> Dict[str, Any]:
        """The slice of context that is safe/useful to send to a model."""
        allowed = ("math_latex", "image_description", "layout_type", "progress", "warnings")
        return {key: self.extra[key] for key in allowed if key in self.extra}


@dataclass
class StrategyResult:
    option_index: int
    confidence: float
    rationale: str
    strategy: SolverStrategy
    latency_ms: float = 0.0
    raw: Dict[str, Any] = field(default_factory=dict)
    samples: Optional[List[SolverSample]] = None


class SolverStrategyBase(abc.ABC):
    """One link of the FR-7.4.1 chain."""

    strategy: SolverStrategy = SolverStrategy.NONE

    def __init__(
        self,
        config: SolverConfig,
        *,
        telemetry: Any = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.config = config
        self.telemetry = telemetry
        self._clock = clock

    def available(self) -> bool:
        return True

    def unavailable_reason(self) -> Optional[str]:
        return None if self.available() else "strategy unavailable"

    @abc.abstractmethod
    def solve(self, question: Question, context: SolverContext) -> Optional[StrategyResult]:
        """Return a result, or ``None`` when this strategy has nothing to say."""

    # -- instrumentation --------------------------------------------------- #
    def run(self, question: Question, context: SolverContext) -> "tuple[Optional[StrategyResult], StrategyAttempt]":
        """Execute with timing, error capture and trace bookkeeping (FR-7.13.4)."""
        started = self._clock()
        if not self.available():
            return None, StrategyAttempt(
                strategy=self.strategy,
                attempted=False,
                skipped_reason=self.unavailable_reason() or "unavailable",
            )
        remaining = context.remaining_s(self._clock)
        if remaining <= 0:
            return None, StrategyAttempt(
                strategy=self.strategy, attempted=False, skipped_reason="solver budget exhausted"
            )
        try:
            result = self.solve(question, context)
        except Exception as exc:  # a strategy failure must not kill the chain (L4)
            latency = (self._clock() - started) * 1000.0
            return None, StrategyAttempt(
                strategy=self.strategy,
                attempted=True,
                produced_answer=False,
                latency_ms=round(latency, 2),
                error=f"{type(exc).__name__}: {exc}",
            )
        latency = (self._clock() - started) * 1000.0
        if result is None:
            return None, StrategyAttempt(
                strategy=self.strategy, attempted=True, produced_answer=False, latency_ms=round(latency, 2)
            )
        attempt = StrategyAttempt(
            strategy=self.strategy,
            attempted=True,
            produced_answer=True,
            option_index=result.option_index,
            confidence=round(float(result.confidence), 4),
            latency_ms=round(latency, 2),
        )
        result.latency_ms = round(latency, 2)
        return result, attempt



# --------------------------------------------------------------------------- #
# answer -> option binding (FR-7.4.2)
# --------------------------------------------------------------------------- #
def bind_answer_to_index(answer: Any, question: Question) -> Optional[int]:
    """Bind a model/knowledge-base answer to an **option index**.

    This is the single place where a letter, an ordinal or an answer string is
    turned into an index against the *validated* Question, which is what kills
    the "answer says B but the options were re-ordered" class of bug.
    """
    if answer is None:
        return None
    count = len(question.options)

    if isinstance(answer, bool):
        return None
    if isinstance(answer, int):
        return answer if 0 <= answer < count else None

    if isinstance(answer, (list, tuple)):
        for item in answer:
            index = bind_answer_to_index(item, question)
            if index is not None:
                return index
        return None

    text = str(answer).strip()
    if not text:
        return None

    # "B", "b.", "(B)", "Option B"
    letter = _extract_letter(text)
    if letter is not None:
        index = ord(letter) - ord("A")
        return index if 0 <= index < count else None

    if re.fullmatch(r"\d+", text):
        value = int(text)
        # A bare number is ambiguous: treat 1..count as 1-based ordinals only
        # when no option text matches it numerically.
        for option in question.options:
            if normalize(option.text) == normalize(text):
                return option.index
        return value - 1 if 1 <= value <= count else (value if 0 <= value < count else None)

    normalized = normalize(text)
    for option in question.options:
        if normalize(option.text) == normalized:
            return option.index
    for option in question.options:
        option_norm = normalize(option.text)
        if option_norm and (option_norm in normalized or normalized in option_norm):
            return option.index
    return None


_LETTER_RE = re.compile(r"^(?:option\s*)?\(?([a-zA-Z])\)?\.?$", re.IGNORECASE)


def _extract_letter(text: str) -> Optional[str]:
    cleaned = text.strip().strip(".").strip()
    match = _LETTER_RE.match(cleaned)
    if match:
        letter = match.group(1).upper()
        return letter if "A" <= letter <= "Z" else None
    if len(cleaned) == 1 and cleaned.isalpha():
        return cleaned.upper()
    return None


def normalize(text: str) -> str:
    return " ".join(re.sub(r"[^0-9a-z\u00c0-\u024f]+", " ", (text or "").lower()).split())


def make_decision(
    question: Question, result: StrategyResult, *, provider: Optional[str] = None, attempts: int = 1
) -> Decision:
    """Build the section 8 ``Decision`` from a strategy result."""
    return Decision(
        question_hash=question.hash,
        option_index=result.option_index,
        strategy=result.strategy,
        confidence=round(float(max(0.0, min(1.0, result.confidence))), 4),
        rationale=result.rationale[:1200],
        samples=result.samples,
        latency_ms=result.latency_ms,
        provider=provider,
        attempts=attempts,
        letter=chr(ord("A") + result.option_index) if result.option_index < 26 else None,
    )


__all__ = [
    "SolverContext",
    "SolverStrategyBase",
    "StrategyResult",
    "bind_answer_to_index",
    "make_decision",
    "normalize",
]
