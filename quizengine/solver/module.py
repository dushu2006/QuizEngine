"""Solver Module (PRD section 7.4).

Runs the FR-7.4.1 cascade, binds the winning answer to an option index
(FR-7.4.2), optionally verifies it by self-consistency (FR-7.4.3) and enforces
the per-question budget (FR-7.4.4).  Emits a complete ``Decision`` plus the
per-strategy attempt trace required by FR-7.13.4.

The module is pure: no capture, no actuator, no filesystem beyond the optional
knowledge base it is handed at construction.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..config import EngineConfig
from ..contracts import (
    Decision,
    FailureCode,
    Question,
    RunEventName,
    SolverStrategy,
    State,
    StrategyAttempt,
)
from ..failures import FailureSignal
from ..models.provider import ModelProvider
from .base import SolverContext, SolverStrategyBase, StrategyResult, make_decision
from .knowledge import KnowledgeBase, LocalExactStrategy, LocalFuzzyStrategy
from .llm import LLMStrategy, VLMStrategy
from .rules import LocalRulesStrategy
from .self_consistency import SelfConsistency

_CHAIN_FACTORIES: Dict[SolverStrategy, Callable[..., SolverStrategyBase]] = {
    SolverStrategy.LOCAL_EXACT: lambda config, provider, knowledge, telemetry, clock: LocalExactStrategy(
        config, knowledge=knowledge, telemetry=telemetry, clock=clock
    ),
    SolverStrategy.LOCAL_FUZZY: lambda config, provider, knowledge, telemetry, clock: LocalFuzzyStrategy(
        config, knowledge=knowledge, telemetry=telemetry, clock=clock
    ),
    SolverStrategy.LOCAL_RULES: lambda config, provider, knowledge, telemetry, clock: LocalRulesStrategy(
        config, telemetry=telemetry, clock=clock
    ),
    SolverStrategy.LLM_REASONING: lambda config, provider, knowledge, telemetry, clock: LLMStrategy(
        config, provider=provider, telemetry=telemetry, clock=clock
    ),
    SolverStrategy.VLM_REANALYSIS: lambda config, provider, knowledge, telemetry, clock: VLMStrategy(
        config, provider=provider, telemetry=telemetry, clock=clock
    ),
}


@dataclass
class SolverOutcome:
    decision: Optional[Decision]
    attempts: List[StrategyAttempt] = field(default_factory=list)
    accepted: Optional[StrategyResult] = None
    latency_ms: float = 0.0
    budget_exceeded: bool = False
    provider_error: Optional[str] = None
    self_consistency_used: bool = False
    warnings: List[str] = field(default_factory=list)

    def describe(self) -> Dict[str, Any]:
        return {
            "option_index": self.decision.option_index if self.decision else None,
            "letter": self.decision.letter if self.decision else None,
            "confidence": self.decision.confidence if self.decision else None,
            "strategy": self.decision.strategy.value if self.decision else None,
            "attempts": [a.model_dump(mode="json", exclude_none=True) for a in self.attempts],
            "latency_ms": round(self.latency_ms, 2),
            "budget_exceeded": self.budget_exceeded,
            "self_consistency_used": self.self_consistency_used,
            "provider_error": self.provider_error,
            "warnings": list(self.warnings),
        }


class SolverModule:
    def __init__(
        self,
        config: EngineConfig,
        *,
        provider: Optional[ModelProvider] = None,
        knowledge: Optional[KnowledgeBase] = None,
        telemetry: Any = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.config = config
        self.solver_config = config.solver
        self.telemetry = telemetry
        self._clock = clock
        self.provider = provider
        self.knowledge = knowledge if knowledge is not None else self._load_knowledge()
        self.self_consistency = SelfConsistency(self.solver_config, provider=provider, clock=clock)
        self.chain: List[SolverStrategyBase] = []
        self.unknown_strategies: List[str] = []
        self._build_chain()
        self.stats: Dict[str, Any] = {"questions": 0, "accepted": 0, "no_answer": 0, "budget_exceeded": 0}

    # -- construction ------------------------------------------------------- #
    def _load_knowledge(self) -> Optional[KnowledgeBase]:
        configured = self.solver_config.knowledge_base
        if not configured:
            return None
        candidates = [Path(configured), Path.cwd() / configured, self.config.paths.knowledge_dir / Path(configured).name]
        for candidate in candidates:
            try:
                if candidate.is_file():
                    return KnowledgeBase.load(candidate, fuzzy_threshold=self.solver_config.fuzzy_threshold)
            except Exception:
                continue
        return None

    def _build_chain(self) -> None:
        for strategy in self.solver_config.strategy_chain:
            factory = _CHAIN_FACTORIES.get(strategy)
            if factory is None:
                self.unknown_strategies.append(strategy.value)
                continue
            self.chain.append(
                factory(self.solver_config, self.provider, self.knowledge, self.telemetry, self._clock)
            )

    # -- main --------------------------------------------------------------- #
    def solve(
        self,
        question: Question,
        context: Optional[SolverContext] = None,
        *,
        correlation_id: Optional[str] = None,
    ) -> SolverOutcome:
        started = self._clock()
        self.stats["questions"] += 1
        context = context or SolverContext()
        if correlation_id and not context.correlation_id:
            context.correlation_id = correlation_id
        if not context.deadline:
            budget = min(self.solver_config.budget_s, self.config.budgets.per_question_budget_s or self.solver_config.budget_s)
            context.deadline = self._clock() + budget

        warnings: List[str] = []
        if question.flags.has_image and not context.crop_b64:
            warnings.append("question has an image flag but no crop was supplied; VLM re-analysis is skipped")

        attempts: List[StrategyAttempt] = []
        best: Optional[StrategyResult] = None
        accepted: Optional[StrategyResult] = None
        provider_error: Optional[str] = None
        budget_exceeded = False

        for strategy in self.chain:
            if context.remaining_s(self._clock) <= 0:
                budget_exceeded = True
                attempts.append(
                    StrategyAttempt(
                        strategy=strategy.strategy, attempted=False, skipped_reason="per-question solver budget exhausted"
                    )
                )
                continue
            result, attempt = strategy.run(question, context)
            attempts.append(attempt)
            if attempt.error:
                provider_error = attempt.error
                if self.telemetry is not None:
                    self.telemetry.event(
                        RunEventName.SOLVER_STRATEGY_FAILED,
                        state=State.DECIDING,
                        module="solver",
                        strategy=strategy.strategy.value,
                        error=attempt.error[:200],
                        correlation_id=context.correlation_id,
                    )
            if result is None:
                continue
            if best is None or result.confidence > best.confidence:
                best = result
            if result.confidence >= self.solver_config.accept_threshold:
                accepted = result
                break

        # FR-7.4.1.4: escalate to the VLM when the content flags say so, even if
        # a text-only answer already looked acceptable.
        if (
            accepted is not None
            and self.solver_config.vlm_on_flags
            and (question.flags.has_math or question.flags.has_image)
            and accepted.strategy is SolverStrategy.LLM_REASONING
            and context.crop_b64
            and context.remaining_s(self._clock) > 1.0
        ):
            vlm = next((s for s in self.chain if isinstance(s, VLMStrategy)), None)
            if vlm is not None:
                vlm_result, vlm_attempt = vlm.run(question, context)
                attempts.append(vlm_attempt)
                if vlm_result is not None and vlm_result.option_index != accepted.option_index:
                    warnings.append(
                        f"VLM re-analysis disagrees with the LLM answer "
                        f"(option {chr(65 + vlm_result.option_index)} vs {chr(65 + accepted.option_index)}); "
                        "keeping the higher-confidence result"
                    )
                    if vlm_result.confidence > accepted.confidence:
                        accepted = vlm_result
                        best = vlm_result

        if accepted is None:
            accepted = best

        self_consistency_used = False
        if accepted is not None and self.self_consistency.applicable(accepted):
            verified = self.self_consistency.run(question, context, accepted)
            if verified is not None:
                self_consistency_used = True
                attempts.append(
                    StrategyAttempt(
                        strategy=SolverStrategy.SELF_CONSISTENCY,
                        attempted=True,
                        produced_answer=True,
                        option_index=verified.option_index,
                        confidence=round(verified.confidence, 4),
                        latency_ms=verified.latency_ms,
                    )
                )
                accepted = verified

        latency_ms = (self._clock() - started) * 1000.0
        if accepted is None:
            self.stats["no_answer"] += 1
            if budget_exceeded:
                self.stats["budget_exceeded"] += 1
            code = FailureCode.SOLVER_TIMEOUT if budget_exceeded else FailureCode.SOLVER_NO_ANSWER
            raise FailureSignal(
                code,
                f"no strategy produced an answer for question {question.hash[:16]} "
                f"({len(attempts)} attempt(s): {', '.join(a.strategy.value for a in attempts) or 'none'})",
                origin_state=State.DECIDING,
                detail={
                    "attempts": [a.model_dump(mode="json", exclude_none=True) for a in attempts],
                    "provider_error": provider_error,
                    "budget_exceeded": budget_exceeded,
                    "latency_ms": round(latency_ms, 2),
                },
            )

        self.stats["accepted"] += 1
        if budget_exceeded:
            self.stats["budget_exceeded"] += 1
        decision = make_decision(
            question,
            accepted,
            provider=(accepted.raw.get("provider") or (self.provider.name if self.provider else None)),
            attempts=len([a for a in attempts if a.attempted]),
        )
        outcome = SolverOutcome(
            decision=decision,
            attempts=attempts,
            accepted=accepted,
            latency_ms=latency_ms,
            budget_exceeded=budget_exceeded,
            provider_error=provider_error,
            self_consistency_used=self_consistency_used,
            warnings=warnings,
        )
        if self.telemetry is not None:
            self.telemetry.event(
                RunEventName.DECISION_MADE,
                state=State.DECIDING,
                module="solver",
                question_hash=question.hash,
                option_index=decision.option_index,
                letter=decision.letter,
                confidence=decision.confidence,
                strategy=decision.strategy.value,
                latency_ms=round(latency_ms, 2),
                correlation_id=context.correlation_id,
            )
        return outcome

    def describe(self) -> Dict[str, Any]:
        return {
            "stats": dict(self.stats),
            "chain": [s.strategy.value for s in self.chain],
            "available": {s.strategy.value: s.available() for s in self.chain},
            "skipped": {s.strategy.value: s.unavailable_reason() for s in self.chain if not s.available()},
            "knowledge_entries": len(self.knowledge) if self.knowledge else 0,
            "unknown_strategies": list(self.unknown_strategies),
            **self.self_consistency.describe(),
        }


__all__ = ["SolverModule", "SolverOutcome"]
