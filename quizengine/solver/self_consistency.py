"""Self-consistency verification (FR-7.4.3).

When the accepted strategy is a model call and its confidence lands below the
configured floor, re-sample the same question ``N`` times at a non-zero
temperature and take the majority vote.  Final confidence is exactly the
formula from the PRD::

    confidence = agreement_fraction * mean(sample_confidences)

This is *verification of reasoning*, not answer selection: it can only lower or
confirm confidence, and it never overrides a local exact/rule answer.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

from ..config import SolverConfig
from ..contracts import Question, SolverSample, SolverStrategy
from ..models.provider import ModelProvider
from ..prompts import build_solver_request
from .base import SolverContext, StrategyResult, bind_answer_to_index
from .llm import _parse_answer


class SelfConsistency:
    def __init__(
        self,
        config: SolverConfig,
        provider: Optional[ModelProvider] = None,
        *,
        clock: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.provider = provider
        self._clock_fn = clock
        self.stats: Dict[str, Any] = {"runs": 0, "samples": 0, "agreements": 0}

    def applicable(self, result: Optional[StrategyResult]) -> bool:
        if result is None or self.provider is None or not self.config.self_consistency:
            return False
        if result.strategy not in {SolverStrategy.LLM_REASONING, SolverStrategy.VLM_REANALYSIS}:
            return False
        return result.confidence < self.config.self_consistency_threshold

    def run(
        self, question: Question, context: SolverContext, base: StrategyResult
    ) -> Optional[StrategyResult]:
        samples_n = max(2, int(self.config.self_consistency_samples))
        self.stats["runs"] += 1
        samples: List[SolverSample] = []
        votes: List[int] = [base.option_index]
        confidences: List[float] = [float(base.confidence)]

        for position in range(samples_n):
            if context.remaining_s(self._clock()) <= 0.5:
                break
            request = build_solver_request(
                question,
                context=context.prompt_context(),
                image_b64=context.crop_b64 if base.strategy is SolverStrategy.VLM_REANALYSIS else None,
                correlation_id=context.correlation_id,
                temperature=self.config.self_consistency_temperature,
                timeout_s=min(self.config.call_timeout_s, max(1.0, context.remaining_s(self._clock()))),
            )
            try:
                response = self.provider.complete(request)
            except Exception as exc:
                samples.append(
                    SolverSample(
                        index=position,
                        option_index=None,
                        confidence=0.0,
                        rationale=f"sample failed: {type(exc).__name__}: {exc}"[:400],
                    )
                )
                continue
            payload = _parse_answer(response)
            index = bind_answer_to_index(payload.get("answer"), question)
            try:
                confidence = float(payload.get("confidence", 0.5))
            except (TypeError, ValueError):
                confidence = 0.5
            confidence = max(0.0, min(1.0, confidence))
            samples.append(
                SolverSample(
                    index=position,
                    option_index=index,
                    confidence=round(confidence, 4),
                    rationale=str(payload.get("rationale") or "")[:400],
                )
            )
            self.stats["samples"] += 1
            if index is not None:
                votes.append(index)
                confidences.append(confidence)

        if not samples or not votes:
            return None
        counter = Counter(votes)
        winner, winner_votes = counter.most_common(1)[0]
        agreement = winner_votes / float(len(votes))
        mean_confidence = sum(confidences) / float(len(confidences))
        final_confidence = agreement * mean_confidence
        self.stats["agreements"] += winner_votes
        rationale = (
            f"self-consistency: {winner_votes}/{len(votes)} of {len(votes)} sample(s) chose "
            f"option {chr(65 + winner)} (agreement {agreement:.2f} x mean confidence {mean_confidence:.2f})"
        )
        return StrategyResult(
            option_index=int(winner),
            confidence=final_confidence,
            rationale=rationale,
            strategy=base.strategy,
            latency_ms=base.latency_ms,
            raw={**base.raw, "self_consistency": {"votes": dict(counter), "agreement": agreement}},
            samples=samples,
        )

    def _clock(self) -> float:
        if self._clock_fn is not None:
            return float(self._clock_fn())
        import time

        return time.perf_counter()

    def describe(self) -> Dict[str, Any]:
        return {"self_consistency": dict(self.stats)}


__all__ = ["SelfConsistency"]
