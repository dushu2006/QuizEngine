"""LLM reasoning strategy (FR-7.4.1 step 3) and VLM re-analysis (step 4).

Both go through the same prompt builder and answer binder; the VLM variant
attaches the pre-computed question crop (never the full frame, FR-16.1) and is
skipped when the configured provider cannot take images.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..config import SolverConfig
from ..contracts import Question, SolverStrategy
from ..models.provider import ModelProvider, extract_json
from ..prompts import build_solver_request
from .base import SolverContext, SolverStrategyBase, StrategyResult, bind_answer_to_index


def _parse_answer(response: Any) -> Dict[str, Any]:
    payload = getattr(response, "parsed", None) or {}
    if not isinstance(payload, dict) or "answer" not in payload:
        extracted = extract_json(getattr(response, "text", "") or "")
        if isinstance(extracted, dict):
            merged = dict(extracted)
            merged.update({k: v for k, v in payload.items() if k in {"confidence", "rationale"}})
            payload = merged
    return payload if isinstance(payload, dict) else {}


def _result_from_payload(
    payload: Dict[str, Any], question: Question, strategy: SolverStrategy, response: Any
) -> Optional[StrategyResult]:
    raw_answer = payload.get("answer")
    index = bind_answer_to_index(raw_answer, question)
    if index is None:
        return None
    try:
        confidence = float(payload.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))
    rationale = str(payload.get("rationale") or "").strip() or f"{strategy.value} answered {chr(65 + index)}"
    return StrategyResult(
        option_index=index,
        confidence=confidence,
        rationale=rationale,
        strategy=strategy,
        raw={
            "answer": raw_answer,
            "model": getattr(response, "model", "") or "",
            "provider": getattr(response, "provider", "") or "",
            "latency_ms": getattr(response, "latency_ms", 0.0),
            "attempts": getattr(response, "attempts", 1),
        },
    )


class LLMStrategy(SolverStrategyBase):
    strategy = SolverStrategy.LLM_REASONING

    def __init__(self, config: SolverConfig, provider: Optional[ModelProvider] = None, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self.provider = provider

    def available(self) -> bool:
        return self.provider is not None

    def unavailable_reason(self) -> str:
        return "no model provider configured (offline mode)"

    def solve(self, question: Question, context: SolverContext) -> Optional[StrategyResult]:
        request = build_solver_request(
            question,
            context=context.prompt_context(),
            image_b64=None,
            correlation_id=context.correlation_id,
            temperature=0.0,
            timeout_s=min(self.config.call_timeout_s, max(1.0, context.remaining_s(self._clock))),
        )
        response = self.provider.complete(request)  # type: ignore[union-attr]
        return _result_from_payload(_parse_answer(response), question, self.strategy, response)


class VLMStrategy(SolverStrategyBase):
    """FR-7.4.1 step 4: re-analyse the question region with the visual model."""

    strategy = SolverStrategy.VLM_REANALYSIS

    def __init__(self, config: SolverConfig, provider: Optional[ModelProvider] = None, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self.provider = provider

    def available(self) -> bool:
        return self.provider is not None and bool(getattr(self.provider, "supports_images", False))

    def unavailable_reason(self) -> str:
        if self.provider is None:
            return "no model provider configured (offline mode)"
        return "configured provider does not accept images"

    def solve(self, question: Question, context: SolverContext) -> Optional[StrategyResult]:
        if not context.crop_b64:
            return None
        payload_context = dict(context.prompt_context())
        payload_context["visual"] = "question crop attached; re-read the options from the image"
        request = build_solver_request(
            question,
            context=payload_context,
            image_b64=context.crop_b64,
            correlation_id=context.correlation_id,
            temperature=0.0,
            timeout_s=min(self.config.call_timeout_s, max(1.0, context.remaining_s(self._clock))),
        )
        response = self.provider.complete(request)  # type: ignore[union-attr]
        return _result_from_payload(_parse_answer(response), question, self.strategy, response)


__all__ = ["LLMStrategy", "VLMStrategy"]
