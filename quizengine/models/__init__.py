"""Model provider layer (FR-6.2)."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from ..config import ModelProviderConfig, ModelsConfig
from ..failures import CapabilityError, ConfigError
from .offline import OfflineProvider, normalize_text, payload_from_request
from .openai_compat import OpenAICompatProvider, encode_image
from .provider import ModelProvider, ResilientProvider, api_key_for, enforce_image_budget, extract_json, request_json
from .orchestrator import ModelOrchestrator
from .http_providers import GeminiProvider, NvidiaNimProvider


def build_raw_provider(
    config: ModelProviderConfig,
    *,
    answer_key: Optional[Dict[str, Any]] = None,
    transport: Optional[Any] = None,
) -> ModelProvider:
    """Instantiate one provider from config (no resilience wrapper)."""
    if config.kind == "offline":
        return OfflineProvider(answer_key=answer_key)
    if config.kind in {"openai_compat", "ollama"}:
        provider_config = config
        if config.kind == "ollama" and not config.base_url:
            provider_config = config.model_copy(update={"base_url": "http://127.0.0.1:11434/v1"})
        if not provider_config.model:
            raise ConfigError(f"models provider '{config.name}' needs a model name")
        return OpenAICompatProvider(provider_config, transport=transport)
    raise ConfigError(f"unknown model provider kind: {config.kind!r}")


class ModelStack:
    """The two provider roles the engine needs: solver and Tier-2 perception.

    ``tier2`` falls back to the primary provider when not configured (section 12).
    """

    def __init__(
        self,
        solver: ModelProvider,
        tier2: ModelProvider,
        *,
        offline_answer_key: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.solver = solver
        self.tier2 = tier2
        self.offline_answer_key = offline_answer_key or {}

    @property
    def is_offline(self) -> bool:
        return getattr(self.solver, "kind", "") == "offline" or getattr(
            getattr(self.solver, "primary", None), "kind", ""
        ) == "offline"

    @property
    def tier2_is_offline(self) -> bool:
        return getattr(self.tier2, "kind", "") == "offline" or getattr(
            getattr(self.tier2, "primary", None), "kind", ""
        ) == "offline"

    def describe(self) -> Dict[str, Any]:
        return {
            "solver": self.solver.describe(),
            "tier2": self.tier2.describe(),
            "offline": self.is_offline,
            "tier2_offline": self.tier2_is_offline,
        }

    def cleanup_iteration(self) -> None:
        seen: set[int] = set()
        for provider in (self.solver, self.tier2):
            if id(provider) in seen:
                continue
            seen.add(id(provider))
            cleanup = getattr(provider, "cleanup_iteration", None)
            if callable(cleanup):
                cleanup()

    def cancel_current(self) -> None:
        seen: set[int] = set()
        for provider in (self.solver, self.tier2):
            if id(provider) in seen:
                continue
            seen.add(id(provider))
            cancel = getattr(provider, "cancel_current", None)
            if callable(cancel):
                cancel()

    def close(self) -> None:
        seen: set[int] = set()
        for provider in (self.solver, self.tier2):
            if id(provider) in seen:
                continue
            seen.add(id(provider))
            close = getattr(provider, "close", None)
            if callable(close):
                close()


def build_model_stack(
    config: ModelsConfig,
    *,
    answer_key: Optional[Dict[str, Any]] = None,
    on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    sleep: Optional[Callable[[float], None]] = None,
    transport: Optional[Any] = None,
) -> ModelStack:
    """Build solver + Tier-2 providers with retry/timeout/fallback (FR-6.2)."""
    import time as _time

    sleep_fn = sleep or _time.sleep

    if config.provider in {"gemini", "nvidia"}:
        routed = ModelOrchestrator(config, on_event=on_event, client=transport)
        return ModelStack(routed, routed, offline_answer_key=answer_key)

    def resilient(provider_config: ModelProviderConfig) -> ResilientProvider:
        primary = build_raw_provider(provider_config, answer_key=answer_key, transport=transport)
        # A configured fallback only.  There is deliberately NO automatic
        # fallback to the offline mock: silently substituting a heuristic for a
        # real model would produce confident-looking wrong answers (L4).
        fallback: Optional[ModelProvider] = None
        if config.fallback is not None and provider_config.name != config.fallback.name:
            fallback = build_raw_provider(config.fallback, answer_key=answer_key, transport=transport)
        return ResilientProvider(
            primary,
            fallback,
            retries=provider_config.retries,
            backoff_base_s=provider_config.backoff_base_s,
            timeout_s=provider_config.timeout_s,
            sleep=sleep_fn,
            on_event=on_event,
        )

    solver = resilient(config.primary)
    tier2 = resilient(config.tier2) if config.tier2 is not None else solver
    return ModelStack(solver, tier2, offline_answer_key=answer_key)


__all__ = [
    "GeminiProvider",
    "ModelOrchestrator",
    "ModelProvider",
    "ModelStack",
    "NvidiaNimProvider",
    "OfflineProvider",
    "OpenAICompatProvider",
    "ResilientProvider",
    "api_key_for",
    "build_model_stack",
    "build_raw_provider",
    "encode_image",
    "enforce_image_budget",
    "extract_json",
    "normalize_text",
    "payload_from_request",
    "request_json",
]
