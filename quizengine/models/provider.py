"""Model provider abstraction (FR-6.2).

Rules enforced here:

* No solver or perception module may import a vendor SDK directly -- they talk to
  :class:`ModelProvider` only.
* Every call has a timeout and a bounded retry count with exponential backoff,
  then an optional fallback provider, then a classified :class:`FailureSignal`
  (``MODEL_TIMEOUT``).  Nothing waits forever (**L3**).
* Cloud providers are opt-in and receive only the crops the task needs
  (FR-16.1); keys come from environment variables only (FR-16.2).
"""

from __future__ import annotations

import abc
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

from ..config import ModelProviderConfig
from ..contracts import FailureCode, ModelRequest, ModelResponse, State
from ..failures import FailureSignal

T = TypeVar("T")

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class ModelProvider(abc.ABC):
    """One completion backend."""

    name: str = "abstract"
    kind: str = "abstract"
    supports_images: bool = False

    @abc.abstractmethod
    def complete(self, request: ModelRequest) -> ModelResponse:
        """Single attempt, no retries -- resilience lives in :class:`ResilientProvider`."""

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "supports_images": self.supports_images}


class ResilientProvider(ModelProvider):
    """Timeout + retry + backoff + fallback (FR-6.2, FR-7.4.4)."""

    def __init__(
        self,
        primary: ModelProvider,
        fallback: Optional[ModelProvider] = None,
        *,
        retries: int = 2,
        backoff_base_s: float = 0.5,
        timeout_s: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.perf_counter,
        on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        max_workers: int = 2,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.retries = max(0, int(retries))
        self.backoff_base_s = float(backoff_base_s)
        self.timeout_s = float(timeout_s)
        self._sleep = sleep
        self._clock = clock
        self._on_event = on_event
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="model")
        self.name = primary.name
        self.kind = primary.kind
        self.supports_images = primary.supports_images
        self.stats: Dict[str, Any] = {"calls": 0, "timeouts": 0, "fallbacks": 0, "errors": 0, "latency_ms": 0.0}

    # -- execution --------------------------------------------------------- #
    def complete(self, request: ModelRequest) -> ModelResponse:
        timeout = float(request.timeout_s or self.timeout_s)
        chain: List[ModelProvider] = [self.primary] + ([self.fallback] if self.fallback is not None else [])
        errors: List[str] = []

        for provider_index, provider in enumerate(chain):
            if provider_index > 0:
                self.stats["fallbacks"] += 1
                self._notify("MODEL_FALLBACK", {"provider": provider.name, "reason": "; ".join(errors[-2:])})
            for attempt in range(1, self.retries + 2):
                started = self._clock()
                self.stats["calls"] += 1
                try:
                    future = self._executor.submit(provider.complete, request)
                    response = future.result(timeout=timeout)
                except FutureTimeout:
                    elapsed = (self._clock() - started) * 1000.0
                    self.stats["timeouts"] += 1
                    errors.append(f"{provider.name} attempt {attempt} timed out after {timeout:.1f}s")
                    self._notify(
                        "MODEL_TIMEOUT",
                        {"provider": provider.name, "attempt": attempt, "timeout_s": timeout, "latency_ms": elapsed},
                    )
                except Exception as exc:
                    elapsed = (self._clock() - started) * 1000.0
                    self.stats["errors"] += 1
                    errors.append(f"{provider.name} attempt {attempt} failed: {type(exc).__name__}: {exc}")
                    self._notify("MODEL_ERROR", {"provider": provider.name, "attempt": attempt, "error": str(exc)})
                else:
                    elapsed = (self._clock() - started) * 1000.0
                    self.stats["latency_ms"] += elapsed
                    response.latency_ms = round(elapsed, 2)
                    response.attempts = attempt
                    self._notify(
                        "MODEL_CALL",
                        {
                            "provider": response.provider,
                            "task": request.task,
                            "attempt": attempt,
                            "latency_ms": round(elapsed, 2),
                            "correlation_id": request.correlation_id,
                        },
                    )
                    return response

                if attempt <= self.retries and self.backoff_base_s > 0:
                    self._sleep(self.backoff_base_s * (2 ** (attempt - 1)))

        raise FailureSignal(
            FailureCode.MODEL_TIMEOUT,
            "all model providers failed: " + "; ".join(errors) if errors else "no provider available",
            origin_state=State.DECIDING,
            detail={"errors": errors, "primary": self.primary.name, "fallback": self.fallback.name if self.fallback else None},
        )

    def _notify(self, event: str, detail: Dict[str, Any]) -> None:
        if self._on_event is not None:
            try:
                self._on_event(event, detail)
            except Exception:  # pragma: no cover - telemetry must never break a call
                pass

    def close(self) -> None:
        self._executor.shutdown(wait=False)

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "supports_images": self.supports_images,
            "primary": self.primary.describe(),
            "fallback": self.fallback.describe() if self.fallback else None,
            "retries": self.retries,
            "timeout_s": self.timeout_s,
            "stats": dict(self.stats),
        }


# --------------------------------------------------------------------------- #
# JSON-mode helpers (FR-7.2.6 strict schema output, FR-7.4.1 structured output)
# --------------------------------------------------------------------------- #
def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort extraction of a JSON object from a model response."""
    if not text:
        return None
    candidates: List[str] = []
    fenced = _JSON_FENCE.findall(text)
    candidates.extend(fenced)
    candidates.append(text)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def request_json(
    provider: ModelProvider,
    request: ModelRequest,
    parse: Callable[[Dict[str, Any]], T],
    *,
    retries: int = 2,
    feedback_template: str = "Your previous output was invalid: {error}. Respond with ONLY a JSON object matching this schema: {schema}",
) -> T:
    """Call a provider and schema-validate the result, with error-feedback re-prompts.

    Used by Tier-2 perception (FR-7.2.6: 2 retries with error feedback, then
    ``PERCEPTION_LOW_CONFIDENCE``) and by the LLM solver (FR-7.4.1).
    """
    schema_text = json.dumps(request.response_schema or {}, ensure_ascii=False)[:2000]
    current = request
    last_error = "no response"
    for attempt in range(retries + 1):
        response = provider.complete(current)
        if not response.ok:
            last_error = response.error or "provider error"
        else:
            payload = response.parsed if response.parsed is not None else extract_json(response.text)
            if payload is None:
                last_error = f"response was not JSON: {response.text[:200]!r}"
            else:
                try:
                    return parse(payload)
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
        if attempt >= retries:
            break
        feedback = feedback_template.format(error=last_error, schema=schema_text)
        messages = list(current.messages) + [
            {"role": "assistant", "content": response.text[:1500] if response.ok else ""},
            {"role": "user", "content": feedback},
        ]
        current = current.model_copy(update={"messages": messages})

    raise FailureSignal(
        FailureCode.MODEL_SCHEMA_FAILURE,
        f"model output failed schema validation after {retries + 1} attempt(s): {last_error}",
        origin_state=State.PERCEIVING,
        detail={"attempts": retries + 1, "last_error": last_error, "task": request.task},
    )


# --------------------------------------------------------------------------- #
# image budget (FR-16.1)
# --------------------------------------------------------------------------- #
def enforce_image_budget(images_b64: List[str], max_images: int = 2, max_bytes: int = 3_000_000) -> List[str]:
    """Never send more crops than the task needs; drop the largest when over budget."""
    if not images_b64:
        return []
    selected = images_b64[:max_images]
    while selected and sum(len(i) for i in selected) > max_bytes:
        selected.pop(int(_argmax([len(i) for i in selected])))
    return selected


def _argmax(values: List[int]) -> int:
    best, best_value = 0, -1
    for index, value in enumerate(values):
        if value > best_value:
            best, best_value = index, value
    return best


def api_key_for(config: ModelProviderConfig) -> Optional[str]:
    """FR-16.2: keys come from the environment, never from config or disk."""
    if not config.api_key_env:
        return None
    return os.environ.get(config.api_key_env) or None


__all__ = [
    "ModelProvider",
    "ResilientProvider",
    "api_key_for",
    "enforce_image_budget",
    "extract_json",
    "request_json",
]
