"""Provider-scoped, latency-aware model routing for runtime inference.

This class implements ``ModelProvider`` and therefore plugs into the existing
Solver/Perception modules without a second vendor-specific code path. It chooses
one role (fast/primary/vision) first, exits immediately when confidence is good,
and only fans out to primary + verifier for an explicitly ambiguous/low-confidence
case. All work stays inside the selected provider family; there is no cross-vendor
fallback.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Any, Dict, List, Optional, Tuple

from ..config import ModelFamilyConfig, ModelsConfig
from ..contracts import ModelRequest, ModelResponse, SolverResult
from ..failures import ConfigError
from .http_providers import GeminiProvider, HTTPProviderError, NvidiaNimProvider
from .offline import payload_from_request
from .provider import ModelProvider, extract_json


class ModelOrchestrator(ModelProvider):
    """One selected vendor family; model roles are sub-routes, not providers."""

    supports_images = True

    def __init__(
        self,
        config: ModelsConfig,
        *,
        on_event: Any = None,
        client: Any = None,
    ) -> None:
        self.config = config
        self.kind = config.provider
        self.name = config.provider
        self.on_event = on_event
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=config.max_parallel_models,
            thread_name_prefix=f"{config.provider}-model",
        )
        self._cancel_event = threading.Event()
        self._closed = False
        self._owns_client = client is None
        self._cache: "OrderedDict[str, ModelResponse]" = OrderedDict()
        self._cache_limit = 128
        self.metrics: Dict[str, Any] = {
            "calls": 0,
            "model_calls_per_question": 0,
            "calls_per_question": [],
            "fast_path_calls": 0,
            "vision_path_calls": 0,
            "verifier_calls": 0,
            "parallel_groups": 0,
            "early_exit_count": 0,
            "cache_hits": 0,
            "retries": 0,
            "failures": 0,
            "disagreements": 0,
            "latency_ms": {},
        }
        if config.provider == "offline":
            self.primary = None
            self.roles: Dict[str, ModelProvider] = {}
            self._owns_client = False
            self._shared_client = None
            return
        if config.provider == "gemini":
            family = config.gemini
            provider_class = GeminiProvider
        elif config.provider == "nvidia":
            family = config.nvidia
            provider_class = NvidiaNimProvider
        else:  # Pydantic normally prevents this; keep factory failure explicit.
            raise ConfigError(f"unsupported model provider {config.provider!r}")

        api_key = _get_key(family)
        shared_client = client
        if shared_client is None:
            try:
                import httpx
            except ImportError as exc:  # pragma: no cover - optional extra
                from ..failures import CapabilityError
                raise CapabilityError("Gemini/NVIDIA providers require httpx; install quizengine[models]") from exc
            shared_client = httpx.Client(
                timeout=httpx.Timeout(config.request_timeout_s, connect=min(3.0, config.request_timeout_s)),
                limits=httpx.Limits(max_connections=config.max_parallel_models + 2,
                                    max_keepalive_connections=config.max_parallel_models + 2,
                                    keepalive_expiry=30.0),
                follow_redirects=False,
            )
            self._shared_client = shared_client
        else:
            self._shared_client = shared_client
        primary_name = family.primary_model or family.fast_model or family.vision_model or family.verifier_model
        if not primary_name:
            raise ConfigError(f"MODEL_PROVIDER={config.provider} requires its PRIMARY_MODEL (FAST_MODEL may serve as primary)")
        role_models: Dict[str, str] = {
            "primary": primary_name,
            "fast": family.fast_model or primary_name,
            "vision": family.vision_model or primary_name,
        }
        if family.verifier_model:
            role_models["verifier"] = family.verifier_model
        self.roles = {
            role: provider_class(
                family,
                model=model_name,
                api_key=api_key,
                timeout_s=config.request_timeout_s,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                client=shared_client,
            )
            for role, model_name in role_models.items()
        }
        self.primary = self.roles["primary"]

    # -- task/provider API ------------------------------------------------- #
    def complete(self, request: ModelRequest) -> ModelResponse:
        if self.kind == "offline":
            from .offline import OfflineProvider
            if not hasattr(self, "_offline"):
                self._offline = OfflineProvider()
            return self._offline.complete(request)
        if self._closed:
            raise RuntimeError("model orchestrator is closed")
        self._cancel_event.clear()
        started = time.perf_counter()
        try:
            if request.task == "solve":
                response = self._solve_routed(request)
            else:
                role = self._route_non_solve(request)
                response = self._call_role(role, request)
            return response
        finally:
            if request.task == "solve":
                elapsed = (time.perf_counter() - started) * 1000.0
                model_ms = float(locals().get("response").latency_ms) if "response" in locals() else 0.0
                self._observe_latency("orchestration_ms", max(0.0, elapsed - model_ms))

    def _solve_routed(self, request: ModelRequest) -> ModelResponse:
        payload = payload_from_request(request)
        flags = payload.get("flags") or {}
        visual = bool(flags.get("has_image") or flags.get("has_diagram") or any(m.images_b64 for m in request.messages))
        complex_text = bool(flags.get("has_math") or flags.get("has_table"))
        selection_started = time.perf_counter()
        family = self.config.gemini if self.kind == "gemini" else self.config.nvidia
        primary_model = family.primary_model or family.fast_model
        fast_available = self.config.enable_fast_path and bool(family.fast_model)
        primary_role = "primary" if primary_model else "fast"

        # Visual content is routed to the configured vision role; text-only stays
        # fast by default. The request contains only the validated question and
        # option list and, when necessary, its already-cropped image.
        if visual:
            role = "vision" if family.vision_model else primary_role
            self.metrics["vision_path_calls"] += 1
        elif fast_available and not complex_text:
            role = "fast"
            self.metrics["fast_path_calls"] += 1
        else:
            role = primary_role

        self._observe_latency("model_selection_ms", (time.perf_counter() - selection_started) * 1000.0)
        if complex_text and family.verifier_model and self.config.enable_parallel_model_calls:
            return self._parallel_consensus(request, primary_role=primary_role, vision_role="vision" if visual else None)

        response = self._call_role(role, request)
        normalized = self._normalize_solver_response(response, payload)
        if normalized is None:
            # Malformed or unbindable output gets one bounded retry, then fails
            # closed into the existing solver strategy/confidence policy.
            response = self._call_role(role, request, force_retry=True)
            normalized = self._normalize_solver_response(response, payload)
        if normalized is None:
            self.metrics["failures"] += 1
            raise HTTPProviderError("model output failed the SolverResult contract")

        confidence = float(normalized["confidence"])
        if confidence >= self.config.primary_confidence_threshold:
            self.metrics["early_exit_count"] += 1
            self._emit("MODEL_EARLY_EXIT", {"provider": self.kind, "role": role, "confidence": confidence})
            return self._as_model_response(normalized, response)

        verifier_model = family.verifier_model
        # A low/medium-confidence response triggers independent primary/verifier
        # calls in parallel when possible; no verifier is called on the fast path
        # when the answer already cleared the confidence threshold.
        if verifier_model and self.config.enable_parallel_model_calls:
            return self._parallel_consensus(
                request,
                primary_role=primary_role,
                vision_role="vision" if visual else None,
                # A low-confidence fast answer is a hint, not a vote. Run the
                # configured primary and verifier as the parallel pair.
                prior=normalized if role != "fast" else None,
            )
        if role == "fast" and primary_role != "fast":
            response = self._call_role(primary_role, request)
            normalized = self._normalize_solver_response(response, payload)
            if normalized is not None and normalized["confidence"] >= self.config.primary_confidence_threshold:
                return self._as_model_response(normalized, response)
        return self._as_model_response(normalized, response)

    def _parallel_consensus(
        self,
        request: ModelRequest,
        *,
        primary_role: str,
        vision_role: Optional[str] = None,
        prior: Optional[Dict[str, Any]] = None,
    ) -> ModelResponse:
        family = self.config.gemini if self.kind == "gemini" else self.config.nvidia
        verifier_model = family.verifier_model
        if not verifier_model:
            role = vision_role or primary_role
            result = self._call_role(role, request)
            normalized = self._normalize_solver_response(result, payload_from_request(request))
            if normalized is None:
                raise HTTPProviderError("model output failed the SolverResult contract")
            return self._as_model_response(normalized, result)

        first_role = vision_role or primary_role
        roles = [first_role]
        verifier_role = "verifier"
        if self._model_for(verifier_role) == self._model_for(roles[0]):
            if prior is not None:
                fallback_response = ModelResponse(
                    provider=self.kind,
                    model=self._model_for(roles[0]),
                    latency_ms=float(prior.get("latency_ms", 0.0)),
                    parsed=prior,
                    text=json.dumps(prior),
                )
                return self._as_model_response(prior, fallback_response)
            single = self._call_role(roles[0], request)
            normalized = self._normalize_solver_response(single, payload_from_request(request))
            if normalized is None:
                raise HTTPProviderError("model output failed the SolverResult contract")
            return self._as_model_response(normalized, single)

        self.metrics["verifier_calls"] += 1
        if self.config.max_parallel_models < 2:
            first = prior
            first_response: Optional[ModelResponse] = None
            if first is None:
                first_response = self._call_role(roles[0], request)
                first = self._normalize_solver_response(first_response, payload_from_request(request))
            verifier_response = self._call_role(verifier_role, request)
            verifier = self._normalize_solver_response(verifier_response, payload_from_request(request))
            if first is None or verifier is None:
                raise HTTPProviderError("primary or verifier output failed the SolverResult contract")
            if first["selected_option_id"] != verifier["selected_option_id"]:
                self.metrics["disagreements"] += 1
                return ModelResponse(
                    text=json.dumps({"answer": None, "confidence": 0.0, "rationale": "configured models disagreed"}),
                    parsed={"answer": None, "confidence": 0.0, "verification_status": "disagreed"},
                    provider=self.kind,
                    model=f"{first['model']}|{verifier['model']}",
                )
            agreement = dict(first)
            agreement["confidence"] = min(0.99, (float(first["confidence"]) + float(verifier["confidence"])) / 2 + 0.05)
            agreement["verification_status"] = "agreed"
            agreement["metadata"] = {"agreement": True, "parallel": False}
            return self._as_model_response(agreement, first_response or verifier_response)

        self.metrics["parallel_groups"] += 1
        if prior is not None:
            futures: Dict[str, Future[ModelResponse]] = {
                verifier_role: self._executor.submit(self._call_role, verifier_role, request)
            }
            primary_result: Optional[ModelResponse] = None
            primary_normalized = prior
        else:
            futures = {
                roles[0]: self._executor.submit(self._call_role, roles[0], request),
                verifier_role: self._executor.submit(self._call_role, verifier_role, request),
            }
            primary_result = None
            primary_normalized = None
        started = time.perf_counter()
        responses: Dict[str, ModelResponse] = {}
        try:
            for name, future in futures.items():
                responses[name] = future.result(timeout=request.timeout_s or self.config.request_timeout_s)
        except FutureTimeout:
            for future in futures.values():
                future.cancel()
            self.metrics["failures"] += 1
            raise HTTPProviderError("parallel model verification timed out") from None
        except Exception:
            for future in futures.values():
                future.cancel()
            self.metrics["failures"] += 1
            raise
        payload = payload_from_request(request)
        if primary_normalized is None:
            primary_result = responses[roles[0]]
            primary_normalized = self._normalize_solver_response(primary_result, payload)
        verify_result = responses[verifier_role]
        verify_normalized = self._normalize_solver_response(verify_result, payload)
        if primary_normalized is None or verify_normalized is None:
            self.metrics["failures"] += 1
            raise HTTPProviderError("primary or verifier output failed the SolverResult contract")

        primary_id = primary_normalized["selected_option_id"]
        verifier_id = verify_normalized["selected_option_id"]
        latency = round((time.perf_counter() - started) * 1000, 2)
        if primary_id != verifier_id:
            self.metrics["disagreements"] += 1
            # Never let a disagreement be turned into a click by a downstream
            # component. The standard LLM parser treats answer=null as no answer.
            return ModelResponse(
                text=json.dumps({"answer": None, "confidence": 0.0, "rationale": "configured models disagreed"}),
                parsed={
                    "answer": None,
                    "selected_option_id": None,
                    "confidence": 0.0,
                    "provider": self.kind,
                    "model": f"{primary_normalized['model']}|{verify_normalized['model']}",
                    "latency_ms": latency,
                    "verification_status": "disagreed",
                },
                provider=self.kind,
                model=f"{primary_normalized['model']}|{verify_normalized['model']}",
                latency_ms=latency,
            )

        confidence = min(0.99, (float(primary_normalized["confidence"]) + float(verify_normalized["confidence"])) / 2 + 0.05)
        result = dict(primary_normalized)
        result["confidence"] = confidence
        result["verification_status"] = "agreed"
        result["metadata"] = {
            "roles": [roles[0], verifier_role],
            "agreement": True,
            "individual_confidences": [primary_normalized["confidence"], verify_normalized["confidence"]],
            "parallel": True,
        }
        result["latency_ms"] = latency
        self._emit("MODEL_CONSENSUS", {"provider": self.kind, "latency_ms": latency, "confidence": confidence})
        return self._as_model_response(result, primary_result or verify_result)

    def _route_non_solve(self, request: ModelRequest) -> str:
        family = self.config.gemini if self.kind == "gemini" else self.config.nvidia
        if request.task in {"perceive", "describe_image", "transcribe_math"} and self.config.enable_tier2_perception:
            return "vision" if family.vision_model else "primary"
        return "primary"

    def _call_role(self, role: str, request: ModelRequest, *, force_retry: bool = False) -> ModelResponse:
        provider = self.roles.get(role) or self.roles.get("primary")
        if provider is None:
            raise ConfigError(f"provider role {role!r} is not configured")
        cache_key = self._cache_key(role, request)
        if self.config.enable_result_cache and not force_retry:
            with self._lock:
                cached = self._cache.get(cache_key)
                if cached is not None:
                    self._cache.move_to_end(cache_key)
                    self.metrics["cache_hits"] += 1
                    return cached.model_copy(deep=True)
        if self._cancel_event.is_set():
            raise HTTPProviderError("model request cancelled")
        limit = 0 if force_retry else self.config.max_retries
        error: Optional[Exception] = None
        for attempt in range(limit + 1):
            start = time.perf_counter()
            try:
                bounded = request.model_copy(update={
                    "timeout_s": min(request.timeout_s, self.config.request_timeout_s),
                    "max_tokens": min(request.max_tokens, self.config.max_tokens),
                    "temperature": self.config.temperature,
                })
                response = provider.complete(bounded)
                latency = round((time.perf_counter() - start) * 1000, 2)
                response.latency_ms = latency
                self.metrics["calls"] += 1
                self.metrics["model_calls_per_question"] += 1
                self.metrics["latency_ms"].setdefault(role, []).append(latency)
                stage_name = {
                    "primary": "primary_model_ms",
                    "fast": "primary_model_ms",
                    "vision": "tier2_perception_ms" if request.task != "solve" else "primary_model_ms",
                    "verifier": "verifier_model_ms",
                }.get(role)
                if stage_name:
                    self._observe_latency(stage_name, latency)
                if self.config.enable_result_cache:
                    with self._lock:
                        self._cache[cache_key] = response.model_copy(deep=True)
                        self._cache.move_to_end(cache_key)
                        while len(self._cache) > self._cache_limit:
                            self._cache.popitem(last=False)
                self._emit("MODEL_ROLE_CALL", {
                    "provider": self.kind,
                    "role": role,
                    "model": self._model_for(role),
                    "latency_ms": latency,
                    "correlation_id": request.correlation_id,
                })
                return response
            except Exception as exc:
                error = exc
                if attempt < limit and _retryable(exc):
                    self.metrics["retries"] += 1
                    time.sleep(min(0.15 * (attempt + 1), 0.3))
                    continue
                break
        self.metrics["failures"] += 1
        if error:
            raise error
        raise HTTPProviderError("provider request failed")

    def _normalize_solver_response(self, response: ModelResponse, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        parsed = response.parsed or extract_json(response.text)
        if not isinstance(parsed, dict):
            return None
        raw_id = parsed.get("selected_option_id")
        answer = parsed.get("answer")
        options = payload.get("options") or []
        index: Optional[int] = None
        if isinstance(raw_id, str):
            match = re.fullmatch(r"option_([1-9][0-9]*)", raw_id)
            if match:
                candidate = int(match.group(1)) - 1
                if 0 <= candidate < len(options):
                    index = candidate
        if index is None and isinstance(answer, str):
            raw = answer.strip().upper()
            match = re.fullmatch(r"(?:OPTION[_ ]?)?([A-Z])(?:\..*)?", raw)
            if match:
                candidate = ord(match.group(1)) - ord("A")
                if 0 <= candidate < len(options):
                    index = candidate
            else:
                match = re.fullmatch(r"(?:OPTION[_ ]?)?([1-9][0-9]*)", raw)
                if match and 0 < int(match.group(1)) <= len(options):
                    index = int(match.group(1)) - 1
        if index is None:
            return None
        try:
            confidence = float(parsed.get("confidence", 0.0))
        except (TypeError, ValueError):
            return None
        if not 0.0 <= confidence <= 1.0:
            return None
        return SolverResult(
            selected_option_id=f"option_{index + 1}",
            confidence=confidence,
            provider=self.kind,
            model=response.model,
            latency_ms=max(0.0, response.latency_ms),
            answer_text=str(parsed.get("answer_text") or "")[:160] or None,
            metadata={"role_response": parsed.get("metadata", {})},
        ).model_dump(mode="json") | {
            "answer": chr(ord("A") + index),
            "rationale": str(parsed.get("rationale") or "")[:120],
        }

    def _as_model_response(self, result: Dict[str, Any], response: ModelResponse) -> ModelResponse:
        data = dict(result)
        data.setdefault("provider", self.kind)
        data.setdefault("model", response.model)
        data.setdefault("latency_ms", response.latency_ms)
        answer = data.get("answer")
        text = json.dumps({
            "answer": answer,
            "confidence": data.get("confidence", 0.0),
            "rationale": data.get("rationale", ""),
        }, ensure_ascii=False)
        return ModelResponse(
            text=text,
            parsed=data,
            provider=self.kind,
            model=str(data["model"]),
            latency_ms=float(data["latency_ms"]),
            finish_reason="stop",
            attempts=response.attempts,
        )

    def _model_for(self, role: str) -> str:
        provider = self.roles.get(role) or self.roles.get("primary")
        return str(getattr(provider, "model", ""))

    @staticmethod
    def _cache_key(role: str, request: ModelRequest) -> str:
        compact = request.model_dump_json(exclude={"correlation_id", "timeout_s"})
        return hashlib.sha256(f"{role}:{compact}".encode("utf-8")).hexdigest()

    def cancel_current(self) -> None:
        self._cancel_event.set()

    def cleanup_iteration(self) -> None:
        """Drop per-question cache and cancellation state in all exit paths."""
        with self._lock:
            self._cache.clear()
            self.metrics["calls_per_question"].append(int(self.metrics["model_calls_per_question"]))
        self.metrics["model_calls_per_question"] = 0
        self._cancel_event.clear()

    def _observe_latency(self, name: str, value_ms: float) -> None:
        self.metrics["latency_ms"].setdefault(name, []).append(max(0.0, float(value_ms)))

    def latency_report(self) -> Dict[str, Dict[str, float | int]]:
        result: Dict[str, Dict[str, float | int]] = {}
        for role, values in self.metrics["latency_ms"].items():
            ordered = sorted(values)
            result[role] = {
                "n": len(ordered),
                "mean_ms": round(sum(ordered) / len(ordered), 2),
                "median_ms": round(_percentile(ordered, 50), 2),
                "p95_ms": round(_percentile(ordered, 95), 2),
            }
        return result

    def describe(self) -> Dict[str, Any]:
        roles = {role: provider.describe() for role, provider in self.roles.items()}
        return {
            "name": self.kind,
            "kind": self.kind,
            "supports_images": True,
            "primary": roles.get("primary", {"name": self.kind, "kind": self.kind}),
            "roles": roles,
            "timeout_s": self.config.request_timeout_s,
            "max_parallel_models": self.config.max_parallel_models,
            "offline": self.kind == "offline",
            "metrics": self.metrics,
            "latency_report": self.latency_report(),
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._cancel_event.set()
        self._executor.shutdown(wait=False, cancel_futures=True)
        seen: set[int] = set()
        for provider in self.roles.values():
            if id(provider) not in seen:
                seen.add(id(provider))
                provider.close()
        if self._owns_client:
            try:
                self._shared_client.close()
            except Exception:
                pass

    def _emit(self, event: str, payload: Dict[str, Any]) -> None:
        if self.on_event:
            try:
                self.on_event(event, payload)
            except Exception:
                pass


def _get_key(family: ModelFamilyConfig) -> str:
    import os
    key = os.environ.get(family.api_key_env, "")
    if not key:
        raise ConfigError(f"selected provider requires environment variable {family.api_key_env}")
    return key


def _retryable(exc: Exception) -> bool:
    if isinstance(exc, HTTPProviderError):
        if exc.status_code is not None:
            return exc.status_code == 429 or exc.status_code >= 500
        return True
    return False


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


__all__ = ["ModelOrchestrator"]
