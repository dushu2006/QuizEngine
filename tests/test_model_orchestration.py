from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from quizengine.config import EngineConfig
from quizengine.contracts import ModelMessage, ModelRequest
from quizengine.failures import ConfigError
from quizengine.models.http_providers import GeminiProvider, NvidiaNimProvider
from quizengine.models.orchestrator import ModelOrchestrator
from quizengine.models import build_model_stack


class FakeHTTPResponse:
    def __init__(self, data, status_code=200):
        self._data = data
        self.status_code = status_code

    def json(self):
        return self._data


class FakeClient:
    def __init__(self, outputs=None, *, delay=0.0):
        self.outputs = outputs or {}
        self.delay = delay
        self.calls = []
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()
        self.closed = False

    def post(self, url, *, headers, json, timeout):
        model = json.get("model") or url.rsplit("/models/", 1)[-1].split(":", 1)[0]
        with self.lock:
            self.calls.append({"url": url, "headers": dict(headers), "payload": json, "model": model})
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                time.sleep(self.delay)
            answer = self.outputs.get(model, {"answer": "B", "confidence": 0.97, "rationale": "short"})
            if answer == "MALFORMED":
                text = "not-json"
            else:
                text = json_module_dumps(answer)
            if "generativelanguage" in url:
                return FakeHTTPResponse({"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}]})
            return FakeHTTPResponse({"choices": [{"message": {"content": text}, "finish_reason": "stop"}]})
        finally:
            with self.lock:
                self.active -= 1

    def close(self):
        self.closed = True


def json_module_dumps(payload):
    return json.dumps(payload)


def solve_request(*, confidence_context=None, image=False):
    payload = {
        "task": "answer_single_choice_question",
        "question": "Which answer is correct?",
        "options": ["A. Alpha", "B. Beta", "C. Gamma"],
        "option_count": 3,
        "flags": confidence_context or {},
        "context": {},
    }
    message = ModelMessage(role="user", content=json.dumps(payload), images_b64=["aGVsbG8="] if image else [])
    return ModelRequest(
        task="solve",
        messages=[ModelMessage(role="system", content="Return only JSON"), message],
        response_schema={"type": "object"},
        timeout_s=1.5,
        max_tokens=80,
    )


def configured(provider="gemini", **model_values):
    cfg = EngineConfig.default().models
    cfg.provider = provider
    cfg.request_timeout_s = 1.5
    cfg.max_retries = 0
    cfg.max_tokens = 80
    family = cfg.gemini if provider == "gemini" else cfg.nvidia
    for key, value in model_values.items():
        setattr(family, key, value)
    return cfg


def test_gemini_provider_uses_json_schema_header_key_and_normalized_response(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-secret")
    client = FakeClient()
    family = EngineConfig.default().models.gemini
    provider = GeminiProvider(family, model="configured-gemini-model", client=client, timeout_s=1.5)
    response = provider.complete(solve_request())
    call = client.calls[0]
    assert response.provider == "gemini"
    assert response.model == "configured-gemini-model"
    assert response.parsed["answer"] == "B"
    assert call["url"].endswith("/models/configured-gemini-model:generateContent")
    assert call["headers"]["x-goog-api-key"] == "not-a-real-secret"
    assert "not-a-real-secret" not in call["url"]
    assert call["payload"]["generationConfig"]["responseMimeType"] == "application/json"
    provider.close()


def test_nvidia_provider_uses_openai_compatible_json_mode(monkeypatch):
    monkeypatch.setenv("NVIDIA_NIM_API_KEY", "not-a-real-secret")
    client = FakeClient()
    family = EngineConfig.default().models.nvidia
    provider = NvidiaNimProvider(family, model="configured-nim-model", client=client, timeout_s=1.5)
    response = provider.complete(solve_request())
    call = client.calls[0]
    assert response.provider == "nvidia"
    assert response.model == "configured-nim-model"
    assert response.parsed["answer"] == "B"
    assert call["url"].endswith("/chat/completions")
    assert call["headers"]["Authorization"] == "Bearer not-a-real-secret"
    assert "not-a-real-secret" not in call["url"]
    assert call["payload"]["response_format"] == {"type": "json_object"}
    provider.close()


def test_selected_provider_requires_only_its_key_and_primary_model(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_NIM_API_KEY", raising=False)
    gemini = configured("gemini", primary_model="gemini-primary")
    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        ModelOrchestrator(gemini)
    nvidia = configured("nvidia", primary_model="nim-primary")
    with pytest.raises(ConfigError, match="NVIDIA_NIM_API_KEY"):
        ModelOrchestrator(nvidia)
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    gemini_stack = build_model_stack(gemini)
    assert gemini_stack.solver.kind == "gemini"
    gemini_stack.close()
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.setenv("NVIDIA_NIM_API_KEY", "nim-key")
    nvidia_stack = build_model_stack(nvidia)
    assert nvidia_stack.solver.kind == "nvidia"
    nvidia_stack.close()


def test_fast_path_returns_early_without_calling_other_roles(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    cfg = configured("gemini", primary_model="primary", fast_model="fast", verifier_model="verify")
    client = FakeClient({"fast": {"answer": "B", "confidence": 0.97, "rationale": "short"}})
    router = ModelOrchestrator(cfg, client=client)
    result = router.complete(solve_request())
    assert result.parsed["selected_option_id"] == "option_2"
    assert result.parsed["provider"] == "gemini"
    assert result.parsed["confidence"] == 0.97
    assert [call["model"] for call in client.calls] == ["fast"]
    assert router.metrics["early_exit_count"] == 1
    router.close()
    assert client.closed is False  # injected client lifecycle belongs to caller


def test_low_confidence_routes_primary_and_verifier_in_parallel(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    cfg = configured("gemini", primary_model="primary", fast_model="fast", verifier_model="verify")
    client = FakeClient({
        "fast": {"answer": "A", "confidence": 0.51, "rationale": "uncertain"},
        "primary": {"answer": "B", "confidence": 0.89, "rationale": "short"},
        "verify": {"answer": "B", "confidence": 0.91, "rationale": "short"},
    }, delay=0.08)
    router = ModelOrchestrator(cfg, client=client)
    result = router.complete(solve_request())
    assert result.parsed["answer"] == "B"
    assert result.parsed["verification_status"] == "agreed"
    assert {call["model"] for call in client.calls} == {"fast", "primary", "verify"}
    assert client.max_active == 2
    assert router.metrics["parallel_groups"] == 1
    assert router.metrics["verifier_calls"] == 1
    router.close()


def test_model_disagreement_returns_no_actionable_answer(monkeypatch):
    monkeypatch.setenv("NVIDIA_NIM_API_KEY", "test")
    cfg = configured("nvidia", primary_model="primary", verifier_model="verify")
    client = FakeClient({
        "primary": {"answer": "A", "confidence": 0.94},
        "verify": {"answer": "C", "confidence": 0.92},
    })
    router = ModelOrchestrator(cfg, client=client)
    result = router.complete(solve_request(confidence_context={"has_math": True}))
    assert result.parsed["answer"] is None
    assert result.parsed["verification_status"] == "disagreed"
    assert router.metrics["disagreements"] == 1
    router.close()


def test_visual_request_routes_to_vision_role(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    cfg = configured("gemini", primary_model="primary", fast_model="fast", vision_model="vision", verifier_model="verify")
    client = FakeClient({"vision": {"answer": "C", "confidence": 0.96}})
    router = ModelOrchestrator(cfg, client=client)
    response = router.complete(solve_request(confidence_context={"has_image": True}, image=True))
    assert response.parsed["answer"] == "C"
    assert [call["model"] for call in client.calls] == ["vision"]
    router.close()


def test_malformed_provider_output_fails_closed(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    cfg = configured("gemini", primary_model="primary")
    client = FakeClient({"primary": "MALFORMED"})
    router = ModelOrchestrator(cfg, client=client)
    with pytest.raises(Exception, match="SolverResult contract"):
        router.complete(solve_request())
    router.close()


def test_iteration_cleanup_clears_question_cache_and_counters(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    cfg = configured("gemini", primary_model="primary")
    client = FakeClient()
    router = ModelOrchestrator(cfg, client=client)
    request = solve_request()
    router.complete(request)
    router.complete(request)
    assert router.metrics["cache_hits"] == 1
    assert router.metrics["model_calls_per_question"] == 1
    router.cleanup_iteration()
    assert not router._cache
    assert router.metrics["model_calls_per_question"] == 0
    router.close()
