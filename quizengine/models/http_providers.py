"""Persistent HTTP providers: Google Gemini and NVIDIA NIM.

Both normalize into :class:`ModelResponse`; vendor response shapes stop at this
module. API keys are headers only, never query parameters, prompts, logs, or
provider descriptions. One ``httpx.Client`` is reused for connection pooling.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from ..config import ModelFamilyConfig, ModelsConfig
from ..contracts import ModelMessage, ModelRequest, ModelResponse
from ..failures import CapabilityError, ConfigError
from .provider import ModelProvider, extract_json


class HTTPProviderError(RuntimeError):
    """Sanitized network/provider error (never includes request headers)."""

    def __init__(self, message: str, *, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class _PersistentHTTPProvider(ModelProvider):
    def __init__(self, *, model: str, api_key: str, base_url: str, timeout_s: float, client: Any = None) -> None:
        if not model.strip():
            raise ConfigError(f"{self.name} needs a configured model name")
        if not api_key:
            raise ConfigError(f"{self.name} API key is missing; set the configured environment variable")
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - optional extra
            raise CapabilityError("HTTP model providers require httpx; install quizengine[models]") from exc
        self.model = model.strip()
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self._owns_client = client is None
        self.client = client or httpx.Client(
            timeout=httpx.Timeout(self.timeout_s, connect=min(3.0, self.timeout_s)),
            limits=httpx.Limits(max_connections=8, max_keepalive_connections=4, keepalive_expiry=30.0),
            follow_redirects=False,
        )

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def _post(self, url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
        try:
            response = self.client.post(url, headers=headers, json=payload, timeout=max(0.1, timeout_s))
        except Exception as exc:
            # Do not forward exception reprs: some HTTP libraries include URLs or
            # request detail. The secret key itself is never part of those URLs.
            raise HTTPProviderError(f"network error ({type(exc).__name__})") from None
        if response.status_code >= 400:
            if response.status_code == 429:
                message = "provider rate limit (HTTP 429)"
            else:
                message = f"provider HTTP {response.status_code}"
            raise HTTPProviderError(message, status_code=response.status_code)
        try:
            data = response.json()
        except Exception:
            raise HTTPProviderError("provider returned invalid JSON") from None
        if not isinstance(data, dict):
            raise HTTPProviderError("provider response must be a JSON object")
        return data

    @staticmethod
    def _extract_content(request: ModelRequest) -> tuple[Optional[str], List[Dict[str, Any]]]:
        system: Optional[str] = None
        messages: List[Dict[str, Any]] = []
        for item in request.messages:
            if item.role == "system":
                system = (system + "\n" if system else "") + item.content
                continue
            parts: List[Dict[str, Any]] = [{"type": "text", "text": item.content}]
            for image_b64 in item.images_b64[:2]:
                if image_b64.startswith("data:"):
                    data_url = image_b64
                else:
                    data_url = f"data:image/png;base64,{image_b64}"
                parts.append({"type": "image_url", "image_url": {"url": data_url}})
            content: Any = parts if len(parts) > 1 else item.content
            messages.append({"role": "assistant" if item.role == "assistant" else "user", "content": content})
        return system, messages


def _json_schema_to_gemini(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Translate the useful JSON Schema subset into Gemini's Schema wire form."""
    if not isinstance(schema, dict):
        return {}
    out: Dict[str, Any] = {}
    type_value = schema.get("type")
    if isinstance(type_value, list):
        type_value = next((item for item in type_value if item != "null"), "object")
    if type_value:
        out["type"] = str(type_value).upper()
    if "description" in schema:
        out["description"] = schema["description"]
    if "enum" in schema:
        out["enum"] = schema["enum"]
    if "required" in schema:
        out["required"] = schema["required"]
    properties = schema.get("properties")
    if isinstance(properties, dict):
        out["properties"] = {key: _json_schema_to_gemini(value) for key, value in properties.items()}
    if isinstance(schema.get("items"), dict):
        out["items"] = _json_schema_to_gemini(schema["items"])
    return out


class GeminiProvider(_PersistentHTTPProvider):
    """Google Gemini ``generateContent`` provider using the stable REST API."""

    name = "gemini"
    kind = "gemini"
    supports_images = True

    def __init__(
        self,
        family: ModelFamilyConfig,
        *,
        model: str,
        api_key: Optional[str] = None,
        timeout_s: float = 8.0,
        max_tokens: int = 256,
        temperature: float = 0.0,
        client: Any = None,
    ) -> None:
        key = api_key if api_key is not None else _env_key(family.api_key_env)
        super().__init__(
            model=model,
            api_key=key,
            base_url=family.base_url or "https://generativelanguage.googleapis.com/v1beta",
            timeout_s=timeout_s,
            client=client,
        )
        self.provider_name = "gemini"
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)

    def complete(self, request: ModelRequest) -> ModelResponse:
        started = time.perf_counter()
        system, conversation = self._extract_content(request)
        contents: List[Dict[str, Any]] = []
        for message in conversation:
            raw = message["content"]
            parts: List[Dict[str, Any]] = []
            if isinstance(raw, str):
                parts.append({"text": raw})
            else:
                for part in raw:
                    if part.get("type") == "text":
                        parts.append({"text": part.get("text", "")})
                    elif part.get("type") == "image_url":
                        data_url = part.get("image_url", {}).get("url", "")
                        encoded = data_url.split(",", 1)[1] if "," in data_url else data_url
                        parts.append({"inline_data": {"mime_type": "image/png", "data": encoded}})
            role = "model" if message["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": parts})
        generation: Dict[str, Any] = {
            "temperature": float(request.temperature if request.temperature is not None else self.temperature),
            "maxOutputTokens": min(int(request.max_tokens), self.max_tokens),
            "responseMimeType": "application/json" if request.response_schema is not None else "text/plain",
        }
        if request.response_schema is not None:
            generation["responseSchema"] = _json_schema_to_gemini(request.response_schema)
        payload: Dict[str, Any] = {"contents": contents, "generationConfig": generation}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        url = f"{self.base_url}/models/{self.model}:generateContent"
        data = self._post(
            url,
            {"Content-Type": "application/json", "x-goog-api-key": self._api_key},
            payload,
            float(request.timeout_s or self.timeout_s),
        )
        candidates = data.get("candidates") or []
        if not candidates:
            raise HTTPProviderError("Gemini returned no candidates")
        candidate = candidates[0]
        content = candidate.get("content") or {}
        parts = content.get("parts") or []
        text = "".join(str(part.get("text", "")) for part in parts if isinstance(part, dict))
        if not text:
            raise HTTPProviderError("Gemini returned an empty candidate")
        parsed = extract_json(text) if request.response_schema is not None else None
        return ModelResponse(
            text=text,
            parsed=parsed,
            provider=self.provider_name,
            model=self.model,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            finish_reason=str(candidate.get("finishReason") or "stop"),
        )

    def describe(self) -> Dict[str, Any]:
        return {"name": "gemini", "kind": self.kind, "model": self.model, "supports_images": True, "base_url": self.base_url, "api_key_present": True}


class NvidiaNimProvider(_PersistentHTTPProvider):
    """NVIDIA NIM provider using its OpenAI-compatible chat-completions API."""

    name = "nvidia"
    kind = "nvidia"
    supports_images = True

    def __init__(
        self,
        family: ModelFamilyConfig,
        *,
        model: str,
        api_key: Optional[str] = None,
        timeout_s: float = 8.0,
        max_tokens: int = 256,
        temperature: float = 0.0,
        client: Any = None,
    ) -> None:
        key = api_key if api_key is not None else _env_key(family.api_key_env)
        super().__init__(
            model=model,
            api_key=key,
            base_url=family.base_url or "https://integrate.api.nvidia.com/v1",
            timeout_s=timeout_s,
            client=client,
        )
        self.provider_name = "nvidia"
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)

    def complete(self, request: ModelRequest) -> ModelResponse:
        started = time.perf_counter()
        system, messages = self._extract_content(request)
        if system:
            messages.insert(0, {"role": "system", "content": system})
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": float(request.temperature if request.temperature is not None else self.temperature),
            "max_tokens": min(int(request.max_tokens), self.max_tokens),
            "stream": False,
        }
        if request.response_schema is not None:
            payload["response_format"] = {"type": "json_object"}
        data = self._post(
            f"{self.base_url}/chat/completions",
            {"Content-Type": "application/json", "Authorization": f"Bearer {self._api_key}"},
            payload,
            float(request.timeout_s or self.timeout_s),
        )
        choices = data.get("choices") or []
        if not choices:
            raise HTTPProviderError("NVIDIA NIM returned no choices")
        choice = choices[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        if isinstance(content, list):
            content = "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
        text = str(content)
        if not text:
            raise HTTPProviderError("NVIDIA NIM returned an empty choice")
        parsed = extract_json(text) if request.response_schema is not None else None
        return ModelResponse(
            text=text,
            parsed=parsed,
            provider=self.provider_name,
            model=self.model,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            finish_reason=str(choice.get("finish_reason") or "stop"),
        )

    def describe(self) -> Dict[str, Any]:
        return {"name": "nvidia", "kind": self.kind, "model": self.model, "supports_images": True, "base_url": self.base_url, "api_key_present": True}


def _env_key(variable: str) -> str:
    import os
    value = os.environ.get(variable, "")
    if not value:
        raise ConfigError(f"selected provider requires {variable}; configure it in the process environment")
    return value


__all__ = ["GeminiProvider", "HTTPProviderError", "NvidiaNimProvider"]
