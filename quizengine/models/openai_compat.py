"""OpenAI-compatible HTTP provider (opt-in, FR-6.2 / FR-16.1 / FR-16.2).

Works against any ``/chat/completions`` endpoint: OpenAI, Azure-style gateways,
vLLM, llama.cpp server, LM Studio, and Ollama's OpenAI-compatible route
(``http://127.0.0.1:11434/v1``).  Uses ``urllib`` from the standard library so
the transport has no hard dependency; ``httpx`` is not required.

Privacy guardrails:

* the API key is read from an environment variable named by config -- never from
  the config file, never logged (FR-16.2);
* only the crops the caller attached are uploaded, and the payload is capped by
  :func:`quizengine.models.provider.enforce_image_budget` (FR-16.1).
"""

from __future__ import annotations

import base64
import json
import re
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional

from ..config import ModelProviderConfig
from ..contracts import ModelRequest, ModelResponse
from .provider import ModelProvider, api_key_for, enforce_image_budget, extract_json

Transport = Callable[[str, Dict[str, str], Dict[str, Any], float], Dict[str, Any]]

DEFAULT_OLLAMA_BASE = "http://127.0.0.1:11434/v1"
_SAFE_MODEL_RE = re.compile(r"[^A-Za-z0-9._:/-]")


class OpenAICompatProvider(ModelProvider):
    name = "openai_compat"
    kind = "openai_compat"
    supports_images = True

    def __init__(
        self,
        config: ModelProviderConfig,
        *,
        api_key: Optional[str] = None,
        transport: Optional[Transport] = None,
        max_images: int = 2,
    ) -> None:
        self.config = config
        self.provider_name = config.name or "openai_compat"
        self.name = self.provider_name
        self.api_key = api_key if api_key is not None else api_key_for(config)
        self.base_url = (config.base_url or DEFAULT_OLLAMA_BASE).rstrip("/")
        self.model = config.model
        self.max_images = int(max_images)
        self._transport = transport or _urllib_transport
        self.last_status: Optional[int] = None

    # -- ModelProvider ----------------------------------------------------- #
    def complete(self, request: ModelRequest) -> ModelResponse:
        if not self.model:
            raise RuntimeError("models.*.model must be set for an HTTP provider")
        payload = self._payload(request)
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = self._transport(url, headers, payload, float(request.timeout_s or self.config.timeout_s))
        self.last_status = int(data.get("_status", 200)) if isinstance(data, dict) else 200
        text, finish_reason = _extract_choice(data)
        parsed = extract_json(text) if request.response_schema is not None else None
        return ModelResponse(
            text=text,
            parsed=parsed,
            provider=self.provider_name,
            model=self.model,
            finish_reason=finish_reason,
        )

    # -- payload ----------------------------------------------------------- #
    def _payload(self, request: ModelRequest) -> Dict[str, Any]:
        messages: List[Dict[str, Any]] = []
        for message in request.messages:
            images = enforce_image_budget(message.images_b64, max_images=self.max_images) if self.config.send_images else []
            if images:
                parts: List[Dict[str, Any]] = [{"type": "text", "text": message.content}]
                for image in images:
                    parts.append({"type": "image_url", "image_url": {"url": _data_url(image)}})
                messages.append({"role": message.role, "content": parts})
            else:
                messages.append({"role": message.role, "content": message.content})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": float(request.temperature),
            "max_tokens": int(min(request.max_tokens, self.config.max_tokens)),
            "stream": False,
        }
        if request.response_schema is not None:
            # JSON mode keeps schema validation meaningful (FR-7.2.6 strict mode).
            payload["response_format"] = {"type": "json_object"}
        return payload

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.provider_name,
            "kind": self.kind,
            "supports_images": self.supports_images,
            "base_url": self.base_url,
            "model": _SAFE_MODEL_RE.sub("", self.model or ""),
            "api_key_present": bool(self.api_key),
            "send_images": self.config.send_images,
        }


def _data_url(image_b64: str) -> str:
    if image_b64.startswith("data:"):
        return image_b64
    return f"data:image/png;base64,{image_b64}"


def _extract_choice(data: Any) -> tuple[str, str]:
    if not isinstance(data, dict):
        raise RuntimeError(f"unexpected provider response type: {type(data).__name__}")
    choices = data.get("choices") or []
    if not choices:
        error = data.get("error")
        message = error.get("message") if isinstance(error, dict) else str(error or data)[:200]
        raise RuntimeError(f"provider returned no choices: {message}")
    first = choices[0]
    message = first.get("message") or {}
    content = message.get("content")
    if isinstance(content, list):  # some gateways return content parts
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content or ""), str(first.get("finish_reason") or "stop")


def _urllib_transport(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=max(1.0, timeout)) as response:
            raw = response.read().decode("utf-8", errors="replace")
            status = int(getattr(response, "status", 200) or 200)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:400] if exc.fp else ""
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"network error contacting {url}: {exc.reason}") from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"provider returned non-JSON: {raw[:200]!r}") from exc
    if isinstance(parsed, dict):
        parsed.setdefault("_status", status)
        return parsed
    return {"_status": status, "choices": [{"message": {"content": raw}, "finish_reason": "stop"}]}


def encode_image(pixels: Any) -> str:
    """Base64-encode an ``H x W x 3`` uint8 array as PNG (FR-16.1 crop upload)."""
    import io

    import numpy as np
    from PIL import Image

    array = np.asarray(pixels)
    if array.ndim == 2:
        mode = "L"
    elif array.shape[2] == 4:
        array = array[:, :, :3]
        mode = "RGB"
    else:
        mode = "RGB"
    buffer = io.BytesIO()
    Image.fromarray(array.astype("uint8"), mode=mode).save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


__all__ = ["OpenAICompatProvider", "encode_image", "DEFAULT_OLLAMA_BASE"]
