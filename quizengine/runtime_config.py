"""Environment and ``.env`` integration for runtime-only provider settings.

Secrets are read directly into process memory from environment variables and are
never copied into the YAML config, logs, telemetry, browser UI, or fingerprints.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

from .config import EngineConfig
from .failures import ConfigError


def load_dotenv(path: str | Path = ".env", *, override: bool = False) -> bool:
    """Load a minimal dotenv file without a dependency; never overwrite by default."""
    file = Path(path)
    if not file.is_file():
        return False
    for line_no, raw in enumerate(file.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not name.replace("_", "").isalnum() or not name:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if override or name not in os.environ:
            os.environ[name] = value
    return True


def _bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"environment variable {name} must be true/false, got {value!r}")


def _number(name: str, default: float, *, integer: bool = False) -> float | int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return int(default) if integer else default
    try:
        return int(value) if integer else float(value)
    except ValueError as exc:
        raise ConfigError(f"environment variable {name} must be numeric, got {value!r}") from exc


def _str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value.strip() if value is not None and value.strip() else default


def apply_environment(config: Optional[EngineConfig] = None) -> EngineConfig:
    """Copy config and apply supported environment values (env wins over YAML)."""
    cfg = (config or EngineConfig.default()).model_copy(deep=True)
    models = cfg.models

    provider = _str("MODEL_PROVIDER", models.provider).lower()
    aliases = {"nvidia_nim": "nvidia", "nim": "nvidia"}
    provider = aliases.get(provider, provider)
    if provider not in {"offline", "gemini", "nvidia"}:
        raise ConfigError("MODEL_PROVIDER must be one of: offline, gemini, nvidia")
    models.provider = provider
    models.request_timeout_s = float(_number("MODEL_REQUEST_TIMEOUT_SECONDS", models.request_timeout_s))
    models.max_retries = int(_number("MODEL_MAX_RETRIES", models.max_retries, integer=True))
    models.temperature = float(_number("MODEL_TEMPERATURE", models.temperature))
    models.max_tokens = int(_number("MODEL_MAX_TOKENS", models.max_tokens, integer=True))
    models.primary_confidence_threshold = float(
        _number("MODEL_PRIMARY_CONFIDENCE_THRESHOLD", models.primary_confidence_threshold)
    )
    models.verification_confidence_threshold = float(
        _number("MODEL_VERIFICATION_CONFIDENCE_THRESHOLD", models.verification_confidence_threshold)
    )
    models.enable_parallel_model_calls = _bool("ENABLE_PARALLEL_MODEL_CALLS", models.enable_parallel_model_calls)
    models.enable_fast_path = _bool("ENABLE_FAST_PATH", models.enable_fast_path)
    models.enable_tier2_perception = _bool("ENABLE_TIER2_PERCEPTION", models.enable_tier2_perception)
    models.enable_result_cache = _bool("ENABLE_RESULT_CACHE", models.enable_result_cache)
    models.enable_screen_cache = _bool("ENABLE_SCREEN_CACHE", models.enable_screen_cache)
    models.max_parallel_models = int(_number("MAX_PARALLEL_MODELS", models.max_parallel_models, integer=True))

    for family_name, env_prefix in (("gemini", "GEMINI"), ("nvidia", "NVIDIA_NIM")):
        family = getattr(models, family_name)
        family.api_key_env = "GEMINI_API_KEY" if family_name == "gemini" else "NVIDIA_NIM_API_KEY"
        family.base_url = _str(f"{env_prefix}_BASE_URL", family.base_url or "") or None
        for field, suffix in (
            ("primary_model", "PRIMARY_MODEL"),
            ("fast_model", "FAST_MODEL"),
            ("vision_model", "VISION_MODEL"),
            ("verifier_model", "VERIFIER_MODEL"),
        ):
            setattr(family, field, _str(f"{env_prefix}_{suffix}", getattr(family, field)))

    cfg.agent.start_hotkey = _str("AGENT_START_HOTKEY", cfg.agent.start_hotkey).lower()
    cfg.agent.stop_hotkey = _str("AGENT_STOP_HOTKEY", cfg.agent.stop_hotkey).lower()
    cfg.agent.debug_mode = _bool("DEBUG_MODE", cfg.agent.debug_mode)
    cfg.agent.retain_debug_artifacts = _bool("RETAIN_DEBUG_ARTIFACTS", cfg.agent.retain_debug_artifacts)
    cfg.telemetry.trace_screenshots = cfg.agent.retain_debug_artifacts
    telemetry_level = _str("TELEMETRY_LEVEL", "normal").lower()
    level_map: Dict[str, str] = {"debug": "DEBUG", "normal": "INFO", "warning": "WARNING", "error": "ERROR"}
    if telemetry_level not in level_map:
        raise ConfigError("TELEMETRY_LEVEL must be debug, normal, warning, or error")
    cfg.telemetry.log_level = level_map[telemetry_level]

    if cfg.agent.start_hotkey == cfg.agent.stop_hotkey:
        raise ConfigError("AGENT_START_HOTKEY and AGENT_STOP_HOTKEY must be different")
    return cfg


def provider_status(config: EngineConfig) -> Dict[str, str | bool]:
    """Safe status only: no key value is returned or logged."""
    provider = config.models.provider
    if provider == "offline":
        return {"provider": "offline", "configured": True, "detail": "deterministic offline provider"}
    family = getattr(config.models, provider)
    key_present = bool(os.environ.get(family.api_key_env))
    primary = family.primary_model or family.fast_model
    configured = key_present and bool(primary)
    return {
        "provider": provider,
        "configured": configured,
        "api_key_present": key_present,
        "primary_model_configured": bool(primary),
        "detail": "ready" if configured else "missing selected-provider API key or primary model",
    }


__all__ = ["apply_environment", "load_dotenv", "provider_status"]
