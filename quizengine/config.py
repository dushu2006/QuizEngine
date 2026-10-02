"""Configuration specification (PRD section 12) + startup validation (NFR-17.4).

Rules enforced here:

* Every threshold is tunable without code changes; defaults mirror section 12
  and the module specs.
* ``run.attest_required`` **cannot** be set to ``false`` (FR-7.14.1) and there is
  deliberately no ``safety.enabled`` key at all (FR-7.14.5): the gatekeeper can
  only be removed by removing code.
* Invalid config -> refuse to run with actionable, field-level errors.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .contracts import SolverStrategy, UncertaintyPolicy
from .failures import ConfigError

CONFIG_SCHEMA_VERSION = 1

#: ``uncertainty_policy`` accepts the section-12 shorthand as aliases for the
#: FR-7.5.2 enum.
_POLICY_ALIASES = {
    "pause": UncertaintyPolicy.PAUSE_HUMAN,
    "pause_human": UncertaintyPolicy.PAUSE_HUMAN,
    "human": UncertaintyPolicy.PAUSE_HUMAN,
    "act": UncertaintyPolicy.ACT,
    "verify_again": UncertaintyPolicy.VERIFY_AGAIN,
    "verify": UncertaintyPolicy.VERIFY_AGAIN,
}


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
class RunConfig(_Cfg):
    """``run:`` -- section 12."""

    attest_required: Literal[True] = True
    attestation: Optional[str] = Field(None, description="operator attestation text (FR-7.14.1)")
    resume: bool = False
    run_id: Optional[str] = None
    platform_profile: str = "generic"
    deterministic: bool = Field(False, description="NFR-17.3: fixed jitter/timing, seeded RNG")
    seed: int = 1234
    dev_mode: bool = Field(True, description="illegal FSM transitions raise instead of halting")
    strict_transitions: bool = True
    #: v1.0 refuses out-of-scope question types instead of guessing (section 3.2).
    refuse_unsupported_question_types: bool = True

    @field_validator("attest_required", mode="before")
    @classmethod
    def _cannot_be_disabled(cls, value: Any) -> Any:
        if value is False or (isinstance(value, str) and value.strip().lower() in {"false", "no", "0", "off"}):
            raise ValueError(
                "run.attest_required cannot be false: the pre-run authorization gate is mandatory (FR-7.14.1). "
                "Remove the key to use the default (true)."
            )
        return True


class CaptureConfig(_Cfg):
    """``capture:`` -- section 7.1 / 12."""

    backend: Literal["mss", "synthetic", "replay"] = "mss"
    monitor: int = Field(0, ge=0)
    region: Optional[Tuple[int, int, int, int]] = Field(None, description="absolute capture region; None = whole monitor")
    max_frame_age_ms: float = Field(750.0, gt=0)
    retries: int = Field(3, ge=0, description="FR-7.1.3 re-capture attempts")
    retry_backoff_ms: float = Field(250.0, ge=0)
    blank_std_threshold: float = Field(2.0, ge=0, description="FR-7.1.2 uniform-blank detector")
    dimension_tolerance_px: int = Field(8, ge=0)
    ring_buffer_frames: int = Field(20, ge=1, description="FR-7.1.7 / FR-15.2 (memory-capped)")
    ring_buffer_memory_mb: int = Field(200, ge=16)
    zoom_crop_scale: float = Field(2.0, gt=1.0, description="FR-7.1.6 capture_zoom_window upscale factor")
    zoom_interpolation: Literal["bicubic", "bilinear", "nearest", "lanczos"] = "bicubic"
    fixture_dir: Optional[str] = Field(None, description="frame source for synthetic/replay backends")
    staleness_check: bool = Field(True, description="FR-7.1.2 reject identical frame when change is expected")
    lock_screen_signatures: List[str] = Field(
        default_factory=lambda: [
            "Press Ctrl+Alt+Del",
            "Sign in to Windows",
            "Lock screen",
            "Screen locked",
            "screensaver",
        ]
    )

    @field_validator("region", mode="before")
    @classmethod
    def _coerce_region(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, dict):  # {x:, y:, w:, h:}
            return (int(value["x"]), int(value["y"]), int(value["w"]), int(value["h"]))
        if isinstance(value, (list, tuple)) and len(value) == 4:
            return tuple(int(v) for v in value)
        raise ValueError("capture.region must be [x, y, w, h] or null")


class PreprocessConfig(_Cfg):
    """FR-7.2.2 mandatory preprocessing sequence."""

    upscale_factor: float = Field(2.0, gt=0)
    grayscale: bool = True
    denoise: bool = True
    denoise_d: int = Field(5, ge=1)
    denoise_sigma: float = Field(25.0, gt=0)
    adaptive_threshold: bool = True
    threshold_auto_select: bool = Field(True, description="Otsu vs Sauvola chosen by contrast score")
    threshold_block_size: int = Field(15, ge=3)
    threshold_c: int = 8
    deskew: bool = Field(True, description="optional deskew step")
    deskew_max_degrees: float = Field(5.0, gt=0)
    #: Raw input to OCR is forbidden (FR-7.2.2) -- kept as an explicit, logged
    #: escape hatch for fixture replay where frames are already clean renders.
    allow_raw_ocr: bool = False


class PerceptionConfig(_Cfg):
    """``perception:`` -- section 7.2 / 12."""

    ocr_engine: Literal["tesseract", "easyocr", "annotation", "none"] = "tesseract"
    ocr_fallback_engine: Literal["tesseract", "easyocr", "annotation", "none"] = "annotation"
    ocr_lang: str = "eng"
    min_text_confidence: float = Field(0.45, ge=0.0, le=1.0)
    tier1_budget_ms: float = Field(500.0, gt=0, description="NFR: Tier-1 p95 < 300ms, hard limit 500ms")
    tier2_trigger: Literal["per_question", "on_ambiguity", "never"] = "per_question"
    tier2_budget_ms: float = Field(8000.0, gt=0)
    tier2_schema_retries: int = Field(2, ge=0, description="FR-7.2.6 error-feedback re-prompts")
    ambiguity_option_count_mismatch: bool = True
    ambiguity_low_ocr_confidence: float = Field(0.6, ge=0.0, le=1.0)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    # Tier-1 heuristics (FR-7.2.3 / FR-7.2.4)
    font_size_clusters: int = Field(4, ge=2)
    column_gap_multiplier: float = Field(1.6, gt=0)
    whitespace_band_min_ratio: float = Field(0.01, gt=0)
    option_row_padding_ratio: float = Field(0.35, gt=0, description="hit-area expansion beyond the text box")
    option_min_height_px: int = Field(12, ge=1)
    reconciliation_tolerance_iou: float = Field(0.5, ge=0.0, le=1.0, description="FR-7.2.7 disagreement tolerance")
    reconciliation_confidence_penalty: float = Field(0.12, ge=0.0, le=1.0)


class ConfidenceWeights(_Cfg):
    ocr: float = Field(0.25, ge=0.0, le=1.0)
    perception_agreement: float = Field(0.25, ge=0.0, le=1.0)
    solver: float = Field(0.50, ge=0.0, le=1.0)
    penalty_reconciliation: float = Field(0.12, ge=0.0, le=1.0)
    penalty_overlay: float = Field(0.20, ge=0.0, le=1.0)
    penalty_low_ocr: float = Field(0.15, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _normalize(self) -> "ConfidenceWeights":
        total = self.ocr + self.perception_agreement + self.solver
        if total <= 0:
            raise ValueError("confidence.weights: ocr + perception_agreement + solver must be > 0")
        return self


class ConfidenceConfig(_Cfg):
    """``confidence:`` -- section 7.5 / 12."""

    high_conf: float = Field(0.85, gt=0.0, le=1.0)
    low_conf: float = Field(0.60, ge=0.0, le=1.0)
    #: FR-7.5.2 default for the mid band: verify again, then act if >= high_conf.
    uncertainty_policy: UncertaintyPolicy = UncertaintyPolicy.VERIFY_AGAIN
    #: Below low_conf the policy is MANDATORY pause-for-human (L7).  This flag
    #: exists only so tests can exercise the branch; it cannot weaken the mid band.
    allow_low_conf_act: bool = False
    weights: ConfidenceWeights = Field(default_factory=ConfidenceWeights)
    verify_again_target: float = Field(0.85, gt=0.0, le=1.0)

    @field_validator("uncertainty_policy", mode="before")
    @classmethod
    def _alias(cls, value: Any) -> Any:
        if isinstance(value, str):
            return _POLICY_ALIASES.get(value.strip().lower(), value)
        return value

    @model_validator(mode="after")
    def _order(self) -> "ConfidenceConfig":
        if self.low_conf >= self.high_conf:
            raise ValueError(f"confidence.low_conf ({self.low_conf}) must be < high_conf ({self.high_conf})")
        return self


class SolverConfig(_Cfg):
    """``solver:`` -- section 7.4 / 12."""

    strategy_chain: List[SolverStrategy] = Field(
        default_factory=lambda: [
            SolverStrategy.LOCAL_EXACT,
            SolverStrategy.LOCAL_FUZZY,
            SolverStrategy.LOCAL_RULES,
            SolverStrategy.LLM_REASONING,
            SolverStrategy.VLM_REANALYSIS,
        ]
    )
    accept_threshold: float = Field(0.85, gt=0.0, le=1.0, description="FR-7.4.1 first confident answer wins")
    fuzzy_threshold: float = Field(0.90, ge=0.0, le=1.0, description="FR-7.4.1.1 normalized-match floor")
    self_consistency: bool = False
    self_consistency_threshold: float = Field(0.85, gt=0.0, le=1.0, description="FR-7.4.3 trigger below this confidence")
    self_consistency_samples: int = Field(3, ge=2)
    self_consistency_temperature: float = Field(0.7, ge=0.0, le=2.0)
    call_timeout_s: float = Field(30.0, gt=0, description="FR-7.4.4")
    call_retries: int = Field(2, ge=0, description="FR-7.4.4 exponential backoff")
    budget_s: float = Field(90.0, gt=0, description="FR-7.4.4 total per-question budget")
    vlm_on_flags: bool = Field(True, description="FR-7.4.1.4 escalate when has_math/has_image")
    knowledge_base: Optional[str] = Field(None, description="QuizForge-provided Q&A pairs (JSON/YAML)")

    @model_validator(mode="after")
    def _chain_valid(self) -> "SolverConfig":
        if not self.strategy_chain:
            raise ValueError("solver.strategy_chain must list at least one strategy")
        seen: set[SolverStrategy] = set()
        for strategy in self.strategy_chain:
            if strategy in {SolverStrategy.NONE, SolverStrategy.HUMAN, SolverStrategy.SELF_CONSISTENCY}:
                raise ValueError(f"solver.strategy_chain cannot contain '{strategy.value}' (not a chain link)")
            if strategy in seen:
                raise ValueError(f"solver.strategy_chain contains duplicate '{strategy.value}'")
            seen.add(strategy)
        return self


class InputProfile(_Cfg):
    """FR-7.6.2 human-plausible input.  Bounds only -- no evasion knobs exist."""

    move_duration_range: Tuple[float, float] = (0.2, 0.4)
    click_jitter_px: int = Field(3, ge=0)
    typing_delay_range: Tuple[float, float] = (0.03, 0.09)
    scroll_pause_range: Tuple[float, float] = (0.15, 0.35)

    @field_validator("move_duration_range", "typing_delay_range", "scroll_pause_range", mode="before")
    @classmethod
    def _coerce_pair(cls, value: Any) -> Any:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            lo, hi = float(value[0]), float(value[1])
            if lo > hi:
                raise ValueError("range must be [low, high]")
            if lo < 0:
                raise ValueError("range values must be >= 0")
            return (lo, hi)
        return value


class ActionConfig(_Cfg):
    """``action:`` -- section 7.6 / 12."""

    backend: Literal["pyautogui", "ahk", "recording", "null"] = "pyautogui"
    input_profile: InputProfile = Field(default_factory=InputProfile)
    focus_check: bool = Field(True, description="FR-7.6.4")
    window_title_match: Optional[str] = Field(None, description="target window title substring for re-focus")
    #: FR-7.7.2 re-resolution thresholds.
    rematch_min_confidence: float = Field(0.8, ge=0.0, le=1.0)
    rematch_max_displacement_px: float = Field(30.0, ge=0)
    rematch_scale_search: float = Field(0.20, ge=0.0, le=1.0)
    failsafe_corner: bool = Field(True, description="pyautogui FAILSAFE: slam mouse to corner to abort")

    @model_validator(mode="after")
    def _no_evasion(self) -> "ActionConfig":
        """FR-7.14.4 input honesty: reject any knob whose only purpose could be
        disguising automation.  Kept as an executable guard so a future edit
        cannot silently add one."""
        forbidden = {"spoof", "stealth", "evade", "humanize_signature", "randomize_device", "hide_cursor"}
        for key in self.model_dump():
            if any(token in key.lower() for token in forbidden):
                raise ValueError(f"action.{key} is prohibited by FR-7.14.4 (input honesty)")
        return self


class VerificationConfig(_Cfg):
    """Section 7.9 closed-loop verification."""

    verify_timeout_ms: float = Field(1500.0, gt=0, description="FR-7.9.1 first evidence deadline")
    verify_recheck_ms: float = Field(3000.0, gt=0, description="FR-7.9.1 second deadline")
    min_region_change_pct: float = Field(5.0, ge=0.0, le=100.0)
    pixel_diff_threshold: int = Field(12, ge=0, description="per-channel delta counted as changed")
    element_aware: bool = Field(True, description="FR-7.9.2: global diff alone is insufficient")
    max_post_action_frames: int = Field(6, ge=1)


class NavigationConfig(_Cfg):
    """``navigation:`` -- section 7.8 / 12."""

    keyboard_nav_enabled: bool = False
    keyboard_keys: List[str] = Field(default_factory=lambda: ["enter", "right", "down"])
    strategy_cascade: List[Literal["next_button", "auto_advance", "scroll_reveal", "keyboard"]] = Field(
        default_factory=lambda: ["next_button", "auto_advance", "scroll_reveal", "keyboard"]
    )
    auto_advance_wait_ms: float = Field(250.0, ge=0, description="FR-7.8.1.b wait for platform auto-advance")
    scroll_increment_px: int = Field(300, gt=0)
    scroll_max_steps: int = Field(6, ge=1)
    min_scroll_motion_pct: float = Field(1.0, ge=0.0, le=100.0, description="FR-7.8.2 exhausted threshold")
    end_state_keywords: List[str] = Field(
        default_factory=lambda: [
            "quiz complete",
            "submission complete",
            "your score",
            "results",
            "finished",
            "thank you",
            "correct answers",
            "you scored",
            "final score",
        ]
    )
    end_state_required_frames: int = Field(2, ge=1, description="FR-7.8.4: consecutive frames with no question region")


class BudgetConfig(_Cfg):
    """``budgets:`` -- section 7.10.2 / 12."""

    max_questions: int = Field(50, ge=1)
    max_runtime_min: float = Field(30.0, gt=0, description="section 12 'max_runtime: 30' is minutes")
    max_consecutive_failures: int = Field(3, ge=1)
    max_cycles: int = Field(400, ge=1, description="absolute loop guard (L3)")
    per_question_budget_s: float = Field(90.0, gt=0, description="section 15 hard limit")


class SafetyConfig(_Cfg):
    """``safety:`` -- section 7.14.  Note: no ``enabled`` key exists (FR-7.14.5)."""

    restricted_scan_interval_s: float = Field(60.0, gt=0)
    scan_at_startup: bool = True
    scan_processes: bool = True
    #: Known proctoring / lockdown / secure-browser signatures (FR-7.14.2).
    #: Detection means *stop*, never evade (section 3.3.1).
    restricted_process_signatures: List[str] = Field(
        default_factory=lambda: [
            "proctorio",
            "proctortrack",
            "proctoru",
            "respondus",
            "lockdownbrowser",
            "lockdown browser",
            "safeexam",
            "safe exam browser",
            "seb.exe",
            "proctor",
            "insperity",
            "examsoft",
            "examplify",
            "honorlock",
            "vericant",
            "smowl",
            "mettl secure",
        ]
    )
    restricted_window_signatures: List[str] = Field(
        default_factory=lambda: [
            "secure browser",
            "lockdown",
            "proctoring",
            "exam mode",
            "kiosk",
        ]
    )
    captcha_overlay_halts: bool = Field(True, description="section 3.3.2: never interact with CAPTCHAs")
    continuous_check: bool = Field(True, description="FR-7.14.3 mid-run policy check")
    halt_on_lock_screen: bool = Field(True, description="FR-7.1.2 secure desktop -> environment event")

    @model_validator(mode="after")
    def _not_disablable(self) -> "SafetyConfig":
        if not self.scan_at_startup:
            raise ValueError("safety.scan_at_startup cannot be disabled (FR-7.14.5): remove the key to use the default")
        if not self.continuous_check:
            raise ValueError("safety.continuous_check cannot be disabled (FR-7.14.3/FR-7.14.5)")
        return self


class TelemetryConfig(_Cfg):
    """``telemetry:`` -- section 7.13 / 12."""

    trace_screenshots: bool = True
    decision_traces: bool = True
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    json_logs: bool = True
    log_file: Optional[str] = None
    metrics_export: Literal["file", "none"] = "file"
    metrics_file: str = "metrics.json"
    metrics_flush_interval_s: float = Field(5.0, gt=0)
    console: bool = Field(True, description="FR-7.13.5 live operator console")
    retention_runs: int = Field(20, ge=1, description="FR-7.12.3")
    artifact_bundle_frames: int = Field(5, ge=1, description="FR-7.11.3")


class ModelProviderConfig(_Cfg):
    """FR-6.2 provider abstraction.  Cloud providers are opt-in (FR-16.1)."""

    name: str = "offline"
    kind: Literal["offline", "openai_compat", "ollama"] = "offline"
    model: str = "deterministic-offline"
    base_url: Optional[str] = None
    api_key_env: str = Field("", description="FR-16.2: keys come from env vars only, never config")
    timeout_s: float = Field(30.0, gt=0)
    retries: int = Field(2, ge=0)
    backoff_base_s: float = Field(0.5, ge=0)
    max_tokens: int = Field(1024, gt=0)
    #: Only question/option crops are ever sent (FR-16.1) -- never full frames.
    send_images: bool = True
    min_crop_px: int = Field(64, ge=8)


class ModelFamilyConfig(_Cfg):
    """One vendor account with independently routed model roles.

    API secrets are referenced by environment-variable name only. An empty model
    role is valid: the orchestrator reuses the configured primary model rather
    than inventing or requiring a vendor-specific model name.
    """

    api_key_env: str
    base_url: Optional[str] = None
    primary_model: str = ""
    fast_model: str = ""
    vision_model: str = ""
    verifier_model: str = ""


class ModelsConfig(_Cfg):
    # Original provider stack remains intact for compatibility with existing YAML.
    primary: ModelProviderConfig = Field(default_factory=ModelProviderConfig)
    fallback: Optional[ModelProviderConfig] = None
    tier2: Optional[ModelProviderConfig] = Field(None, description="VLM used for Tier-2 perception / re-analysis")

    # Explicit family selection. There is deliberately no cross-family fallback.
    provider: Literal["offline", "gemini", "nvidia"] = "offline"
    request_timeout_s: float = Field(8.0, gt=0)
    max_retries: int = Field(1, ge=0, le=5)
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(256, ge=1, le=4096)
    primary_confidence_threshold: float = Field(0.75, ge=0.0, le=1.0)
    verification_confidence_threshold: float = Field(0.85, ge=0.0, le=1.0)
    enable_parallel_model_calls: bool = True
    enable_fast_path: bool = True
    enable_tier2_perception: bool = True
    enable_result_cache: bool = True
    enable_screen_cache: bool = True
    max_parallel_models: int = Field(2, ge=1, le=8)
    gemini: ModelFamilyConfig = Field(
        default_factory=lambda: ModelFamilyConfig(api_key_env="GEMINI_API_KEY", base_url="https://generativelanguage.googleapis.com/v1beta")
    )
    nvidia: ModelFamilyConfig = Field(
        default_factory=lambda: ModelFamilyConfig(api_key_env="NVIDIA_NIM_API_KEY", base_url="https://integrate.api.nvidia.com/v1")
    )


class AgentConfig(_Cfg):
    """Manual, user-triggered desktop agent controls (Windows global hotkeys)."""

    start_hotkey: str = "win+alt+q"
    stop_hotkey: str = "win+alt+x"
    retain_debug_artifacts: bool = False
    debug_mode: bool = False


class PathsConfig(_Cfg):
    runs_dir: str = "runs"
    session_file: str = "session.json"
    fixtures_dir: str = "tests/fixtures/frames"
    knowledge_dir: str = "knowledge"


class ExtractionConfig(_Cfg):
    """Section 7.3 validation gates."""

    min_options: int = Field(2, ge=2)
    max_options: int = Field(6, le=12, description="section 3.1: single-choice 2-6 options")
    question_text_min_chars: int = Field(5, ge=1)
    question_text_max_chars: int = Field(1000, ge=10)
    min_ocr_confidence: float = Field(0.60, ge=0.0, le=1.0)
    min_confident_text_fraction: float = Field(0.80, ge=0.0, le=1.0, description="FR-7.3.2")
    require_distinct_options: bool = True
    retries: int = Field(2, ge=0)
    math_keywords: List[str] = Field(
        default_factory=lambda: ["=", "+", "-", "*", "/", "^", "sqrt", "sum", "integral", "frac", "%", "pi"]
    )
    table_keywords: List[str] = Field(default_factory=lambda: ["|", "row", "column", "table"])
    math_vlm_confidence_floor: float = Field(
        0.7, ge=0.0, le=1.0, description="FR-7.3.4: escalate math to the VLM below this extraction confidence"
    )

    @model_validator(mode="after")
    def _bounds(self) -> "ExtractionConfig":
        if self.min_options > self.max_options:
            raise ValueError("extraction.min_options must be <= max_options")
        if self.question_text_min_chars >= self.question_text_max_chars:
            raise ValueError("extraction.question_text_min_chars must be < max_chars")
        return self


# --------------------------------------------------------------------------- #
# Root
# --------------------------------------------------------------------------- #
class EngineConfig(_Cfg):
    """Root configuration object (section 12)."""

    config_schema_version: int = CONFIG_SCHEMA_VERSION
    run: RunConfig = Field(default_factory=RunConfig)
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    perception: PerceptionConfig = Field(default_factory=PerceptionConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    confidence: ConfidenceConfig = Field(default_factory=ConfidenceConfig)
    solver: SolverConfig = Field(default_factory=SolverConfig)
    action: ActionConfig = Field(default_factory=ActionConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    navigation: NavigationConfig = Field(default_factory=NavigationConfig)
    budgets: BudgetConfig = Field(default_factory=BudgetConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    source_path: Optional[str] = None

    # -- construction ------------------------------------------------------ #
    @classmethod
    def default(cls) -> "EngineConfig":
        """Defaults documented in the README (section 12)."""
        return cls()

    @classmethod
    def from_dict(cls, data: Dict[str, Any], *, source_path: Optional[str] = None) -> "EngineConfig":
        payload = _deep_copy_dict(data or {})
        payload.pop("source_path", None)
        try:
            config = cls.model_validate(payload)
        except Exception as exc:  # pydantic ValidationError and friends
            raise ConfigError(_format_validation_error(exc, source_path), detail={"source": source_path}) from exc
        config.source_path = source_path
        _post_validate(config)
        return config

    @classmethod
    def load(cls, path: str | Path) -> "EngineConfig":
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"config file not found: {path}", detail={"path": str(path)})
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"config file is not valid YAML: {path}: {exc}", detail={"path": str(path)}) from exc
        if not isinstance(data, dict):
            raise ConfigError(f"config root must be a mapping, got {type(data).__name__}", detail={"path": str(path)})
        return cls.from_dict(data, source_path=str(path))

    # -- serialization ----------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=False)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False)

    def fingerprint(self) -> str:
        """Stable digest used to refuse resuming a session under a different config."""
        payload = self.model_dump(mode="json", exclude={"source_path", "run"})
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return "sha1:" + hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]

    # -- runtime helpers --------------------------------------------------- #
    @property
    def runs_dir(self) -> Path:
        return Path(self.paths.runs_dir)

    @property
    def max_runtime_s(self) -> float:
        return self.budgets.max_runtime_min * 60.0

    def apply_overrides(self, overrides: Dict[str, Any]) -> "EngineConfig":
        """Apply dotted-path overrides (``{"run.attestation": "..."}``)."""
        data = self.to_dict()
        for dotted, value in (overrides or {}).items():
            _set_dotted(data, dotted, value)
        return EngineConfig.from_dict(data, source_path=self.source_path)

    def describe_defaults(self) -> str:
        lines = ["RESOLVED CONFIGURATION", "-" * 72]
        for section, value in self.to_dict().items():
            if isinstance(value, dict):
                lines.append(f"{section}:")
                for key, item in value.items():
                    lines.append(f"  {key} = {item!r}")
            else:
                lines.append(f"{section} = {value!r}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _deep_copy_dict(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: _deep_copy_dict(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_deep_copy_dict(v) for v in data]
    return data


def _set_dotted(data: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cursor = data
    for part in parts[:-1]:
        nxt = cursor.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cursor[part] = nxt
        cursor = nxt
    leaf = parts[-1]
    if isinstance(value, str):
        value = _coerce_scalar(value)
    cursor[leaf] = value


def _coerce_scalar(value: str) -> Any:
    lowered = value.strip().lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"null", "none", "~"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if value.startswith("[") and value.endswith("]"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return [v.strip() for v in value[1:-1].split(",") if v.strip()]
    return value


def _format_validation_error(exc: Exception, source_path: Optional[str]) -> str:
    errors = getattr(exc, "errors", None)
    where = f" in {source_path}" if source_path else ""
    if not callable(errors):
        return f"invalid configuration{where}: {exc}"
    lines = [f"invalid configuration{where}:"]
    for error in errors():
        location = ".".join(str(p) for p in error.get("loc", ())) or "<root>"
        lines.append(f"  - {location}: {error.get('msg', 'invalid value')}")
        ctx = error.get("ctx")
        if ctx:
            lines.append(f"      context: {ctx}")
    lines.append("Fix the config and re-run (NFR-17.4). See config/default.yaml for documented defaults.")
    return "\n".join(lines)


def _post_validate(config: EngineConfig) -> None:
    """Cross-section checks pydantic cannot express per-model."""
    problems: List[str] = []

    if config.run.attestation is None and config.run.attest_required:
        # Not fatal at config-parse time: the CLI may supply --attest later.
        # The gatekeeper (section 7.14) is what actually refuses the run.
        pass

    if config.capture.backend in {"synthetic", "replay"} and not config.capture.fixture_dir:
        problems.append(
            f"capture.backend='{config.capture.backend}' requires capture.fixture_dir to be set"
        )

    if config.perception.ocr_engine == "none" and config.perception.tier2_trigger == "never":
        problems.append(
            "perception.ocr_engine='none' with tier2_trigger='never' leaves the agent blind; "
            "choose an OCR engine or enable Tier-2"
        )

    if config.perception.preprocess.allow_raw_ocr and config.perception.ocr_engine in {"tesseract", "easyocr"}:
        problems.append(
            "perception.preprocess.allow_raw_ocr=true violates FR-7.2.2 (raw input to OCR is forbidden) "
            "for real OCR engines; it is only permitted for the deterministic annotation backend"
        )

    if config.action.backend in {"pyautogui", "ahk"} and config.run.deterministic:
        # Deterministic mode fixes jitter/timing; that is allowed, but warn-level
        # documentation lives in the README.  No problem recorded.
        pass

    if config.verification.verify_recheck_ms < config.verification.verify_timeout_ms:
        problems.append("verification.verify_recheck_ms must be >= verify_timeout_ms")

    if config.models.primary.kind != "offline" and not config.models.primary.api_key_env:
        problems.append(
            f"models.primary.kind='{config.models.primary.kind}' requires models.primary.api_key_env "
            "(FR-16.2: keys come from environment variables only)"
        )
    if config.models.primary.api_key_env and config.models.primary.kind == "offline":
        problems.append("models.primary.api_key_env is meaningless for kind='offline'")

    if config.navigation.keyboard_nav_enabled and "keyboard" not in config.navigation.strategy_cascade:
        problems.append("navigation.keyboard_nav_enabled=true but 'keyboard' is missing from strategy_cascade")
    if not config.navigation.keyboard_nav_enabled and "keyboard" in config.navigation.strategy_cascade:
        # Keyboard nav is opt-in (FR-7.8.1.d): silently drop it from the cascade.
        config.navigation.strategy_cascade = [s for s in config.navigation.strategy_cascade if s != "keyboard"]

    if problems:
        raise ConfigError(
            "invalid configuration:\n" + "\n".join(f"  - {p}" for p in problems),
            detail={"problems": problems},
        )


def config_from_env(base: Optional[EngineConfig] = None) -> EngineConfig:
    """Layer ``QUIZENGINE_*`` environment overrides on top of ``base``.

    ``QUIZENGINE_CONFIG`` selects the file; ``QUIZENGINE_ATTENTION``-style
    variables use dotted paths, e.g. ``QUIZENGINE_CAPTURE__MONITOR=1``.
    """
    path = os.environ.get("QUIZENGINE_CONFIG")
    config = EngineConfig.load(path) if path else (base or EngineConfig.default())
    overrides: Dict[str, Any] = {}
    prefix = "QUIZENGINE_"
    for key, value in os.environ.items():
        if not key.startswith(prefix) or key == "QUIZENGINE_CONFIG":
            continue
        dotted = key[len(prefix) :].lower().replace("__", ".")
        if "." in dotted:
            overrides[dotted] = value
    return config.apply_overrides(overrides) if overrides else config


__all__ = [
    "EngineConfig",
    "RunConfig",
    "CaptureConfig",
    "PerceptionConfig",
    "PreprocessConfig",
    "ExtractionConfig",
    "ConfidenceConfig",
    "ConfidenceWeights",
    "SolverConfig",
    "ActionConfig",
    "InputProfile",
    "VerificationConfig",
    "NavigationConfig",
    "BudgetConfig",
    "SafetyConfig",
    "TelemetryConfig",
    "ModelsConfig",
    "ModelProviderConfig",
    "PathsConfig",
    "CONFIG_SCHEMA_VERSION",
]
