"""Tier 2 -- VLM structural analysis (FR-7.2.5, FR-7.2.6).

Runs when Tier 1 is ambiguous, on layout change, and at least once per question
(configurable via ``perception.tier2_trigger``).  Output is schema-validated in
strict mode; invalid output triggers up to two error-feedback re-prompts, after
which the caller treats the result as ``PERCEPTION_LOW_CONFIDENCE``.

Privacy (FR-16.1): only the minimal crop containing the question, options and
navigation is uploaded -- never a full-screen dump.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from pydantic import ConfigDict, Field, ValidationError, field_validator

from ..config import PerceptionConfig
from ..contracts import (
    Frame,
    LayoutType,
    OverlayKind,
    SchemaModel,
    SelectedMarker,
)
from ..geometry import Box, as_box
from ..models.provider import ModelProvider, request_json
from ..prompts import build_perception_request, perception_crop_box

try:  # pragma: no cover - optional transport
    from ..models.openai_compat import encode_image as _encode_image
except ImportError:  # pragma: no cover
    _encode_image = None  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# strict output model
# --------------------------------------------------------------------------- #
class _Tier2Button(SchemaModel):
    model_config = ConfigDict(extra="ignore")
    handle: str
    box: List[float] = Field(..., min_length=4, max_length=4)
    text: str = ""
    enabled: bool = True

    @field_validator("box", mode="before")
    @classmethod
    def _box(cls, value: Any) -> Any:
        return list(value) if isinstance(value, (list, tuple)) else value


class _Tier2Option(SchemaModel):
    model_config = ConfigDict(extra="ignore")
    index: int = Field(..., ge=0)
    handle: str
    text: str = ""
    hit_box: List[float] = Field(..., min_length=4, max_length=4)
    text_box: Optional[List[float]] = Field(None, min_length=4, max_length=4)
    text_conf: float = Field(0.9, ge=0.0, le=1.0)
    selected_marker: SelectedMarker = SelectedMarker.NONE


class _Tier2Navigation(SchemaModel):
    model_config = ConfigDict(extra="ignore")
    next_btn: Optional[_Tier2Button] = None
    prev_btn: Optional[_Tier2Button] = None
    submit_btn: Optional[_Tier2Button] = None
    progress_text: Optional[str] = None
    progress_current: Optional[int] = None
    progress_total: Optional[int] = None


class _Tier2Overlay(SchemaModel):
    model_config = ConfigDict(extra="ignore")
    handle: str
    box: List[float] = Field(..., min_length=4, max_length=4)
    text: str = ""
    kind: OverlayKind = OverlayKind.UNKNOWN
    dismissible: Optional[bool] = None


class Tier2Analysis(SchemaModel):
    """Schema-validated Tier-2 output (FR-7.2.6)."""

    model_config = ConfigDict(extra="ignore")

    layout_type: LayoutType = LayoutType.UNKNOWN
    question_region: Optional[List[float]] = Field(None, min_length=4, max_length=4)
    question_text: str = ""
    verbatim_element_indices: Optional[List[int]] = None
    options: List[_Tier2Option] = Field(default_factory=list)
    navigation: _Tier2Navigation = Field(default_factory=_Tier2Navigation)
    overlays: List[_Tier2Overlay] = Field(default_factory=list)
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    notes: str = ""
    #: ADDITIVE: tells reconciliation whether real vision happened at all.
    source: str = "vlm"

    @field_validator("question_region", mode="before")
    @classmethod
    def _box(cls, value: Any) -> Any:
        if value is None:
            return None
        return list(value) if isinstance(value, (list, tuple)) else value

    @property
    def is_echo(self) -> bool:
        """True when the provider could not actually see the crop."""
        return self.source != "vlm"


@dataclass
class Tier2Outcome:
    analysis: Tier2Analysis
    crop_box: Box
    latency_ms: float = 0.0
    flags: List[str] = field(default_factory=list)
    provider: str = ""
    text_only: bool = False
    raw: Dict[str, Any] = field(default_factory=dict)


class Tier2Perception:
    def __init__(
        self,
        provider: ModelProvider,
        config: PerceptionConfig,
        *,
        telemetry: Any = None,
        encoder: Optional[Callable[[Any], str]] = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.provider = provider
        self.config = config
        self.telemetry = telemetry
        self._encoder = encoder or _encode_image
        self._clock = clock
        self.stats: Dict[str, Any] = {"calls": 0, "schema_failures": 0, "text_only": 0, "echo": 0}

    # -- main --------------------------------------------------------------- #
    def analyze(
        self,
        frame: Frame,
        tier1: Any,
        *,
        ambiguity: Sequence[str] = (),
        correlation_id: Optional[str] = None,
    ) -> Tier2Outcome:
        """Run Tier 2 on one frame.  Raises ``FailureSignal`` on schema/timeout failure."""
        started = self._clock()
        self.stats["calls"] += 1
        crop_box = perception_crop_box(tier1, frame.full_box)
        crop_b64: Optional[str] = None
        flags: List[str] = []
        text_only = not bool(getattr(self.provider, "supports_images", False))

        if not text_only and self._encoder is not None and frame.pixels is not None:
            try:
                from ..render import crop

                pixels = crop(np.asarray(frame.pixels), crop_box)
                crop_b64 = self._encoder(pixels) if pixels.size else None
            except Exception as exc:  # a crop failure must not kill perception
                flags.append(f"crop encode failed: {type(exc).__name__}: {exc}")
                crop_b64 = None
            if crop_b64 is None:
                text_only = True
        if text_only:
            self.stats["text_only"] += 1
            flags.append("tier2 ran without an image (provider has no vision): structure comes from Tier-1 hints")

        request = build_perception_request(
            tier1=tier1,
            crop_b64=crop_b64,
            crop_box=crop_box,
            ambiguity=ambiguity,
            correlation_id=correlation_id,
            timeout_s=self.config.tier2_budget_ms / 1000.0,
        )

        def parse(payload: Dict[str, Any]) -> Tier2Analysis:
            return Tier2Analysis.model_validate(payload)

        try:
            analysis = request_json(self.provider, request, parse, retries=self.config.tier2_schema_retries)
        except ValidationError as exc:  # pragma: no cover - request_json wraps these
            self.stats["schema_failures"] += 1
            raise
        except Exception:
            self.stats["schema_failures"] += 1
            raise

        _translate_to_frame(analysis, crop_box)
        if analysis.is_echo:
            self.stats["echo"] += 1
            flags.append(f"tier2 source={analysis.source!r}: penalize during reconciliation")

        latency_ms = (self._clock() - started) * 1000.0
        if self.telemetry is not None:
            self.telemetry.observe_latency("tier2_latency_ms", latency_ms)
        return Tier2Outcome(
            analysis=analysis,
            crop_box=crop_box,
            latency_ms=latency_ms,
            flags=flags,
            provider=getattr(self.provider, "name", "unknown"),
            text_only=text_only,
            raw=analysis.to_wire(),
        )

    # -- trigger policy (FR-7.2.5) ------------------------------------------- #
    def should_run(self, *, tier1_ambiguity: Sequence[str], layout_changed: bool, questions_since_last: int) -> bool:
        trigger = self.config.tier2_trigger
        if trigger == "never":
            return False
        if trigger == "on_ambiguity":
            return bool(tier1_ambiguity) or layout_changed
        # per_question: at least once per question, plus whenever Tier 1 is unsure.
        return bool(tier1_ambiguity) or layout_changed or questions_since_last >= 1

    def describe(self) -> Dict[str, Any]:
        return {"provider": getattr(self.provider, "name", "unknown"), "stats": dict(self.stats)}


def _translate_to_frame(analysis: Tier2Analysis, crop_box: Box) -> None:
    """Tier 2 answers in crop coordinates; move everything into frame space."""
    dx, dy = int(round(crop_box[0])), int(round(crop_box[1]))
    if dx == 0 and dy == 0:
        return
    if analysis.question_region is not None:
        analysis.question_region = [
            analysis.question_region[0] + dx,
            analysis.question_region[1] + dy,
            analysis.question_region[2],
            analysis.question_region[3],
        ]
    for option in analysis.options:
        option.hit_box = _shift(option.hit_box, dx, dy)
        if option.text_box is not None:
            option.text_box = _shift(option.text_box, dx, dy)
    for button in (analysis.navigation.next_btn, analysis.navigation.prev_btn, analysis.navigation.submit_btn):
        if button is not None:
            button.box = _shift(button.box, dx, dy)
    for overlay in analysis.overlays:
        overlay.box = _shift(overlay.box, dx, dy)


def _shift(box: Sequence[float], dx: int, dy: int) -> List[float]:
    return [float(box[0]) + dx, float(box[1]) + dy, float(box[2]), float(box[3])]


def to_int_box(box: Optional[Sequence[float]]) -> Optional[Box]:
    return as_box(box) if box is not None else None


__all__ = ["Tier2Analysis", "Tier2Outcome", "Tier2Perception", "to_int_box"]
