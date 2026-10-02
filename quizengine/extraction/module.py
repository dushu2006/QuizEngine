"""Extraction & Validation Module (PRD section 7.3).

Turns a :class:`~quizengine.contracts.PerceptionResult` into a validated
:class:`~quizengine.contracts.Question`, or reports exactly which gate failed so
the orchestrator can run the bounded recovery (re-crop -> re-OCR -> VLM).

Pure data transformation plus (optional) model escalations for math and images.
It never touches the screen (**L5**): re-capture is the orchestrator's job.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..config import EngineConfig
from ..contracts import (
    SUPPORTED_QUESTION_TYPES,
    ContentFlags,
    FailureCode,
    Frame,
    LayoutType,
    OverlayKind,
    PerceptionResult,
    Question,
    QuestionOption,
    QuestionType,
    RunEventName,
    ScreenTransition,
    SelectedMarker,
    State,
    TransitionAssessment,
)
from ..failures import FailureSignal
from ..geometry import Box, box_area, box_union_all
from ..models.provider import ModelProvider, extract_json

_MATH_TOKEN_RE = re.compile(r"(=|\+|\*|/|\^|\u221a|\u222b|\u2211|\\frac|\\sqrt|sqrt\()")
_NUMBER_RE = re.compile(r"\d")
_TABLE_HINT_RE = re.compile(r"(\||\u2502|\u2500{3,})")


def normalize_text(text: str) -> str:
    """Canonical form used for hashing and duplicate detection."""
    lowered = (text or "").lower().strip()
    lowered = re.sub(r"[^a-z0-9\u00c0-\u024f\u0900-\u097f]+", " ", lowered)
    return " ".join(lowered.split())


def content_hash(text: str) -> str:
    """sha1 of the normalized question text (order-independent)."""
    return "sha1:" + hashlib.sha1(normalize_text(text).encode("utf-8")).hexdigest()


def question_hash(text: str, ordinal: int) -> str:
    """Stable id: hash of question text + ordinal (FR-7.3.1)."""
    payload = f"{normalize_text(text)}|{int(ordinal)}"
    return "sha1:" + hashlib.sha1(payload.encode("utf-8")).hexdigest()


@dataclass
class ExtractionOutcome:
    question: Optional[Question]
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    transition: Optional[TransitionAssessment] = None
    needs_recapture: bool = False
    math_latex: Optional[str] = None
    image_description: Optional[str] = None
    solver_context: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    ordinal: int = 0
    hashes: Dict[str, str] = field(default_factory=dict)

    def describe(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "ordinal": self.ordinal,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "hashes": dict(self.hashes),
            "transition": self.transition.transition.value if self.transition else None,
            "already_answered": bool(self.transition.already_answered) if self.transition else False,
            "needs_recapture": self.needs_recapture,
            "math_latex": self.math_latex,
            "image_description": (self.image_description or "")[:80] or None,
        }


class ExtractionModule:
    def __init__(
        self,
        config: EngineConfig,
        *,
        telemetry: Any = None,
        solver_provider: Optional[ModelProvider] = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.config = config
        self.extraction_config = config.extraction
        self.telemetry = telemetry
        self.provider = solver_provider
        self._clock = clock
        self.stats: Dict[str, Any] = {
            "extracted": 0,
            "invalid": 0,
            "math_escalations": 0,
            "image_escalations": 0,
            "duplicate_suppressions": 0,
        }

    # -- main --------------------------------------------------------------- #
    def extract(
        self,
        perception: PerceptionResult,
        *,
        ordinal_hint: Optional[int] = None,
        previous: Optional[Question] = None,
        answered_hashes: Optional[set] = None,
        answered_content_hashes: Optional[set] = None,
        frame: Optional[Frame] = None,
        hints: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ) -> ExtractionOutcome:
        started = self._clock()
        hints = hints or {}
        answered_hashes = answered_hashes or set()
        answered_content_hashes = answered_content_hashes or set()

        errors: List[str] = []
        warnings: List[str] = []

        question_text = (perception.question_text or "").strip()
        options = list(perception.options)
        text_confidence = self._text_confidence(perception)

        ordinal = self._ordinal(perception, ordinal_hint, previous)
        c_hash = content_hash(question_text)
        q_hash = question_hash(question_text, ordinal)

        # -- overlays first: a popup is a transition, not a question (FR-7.3.3)
        blocking = [o for o in perception.overlays if o.kind in {OverlayKind.MODAL, OverlayKind.UNKNOWN}]
        restricted = [o for o in perception.overlays if o.kind in {OverlayKind.CAPTCHA, OverlayKind.HUMAN_VERIFICATION}]
        if restricted:
            raise FailureSignal(
                FailureCode.RESTRICTED_ENVIRONMENT,
                f"human-verification/CAPTCHA overlay detected ({restricted[0].text[:60]!r}); refusing to interact",
                origin_state=State.EXTRACTING,
                detail={"overlays": [o.kind.value for o in restricted]},
            )

        # -- validation gates (FR-7.3.2) ------------------------------------ #
        if len(options) < self.extraction_config.min_options:
            errors.append(
                f"only {len(options)} option(s) detected; minimum is {self.extraction_config.min_options}"
            )
        if len(options) > self.extraction_config.max_options:
            warnings.append(
                f"{len(options)} options detected; section 3.1 expects 2-{self.extraction_config.max_options} "
                "for single-choice (proceeding, not fatal)"
            )
        empty = [o.index for o in options if not o.text.strip()]
        if empty:
            errors.append(f"option(s) with empty text at index/indices {empty}")
        if self.extraction_config.require_distinct_options:
            normalized = [normalize_text(o.text) for o in options if o.text.strip()]
            duplicates = sorted({t for t in normalized if normalized.count(t) > 1})
            if duplicates:
                errors.append(f"duplicate option text detected: {duplicates[:3]}")
        if not question_text:
            errors.append("question text is empty")
        elif not (
            self.extraction_config.question_text_min_chars
            <= len(question_text)
            <= self.extraction_config.question_text_max_chars
        ):
            errors.append(
                f"question text length {len(question_text)} outside sanity bounds "
                f"[{self.extraction_config.question_text_min_chars}, {self.extraction_config.question_text_max_chars}]"
            )
        confident_fraction = self._confident_fraction(perception)
        if confident_fraction < self.extraction_config.min_confident_text_fraction:
            errors.append(
                f"only {confident_fraction:.0%} of text is at or above OCR confidence "
                f"{self.extraction_config.min_ocr_confidence} (need {self.extraction_config.min_confident_text_fraction:.0%})"
            )
        if text_confidence < self.extraction_config.min_ocr_confidence * 0.75:
            warnings.append(f"mean text confidence is low ({text_confidence:.2f})")

        transition = self.assess_transition(
            previous=previous,
            perception=perception,
            question_text=question_text,
            c_hash=c_hash,
            q_hash=q_hash,
            answered_hashes=answered_hashes,
            answered_content_hashes=answered_content_hashes,
        )
        if transition.already_answered:
            self.stats["duplicate_suppressions"] += 1
            warnings.append(f"question already answered this session (L9): {q_hash[:16]}")

        flags = self._content_flags(perception, question_text, options, hints)

        if errors:
            self.stats["invalid"] += 1
            latency_ms = (self._clock() - started) * 1000.0
            outcome = ExtractionOutcome(
                question=None,
                valid=False,
                errors=errors,
                warnings=warnings,
                transition=transition,
                needs_recapture=True,
                latency_ms=latency_ms,
                ordinal=ordinal,
                hashes={"content": c_hash, "question": q_hash},
            )
            if self.telemetry is not None:
                self.telemetry.event(
                    RunEventName.EXTRACTION_FAILURE,
                    state=State.EXTRACTING,
                    module="extraction",
                    code=FailureCode.EXTRACTION_FAILURE,
                    latency_ms=latency_ms,
                    correlation_id=correlation_id,
                    errors=len(errors),
                    first_error=errors[0][:120],
                )
            return outcome

        q_type = self._question_type(perception, options)
        if q_type not in SUPPORTED_QUESTION_TYPES and self.config.run.refuse_unsupported_question_types:
            raise FailureSignal(
                FailureCode.UNSUPPORTED_QUESTION_TYPE,
                f"question type '{q_type.value}' is out of scope for v1.0 (section 3.2)",
                origin_state=State.EXTRACTING,
                detail={"type": q_type.value, "question": question_text[:120]},
            )

        question = Question(
            hash=q_hash,
            ordinal=ordinal,
            text=question_text,
            type=q_type,
            options=[
                QuestionOption(
                    index=index,
                    text=option.text.strip(),
                    handle=option.handle,
                    hit_box=option.hit_box,
                    text_conf=option.text_conf,
                    selected_marker=option.selected_marker,
                )
                for index, option in enumerate(sorted(options, key=lambda o: o.index))
            ],
            flags=flags,
            extraction_confidence=round(self._extraction_confidence(perception, confident_fraction, warnings), 4),
            content_hash=c_hash,
            frame_seq=perception.frame_seq,
            question_region=perception.question_region,
            layout_type=perception.layout_type,
            source="reconciled" if perception.tier2_used else "tier1",
        )

        solver_context: Dict[str, Any] = {
            "layout_type": perception.layout_type.value,
            "progress": perception.navigation.progress_text,
            "overlays": [o.kind.value for o in perception.overlays],
            "warnings": list(warnings),
        }
        math_latex = None
        image_description = None
        if flags.has_math:
            math_latex = self._escalate_math(question, perception, frame, correlation_id)
            if math_latex:
                solver_context["math_latex"] = math_latex
        if flags.has_image or flags.has_chart:
            image_description = self._escalate_image(question, perception, frame, correlation_id)
            if image_description:
                solver_context["image_description"] = image_description

        self.stats["extracted"] += 1
        latency_ms = (self._clock() - started) * 1000.0
        return ExtractionOutcome(
            question=question,
            valid=True,
            warnings=warnings,
            transition=transition,
            math_latex=math_latex,
            image_description=image_description,
            solver_context=solver_context,
            latency_ms=latency_ms,
            ordinal=ordinal,
            hashes={"content": c_hash, "question": q_hash},
        )

    # -- transition classification (FR-7.3.3) -------------------------------- #
    def assess_transition(
        self,
        *,
        previous: Optional[Question],
        perception: PerceptionResult,
        question_text: str,
        c_hash: str,
        q_hash: str,
        answered_hashes: set,
        answered_content_hashes: set,
    ) -> TransitionAssessment:
        reasons: List[str] = []
        already = q_hash in answered_hashes or (bool(c_hash) and c_hash in answered_content_hashes)
        if already:
            reasons.append("hash present in the session answered-set (L9)")

        overlays = [o for o in perception.overlays if o.kind not in {OverlayKind.TOAST, OverlayKind.LOADING}]
        if overlays:
            reasons.append(f"{len(overlays)} blocking overlay(s) present")
            transition = ScreenTransition.POPUP
        elif not question_text.strip() or len(perception.options) < self.extraction_config.min_options:
            if perception.end_state_evidence:
                reasons.extend(perception.end_state_evidence[:3])
                transition = ScreenTransition.END_STATE
            else:
                reasons.append("no question-like region on screen")
                transition = ScreenTransition.UNKNOWN
        elif previous is None:
            transition = ScreenTransition.NEW_QUESTION
            reasons.append("first question of the session")
        elif previous.content_hash == c_hash:
            previous_options = [normalize_text(o.text) for o in previous.options]
            current_options = [normalize_text(o.text) for o in perception.options]
            if previous_options == current_options:
                transition = ScreenTransition.SAME_QUESTION
                reasons.append("identical question text and option set")
            else:
                overlap = len(set(previous_options) & set(current_options))
                transition = ScreenTransition.PARTIAL_SCROLL
                reasons.append(f"same question text but option set differs ({overlap} shared)")
        else:
            previous_options = {normalize_text(o.text) for o in previous.options}
            current_options = {normalize_text(o.text) for o in perception.options}
            shared = previous_options & current_options
            if shared and len(shared) >= max(2, int(0.5 * min(len(previous_options), len(current_options)))):
                transition = ScreenTransition.PARTIAL_SCROLL
                reasons.append(f"{len(shared)} option(s) shared with the previous screen (partial scroll)")
            else:
                transition = ScreenTransition.NEW_QUESTION
                reasons.append("question text changed")

        return TransitionAssessment(
            transition=transition,
            previous_question_hash=previous.hash if previous else None,
            question_hash=q_hash if question_text.strip() else None,
            already_answered=already,
            reasons=reasons,
        )

    # -- gates / heuristics --------------------------------------------------- #
    def _ordinal(self, perception: PerceptionResult, hint: Optional[int], previous: Optional[Question]) -> int:
        if perception.navigation.progress_current is not None:
            return int(perception.navigation.progress_current)
        if hint is not None:
            return int(hint)
        if previous is not None:
            return int(previous.ordinal) + 1
        return 1

    def _text_confidence(self, perception: PerceptionResult) -> float:
        values: List[float] = []
        if perception.question_text:
            values.extend(b.confidence for b in perception.text_blocks if b.text.strip())
        values.extend(o.text_conf for o in perception.options)
        return float(sum(values) / len(values)) if values else 0.0

    def _confident_fraction(self, perception: PerceptionResult) -> float:
        threshold = self.extraction_config.min_ocr_confidence
        relevant = [b.confidence for b in perception.text_blocks if b.text.strip()]
        relevant.extend(o.text_conf for o in perception.options)
        if not relevant:
            return 0.0
        good = sum(1 for value in relevant if value >= threshold)
        return good / float(len(relevant))

    def _extraction_confidence(
        self, perception: PerceptionResult, confident_fraction: float, warnings: Sequence[str]
    ) -> float:
        text_conf = self._text_confidence(perception)
        structure = 0.5 + 0.1 * min(4, len(perception.options))
        if perception.tier2_used:
            structure += 0.15
        score = 0.5 * text_conf + 0.3 * confident_fraction + 0.2 * min(1.0, structure / 1.15)
        score -= 0.05 * min(3, len(warnings))
        return float(max(0.0, min(1.0, score)))

    def _question_type(self, perception: PerceptionResult, options: Sequence[Any]) -> QuestionType:
        markers = [o.selected_marker for o in options]
        selected = [m for m in markers if m != SelectedMarker.NONE]
        if len(selected) > 1:
            return QuestionType.MULTI_SELECT
        if any(m == SelectedMarker.CHECK for m in markers):
            return QuestionType.CHECKBOX_TILE
        if perception.layout_type in {LayoutType.CARD_GRID, LayoutType.TILE}:
            return QuestionType.CHECKBOX_TILE if selected else QuestionType.SINGLE_CHOICE
        return QuestionType.SINGLE_CHOICE

    def _content_flags(
        self,
        perception: PerceptionResult,
        question_text: str,
        options: Sequence[Any],
        hints: Dict[str, Any],
    ) -> ContentFlags:
        haystack = " ".join([question_text] + [o.text for o in options])
        has_math = bool(_MATH_TOKEN_RE.search(haystack)) and bool(_NUMBER_RE.search(haystack))
        has_table = bool(_TABLE_HINT_RE.search(haystack)) or self._detect_table(perception)
        has_image = self._detect_image_region(perception)
        contrast = float(hints.get("contrast_score", 1.0) or 1.0)
        return ContentFlags(
            has_math=has_math,
            has_image=has_image,
            has_table=has_table,
            has_chart=bool(hints.get("has_chart", False)) or (has_image and _NUMBER_RE.search(haystack) is not None),
            low_contrast=contrast < 0.34,
        )

    @staticmethod
    def _detect_table(perception: PerceptionResult) -> bool:
        """Three or more text runs sharing both row and column alignment."""
        blocks = [b for b in perception.text_blocks if b.text.strip()]
        if len(blocks) < 4:
            return False
        rows: Dict[int, int] = {}
        cols: Dict[int, int] = {}
        for block in blocks:
            rows[int(round(block.box[1] / 12.0))] = rows.get(int(round(block.box[1] / 12.0)), 0) + 1
            cols[int(round(block.box[0] / 24.0))] = cols.get(int(round(block.box[0] / 24.0)), 0) + 1
        return sum(1 for v in rows.values() if v >= 3) >= 2 and sum(1 for v in cols.values() if v >= 3) >= 2

    @staticmethod
    def _detect_image_region(perception: PerceptionResult) -> bool:
        """A sizeable graphic region with no text inside it.

        Tier 1 files unexplained visual mass as a NOISE region proposal; a large
        NOISE box that contains no text run is an image, chart or diagram.
        """
        text_boxes = [b.box for b in perception.text_blocks if b.text.strip()]
        for proposal in perception.regions:
            if proposal.kind.value != "noise":
                continue
            if box_area(proposal.box) < 60 * 60:
                continue
            if not any(box_area(b) and _inside(b, proposal.box) for b in text_boxes):
                return True
        return False

    # -- model escalations (FR-7.3.4 / FR-7.3.5) ------------------------------ #
    def _escalate_math(
        self, question: Question, perception: PerceptionResult, frame: Optional[Frame], correlation_id: Optional[str]
    ) -> Optional[str]:
        """Low-confidence math -> ask the VLM for LaTeX."""
        if self.provider is None:
            return None
        region = question.question_region or perception.question_region
        confidence = question.extraction_confidence
        if confidence >= self.config.extraction.math_vlm_confidence_floor and region is None:
            return None
        self.stats["math_escalations"] += 1
        from ..prompts import build_math_request

        image_b64 = _crop_b64(frame, region)
        request = build_math_request(
            ocr_text=question.text, image_b64=image_b64, correlation_id=correlation_id,
            timeout_s=self.config.solver.call_timeout_s,
        )
        try:
            response = self.provider.complete(request)
        except FailureSignal:
            return None
        payload = response.parsed or extract_json(response.text) or {}
        latex = str(payload.get("latex", "")).strip()
        return latex or None

    def _escalate_image(
        self, question: Question, perception: PerceptionResult, frame: Optional[Frame], correlation_id: Optional[str]
    ) -> Optional[str]:
        if self.provider is None:
            return None
        self.stats["image_escalations"] += 1
        from ..prompts import build_image_request

        region = _image_region(perception) or question.question_region
        image_b64 = _crop_b64(frame, region)
        request = build_image_request(
            ocr_text=question.text, image_b64=image_b64, correlation_id=correlation_id,
            timeout_s=self.config.solver.call_timeout_s,
        )
        try:
            response = self.provider.complete(request)
        except FailureSignal:
            return None
        payload = response.parsed or extract_json(response.text) or {}
        description = str(payload.get("description", "")).strip()
        return description or None

    def describe(self) -> Dict[str, Any]:
        return {"stats": dict(self.stats), "config": self.extraction_config.model_dump()}


def _inside(inner: Box, outer: Box) -> bool:
    ix, iy, iw, ih = inner
    ox, oy, ow, oh = outer
    return ix >= ox and iy >= oy and ix + iw <= ox + ow and iy + ih <= oy + oh


def _image_region(perception: PerceptionResult) -> Optional[Box]:
    boxes = [p.box for p in perception.regions if p.kind.value == "noise" and box_area(p.box) > 60 * 60]
    return box_union_all(boxes) if boxes else None


def _crop_b64(frame: Optional[Frame], region: Optional[Box]) -> Optional[str]:
    """Crop for a model call (FR-16.1: crops only, never full frames)."""
    from ..render import crop_b64

    return crop_b64(frame, region)


__all__ = [
    "ExtractionModule",
    "ExtractionOutcome",
    "content_hash",
    "normalize_text",
    "question_hash",
]
