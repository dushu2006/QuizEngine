"""Tier 1 -- fast classical perception (FR-7.2.1 .. FR-7.2.4).

Runs on every frame within the Tier-1 budget (NFR: p95 < 300 ms, hard limit
500 ms).  Pipeline::

    frame -> FR-7.2.2 preprocessing -> OCR (swappable engine)
          -> pixel evidence (background, panels, outlines)
          -> FR-7.2.3 region proposals
          -> FR-7.2.4 option hit-area localization + selected-state markers
          -> navigation affordances, overlays, end-state evidence

Tier 1 never calls a model.  It also reports *why* it is unsure, which is what
triggers Tier 2 (FR-7.2.5).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import PerceptionConfig
from ..contracts import (
    Frame,
    LayoutType,
    NavButton,
    NavigationPerception,
    OptionPerception,
    Overlay,
    OverlayKind,
    PerceptionResult,
    RegionKind,
    RegionProposal,
    SelectedMarker,
    TextBlock,
)
from ..geometry import Box, Geometry, box_area, box_contains_box, box_intersection, box_iou, box_union_all
from . import pixels as px
from .localization import LocalizationResult, OptionLocator
from .ocr.base import NullOCR, OCREngine, OCRRequest, filter_blocks
from .preprocessing import PreprocessResult, prepare_for_ocr
from .segmentation import RegionSegmenter, SegmentationContext, is_nav_text

_PROGRESS_RE = re.compile(r"(\d+)\s*(?:of|/)\s*(\d+)", re.IGNORECASE)
_QUESTION_NUMBER_RE = re.compile(r"question\s*(\d+)", re.IGNORECASE)

#: Generic overlay vocabulary -> OverlayKind.  CAPTCHA / human-verification hits
#: are restricted-environment indicators (section 3.3.2) and are escalated by the
#: gatekeeper; Tier 1 only classifies, it never interacts.
_OVERLAY_KEYWORDS: Tuple[Tuple[str, OverlayKind], ...] = (
    ("captcha", OverlayKind.CAPTCHA),
    ("recaptcha", OverlayKind.CAPTCHA),
    ("hcaptcha", OverlayKind.CAPTCHA),
    ("i'm not a robot", OverlayKind.HUMAN_VERIFICATION),
    ("i am not a robot", OverlayKind.HUMAN_VERIFICATION),
    ("verify you are human", OverlayKind.HUMAN_VERIFICATION),
    ("are you a human", OverlayKind.HUMAN_VERIFICATION),
    ("human verification", OverlayKind.HUMAN_VERIFICATION),
    ("security check", OverlayKind.HUMAN_VERIFICATION),
    ("cookie", OverlayKind.COOKIE_BANNER),
    ("loading", OverlayKind.LOADING),
    ("please wait", OverlayKind.LOADING),
    ("saved", OverlayKind.TOAST),
    ("retry", OverlayKind.TOAST),
    ("connection", OverlayKind.TOAST),
)

_DISMISS_KEYWORDS = ("close", "dismiss", "x", "\u00d7", "got it", "ok", "okay", "no thanks", "accept")
_NEXT_KEYWORDS = ("next", "continue", "->", "\u2192", "next question")
_SUBMIT_KEYWORDS = ("submit", "finish", "complete quiz", "check answer", "grade")
_PREV_KEYWORDS = ("previous", "prev", "back", "<-", "\u2190")


@dataclass
class Tier1Result:
    perception: PerceptionResult
    preprocess: Optional[PreprocessResult]
    context: SegmentationContext
    localization: LocalizationResult
    proposals: List[RegionProposal]
    ambiguity: List[str] = field(default_factory=list)
    confidence: float = 0.0
    latency_ms: float = 0.0
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def needs_tier2(self) -> bool:
        return bool(self.ambiguity)


class Tier1Perception:
    def __init__(
        self,
        config: PerceptionConfig,
        ocr: OCREngine,
        *,
        end_state_keywords: Optional[Sequence[str]] = None,
        clock: Any = time.perf_counter,
    ) -> None:
        self.config = config
        self.ocr = ocr
        self.segmenter = RegionSegmenter(config)
        self.locator = OptionLocator(config)
        self.end_state_keywords = tuple(k.lower() for k in (end_state_keywords or ()))
        self._clock = clock

    # -- main --------------------------------------------------------------- #
    def analyze(self, frame: Frame) -> Tier1Result:
        started = self._clock()
        pixels = _as_array(frame)
        geometry = Geometry(width=frame.width, height=frame.height, dpi_scale=frame.dpi_scale)
        ambiguity: List[str] = []
        diagnostics: Dict[str, Any] = {"ocr_engine": self.ocr.name, "pixel_based": self.ocr.pixel_based}

        preprocess: Optional[PreprocessResult] = None
        blocks: List[TextBlock] = []
        if pixels is not None:
            preprocess = prepare_for_ocr(
                pixels, self.config.preprocess, allow_raw=not self.ocr.pixel_based and self.config.preprocess.allow_raw_ocr
            )
            diagnostics["preprocess"] = preprocess.describe()
            try:
                raw_blocks = self.ocr.read(
                    OCRRequest(frame=frame, image=preprocess.image, preprocess=preprocess, lang=self.config.ocr_lang)
                )
            except Exception as exc:
                ambiguity.append(f"ocr engine '{self.ocr.name}' failed: {type(exc).__name__}: {exc}")
                raw_blocks = []
            blocks = _clip_to_frame(raw_blocks, geometry)
            diagnostics["ocr_blocks"] = len(blocks)
            diagnostics["ocr_blocks_raw"] = len(raw_blocks)
        else:
            ambiguity.append("frame carried no pixel data")

        kept = filter_blocks(blocks, self.config.min_text_confidence)
        dropped = len(blocks) - len(kept)
        if dropped:
            diagnostics["low_confidence_blocks_dropped"] = dropped

        background = px.dominant_color(pixels) if pixels is not None else (255, 255, 255)
        panels = px.foreign_panels(pixels, background) if pixels is not None else []
        outlines = px.rectangular_outlines(pixels, background) if pixels is not None else []
        accent = px.accent_color(pixels, [p.box for p in panels[:6]], background) if pixels is not None else None

        blocking = _modal_panel_boxes(panels, geometry)
        context = SegmentationContext(
            geometry=geometry,
            blocks=kept,
            background=background,
            pixels=pixels,
            binary=preprocess.image if preprocess is not None else None,
            panels=panels,
            outlines=outlines,
            accent=accent,
            frame_seq=frame.seq,
            blocking_panels=blocking,
        )
        diagnostics.update(
            {
                "background": list(background),
                "accent": list(accent) if accent else None,
                "panels": len(panels),
                "blocking_panels": len(blocking),
                "outlines": len(outlines),
                "text_blocks": len(kept),
            }
        )

        proposals = self.segmenter.propose(context)
        localization = self.locator.localize(context, proposals)
        question_region, question_text, question_confidence = self._question(context, proposals, localization)
        navigation, progress_blocks = self._navigation(context, proposals, localization)
        overlays = self._overlays(context, proposals, localization)
        end_state = self._end_state_evidence(context, question_text, localization, overlays)

        layout = localization.layout_type if localization.options else LayoutType.UNKNOWN
        confidence = self._confidence(kept, localization, proposals, question_text, overlays)
        ambiguity.extend(self._ambiguity(kept, localization, question_text, overlays, layout))

        perception = PerceptionResult(
            frame_seq=frame.seq,
            layout_type=layout,
            question_region=question_region,
            question_text=question_text,
            options=localization.options,
            navigation=navigation,
            overlays=overlays,
            tier2_used=False,
            reconciliation_flags=[],
            text_blocks=kept,
            regions=proposals,
            ocr_engine=self.ocr.name,
            latency_ms=0.0,  # filled below
            tier1_confidence=confidence,
            end_state_evidence=end_state,
        )
        latency_ms = (self._clock() - started) * 1000.0
        perception.latency_ms = round(latency_ms, 2)
        diagnostics["question_confidence"] = round(question_confidence, 3)
        diagnostics["progress_blocks"] = len(progress_blocks)

        return Tier1Result(
            perception=perception,
            preprocess=preprocess,
            context=context,
            localization=localization,
            proposals=proposals,
            ambiguity=ambiguity,
            confidence=confidence,
            latency_ms=latency_ms,
            diagnostics=diagnostics,
        )

    # -- question ----------------------------------------------------------- #
    def _question(
        self, ctx: SegmentationContext, proposals: Sequence[RegionProposal], localization: LocalizationResult
    ) -> Tuple[Optional[Box], str, float]:
        proposal = next((p for p in proposals if p.kind == RegionKind.QUESTION), None)
        option_boxes = [o.hit_box for o in localization.options]

        def in_question(block: TextBlock) -> bool:
            if proposal is not None and _overlap(block.box, proposal.box) > 0.5:
                return True
            if any(_overlap(block.box, box) > 0.5 for box in option_boxes):
                return False
            return proposal is None and block.box[1] < ctx.height * 0.5 and not is_nav_text(block.text)

        members = [b for b in ctx.visible_blocks() if in_question(b) and not is_nav_text(b.text)]
        if not members:
            return (proposal.box if proposal else None, "", 0.0)
        members.sort(key=lambda b: (b.box[1], b.box[0]))
        text = _join_lines([b.text for b in members])
        region = box_union_all([b.box for b in members]) or (proposal.box if proposal else None)
        confidence = float(np.mean([b.confidence for b in members])) if members else 0.0
        return (region, text, confidence)

    # -- navigation --------------------------------------------------------- #
    def _navigation(
        self, ctx: SegmentationContext, proposals: Sequence[RegionProposal], localization: LocalizationResult
    ) -> Tuple[NavigationPerception, List[TextBlock]]:
        nav_proposal = next((p for p in proposals if p.kind == RegionKind.NAVIGATION), None)
        option_boxes = [o.hit_box for o in localization.options]
        next_btn: Optional[NavButton] = None
        prev_btn: Optional[NavButton] = None
        submit_btn: Optional[NavButton] = None
        progress_blocks: List[TextBlock] = []
        progress_text: Optional[str] = None
        progress_current = progress_total = None

        for block in ctx.blocks:
            if any(_overlap(block.box, box) > 0.6 for box in option_boxes):
                continue
            match = _PROGRESS_RE.search(block.text)
            if match:
                progress_blocks.append(block)
                progress_text = block.text.strip()
                progress_current, progress_total = int(match.group(1)), int(match.group(2))
                continue
            lowered = block.text.strip().lower()
            if not is_nav_text(lowered):
                continue
            box = _button_box(block, ctx)
            handle, kind = _classify_nav(lowered)
            button = NavButton(handle=handle, box=box, text=block.text.strip(), enabled=_looks_enabled(block, ctx))
            if kind == "next" and next_btn is None:
                next_btn = button
            elif kind == "prev" and prev_btn is None:
                prev_btn = button
            elif kind == "submit" and submit_btn is None:
                submit_btn = button

        if nav_proposal is not None and next_btn is None and submit_btn is None and prev_btn is None:
            # A button-like panel exists but carries no recognizable label.
            for panel in ctx.panels:
                if _overlap(panel.box, nav_proposal.box) > 0.5 and box_area(panel.box) < ctx.geometry.area * 0.08:
                    next_btn = NavButton(handle="nav_next", box=panel.box, text="", enabled=True)
                    break

        navigation = NavigationPerception(
            next_btn=next_btn,
            prev_btn=prev_btn,
            submit_btn=submit_btn,
            progress_text=progress_text,
            progress_current=progress_current,
            progress_total=progress_total,
        )
        return navigation, progress_blocks

    # -- overlays ------------------------------------------------------------ #
    def _overlays(
        self, ctx: SegmentationContext, proposals: Sequence[RegionProposal], localization: LocalizationResult
    ) -> List[Overlay]:
        option_boxes = [o.hit_box for o in localization.options]
        nav_proposal = next((p for p in proposals if p.kind == RegionKind.NAVIGATION), None)
        excluded: List[Box] = list(option_boxes)
        if nav_proposal is not None:
            excluded.append(nav_proposal.box)
        excluded.extend(
            _button_panel_boxes(ctx, [o.hit_box for o in localization.options])
        )

        overlays: List[Overlay] = []
        for index, panel in enumerate(ctx.panels):
            if any(_overlap(panel.box, box) > 0.55 for box in excluded):
                continue  # an option tile / a navigation button is not an overlay
            if any(
                other is not panel and _overlap(panel.box, other.box) > 0.9
                for other in ctx.panels
            ):
                continue  # a control *inside* another panel (a dialog's Close button)
            if panel.box[3] > ctx.height * 0.85 and panel.box[2] > ctx.width * 0.85:
                continue  # the page itself
            inside = [b for b in ctx.blocks if _overlap(b.box, panel.box) > 0.6]
            if not inside and box_area(panel.box) < ctx.geometry.area * 0.004:
                continue  # a small colour swatch with no text is not an overlay
            text = _join_lines([b.text for b in sorted(inside, key=lambda b: (b.box[1], b.box[0]))])
            kind = _classify_overlay(text)
            # Geometry outranks copy: a large centred singleton panel is a dialog
            # even when its text says "your progress has been saved" (which the
            # keyword table reads as a toast).  Restricted kinds still win -- they
            # are a safety decision, never a layout decision (section 3.3.2).
            if any(box_iou(panel.box, blocking) > 0.6 for blocking in ctx.blocking_panels) and kind not in {
                OverlayKind.CAPTCHA,
                OverlayKind.HUMAN_VERIFICATION,
            }:
                kind = OverlayKind.MODAL
            close_btn = _find_close_button(panel.box, inside, ctx)
            dismissible: Optional[bool]
            if kind in {OverlayKind.CAPTCHA, OverlayKind.HUMAN_VERIFICATION}:
                dismissible = False  # never interact (section 3.3.2)
            elif kind in {OverlayKind.TOAST, OverlayKind.LOADING, OverlayKind.COOKIE_BANNER}:
                dismissible = True
            else:
                dismissible = None if close_btn is None else True
            overlays.append(
                Overlay(
                    handle=f"overlay_{index}",
                    box=panel.box,
                    text=text[:300],
                    kind=kind,
                    dismissible=dismissible,
                    close_btn=close_btn,
                )
            )
        return overlays[:6]

    # -- end state (FR-7.8.4 evidence) --------------------------------------- #
    def _end_state_evidence(
        self, ctx: SegmentationContext, question_text: str, localization: LocalizationResult, overlays: Sequence[Overlay]
    ) -> List[str]:
        evidence: List[str] = []
        if not localization.options:
            evidence.append("no option-like regions detected")
        if not question_text.strip():
            evidence.append("no question text detected")
        haystack = " | ".join([b.text for b in ctx.blocks]).lower()
        for keyword in self.end_state_keywords:
            if keyword and keyword in haystack:
                evidence.append(f"end-state keyword {keyword!r} present")
        return evidence[:6]

    # -- confidence / ambiguity ---------------------------------------------- #
    def _confidence(
        self,
        blocks: Sequence[TextBlock],
        localization: LocalizationResult,
        proposals: Sequence[RegionProposal],
        question_text: str,
        overlays: Sequence[Overlay],
    ) -> float:
        ocr_conf = float(np.mean([b.confidence for b in blocks])) if blocks else 0.0
        question_score = 0.25 if question_text.strip() else 0.0
        proposal_score = float(np.mean([p.score for p in proposals if p.kind != RegionKind.NOISE])) if proposals else 0.0
        score = 0.35 * ocr_conf + 0.45 * localization.confidence + question_score + 0.20 * proposal_score
        if overlays:
            score -= 0.10
        return float(max(0.0, min(0.99, score)))

    def _ambiguity(
        self,
        blocks: Sequence[TextBlock],
        localization: LocalizationResult,
        question_text: str,
        overlays: Sequence[Overlay],
        layout: LayoutType,
    ) -> List[str]:
        reasons: List[str] = []
        if len(localization.options) < 2:
            reasons.append(f"only {len(localization.options)} option(s) localized")
        if not question_text.strip():
            reasons.append("no question text recovered")
        if layout == LayoutType.UNKNOWN and localization.options:
            reasons.append("layout type unresolved")
        if blocks:
            mean_conf = float(np.mean([b.confidence for b in blocks]))
            if mean_conf < self.config.ambiguity_low_ocr_confidence:
                reasons.append(f"mean OCR confidence {mean_conf:.2f} < {self.config.ambiguity_low_ocr_confidence}")
        for overlay in overlays:
            if overlay.kind in {OverlayKind.CAPTCHA, OverlayKind.HUMAN_VERIFICATION}:
                reasons.append(f"restricted overlay kind {overlay.kind.value} detected")
            elif overlay.dismissible is None:
                reasons.append(f"unclassified overlay {overlay.handle!r}")
        selected = [o for o in localization.options if o.selected_marker != SelectedMarker.NONE]
        if len(selected) > 1:
            reasons.append(f"{len(selected)} options show a selected marker (single-choice expects 0 or 1)")
        if not blocks:
            reasons.append("OCR returned no text")
        return reasons


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _as_array(frame: Frame) -> Optional[np.ndarray]:
    if frame.pixels is None:
        return None
    array = np.asarray(frame.pixels)
    if array.ndim == 3 and array.shape[2] == 4:
        array = array[:, :, :3]
    return array if array.size else None


def _clip_to_frame(blocks: Sequence[TextBlock], geometry: Geometry) -> List[TextBlock]:
    from ..geometry import box_clip

    out: List[TextBlock] = []
    for block in blocks:
        clipped = box_clip(block.box, geometry.width, geometry.height)
        if box_area(clipped) < 8:
            continue
        out.append(block.model_copy(update={"box": clipped}) if clipped != block.box else block)
    return out


def _overlap(inner: Box, outer: Box) -> float:
    if box_area(inner) <= 0:
        return 0.0
    return box_area(box_intersection(inner, outer)) / float(box_area(inner))


def _join_lines(lines: Sequence[str]) -> str:
    cleaned = [line.strip() for line in lines if line and line.strip()]
    if not cleaned:
        return ""
    text = cleaned[0]
    for line in cleaned[1:]:
        if text.endswith(("-", "\u2013")):
            text = text[:-1] + line
        else:
            text = f"{text} {line}"
    return re.sub(r"\s+", " ", text).strip()


def _button_box(block: TextBlock, ctx: SegmentationContext) -> Box:
    """Prefer the enclosing panel/outline as the clickable box (FR-7.2.4).

    The shape must roughly contain the label but must not be vastly larger --
    otherwise a full-width card would be reported as the button.
    """
    shapes = list(ctx.outlines) + [panel.box for panel in ctx.panels]
    label_area = max(1, box_area(block.box))
    best: Optional[Box] = None
    best_area = 0
    for shape in shapes:
        if _overlap(block.box, shape) < 0.7:
            continue
        area = box_area(shape)
        if area < label_area * 0.8 or area > label_area * 12:
            continue
        if best is None or area < best_area:
            best, best_area = shape, area
    if best is not None:
        return best
    from ..geometry import box_clip, box_pad_relative

    return box_clip(box_pad_relative(block.box, 0.35, 0.5), ctx.width, ctx.height)


def _button_panel_boxes(ctx: SegmentationContext, option_boxes: Sequence[Box]) -> List[Box]:
    """Panels that are almost certainly buttons rather than floating overlays.

    Evidence used: small relative area, positioned in the lower half of the
    viewport, and carrying navigation wording (or no text at all next to a
    labelled sibling -- icon buttons).
    """
    boxes: List[Box] = []
    area_limit = ctx.geometry.area * 0.10
    for panel in ctx.panels:
        if box_area(panel.box) > area_limit:
            continue
        if panel.box[1] + panel.box[3] < ctx.height * 0.45:
            continue
        if any(_overlap(panel.box, option_box) > 0.5 for option_box in option_boxes):
            continue
        inside = [b.text.strip().lower() for b in ctx.blocks if _overlap(b.box, panel.box) > 0.6]
        if not inside or any(is_nav_text(text) for text in inside):
            boxes.append(panel.box)
    return boxes


def _classify_nav(text: str) -> Tuple[str, str]:
    lowered = text.strip().lower()
    if any(keyword in lowered for keyword in _SUBMIT_KEYWORDS):
        return ("nav_submit", "submit")
    if any(keyword in lowered for keyword in _PREV_KEYWORDS):
        return ("nav_prev", "prev")
    if any(keyword in lowered for keyword in _NEXT_KEYWORDS):
        return ("nav_next", "next")
    return ("nav_other", "other")


def _looks_enabled(block: TextBlock, ctx: SegmentationContext) -> bool:
    """A washed-out button is usually disabled; use colour contrast as evidence."""
    color = px.region_mean_color(ctx.pixels, block.box) if ctx.pixels is not None else (0, 0, 0)
    luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
    background_luminance = 0.299 * ctx.background[0] + 0.587 * ctx.background[1] + 0.114 * ctx.background[2]
    return abs(luminance - background_luminance) > 24.0


def _classify_overlay(text: str) -> OverlayKind:
    lowered = (text or "").lower()
    for keyword, kind in _OVERLAY_KEYWORDS:
        if keyword in lowered:
            return kind
    return OverlayKind.UNKNOWN


def _similar_size(a: Box, b: Box) -> bool:
    return abs(a[2] - b[2]) <= 0.15 * max(a[2], b[2]) and abs(a[3] - b[3]) <= 0.15 * max(a[3], b[3])


def _modal_panel_boxes(panels: Sequence[Any], geometry: Geometry) -> List[Box]:
    """Dialog-like panels: large, centred, and *unique* on the page.

    A card grid is also made of large panels, but its tiles come in same-sized
    siblings; a dialog is a singleton covering the middle of the viewport.  Only
    a dialog hides the question, so only a dialog blocks localization.
    """
    candidates: List[Box] = []
    for panel in panels:
        box = panel.box
        if box_area(box) < geometry.area * 0.06:
            continue
        if box[2] > geometry.width * 0.9 and box[3] > geometry.height * 0.9:
            continue  # the page background itself
        centre_x = box[0] + box[2] / 2.0
        centre_y = box[1] + box[3] / 2.0
        if abs(centre_x - geometry.width / 2.0) > geometry.width * 0.15:
            continue
        if abs(centre_y - geometry.height / 2.0) > geometry.height * 0.22:
            continue
        candidates.append(box)
    unique: List[Box] = []
    for box in candidates:
        if any(other is not box and _similar_size(box, other) for other in candidates):
            continue  # an option card / tile in a grid
        unique.append(box)
    return unique[:3]


def _find_close_button(panel_box: Box, inside: Sequence[TextBlock], ctx: SegmentationContext) -> Optional[NavButton]:
    for block in inside:
        lowered = block.text.strip().lower()
        if lowered in _DISMISS_KEYWORDS or any(lowered.startswith(k) for k in ("close", "dismiss", "no thanks")):
            return NavButton(handle="overlay_close", box=_button_box(block, ctx), text=block.text.strip())
    # A tiny square panel in the top-right corner is the classic close affordance.
    x, y, w, h = panel_box
    corner = (x + w - max(28, int(h * 0.5)), y, max(20, int(h * 0.45)), max(20, int(h * 0.45)))
    for outline in ctx.outlines:
        if box_iou(outline, corner) > 0.2 and box_area(outline) < 40 * 40:
            return NavButton(handle="overlay_close", box=outline, text="")
    return None


def question_number_from(blocks: Sequence[TextBlock]) -> Optional[int]:
    for block in blocks:
        match = _QUESTION_NUMBER_RE.search(block.text)
        if match:
            return int(match.group(1))
    return None


__all__ = ["Tier1Perception", "Tier1Result"]
