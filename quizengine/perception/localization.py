"""Option element localization (FR-7.2.4) + layout classification.

For every candidate option this produces a pixel-precise bounding box around the
**clickable hit area** -- the full row/tile, not just the text -- via
connected-component analysis on text blocks plus border/edge detection for card
outlines.  Handles (``opt_0``, ``opt_1``, ...) are the semantic ids every action
must reference (FR-7.7.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import PerceptionConfig
from ..contracts import LayoutType, OptionPerception, RegionKind, RegionProposal, SelectedMarker, TextBlock
from ..geometry import (
    Box,
    Geometry,
    box_area,
    box_center,
    box_clip,
    box_intersection,
    box_iou,
    box_union,
    box_union_all,
    horizontal_gap,
    sort_boxes_reading_order,
    vertical_gap,
)
from . import pixels as px
from .segmentation import (
    SegmentationContext,
    column_gaps,
    group_by_rhythm,
    is_nav_text,
    maximal_rhythm_union,
)


@dataclass
class LocalizationResult:
    options: List[OptionPerception] = field(default_factory=list)
    layout_type: LayoutType = LayoutType.UNKNOWN
    evidence: List[str] = field(default_factory=list)
    confidence: float = 0.0
    style_hint: str = "radio"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "count": len(self.options),
            "layout_type": self.layout_type.value,
            "confidence": round(self.confidence, 3),
            "style_hint": self.style_hint,
            "evidence": list(self.evidence),
            "options": [
                {"index": o.index, "handle": o.handle, "text": o.text, "hit_box": list(o.hit_box)}
                for o in self.options
            ],
        }


class OptionLocator:
    def __init__(self, config: PerceptionConfig) -> None:
        self.config = config

    # -- public API --------------------------------------------------------- #
    def localize(self, ctx: SegmentationContext, proposals: Sequence[RegionProposal]) -> LocalizationResult:
        candidates, source = self._candidate_blocks(ctx, proposals)
        evidence: List[str] = [f"candidate source: {source}", f"{len(candidates)} candidate text run(s)"]
        if len(candidates) < 2:
            return LocalizationResult(evidence=evidence + ["fewer than 2 candidates -> no options"], confidence=0.0)

        hit_boxes = self._hit_boxes(candidates, ctx, proposals)
        merged = _merge_hits(hit_boxes, candidates, ctx)
        if len(merged) < 2:
            evidence.append(f"hit boxes collapsed to {len(merged)} after merging")
            return LocalizationResult(evidence=evidence, confidence=0.1)

        style_hint, panels_covering = self._style_hint(merged, ctx)
        layout = classify_layout([h["box"] for h in merged], ctx.geometry, style_hint, panels_covering)
        ordered = order_hits(merged, layout)
        markers = self._markers(ordered, ctx, style_hint)

        options: List[OptionPerception] = []
        for index, hit in enumerate(ordered):
            text_box = hit["text_box"]
            options.append(
                OptionPerception(
                    index=index,
                    handle=f"opt_{index}",
                    text=hit["text"].strip(),
                    hit_box=hit["box"],
                    text_box=text_box,
                    text_conf=float(np.mean([b.confidence for b in hit["blocks"]])) if hit["blocks"] else 0.0,
                    selected_marker=markers[index],
                )
            )

        confidence = self._confidence(ordered, layout, panels_covering, ctx)
        evidence.extend(
            [
                f"{len(options)} options localized",
                f"layout={layout.value}",
                f"style={style_hint}",
                f"panels covering {panels_covering}/{len(options)} hit areas",
                f"selected={[o.index for o in options if o.selected_marker != SelectedMarker.NONE]}",
            ]
        )
        return LocalizationResult(
            options=options, layout_type=layout, evidence=evidence, confidence=confidence, style_hint=style_hint
        )

    # -- candidates --------------------------------------------------------- #
    def _candidate_blocks(
        self, ctx: SegmentationContext, proposals: Sequence[RegionProposal]
    ) -> Tuple[List[TextBlock], str]:
        options_region = next((p for p in proposals if p.kind == RegionKind.OPTIONS), None)
        excluded = [p.box for p in proposals if p.kind in {RegionKind.QUESTION, RegionKind.NAVIGATION, RegionKind.HEADER}]

        def usable(block: TextBlock) -> bool:
            if is_nav_text(block.text):
                return False
            if block.box[3] < self.config.option_min_height_px:
                return False
            if any(_overlap_ratio(block.box, box) > 0.6 for box in excluded):
                return False
            return True

        if options_region is not None:
            inside = [b for b in ctx.visible_blocks() if usable(b) and _overlap_ratio(b.box, options_region.box) > 0.35]
            if len(inside) >= 2:
                return sorted(inside, key=lambda b: (b.box[1], b.box[0])), "OPTIONS region proposal"

        pool = [b for b in ctx.visible_blocks() if usable(b)]
        union = maximal_rhythm_union(pool)
        if len(union) >= 2:
            return union, f"typography rhythm union ({len(union)} run(s))"
        return sorted(pool, key=lambda b: (b.box[1], b.box[0])), "all non-navigation text runs"

    # -- hit areas ---------------------------------------------------------- #
    def _hit_boxes(
        self, candidates: Sequence[TextBlock], ctx: SegmentationContext, proposals: Sequence[RegionProposal]
    ) -> List[Dict[str, Any]]:
        """Expand each text run to its clickable row/tile (FR-7.2.4).

        Orientation is detected from the candidate layout, because the expansion
        direction differs: a vertical list shares a column extent, a horizontal
        row shares a row extent, and a grid shares both per cell.
        """
        shapes = list(ctx.outlines) + [panel.box for panel in ctx.panels]
        boxes = [b.box for b in candidates]
        rows = _cluster_axis(boxes, "y")
        cols = _cluster_axis(boxes, "x")
        if len(rows) == 1 and len(candidates) >= 2:
            orientation = "horizontal"
        elif len(cols) == 1:
            orientation = "vertical"
        else:
            orientation = "grid"

        median_height = float(np.median([b.box[3] for b in candidates])) or 16.0
        pad_ratio = float(self.config.option_row_padding_ratio)
        row_of = {id(box): index for index, row in enumerate(rows) for box in row}
        col_of = {id(box): index for index, col in enumerate(cols) for box in col}

        def extent(group: Sequence[Box]) -> Tuple[int, int, int, int]:
            left = min(b[0] for b in group)
            right = max(b[0] + b[2] for b in group)
            top = min(b[1] for b in group)
            bottom = max(b[1] + b[3] for b in group)
            return left, top, right, bottom

        hits: List[Dict[str, Any]] = []
        for block in candidates:
            enclosing = _enclosing_shape(block.box, shapes)
            if enclosing is not None:
                hits.append(
                    {
                        "box": box_clip(enclosing, ctx.width, ctx.height),
                        "text_box": block.box,
                        "text": block.text,
                        "blocks": [block],
                        "source": "panel/outline",
                    }
                )
                continue

            if orientation == "horizontal":
                row_boxes = rows[row_of[id(block.box)]]
                left, top, right, bottom = extent(row_boxes)
                gaps = [horizontal_gap(a, b) for a, b in zip(sorted(row_boxes, key=lambda b: b[0]), sorted(row_boxes, key=lambda b: b[0])[1:])]
                gap = float(np.median(gaps)) if gaps else 0.0
                pad_x = int(min(max(6.0, gap * 0.4), median_height * 1.2)) if gap > 0 else int(median_height * 0.5)
                pad_y = int(min(median_height * pad_ratio, max(4.0, gap * 0.25)))
                box = (block.box[0] - pad_x, top - pad_y, block.box[2] + 2 * pad_x, (bottom - top) + 2 * pad_y)
                source = "row expansion"
            elif orientation == "grid":
                cell_boxes = [b for b in boxes if row_of.get(id(b)) == row_of[id(block.box)] or col_of.get(id(b)) == col_of[id(block.box)]]
                left, top, right, bottom = extent(cell_boxes)
                pad_x = int(median_height * 0.4)
                pad_y = int(median_height * pad_ratio)
                box = (block.box[0] - pad_x, block.box[1] - pad_y, block.box[2] + 2 * pad_x, block.box[3] + 2 * pad_y)
                source = "grid cell expansion"
            else:
                column_boxes = cols[col_of[id(block.box)]]
                left, _top, right, _bottom = extent(column_boxes)
                # A plain radio/checkbox row has no border and no fill, so the
                # only honest evidence for how far the clickable row reaches is
                # the content column it sits in, bounded by the whitespace gutter
                # to its right (L6: never invent a width the pixels do not show).
                right = max(right, _column_right_edge(block.box, ctx))
                glyph_inset = int(median_height * 1.1)
                box_left = max(0, min(left, block.box[0] - glyph_inset))
                top = int(block.box[1] - median_height * pad_ratio)
                bottom = int(block.box[1] + block.box[3] + median_height * pad_ratio)
                box = (box_left, top, max(1, right - box_left), max(1, bottom - top))
                source = "column expansion"

            hits.append(
                {
                    "box": box_clip(box, ctx.width, ctx.height),
                    "text_box": block.box,
                    "text": block.text,
                    "blocks": [block],
                    "source": source,
                }
            )
        return hits

    def _style_hint(self, hits: Sequence[Dict[str, Any]], ctx: SegmentationContext) -> Tuple[str, int]:
        boxes = [h["box"] for h in hits]
        covering = sum(1 for box in boxes if any(_overlap_ratio(box, shape) > 0.85 for shape in ctx.outlines + [p.box for p in ctx.panels]))
        frame_area = float(ctx.geometry.area) or 1.0
        mean_area_ratio = float(np.mean([box_area(b) / frame_area for b in boxes])) if boxes else 0.0
        if covering >= max(2, int(len(boxes) * 0.6)):
            if mean_area_ratio > 0.05:
                return ("tile", covering)
            return ("card", covering)
        if mean_area_ratio > 0.09:
            return ("tile", covering)
        return ("radio", covering)

    # -- selection markers -------------------------------------------------- #
    def _markers(
        self, hits: Sequence[Dict[str, Any]], ctx: SegmentationContext, style_hint: str
    ) -> List[SelectedMarker]:
        if ctx.pixels is None:
            return [SelectedMarker.NONE] * len(hits)
        features: List[px.MarkerFeatures] = []
        boxes = [tuple(hit["box"]) for hit in hits]
        for index, hit in enumerate(hits):
            feature = px.analyze_marker(
                ctx.pixels,
                hit["box"],
                ctx.background,
                style_hint,
                text_box=hit.get("text_box"),
                left_limit=_zone_left_limit(tuple(hit["box"]), boxes),
            )
            feature.index = index
            features.append(feature)
        return px.infer_selected_markers(features, style_hint=style_hint)

    # -- confidence --------------------------------------------------------- #
    def _confidence(
        self, hits: Sequence[Dict[str, Any]], layout: LayoutType, panels_covering: int, ctx: SegmentationContext
    ) -> float:
        boxes = [h["box"] for h in hits]
        if len(boxes) < 2:
            return 0.1
        score = 0.45
        regularity = _rhythm_score(boxes, layout)
        score += 0.25 * regularity
        overlaps = sum(1 for i, a in enumerate(boxes) for b in boxes[i + 1 :] if box_iou(a, b) > 0.15)
        score -= min(0.3, 0.08 * overlaps)
        if panels_covering >= max(2, int(len(boxes) * 0.6)):
            score += 0.15
        if layout != LayoutType.UNKNOWN:
            score += 0.10
        mean_conf = float(np.mean([np.mean([b.confidence for b in h["blocks"]]) for h in hits])) if hits else 0.0
        score += 0.15 * max(0.0, min(1.0, mean_conf))
        return float(max(0.0, min(0.98, score)))


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def classify_layout(
    boxes: Sequence[Box], geometry: Geometry, style_hint: str = "radio", panels_covering: int = 0
) -> LayoutType:
    """Vertical / horizontal / card grid / tile / text-only (section 3.1 layouts)."""
    if len(boxes) < 2:
        return LayoutType.UNKNOWN

    rows = _cluster_axis(boxes, axis="y")
    cols = _cluster_axis(boxes, axis="x")
    frame_area = float(geometry.area) or 1.0
    mean_area_ratio = float(np.mean([box_area(b) / frame_area for b in boxes]))

    if len(rows) >= 2 and len(cols) >= 2:
        return LayoutType.CARD_GRID
    if len(rows) == 1:
        return LayoutType.HORIZONTAL_OPTIONS
    if len(cols) == 1:
        # A single column is a vertical stack even when each row is a card; only
        # genuinely large tiles earn the TILE label (section 3.1 layout list).
        if mean_area_ratio > 0.07 or (style_hint == "tile" and mean_area_ratio > 0.05):
            return LayoutType.TILE
        return LayoutType.VERTICAL_OPTIONS
    if panels_covering == 0 and mean_area_ratio < 0.02:
        return LayoutType.TEXT_ONLY_BUTTONS
    return LayoutType.VERTICAL_OPTIONS


def _zone_left_limit(box: Box, others: Sequence[Box]) -> int:
    """Right edge of the nearest sibling sharing this option's row.

    The marker zone reaches left of the label (that is where the glyph lives), so
    without this limit it can slide into a selected neighbour and read *its* accent
    ink as this option's marker.
    """
    limit = 0
    middle = box[0] + box[2] // 2
    for other in others:
        if other == box:
            continue
        same_row = other[1] < box[1] + box[3] and box[1] < other[1] + other[3]
        if same_row and other[0] + other[2] <= middle:
            limit = max(limit, int(other[0] + other[2]))
    return limit


def _column_right_edge(box: Box, ctx: SegmentationContext, *, min_gap_px: int = 24) -> int:
    """How far right a plain option row plausibly extends.

    Evidence used, in order: the widest text run sharing the row's column (the
    question paragraph, a longer label), then the first whitespace gutter to the
    right of the run.  The gutter caps the result so a row never reaches into the
    next column.
    """
    text_right = int(box[0] + box[2])
    visible = ctx.visible_blocks()
    column_right = max(
        (int(b.box[0] + b.box[2]) for b in visible if b.box[0] <= text_right and b.box[0] + b.box[2] >= int(box[0])),
        default=text_right,
    )
    gutter: Optional[int] = None
    for start, _end in column_gaps(visible, ctx.width, min_gap_px=min_gap_px):
        if start >= text_right - 2:
            gutter = int(start)
            break
    right = max(column_right, text_right)
    return right if gutter is None else min(gutter, right)


def _cluster_axis(boxes: Sequence[Box], axis: str) -> List[List[Box]]:
    """Group boxes that overlap along the given axis (y -> rows, x -> columns)."""
    ordered = sorted(boxes, key=lambda b: b[1] if axis == "y" else b[0])
    clusters: List[List[Box]] = []
    for box in ordered:
        placed = False
        for cluster in clusters:
            if any(_axis_overlap(box, other, axis) for other in cluster):
                cluster.append(box)
                placed = True
                break
        if not placed:
            clusters.append([box])
    return clusters


def _axis_overlap(a: Box, b: Box, axis: str) -> bool:
    if axis == "y":
        a0, a1 = a[1], a[1] + a[3]
        b0, b1 = b[1], b[1] + b[3]
        reference = min(a[3], b[3]) or 1
    else:
        a0, a1 = a[0], a[0] + a[2]
        b0, b1 = b[0], b[0] + b[2]
        reference = min(a[2], b[2]) or 1
    overlap = min(a1, b1) - max(a0, b0)
    return overlap > 0.45 * reference


def order_hits(hits: List[Dict[str, Any]], layout: LayoutType) -> List[Dict[str, Any]]:
    """Deterministic option order (the order the solver's A-D letters refer to)."""
    if layout == LayoutType.HORIZONTAL_OPTIONS:
        return sorted(hits, key=lambda h: h["box"][0])
    if layout == LayoutType.CARD_GRID:
        rows = _cluster_axis([h["box"] for h in hits], axis="y")
        row_of: Dict[int, int] = {}
        for row_index, row in enumerate(sorted(rows, key=lambda r: min(b[1] for b in r))):
            for box in row:
                row_of[id(box)] = row_index
        return sorted(hits, key=lambda h: (row_of.get(id(h["box"]), 0), h["box"][0]))
    return sorted(hits, key=lambda h: (h["box"][1], h["box"][0]))


def _merge_hits(hits: List[Dict[str, Any]], candidates: Sequence[TextBlock], ctx: SegmentationContext) -> List[Dict[str, Any]]:
    """Merge hit areas that describe the same clickable element."""
    merged: List[Dict[str, Any]] = []
    for hit in sorted(hits, key=lambda h: (h["box"][1], h["box"][0])):
        target: Optional[Dict[str, Any]] = None
        for existing in merged:
            if box_iou(existing["box"], hit["box"]) > 0.45 or _contains(existing["box"], hit["box"]) or _contains(hit["box"], existing["box"]):
                target = existing
                break
        if target is None:
            merged.append(dict(hit))
            continue
        # Same row: "A." and "Paris" are one option, not two.
        target["box"] = box_union(target["box"], hit["box"])
        target["text"] = f"{target['text']} {hit['text']}".strip()
        target["blocks"] = list(target["blocks"]) + list(hit["blocks"])
        if target.get("text_box") is None or box_area(hit["text_box"]) > box_area(target["text_box"]):
            target["text_box"] = hit["text_box"]
    clipped = []
    for hit in merged:
        box = box_clip(hit["box"], ctx.width, ctx.height)
        if box_area(box) < 16:
            continue
        hit["box"] = box
        clipped.append(hit)
    return clipped


def _contains(outer: Box, inner: Box) -> bool:
    ox, oy, ow, oh = outer
    ix, iy, iw, ih = inner
    return ix >= ox and iy >= oy and ix + iw <= ox + ow and iy + ih <= oy + oh


def _enclosing_shape(box: Box, shapes: Sequence[Box]) -> Optional[Box]:
    best: Optional[Box] = None
    best_area = 0
    for shape in shapes:
        ratio = _overlap_ratio(box, shape)
        area = box_area(shape)
        if ratio > 0.8 and area > box_area(box) and (best is None or area < best_area):
            best, best_area = shape, area
    return best


def _overlap_ratio(inner: Box, outer: Box) -> float:
    if box_area(inner) <= 0:
        return 0.0
    return box_area(box_intersection(inner, outer)) / float(box_area(inner))


def _rhythm_score(boxes: Sequence[Box], layout: LayoutType) -> float:
    if len(boxes) < 3:
        return 0.5
    ordered = sorted(boxes, key=lambda b: (b[1], b[0]))
    if layout == LayoutType.HORIZONTAL_OPTIONS:
        gaps = [horizontal_gap(a, b) for a, b in zip(ordered, ordered[1:])]
        sizes = [b[2] for b in ordered]
    else:
        gaps = [vertical_gap(a, b) for a, b in zip(ordered, ordered[1:])]
        sizes = [b[3] for b in ordered]
    gap_std = float(np.std(gaps)) if gaps else 1.0
    gap_mean = float(np.mean(gaps)) if gaps else 1.0
    size_cv = float(np.std(sizes) / (np.mean(sizes) or 1.0))
    return float(max(0.0, min(1.0, 1.0 - (gap_std / (gap_mean + 1e-6)) * 0.5 - size_cv)))


__all__ = ["LocalizationResult", "OptionLocator", "classify_layout", "order_hits"]
