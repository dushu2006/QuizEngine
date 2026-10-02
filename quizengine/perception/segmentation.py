"""Region segmentation heuristic (FR-7.2.3).

Classifies candidate regions as ``{QUESTION, OPTIONS, NAVIGATION, HEADER, NOISE}``
from typography statistics alone -- font-size clusters, column gaps, whitespace
bands and the position of interactive-looking elements.

This module is explicitly a **proposal generator**: Tier 2 confirms or overrides
(FR-7.2.5), and reconciliation decides who wins on which axis (FR-7.2.7).  No
selector, keyword-per-site or coordinate assumption is used anywhere; the only
vocabulary is generic UI wording ("next", "back", "submit") which is a property
of the language, not of a platform.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import PerceptionConfig
from ..contracts import RegionKind, RegionProposal, TextBlock
from ..geometry import (
    Box,
    Geometry,
    box_area,
    box_center,
    box_contains_box,
    box_intersection,
    box_iou,
    box_union,
    box_union_all,
    horizontal_gap,
    vertical_gap,
)
from . import pixels as px

#: Generic navigation vocabulary (language-level, not platform-level).
NAV_KEYWORDS = (
    "next",
    "previous",
    "prev",
    "back",
    "continue",
    "submit",
    "finish",
    "check answer",
    "check",
    "save",
    "skip",
    "restart",
    "->",
    "\u2192",
    "\u2190",
)

HEADER_HINTS = ("quiz", "test", "exam", "score", "authorized", "environment", "question", "of")


@dataclass
class SegmentationContext:
    """Everything Tier-1 segmentation may look at for one frame."""

    geometry: Geometry
    blocks: List[TextBlock]
    background: Tuple[int, int, int]
    #: Raw RGB pixels of the frame -- colour evidence (selection markers, panels).
    pixels: Optional[np.ndarray] = None
    #: Binarized text mask from the FR-7.2.2 pipeline.
    binary: Optional[np.ndarray] = None
    panels: List[px.Panel] = field(default_factory=list)
    outlines: List[Box] = field(default_factory=list)
    accent: Optional[Tuple[int, int, int]] = None
    frame_seq: int = -1
    #: Boxes of dialog-like panels (FR-7.2.5).  Text underneath one of these is
    #: *not* the question and must not be localized until the dialog is gone.
    blocking_panels: List[Box] = field(default_factory=list)

    @property
    def width(self) -> int:
        return self.geometry.width

    def obscured(self, block: TextBlock) -> bool:
        """True when a dialog-like panel covers this text run."""
        return any(_overlap_ratio(block.box, panel) > 0.6 for panel in self.blocking_panels)

    def visible_blocks(self) -> List[TextBlock]:
        """Text runs that are not hidden behind a dialog (FR-7.2.5)."""
        if not self.blocking_panels:
            return list(self.blocks)
        return [block for block in self.blocks if not self.obscured(block)]

    @property
    def height(self) -> int:
        return self.geometry.height


@dataclass
class FontCluster:
    """One font-size class: a median glyph height and its members."""

    median_height: float
    blocks: List[TextBlock]

    @property
    def size(self) -> int:
        return len(self.blocks)

    def union_box(self) -> Optional[Box]:
        return box_union_all([b.box for b in self.blocks])


class RegionSegmenter:
    def __init__(self, config: PerceptionConfig) -> None:
        self.config = config

    # -- public API --------------------------------------------------------- #
    def propose(self, ctx: SegmentationContext) -> List[RegionProposal]:
        proposals: List[RegionProposal] = []
        visible = ctx.visible_blocks()
        clusters = font_size_clusters(visible, max_clusters=self.config.font_size_clusters)
        nav_blocks = [b for b in visible if is_nav_text(b.text)]
        header = self._header_region(ctx, clusters, nav_blocks)
        if header is not None:
            proposals.append(header)
        navigation = self._navigation_region(ctx, nav_blocks)
        if navigation is not None:
            proposals.append(navigation)
        question = self._question_region(ctx, clusters, proposals)
        if question is not None:
            proposals.append(question)
        options = self._options_region(ctx, question, navigation, header)
        if options is not None:
            proposals.append(options)
        noise = self._noise_regions(ctx, proposals)
        proposals.extend(noise)
        return _dedupe(proposals)

    # -- regions ------------------------------------------------------------ #
    def _header_region(
        self, ctx: SegmentationContext, clusters: List[FontCluster], nav_blocks: List[TextBlock]
    ) -> Optional[RegionProposal]:
        band = int(ctx.height * 0.16)
        candidates = [
            b
            for b in ctx.visible_blocks()
            if b.box[1] + b.box[3] <= band
            and b not in nav_blocks
            and not (clusters and clusters[0].median_height and b.box[3] >= clusters[0].median_height * 1.15)
        ]
        if not candidates:
            return None
        box = box_union_all([b.box for b in candidates])
        if box is None:
            return None
        evidence = [f"{len(candidates)} small text run(s) in the top {band}px band"]
        if any(any(hint in b.text.lower() for hint in HEADER_HINTS) for b in candidates):
            evidence.append("header-like wording")
        return RegionProposal(kind=RegionKind.HEADER, box=box, score=0.55, evidence=evidence)

    def _navigation_region(self, ctx: SegmentationContext, nav_blocks: List[TextBlock]) -> Optional[RegionProposal]:
        boxes: List[Box] = []
        evidence: List[str] = []
        for block in nav_blocks:
            panel = _enclosing_panel(block.box, ctx.panels + [px.Panel(o, 0.3, ctx.background, 200.0) for o in ctx.outlines])
            boxes.append(panel if panel is not None else block.box)
        button_panels = [
            panel.box
            for panel in ctx.panels
            if panel.box[1] > ctx.height * 0.55 and box_area(panel.box) < ctx.geometry.area * 0.12
        ]
        boxes.extend(button_panels)
        if not boxes:
            return None
        box = box_union_all(boxes)
        if box is None:
            return None
        if nav_blocks:
            evidence.append(f"{len(nav_blocks)} navigation keyword run(s): " + ", ".join(repr(b.text) for b in nav_blocks[:3]))
        if button_panels:
            evidence.append(f"{len(button_panels)} button-like panel(s) in the lower half")
        score = 0.5 + (0.25 if nav_blocks else 0.0) + (0.2 if button_panels else 0.0)
        return RegionProposal(kind=RegionKind.NAVIGATION, box=box, score=min(0.95, score), evidence=evidence)

    def _question_region(
        self, ctx: SegmentationContext, clusters: List[FontCluster], proposals: List[RegionProposal]
    ) -> Optional[RegionProposal]:
        excluded = [p.box for p in proposals if p.kind in {RegionKind.NAVIGATION}]
        headline = clusters[0] if clusters else None
        candidates: List[TextBlock] = []
        evidence: List[str] = []

        if headline is not None and headline.size:
            pool = [
                b
                for b in headline.blocks
                if not is_nav_text(b.text)
                and not ctx.obscured(b)
                and not any(box_contains_box(x, b.box) for x in excluded)
            ]
            # A question paragraph is *contiguous*; an option list is a repeated
            # rhythm.  Drop rhythm groups, otherwise a card grid whose tiles are
            # as tall as the headline gets absorbed into the question region.
            option_like = option_like_groups(pool)
            candidates = [b for b in pool if b not in option_like]
            evidence.append(
                f"largest font cluster (median height {headline.median_height:.0f}px, {headline.size} run(s), "
                f"{len(option_like)} excluded as option-like)"
            )

        if not candidates:
            # Fall back to the longest non-navigation text in the upper 70%.
            pool = [
                b
                for b in ctx.visible_blocks()
                if not is_nav_text(b.text)
                and b.box[1] < ctx.height * 0.7
                and not any(box_contains_box(x, b.box) for x in excluded)
            ]
            if pool:
                pool.sort(key=lambda b: len(b.text), reverse=True)
                candidates = pool[:1]
                evidence.append(f"longest text run in the upper viewport ({len(candidates[0].text)} chars)")

        if not candidates:
            # Last resort: the topmost run of whatever the headline cluster held.
            # Only after the longest-text search above, otherwise a page whose
            # largest font cluster *is* the option list (no Next button, short
            # labels) yields an option as the "question" and the remaining options
            # get renumbered -- silently binding the answer to the wrong one.
            headline_pool = headline.blocks if headline is not None else []
            usable = [b for b in headline_pool if not is_nav_text(b.text) and not ctx.obscured(b)]
            if not usable:
                return None
            candidates = sorted(usable, key=lambda b: (b.box[1], b.box[0]))[:1]
            evidence.append("last resort: topmost run of the largest font cluster")

        # Merge continuation lines: same column, adjacent rows, similar height.
        merged = _merge_paragraph(candidates, ctx.visible_blocks(), excluded)
        box = box_union_all([b.box for b in merged])
        if box is None:
            return None
        score = 0.6 + min(0.3, 0.05 * len(merged))
        if headline is not None and headline.size:
            score += 0.1
        evidence.append(f"{len(merged)} text run(s) merged into a paragraph")
        return RegionProposal(kind=RegionKind.QUESTION, box=_pad(box, ctx, 0.12), score=min(0.95, score), evidence=evidence)

    def _options_region(
        self,
        ctx: SegmentationContext,
        question: Optional[RegionProposal],
        navigation: Optional[RegionProposal],
        header: Optional[RegionProposal],
    ) -> Optional[RegionProposal]:
        excluded = [p.box for p in (question, navigation, header) if p is not None]
        candidates = [
            b
            for b in ctx.visible_blocks()
            if not is_nav_text(b.text) and not any(_overlap_ratio(b.box, x) > 0.6 for x in excluded)
        ]
        if len(candidates) < 2:
            return None
        best = maximal_rhythm_union(candidates)
        if len(best) < 2:
            return None
        box = box_union_all([b.box for b in best])
        if box is None:
            return None
        gaps = _regularity(best)
        evidence = [
            f"{len(best)} repeated text runs",
            f"rhythm regularity {gaps:.2f}",
            f"median height {float(np.median([b.box[3] for b in best])):.0f}px",
        ]
        score = 0.5 + 0.3 * gaps + min(0.15, 0.03 * len(best))
        return RegionProposal(kind=RegionKind.OPTIONS, box=_pad(box, ctx, 0.18), score=min(0.95, score), evidence=evidence)

    def _noise_regions(self, ctx: SegmentationContext, proposals: List[RegionProposal]) -> List[RegionProposal]:
        covered = [p.box for p in proposals if p.kind != RegionKind.NOISE]
        noise: List[RegionProposal] = []
        for block in ctx.blocks:
            if any(_overlap_ratio(block.box, box) > 0.5 for box in covered):
                continue
            noise.append(
                RegionProposal(
                    kind=RegionKind.NOISE,
                    box=block.box,
                    score=0.3,
                    evidence=[f"text run {block.text[:32]!r} not covered by any semantic region"],
                )
            )
        return noise[:12]


# --------------------------------------------------------------------------- #
# typography statistics
# --------------------------------------------------------------------------- #
def font_size_clusters(blocks: Sequence[TextBlock], max_clusters: int = 4) -> List[FontCluster]:
    """1-D clustering of text heights; largest-first."""
    heights = sorted((b.box[3] for b in blocks if b.box[3] > 0), reverse=True)
    if not heights:
        return []
    buckets: List[List[float]] = [[heights[0]]]
    for height in heights[1:]:
        if height >= buckets[-1][0] * 0.72:
            buckets[-1].append(height)
        else:
            buckets.append([height])
    clusters: List[FontCluster] = []
    for bucket in buckets[:max_clusters]:
        median = float(np.median(bucket))
        members = [b for b in blocks if abs(b.box[3] - median) <= median * 0.28]
        clusters.append(FontCluster(median_height=median, blocks=members))
    clusters.sort(key=lambda c: c.median_height, reverse=True)
    # De-duplicate members claimed by a larger cluster.
    claimed: set[int] = set()
    unique: List[FontCluster] = []
    for cluster in clusters:
        members = [b for b in cluster.blocks if id(b) not in claimed]
        claimed.update(id(b) for b in members)
        if members:
            unique.append(FontCluster(median_height=cluster.median_height, blocks=members))
    return unique


def whitespace_bands(height: int, blocks: Sequence[TextBlock], min_band_px: int = 8) -> List[Tuple[int, int]]:
    """Horizontal whitespace bands -- layout separators (FR-7.2.3)."""
    occupied = np.zeros(height, dtype=bool)
    for block in blocks:
        y0 = max(0, block.box[1])
        y1 = min(height, block.box[1] + block.box[3])
        if y1 > y0:
            occupied[y0:y1] = True
    bands: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for index in range(height):
        if not occupied[index]:
            if start is None:
                start = index
        elif start is not None:
            if index - start >= min_band_px:
                bands.append((start, index))
            start = None
    if start is not None and height - start >= min_band_px:
        bands.append((start, height))
    return bands


def column_gaps(blocks: Sequence[TextBlock], width: int, min_gap_px: int = 24) -> List[Tuple[int, int]]:
    """Vertical whitespace gutters -- the column structure signal."""
    occupied = np.zeros(width, dtype=bool)
    for block in blocks:
        x0 = max(0, block.box[0])
        x1 = min(width, block.box[0] + block.box[2])
        if x1 > x0:
            occupied[x0:x1] = True
    gaps: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for index in range(width):
        if not occupied[index]:
            if start is None:
                start = index
        elif start is not None:
            if index - start >= min_gap_px:
                gaps.append((start, index))
            start = None
    return gaps


def is_nav_text(text: str) -> bool:
    lowered = (text or "").strip().lower()
    if not lowered or len(lowered) > 40:
        return False
    return any(keyword == lowered or lowered.startswith(keyword + " ") or keyword in lowered for keyword in NAV_KEYWORDS)


def group_by_rhythm(blocks: Sequence[TextBlock]) -> List[List[TextBlock]]:
    """Group text runs that share a repeated vertical/horizontal rhythm."""
    if len(blocks) < 2:
        return [list(blocks)] if blocks else []
    heights = np.array([b.box[3] for b in blocks], dtype=float)
    median_height = float(np.median(heights)) if heights.size else 0.0
    similar = [b for b in blocks if median_height and abs(b.box[3] - median_height) <= median_height * 0.45]
    if len(similar) < 2:
        return []

    # Vertical stack: similar left edges, non-overlapping y, repeated gap.
    by_left: Dict[int, List[TextBlock]] = {}
    for block in similar:
        key = int(round(block.box[0] / 24.0))
        by_left.setdefault(key, []).append(block)
    groups: List[List[TextBlock]] = []
    for column in by_left.values():
        if len(column) >= 2:
            groups.append(sorted(column, key=lambda b: b.box[1]))

    # Horizontal row: similar tops, non-overlapping x.
    by_top: Dict[int, List[TextBlock]] = {}
    for block in similar:
        key = int(round(block.box[1] / 16.0))
        by_top.setdefault(key, []).append(block)
    for row in by_top.values():
        if len(row) >= 2:
            groups.append(sorted(row, key=lambda b: b.box[0]))

    if not groups:
        groups = [sorted(similar, key=lambda b: (b.box[1], b.box[0]))]
    # Prefer the largest, most regular group.
    groups.sort(key=lambda g: (len(g), _regularity(g)), reverse=True)
    return groups


def option_like_groups(blocks: Sequence[TextBlock]) -> List[TextBlock]:
    """Blocks that belong to a repeated option rhythm rather than a paragraph.

    Signal: two or more same-height runs separated by a gap larger than half a
    line height, or arranged side by side in one row.  Wrapped question text has
    near-zero gaps, so it is not caught by this.
    """
    if len(blocks) < 2:
        return []
    heights = [b.box[3] for b in blocks]
    median_height = float(np.median(heights)) or 1.0
    similar = [b for b in blocks if abs(b.box[3] - median_height) <= median_height * 0.45]
    if len(similar) < 2:
        return []
    ordered = sorted(similar, key=lambda b: (b.box[1], b.box[0]))
    gaps = [vertical_gap(a.box, b.box) for a, b in zip(ordered, ordered[1:])]
    if gaps and max(gaps) > median_height * 0.5:
        return list(similar)
    rows: Dict[int, List[TextBlock]] = {}
    for block in similar:
        rows.setdefault(int(round(block.box[1] / max(8.0, median_height * 0.5))), []).append(block)
    if any(len(row) >= 2 for row in rows.values()):
        return list(similar)
    return []


def _cluster_1d(values: Sequence[float], tolerance: float) -> List[float]:
    """Cluster sorted scalar values; returns one representative per cluster."""
    clusters: List[List[float]] = []
    for value in sorted(values):
        if clusters and abs(value - clusters[-1][-1]) <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [sum(group) / len(group) for group in clusters]


def complete_grid(selected: Sequence[TextBlock], pool: Sequence[TextBlock]) -> List[TextBlock]:
    """Add same-sized blocks that align with the selected grid's columns/rows.

    A 3x2 tile grid holding four options produces one full row of three plus a
    lone tile underneath.  Rhythm grouping only ever returns *maximal* groups, so
    the incomplete last row would be dropped -- this puts it back.
    """
    if len(selected) < 2:
        return list(selected)
    heights = [block.box[3] for block in selected]
    median_height = float(np.median(heights)) or 1.0
    tolerance = max(8.0, median_height * 0.6)
    columns = _cluster_1d([block.box[0] for block in selected], tolerance)
    rows = _cluster_1d([block.box[1] for block in selected], tolerance)
    if len(columns) < 2 or len(rows) < 1:
        return list(selected)
    chosen = list(selected)
    chosen_ids = {id(block) for block in chosen}
    top_row = min(rows)
    for block in pool:
        if id(block) in chosen_ids:
            continue
        if abs(block.box[3] - median_height) > median_height * 0.45:
            continue
        aligned_column = any(abs(block.box[0] - column) <= tolerance for column in columns)
        on_grid_row = any(abs(block.box[1] - row) <= tolerance for row in rows) or block.box[1] >= top_row - tolerance
        if aligned_column and on_grid_row:
            chosen.append(block)
            chosen_ids.add(id(block))
    chosen.sort(key=lambda b: (b.box[1], b.box[0]))
    return chosen


def maximal_rhythm_union(blocks: Sequence[TextBlock]) -> List[TextBlock]:
    """Union of every rhythm group of the maximal size, grid-completed.

    A vertical list produces one big column group; a card grid produces several
    equal-sized column *and* row groups.  Taking only the first group would drop
    half the options in a grid, so all maximal groups are unioned -- and then
    :func:`complete_grid` restores any short final row.
    """
    groups = group_by_rhythm(blocks)
    if not groups:
        return []
    best_size = max(len(group) for group in groups)
    if best_size < 2:
        return []
    union: List[TextBlock] = []
    seen: set = set()
    for group in groups:
        if len(group) != best_size:
            continue
        for block in group:
            if id(block) not in seen:
                seen.add(id(block))
                union.append(block)
    union.sort(key=lambda b: (b.box[1], b.box[0]))
    return complete_grid(union, list(blocks))


def _regularity(blocks: Sequence[TextBlock]) -> float:
    """0..1: how evenly spaced the group is (the option-rhythm signal)."""
    if len(blocks) < 3:
        return 0.6 if len(blocks) == 2 else 0.0
    ordered = sorted(blocks, key=lambda b: (b.box[1], b.box[0]))
    vertical = [vertical_gap(a.box, b.box) for a, b in zip(ordered, ordered[1:])]
    horizontal = [horizontal_gap(a.box, b.box) for a, b in zip(ordered, ordered[1:])]
    gaps = vertical if max(vertical) >= max(horizontal) else horizontal
    if not gaps or max(gaps) <= 0:
        return 0.4
    std = float(np.std(gaps))
    mean = float(np.mean(gaps)) or 1.0
    return float(max(0.0, min(1.0, 1.0 - std / (mean + 1e-6))))


def _spread(blocks: Sequence[TextBlock]) -> float:
    box = box_union_all([b.box for b in blocks])
    return float(box_area(box)) if box else 0.0


def _merge_paragraph(
    seeds: Sequence[TextBlock], blocks: Sequence[TextBlock], excluded: Sequence[Box]
) -> List[TextBlock]:
    """Greedily absorb continuation lines into the question paragraph."""
    merged: List[TextBlock] = list(seeds)
    if not merged:
        return merged
    merged.sort(key=lambda b: b.box[1])
    anchor = merged[0]
    for block in blocks:
        if block in merged or is_nav_text(block.text):
            continue
        if any(_overlap_ratio(block.box, box) > 0.5 for box in excluded):
            continue
        same_column = abs(box_center(block.box)[0] - box_center(anchor.box)[0]) <= max(anchor.box[2], block.box[2]) * 0.6
        adjacent = 0 <= vertical_gap(anchor.box, block.box) <= max(6, int(anchor.box[3] * 0.55))
        below = block.box[1] >= anchor.box[1]
        similar_height = block.box[3] <= anchor.box[3] * 1.25
        if same_column and adjacent and below and similar_height:
            merged.append(block)
            anchor = block
    return sorted(merged, key=lambda b: b.box[1])


def _enclosing_panel(box: Box, panels: Sequence[Any]) -> Optional[Box]:
    best: Optional[Box] = None
    best_ratio = 0.0
    for panel in panels:
        candidate = panel.box if hasattr(panel, "box") else panel
        ratio = _overlap_ratio(box, candidate)
        if ratio > 0.75 and box_area(candidate) > box_area(box) and ratio > best_ratio:
            best, best_ratio = candidate, ratio
    return best


def _overlap_ratio(inner: Box, outer: Box) -> float:
    if box_area(inner) <= 0:
        return 0.0
    return box_area(box_intersection(inner, outer)) / float(box_area(inner))


def _pad(box: Box, ctx: SegmentationContext, ratio: float) -> Box:
    from ..geometry import box_clip, box_pad_relative

    return box_clip(box_pad_relative(box, ratio * 0.35, ratio), ctx.width, ctx.height)


def _dedupe(proposals: List[RegionProposal]) -> List[RegionProposal]:
    """Keep the best proposal per region kind, plus distinct NOISE entries."""
    best: Dict[RegionKind, RegionProposal] = {}
    noise: List[RegionProposal] = []
    for proposal in proposals:
        if proposal.kind == RegionKind.NOISE:
            noise.append(proposal)
            continue
        existing = best.get(proposal.kind)
        if existing is None or proposal.score > existing.score:
            best[proposal.kind] = proposal
    ordered = [best[kind] for kind in (RegionKind.HEADER, RegionKind.QUESTION, RegionKind.OPTIONS, RegionKind.NAVIGATION) if kind in best]
    return ordered + noise


def summarize(proposals: Sequence[RegionProposal]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for proposal in proposals:
        out.setdefault(proposal.kind.value, []).append(
            {"box": list(proposal.box), "score": round(proposal.score, 3), "evidence": proposal.evidence}
        )
    return out


__all__ = [
    "FontCluster",
    "NAV_KEYWORDS",
    "RegionSegmenter",
    "SegmentationContext",
    "column_gaps",
    "complete_grid",
    "font_size_clusters",
    "group_by_rhythm",
    "is_nav_text",
    "maximal_rhythm_union",
    "option_like_groups",
    "summarize",
    "whitespace_bands",
]
