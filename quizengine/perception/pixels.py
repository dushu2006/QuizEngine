"""Pixel-level evidence used by Tier-1 perception.

Everything here is *classical* computer vision on the current frame: no learned
models, no cached geometry, no assumptions about theme or resolution (L1, L6).
OpenCV is preferred; NumPy fallbacks keep the heuristics alive on minimal
installs (degraded accuracy, never a crash).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..contracts import SelectedMarker
from ..geometry import Box, box_area, box_center, box_intersection, box_iou, box_union

try:  # pragma: no cover - import guard
    import cv2 as _cv2
except ImportError:  # pragma: no cover
    _cv2 = None

Color = Tuple[int, int, int]


# --------------------------------------------------------------------------- #
# background / colour
# --------------------------------------------------------------------------- #
def dominant_color(image: np.ndarray, bins: int = 16) -> Color:
    """Most common quantized colour -- the page background in practice."""
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.size == 0:
        return (255, 255, 255)
    flat = array.reshape(-1, 3).astype(np.int32)
    if flat.shape[0] > 120_000:
        flat = flat[:: max(1, flat.shape[0] // 120_000)]
    step = max(1, 256 // bins)
    quantized = (flat // step) * step + step // 2
    keys = (quantized[:, 0] << 16) | (quantized[:, 1] << 8) | quantized[:, 2]
    values, counts = np.unique(keys, return_counts=True)
    winner = int(values[int(np.argmax(counts))])
    return ((winner >> 16) & 0xFF, (winner >> 8) & 0xFF, winner & 0xFF)


def color_distance_mask(image: np.ndarray, color: Color, tolerance: int = 26) -> np.ndarray:
    """Boolean mask of pixels that are NOT the given colour (i.e. 'ink/foreign')."""
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    target = np.array(color, dtype=np.int16)
    delta = np.abs(array.astype(np.int16) - target).max(axis=-1)
    return delta > int(tolerance)


def accent_color(image: np.ndarray, boxes: Sequence[Box], background: Color) -> Optional[Color]:
    """Most common saturated non-background colour inside ``boxes`` (selection accent)."""
    array = np.asarray(image)
    if array.ndim == 2 or array.size == 0:
        return None
    pixels: List[np.ndarray] = []
    height, width = array.shape[:2]
    for box in boxes:
        x, y, w, h = box
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(width, x + w), min(height, y + h)
        if x1 <= x0 or y1 <= y0:
            continue
        region = array[y0:y1, x0:x1].reshape(-1, 3).astype(np.int16)
        delta = np.abs(region - np.array(background, dtype=np.int16)).max(axis=-1)
        saturated = region[delta > 40]
        if saturated.size:
            pixels.append(saturated)
    if not pixels:
        return None
    stacked = np.concatenate(pixels, axis=0)
    if stacked.shape[0] > 20_000:
        stacked = stacked[:: max(1, stacked.shape[0] // 20_000)]
    quantized = (stacked // 16) * 16 + 8
    keys = (quantized[:, 0].astype(np.int32) << 16) | (quantized[:, 1].astype(np.int32) << 8) | quantized[:, 2].astype(np.int32)
    values, counts = np.unique(keys, return_counts=True)
    winner = int(values[int(np.argmax(counts))])
    return ((winner >> 16) & 0xFF, (winner >> 8) & 0xFF, winner & 0xFF)


def region_mean_color(image: np.ndarray, box: Box) -> Color:
    region = crop_region(image, box)
    if region.size == 0:
        return (0, 0, 0)
    flat = region.reshape(-1, region.shape[-1]) if region.ndim == 3 else region.reshape(-1, 1)
    mean = flat.mean(axis=0)
    if mean.shape[0] == 1:
        return (int(mean[0]), int(mean[0]), int(mean[0]))
    return (int(mean[0]), int(mean[1]), int(mean[2]))


def crop_region(image: np.ndarray, box: Box) -> np.ndarray:
    array = np.asarray(image)
    height, width = array.shape[:2]
    x, y, w, h = box
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + w), min(height, y + h)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((0, 0) + array.shape[2:], dtype=array.dtype)
    return array[y0:y1, x0:x1]


# --------------------------------------------------------------------------- #
# components / structure
# --------------------------------------------------------------------------- #
def connected_boxes(mask: np.ndarray, min_area: int = 24) -> List[Box]:
    """Bounding boxes of connected components in a boolean mask."""
    if mask.size == 0 or not mask.any():
        return []
    if _cv2 is not None:
        binary = (mask.astype(np.uint8)) * 255
        count, _labels, stats, _rects = _cv2.connectedComponentsWithStats(binary, connectivity=8)
        boxes: List[Box] = []
        for index in range(1, count):
            area = int(stats[index, _cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            x = int(stats[index, _cv2.CC_STAT_LEFT])
            y = int(stats[index, _cv2.CC_STAT_TOP])
            w = int(stats[index, _cv2.CC_STAT_WIDTH])
            h = int(stats[index, _cv2.CC_STAT_HEIGHT])
            boxes.append((x, y, w, h))
        return boxes
    return _projection_boxes(mask, min_area)


def _projection_boxes(mask: np.ndarray, min_area: int) -> List[Box]:
    """NumPy fallback: row-band then column-band projection clustering."""
    rows = mask.any(axis=1)
    bands: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for index, active in enumerate(rows):
        if active and start is None:
            start = index
        elif not active and start is not None:
            bands.append((start, index))
            start = None
    if start is not None:
        bands.append((start, len(rows)))
    boxes: List[Box] = []
    for top, bottom in bands:
        strip = mask[top:bottom]
        cols = strip.any(axis=0)
        col_start: Optional[int] = None
        for index, active in enumerate(cols):
            if active and col_start is None:
                col_start = index
            elif not active and col_start is not None:
                width = index - col_start
                if (bottom - top) * width >= min_area:
                    boxes.append((col_start, top, width, bottom - top))
                col_start = None
        if col_start is not None:
            width = len(cols) - col_start
            if (bottom - top) * width >= min_area:
                boxes.append((col_start, top, width, bottom - top))
    return boxes


def ink_text_runs(binary: np.ndarray, min_height: int = 6, max_height_ratio: float = 0.35) -> List[Box]:
    """Text-line candidates from a binarized image (ink = dark pixels).

    Used to recover structure when the OCR engine returns text without boxes,
    and as an independent cross-check for option localization (FR-7.2.4).
    """
    array = np.asarray(binary)
    if array.ndim == 3:
        array = array[:, :, 0]
    ink = array < 128
    if not ink.any():
        return []
    max_height = max(min_height + 1, int(array.shape[0] * max_height_ratio))
    boxes = connected_boxes(ink, min_area=max(8, min_height * 2))
    # Merge components that sit on the same text line.
    lines: List[Box] = []
    for box in sorted(boxes, key=lambda b: (b[1], b[0])):
        if box[3] < min_height or box[3] > max_height:
            continue
        merged = False
        for index, line in enumerate(lines):
            if _same_line(line, box):
                lines[index] = box_union(line, box)
                merged = True
                break
        if not merged:
            lines.append(box)
    return sorted(lines, key=lambda b: (b[1], b[0]))


def _same_line(a: Box, b: Box, tolerance: float = 0.6) -> bool:
    _, ay, _, ah = a
    _, by, _, bh = b
    center_a, center_b = ay + ah / 2.0, by + bh / 2.0
    return abs(center_a - center_b) <= tolerance * max(ah, bh)


def rectangular_outlines(image: np.ndarray, background: Color, min_area: int = 900) -> List[Box]:
    """Card / button outlines: rectangles whose border differs from the page.

    Two shapes qualify.  The hollow one is a bordered card on the page colour
    (only its border and label differ from the background).  The solid one is a
    *selected* card or tile: selection state paints the whole element with a
    subtle tint and an accent border, so the interior differs from the page too.
    Without the second case a selected card stops being an option container the
    moment it is selected -- its hit box collapses to the label and the marker
    evidence (the accent border) is measured on the wrong region.
    """
    mask = color_distance_mask(image, background, tolerance=22)
    # A 1px border on a 250x120 button is only ~740 inked pixels, so the area
    # floor alone would drop every small bordered control.  Collect cheaply and
    # let the shape tests below decide.
    boxes = connected_boxes(mask, min_area=min(120, min_area))
    result: List[Box] = []
    height, width = mask.shape[:2]
    for box in boxes:
        x, y, w, h = box
        if w < 24 or h < 14 or w > width or h > height:
            continue
        region = mask[y : y + h, x : x + w]
        fill_ratio = float(region.mean())
        ink = float(region.sum())
        # Cards/buttons are mostly hollow (border + a little text): 0.02 .. 0.55.
        if 0.02 <= fill_ratio <= 0.55:
            perimeter = 2.0 * (w + h)
            if ink >= min_area or (
                ink >= 0.5 * perimeter and _border_ring_score(region) >= 0.8
            ):
                result.append(box)
        elif fill_ratio > 0.55 and _edge_accent_ratio(image, box, background) > 0.05:
            result.append(box)  # a tinted, accent-bordered card/tile
    return result


def _border_ring_score(region_mask: np.ndarray, band: int = 2) -> float:
    """Fraction of a component's ink lying on its own bounding-box edge.

    A border ring is ~1.0 (all of its ink traces the box edge); a word of text is
    scattered inside its tight bounding box, which is what keeps labels out of the
    outline list.
    """
    if region_mask.size == 0:
        return 0.0
    ink = float(region_mask.sum())
    if ink <= 0:
        return 0.0
    width = max(1, min(int(band), min(region_mask.shape) // 2))
    edge = np.zeros(region_mask.shape, dtype=bool)
    edge[:width, :] = True
    edge[-width:, :] = True
    edge[:, :width] = True
    edge[:, -width:] = True
    return float(region_mask[edge].sum() / ink)


def foreign_panels(image: np.ndarray, background: Color, min_area: int = 1200) -> List["Panel"]:
    """Filled panels whose colour is far from the page background.

    This is how Tier 1 proposes overlays/toasts (FR-7.2.5) without any DOM or
    site-specific knowledge: a dark rounded rectangle on a light page is an
    anomaly, whatever site it came from.
    """
    mask = color_distance_mask(image, background, tolerance=48)
    boxes = connected_boxes(mask, min_area=min_area)
    panels: List[Panel] = []
    for box in boxes:
        x, y, w, h = box
        region_mask = mask[y : y + h, x : x + w]
        fill = float(region_mask.mean()) if region_mask.size else 0.0
        if fill < 0.62:  # not a solid panel -> probably text on the page background
            continue
        region = crop_region(image, box)
        panels.append(
            Panel(
                box=box,
                fill_ratio=fill,
                mean_color=region_mean_color(image, box),
                luminance=float(_luminance(region)),
            )
        )
    return sorted(panels, key=lambda p: box_area(p.box), reverse=True)


@dataclass(frozen=True)
class Panel:
    box: Box
    fill_ratio: float
    mean_color: Color
    luminance: float


def _luminance(region: np.ndarray) -> float:
    """Mean luminance of a region (0..255)."""
    if region.size == 0:
        return 0.0
    array = region.astype(np.float32)
    if array.ndim == 3:
        luminance = array[:, :, :3] @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        return float(luminance.mean())
    return float(array.mean())


# --------------------------------------------------------------------------- #
# selection-state markers (FR-7.2.5 selected-state markers, FR-7.5.3)
# --------------------------------------------------------------------------- #
@dataclass
class MarkerFeatures:
    index: int
    marker_zone: Box
    ink_ratio: float = 0.0
    filled_ratio: float = 0.0
    circularity: float = 0.0
    extent: float = 0.0
    accent_pixels: float = 0.0
    edge_accent_ratio: float = 0.0
    panel_deviation: float = 0.0
    text_color: Color = (0, 0, 0)
    panel_color: Color = (255, 255, 255)
    extra: Dict[str, Any] = field(default_factory=dict)


def marker_zone(
    box: Box, style_hint: str = "radio", text_box: Optional[Box] = None, left_limit: int = 0
) -> Box:
    """The small region where a radio/checkbox glyph would be drawn.

    When the option's text box is known the zone stops *before* it: otherwise the
    first glyph of the label lands in the zone and the circularity test classifies
    a radio dot as a checkbox tick.
    """
    _x, _y, _w, _h = box
    x, y, w, h = box
    if style_hint in {"card", "tile", "text_only", "button"}:
        # No dedicated glyph: the whole tile is the evidence.
        return box
    if text_box is not None:
        # Anchor the zone to the *label* and reach left into the whitespace.
        # An undecorated row (a bare radio list) gives no evidence wider than its
        # text, so the perceived option box can start at the label itself; the
        # glyph then sits just outside that box, and confining the zone to the box
        # would miss it -- reading a selected option as unselected.
        text_height = max(8, int(text_box[3]))
        size = max(14, min(int(text_height * 3.0), 56))
        # Never reach past the neighbour on the left: a selected sibling paints
        # accent ink there, and picking it up would credit the wrong option.
        zone_x = max(0, int(left_limit) + 2, int(text_box[0]) - size - 2)
        size = max(8, min(size, int(text_box[0]) - zone_x - 1))
        centre_y = int(text_box[1] + text_box[3] / 2.0)
        zone_y = max(0, centre_y - size // 2)
        return (zone_x, zone_y, size, size)
    size = max(8, min(int(h * 0.6), 40))
    inset_x = max(4, int(h * 0.25))
    return (x + inset_x, y + max(1, (h - size) // 2), size, size)


def analyze_marker(
    image: np.ndarray,
    option_box: Box,
    background: Color,
    style_hint: str = "radio",
    text_box: Optional[Box] = None,
    left_limit: int = 0,
) -> MarkerFeatures:
    zone = marker_zone(option_box, style_hint, text_box, left_limit=left_limit)
    zone_pixels = crop_region(image, zone)
    features = MarkerFeatures(index=-1, marker_zone=zone, panel_color=region_mean_color(image, option_box))
    if zone_pixels.size == 0:
        return features

    array = zone_pixels.astype(np.int16)
    if array.ndim == 3:
        delta = np.abs(array - np.array(background, dtype=np.int16)).max(axis=-1)
    else:
        delta = np.abs(array[:, :, 0] - int(sum(background) / 3))
    ink = delta > 40
    features.ink_ratio = float(ink.mean()) if ink.size else 0.0

    strong = delta > 90
    features.filled_ratio = float(strong.mean()) if strong.size else 0.0

    accent_mask = _accent_mask(zone_pixels, background)
    features.accent_pixels = float(accent_mask.mean()) if accent_mask.size else 0.0
    features.edge_accent_ratio = _edge_accent_ratio(image, option_box, background)

    page_lum = _luminance(np.array([[background]], dtype=np.uint8))
    features.panel_deviation = abs(_luminance(crop_region(image, option_box)) - page_lum)
    features.circularity, features.extent = _shape_metrics(strong if strong.any() else ink)
    features.text_color = region_mean_color(image, option_box)
    return features


def _edge_accent_ratio(image: np.ndarray, box: Box, background: Color, band: int = 4) -> float:
    """Fraction of the option's *perimeter band* painted in a saturated accent.

    Cards, tiles and text-only buttons show selection as a border or a tint that
    touches the element edge; radio dots and checkbox ticks do not.  This is the
    evidence that separates HIGHLIGHT from DOT/CHECK.
    """
    region = crop_region(image, box)
    if region.size == 0 or region.ndim == 2:
        return 0.0
    height, width = region.shape[:2]
    band = max(1, min(int(band), height // 3, width // 3))
    mask = np.zeros((height, width), dtype=bool)
    mask[:band, :] = True
    mask[-band:, :] = True
    mask[:, :band] = True
    mask[:, -band:] = True
    accent = _accent_mask(region, background)
    edge_pixels = int(mask.sum())
    if edge_pixels == 0:
        return 0.0
    return float(accent[mask].mean())


def _outlier_option(features: Sequence[MarkerFeatures]) -> Tuple[int, bool]:
    """Scale-free outlier test for glyph-less option styles.

    A selected tile/card/text-button differs from its siblings; nothing else does.
    Comparing each option against the *median* option makes the test independent
    of theme, accent colour, absolute contrast and tile size (L6).
    """
    if len(features) < 2:
        return (0, False)
    vectors = np.array(
        [
            [
                f.panel_deviation / 40.0,
                f.accent_pixels * 6.0,
                f.filled_ratio,
                f.ink_ratio * 0.5,
            ]
            for f in features
        ],
        dtype=np.float64,
    )
    median = np.median(vectors, axis=0)
    distances = np.abs(vectors - median).sum(axis=1)
    winner = int(np.argmax(distances))
    others = np.delete(distances, winner)
    baseline = float(np.median(others)) if others.size else 0.0
    # The winner must stand apart from its siblings, not merely be the largest of
    # a set of equal values.
    return winner, bool(distances[winner] > max(0.06, 3.0 * baseline))


def _accent_mask(region: np.ndarray, background: Color) -> np.ndarray:
    if region.size == 0 or region.ndim == 2:
        return np.zeros(region.shape[:2], dtype=bool) if region.ndim >= 2 else region.astype(bool)
    array = region.astype(np.int16)
    max_delta = array.max(axis=-1)
    min_delta = array.min(axis=-1)
    saturation = max_delta - min_delta
    distance = np.abs(array - np.array(background, dtype=np.int16)).max(axis=-1)
    return (saturation > 45) & (distance > 45)


def _shape_metrics(mask: np.ndarray) -> Tuple[float, float]:
    """``(circularity, extent)`` of the dominant ink blob.

    ``circularity = 4*pi*area / perimeter^2`` (1.0 for a disc, ~0.79 for a square)
    ``extent      = contour area / bounding-box area`` (~0.79 for a disc, ~1.0 for
    a square).  The pair separates a radio dot from a checkbox outline, which
    circularity alone cannot do reliably.
    """
    if mask.size == 0 or not mask.any():
        return (0.0, 0.0)
    if _cv2 is not None:
        binary = mask.astype(np.uint8) * 255
        contours, _ = _cv2.findContours(binary, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return (0.0, 0.0)
        largest = max(contours, key=_cv2.contourArea)
        area = float(_cv2.contourArea(largest))
        perimeter = float(_cv2.arcLength(largest, True))
        _x, _y, box_w, box_h = _cv2.boundingRect(largest)
        extent = area / float(box_w * box_h) if box_w * box_h > 0 else 0.0
        circularity = min(1.0, 4.0 * np.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0.0
        return (float(circularity), float(extent))
    ys, xs = np.nonzero(mask)
    height = int(ys.max() - ys.min() + 1)
    width = int(xs.max() - xs.min() + 1)
    if height <= 0 or width <= 0:
        return (0.0, 0.0)
    fill = float(mask.sum()) / float(height * width)
    aspect = min(height, width) / float(max(height, width))
    return (float(min(1.0, fill * aspect * 1.27)), float(fill))


def infer_selected_markers(
    features: Sequence[MarkerFeatures], *, style_hint: str = "radio"
) -> List[SelectedMarker]:
    """Decide which option (if any) shows a selected-state marker.

    Purely comparative and scale-free (L6): the selected option is the *outlier*
    among its siblings, measured on a feature vector that works for radio dots,
    checkbox ticks, filled cards, tinted tiles and text-only buttons alike.

    The marker *kind* then follows from where the evidence sits: accent ink on
    the element perimeter means a highlight/border, a compact glyph inside the
    marker zone means a dot (disc-shaped) or a check (anything else).
    """
    markers: List[SelectedMarker] = [SelectedMarker.NONE] * len(features)
    if len(features) < 2:
        return markers

    # The outlier test is for *glyph-less* styles (cards, tiles, text buttons),
    # where selection shows up only as a difference from the siblings.  A radio or
    # checkbox row states it directly with a glyph, and its perceived boxes differ
    # in width with the label length -- enough for a panel-luminance outlier to
    # fire on a screen where nothing is selected at all.
    glyph_less = str(style_hint) in {"card", "tile", "text_only", "button"}
    outlier_index, is_outlier = _outlier_option(features) if glyph_less else (0, False)
    glyph_scores = [max(f.filled_ratio, f.ink_ratio * 0.6) + 1.5 * f.accent_pixels for f in features]
    glyph_index = int(np.argmax(glyph_scores))
    others = sorted(glyph_scores)[: max(1, len(glyph_scores) - 1)]
    glyph_baseline = float(np.median(others))
    glyph_stands_out = glyph_scores[glyph_index] > glyph_baseline + 0.045

    if is_outlier and glyph_stands_out and outlier_index != glyph_index:
        # Two different options win on two different signals -> not trustworthy.
        return markers
    winner = outlier_index if is_outlier else (glyph_index if glyph_stands_out else None)
    if winner is None:
        return markers

    winner_features = features[winner]
    if winner_features.edge_accent_ratio > 0.04:
        markers[winner] = SelectedMarker.HIGHLIGHT
        return markers
    # A disc reads as high circularity *and* an extent near pi/4 (0.785); a square
    # checkbox outline has extent ~0.9-1.0 at similar circularity.
    is_disc = winner_features.circularity >= 0.60 and winner_features.extent <= 0.82
    markers[winner] = SelectedMarker.DOT if is_disc else SelectedMarker.CHECK
    return markers


# --------------------------------------------------------------------------- #
# frame differencing (used by section 7.9 verification and FR-7.8.2 scroll checks)
# --------------------------------------------------------------------------- #
def diff_ratio(
    before: np.ndarray,
    after: np.ndarray,
    box: Optional[Box] = None,
    *,
    threshold: int = 12,
) -> float:
    """Percentage of changed pixels inside ``box`` (whole frame when omitted).

    FR-7.9.2: verification is element-aware, so callers pass the target
    element's region -- a change elsewhere must not satisfy verification.
    """
    a = crop_region(before, box) if box is not None else np.asarray(before)
    b = crop_region(after, box) if box is not None else np.asarray(after)
    if a.size == 0 or b.size == 0:
        return 0.0
    if a.shape != b.shape:
        height = min(a.shape[0], b.shape[0])
        width = min(a.shape[1], b.shape[1])
        a = a[:height, :width]
        b = b[:height, :width]
    if a.ndim == 3:
        delta = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=-1)
    else:
        delta = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return float((delta > int(threshold)).mean() * 100.0)


def mean_shift_vector(before: np.ndarray, after: np.ndarray, *, downsample: int = 4) -> Tuple[float, float, float]:
    """Estimate a global (dx, dy) content shift + confidence, for scroll checks.

    Coarse block matching: good enough to tell "content moved" from "nothing
    moved" (FR-7.8.2's 1% viewport rule) without a full optical-flow dependency.
    """
    a = np.asarray(before)
    b = np.asarray(after)
    if a.ndim == 3:
        a = a.mean(axis=-1)
    if b.ndim == 3:
        b = b.mean(axis=-1)
    step = max(1, int(downsample))
    a_small = a[::step, ::step].astype(np.float32)
    b_small = b[::step, ::step].astype(np.float32)
    if a_small.shape != b_small.shape:
        height = min(a_small.shape[0], b_small.shape[0])
        width = min(a_small.shape[1], b_small.shape[1])
        a_small = a_small[:height, :width]
        b_small = b_small[:height, :width]
    if a_small.size == 0:
        return (0.0, 0.0, 0.0)
    best = (0.0, 0.0, float("inf"))
    search = max(2, int(64 / step))
    for dy in range(-search, search + 1, max(1, search // 4)):
        for dx in range(-search, search + 1, max(1, search // 4)):
            shifted = np.roll(np.roll(a_small, dy, axis=0), dx, axis=1)
            error = float(np.abs(shifted - b_small).mean())
            if error < best[2]:
                best = (float(dx * step), float(dy * step), error)
    baseline = float(np.abs(a_small - b_small).mean())
    confidence = 0.0 if baseline <= 0 else max(0.0, min(1.0, (baseline - best[2]) / baseline))
    return (best[0], best[1], confidence)


__all__ = [
    "Panel",
    "MarkerFeatures",
    "accent_color",
    "analyze_marker",
    "color_distance_mask",
    "connected_boxes",
    "diff_ratio",
    "dominant_color",
    "foreign_panels",
    "infer_selected_markers",
    "ink_text_runs",
    "marker_zone",
    "mean_shift_vector",
    "rectangular_outlines",
    "region_mean_color",
]
