"""Geometry primitives shared by every module.

Design law **[L6] never assume**: nothing in the codebase may hard-code a
resolution, DPI scale or window position.  All *stored* geometry is therefore
expressed in one of two forms:

``Box``    absolute pixels ``(x, y, w, h)`` -- valid for exactly one frame.
``RelBox`` fractions of the frame ``(fx, fy, fw, fh)`` in ``[0, 1]`` -- the only
           form that may be persisted across frames (FR-7.7.3).

Conversion between the two always goes through :class:`Geometry`, which is built
from the *current* frame's metadata, so resolution / DPI / zoom changes are
absorbed automatically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

Box = Tuple[int, int, int, int]
"""Absolute pixel box ``(x, y, w, h)``."""

RelBox = Tuple[float, float, float, float]
"""Relative box ``(fx, fy, fw, fh)`` as fractions of frame width/height."""

Point = Tuple[int, int]


def as_box(value: Sequence[float]) -> Box:
    """Coerce any 4-sequence into an int ``(x, y, w, h)`` box."""
    if value is None or len(value) != 4:
        raise ValueError(f"box must have 4 components, got {value!r}")
    x, y, w, h = (int(round(float(v))) for v in value)
    return (x, y, max(0, w), max(0, h))


def as_rel_box(value: Sequence[float]) -> RelBox:
    if value is None or len(value) != 4:
        raise ValueError(f"rel box must have 4 components, got {value!r}")
    return tuple(max(0.0, float(v)) for v in value)  # type: ignore[return-value]


def box_x2y2(box: Box) -> Tuple[int, int, int, int]:
    """``(x, y, w, h)`` -> ``(left, top, right, bottom)``."""
    x, y, w, h = box
    return x, y, x + w, y + h


def box_from_x2y2(left: int, top: int, right: int, bottom: int) -> Box:
    return (
        int(min(left, right)),
        int(min(top, bottom)),
        int(abs(right - left)),
        int(abs(bottom - top)),
    )


def box_center(box: Box) -> Point:
    """Centroid of a box -- the click target mandated by FR-7.6.3."""
    x, y, w, h = box
    return (int(round(x + w / 2.0)), int(round(y + h / 2.0)))


def box_area(box: Box) -> int:
    return int(box[2]) * int(box[3])


def box_intersection(a: Box, b: Box) -> Box:
    al, at, ar, ab = box_x2y2(a)
    bl, bt, br, bb = box_x2y2(b)
    left, top = max(al, bl), max(at, bt)
    right, bottom = min(ar, br), min(ab, bb)
    if right <= left or bottom <= top:
        return (0, 0, 0, 0)
    return (left, top, right - left, bottom - top)


def box_union(a: Box, b: Box) -> Box:
    if box_area(a) == 0:
        return b
    if box_area(b) == 0:
        return a
    al, at, ar, ab = box_x2y2(a)
    bl, bt, br, bb = box_x2y2(b)
    return box_from_x2y2(min(al, bl), min(at, bt), max(ar, br), max(ab, bb))


def box_union_all(boxes: Iterable[Box]) -> Box | None:
    result: Box | None = None
    for box in boxes:
        result = box if result is None else box_union(result, box)
    return result


def box_iou(a: Box, b: Box) -> float:
    """Intersection-over-union.  Acceptance criterion AC-7.2.1 is measured with this."""
    inter = box_area(box_intersection(a, b))
    union = box_area(a) + box_area(b) - inter
    if union <= 0:
        return 0.0
    return inter / union


def box_contains_point(box: Box, point: Point) -> bool:
    x, y, w, h = box
    px, py = point
    return x <= px < x + w and y <= py < y + h


def box_contains_box(outer: Box, inner: Box) -> bool:
    ol, ot, orr, ob = box_x2y2(outer)
    il, it, ir, ib = box_x2y2(inner)
    return il >= ol and it >= ot and ir <= orr and ib <= ob


def box_expand(box: Box, px: int) -> Box:
    """Grow a box by ``px`` on every side (never below zero size)."""
    x, y, w, h = box
    return (x - px, y - px, w + 2 * px, h + 2 * px)


def box_pad_relative(box: Box, frac_x: float, frac_y: float) -> Box:
    """Grow a box by a fraction of its own size (used for +/-20% scale search)."""
    x, y, w, h = box
    dx, dy = int(round(w * frac_x)), int(round(h * frac_y))
    return (x - dx, y - dy, w + 2 * dx, h + 2 * dy)


def box_clip(box: Box, width: int, height: int) -> Box:
    """Clip a box to a frame of ``width`` x ``height`` (FR-7.6.4 viewport check)."""
    x, y, w, h = box
    left, top = max(0, x), max(0, y)
    right, bottom = min(width, x + w), min(height, y + h)
    if right <= left or bottom <= top:
        return (0, 0, 0, 0)
    return (left, top, right - left, bottom - top)


def box_is_visible(box: Box, width: int, height: int, min_area: int = 16) -> bool:
    """True when the box is fully-or-partially inside the viewport and clickable."""
    clipped = box_clip(box, width, height)
    if box_area(clipped) < min_area:
        return False
    # A box that had to be clipped lost part of its hit area -> its centroid may
    # now be outside the element, so treat heavy clipping as not-visible.
    if box_area(box) > 0 and box_area(clipped) / box_area(box) < 0.75:
        return False
    return True


def box_to_rel(box: Box, width: int, height: int) -> RelBox:
    """Absolute -> relative.  Only relative boxes may be stored across frames."""
    if width <= 0 or height <= 0:
        raise ValueError("frame dimensions must be positive")
    x, y, w, h = box
    return (x / width, y / height, w / width, h / height)


def rel_to_box(rel: RelBox, width: int, height: int) -> Box:
    """Relative -> absolute, using the *current* geometry (FR-7.7.3)."""
    fx, fy, fw, fh = rel
    return (
        int(round(fx * width)),
        int(round(fy * height)),
        int(round(fw * width)),
        int(round(fh * height)),
    )


def box_distance(a: Box, b: Box) -> float:
    """Euclidean distance between centroids -- used for displacement checks."""
    ax, ay = box_center(a)
    bx, by = box_center(b)
    return math.hypot(ax - bx, ay - by)


def box_center_distance_pt(a: Box, point: Point) -> float:
    ax, ay = box_center(a)
    return math.hypot(ax - point[0], ay - point[1])


@dataclass(frozen=True)
class Geometry:
    """Frame geometry + DPI scale.  Built fresh from every captured frame."""

    width: int
    height: int
    dpi_scale: float = 1.0

    @property
    def size_px(self) -> Box:
        return (0, 0, self.width, self.height)

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_rel(self, box: Box) -> RelBox:
        return box_to_rel(box, self.width, self.height)

    def to_abs(self, rel: RelBox) -> Box:
        return rel_to_box(rel, self.width, self.height)

    def is_visible(self, box: Box) -> bool:
        return box_is_visible(box, self.width, self.height)

    def clip(self, box: Box) -> Box:
        return box_clip(box, self.width, self.height)

    def rescale(self, other: "Geometry") -> "Geometry":
        """Return a Geometry mapping boxes from ``other`` into ``self``."""
        return _RescaledGeometry(source=other, target=self)  # type: ignore[return-value]

    def scale_factor_to(self, other: "Geometry") -> Tuple[float, float]:
        if other.width <= 0 or other.height <= 0:
            return (1.0, 1.0)
        return (self.width / other.width, self.height / other.height)


class _RescaledGeometry(Geometry):  # pragma: no cover - thin helper
    def __init__(self, source: Geometry, target: Geometry) -> None:
        object.__setattr__(self, "width", target.width)
        object.__setattr__(self, "height", target.height)
        object.__setattr__(self, "dpi_scale", target.dpi_scale)
        object.__setattr__(self, "_source", source)

    def map_box(self, box: Box) -> Box:
        source: Geometry = getattr(self, "_source")
        return rel_to_box(box_to_rel(box, source.width, source.height), self.width, self.height)


def vertical_gap(a: Box, b: Box) -> int:
    """Vertical whitespace between two boxes (0 when they overlap vertically)."""
    _, ay, _, ah = a
    _, by, _, bh = b
    if by >= ay + ah:
        return by - (ay + ah)
    if ay >= by + bh:
        return ay - (by + bh)
    return 0


def horizontal_gap(a: Box, b: Box) -> int:
    ax, _, aw, _ = a
    bx, _, bw, _ = b
    if bx >= ax + aw:
        return bx - (ax + aw)
    if ax >= bx + bw:
        return ax - (bx + bw)
    return 0


def sort_boxes_reading_order(boxes: Iterable[Box], row_tolerance: int = 8) -> list[Box]:
    """Sort boxes top-to-bottom, left-to-right, grouping near-equal tops into rows."""
    ordered = sorted(boxes, key=lambda b: (b[1], b[0]))
    rows: list[list[Box]] = []
    for box in ordered:
        if rows and abs(rows[-1][0][1] - box[1]) <= row_tolerance:
            rows[-1].append(box)
        else:
            rows.append([box])
    return [b for row in rows for b in sorted(row, key=lambda b: b[0])]
