"""Screen scene descriptions: the fixture format behind deterministic testing.

A :class:`Scene` is a JSON-serializable description of one quiz screen.  It is
rendered to *real pixels* (Pillow) so that the perception pipeline runs against
actual images, and it carries a **hand-labeled ground-truth annotation** so that
localization IoU (AC-7.2.1) and transcription accuracy (AC-7.2.2) can be scored.

This is a test/replay artifact, not a production input path: a real run gets its
frames from the ``mss`` backend and has no annotation, which is exactly why the
``annotation`` OCR engine is refused for real backends (see
:mod:`quizengine.perception.ocr`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import Field, field_validator

from .contracts import LayoutType, OverlayKind, QuestionType, SchemaModel, SelectedMarker
from .geometry import Box, as_box

Color = Tuple[int, int, int]

Role = Literal[
    "header",
    "progress",
    "question",
    "option",
    "nav",
    "toast",
    "overlay",
    "image",
    "math",
    "table",
    "result",
    "noise",
    "lock",
]

OptionStyle = Literal["radio", "checkbox", "card", "tile", "button", "text_only"]


class SceneText(SchemaModel):
    """A rendered text run with its pixel box."""

    text: str
    box: Box
    font_px: int = Field(16, ge=6, le=96)
    color: Color = (17, 17, 17)
    role: Role = "noise"
    bold: bool = False
    align: Literal["left", "center", "right"] = "left"
    #: Ground-truth OCR confidence the fixture engine should report for this run.
    confidence: float = Field(0.97, ge=0.0, le=1.0)
    contrast: float = Field(1.0, ge=0.0, le=1.0, description="FR-13.3 OCR-hostile tier: <1 washes the glyph out")

    @field_validator("box", mode="before")
    @classmethod
    def _box(cls, value: Any) -> Any:
        return as_box(value) if isinstance(value, (list, tuple)) else value

    @field_validator("color", mode="before")
    @classmethod
    def _color(cls, value: Any) -> Any:
        if isinstance(value, str):
            return _hex_to_rgb(value)
        if isinstance(value, (list, tuple)) and len(value) == 3:
            return tuple(int(v) for v in value)
        return value


class SceneOption(SchemaModel):
    """Ground truth for one answer option (FR-7.2.4 hit area vs text box)."""

    index: int = Field(..., ge=0)
    text: str
    hit_box: Box
    text_box: Optional[Box] = None
    style: OptionStyle = "radio"
    selected_marker: SelectedMarker = SelectedMarker.NONE
    font_px: int = Field(16, ge=6, le=96)
    color: Color = (17, 17, 17)
    confidence: float = Field(0.97, ge=0.0, le=1.0)
    contrast: float = Field(1.0, ge=0.0, le=1.0)
    marker_box: Optional[Box] = None

    @field_validator("hit_box", "text_box", "marker_box", mode="before")
    @classmethod
    def _box(cls, value: Any) -> Any:
        if value is None:
            return None
        return as_box(value) if isinstance(value, (list, tuple)) else value

    @field_validator("color", mode="before")
    @classmethod
    def _color(cls, value: Any) -> Any:
        if isinstance(value, str):
            return _hex_to_rgb(value)
        return value


class SceneButton(SchemaModel):
    text: str
    box: Box
    enabled: bool = True
    font_px: int = Field(15, ge=6, le=96)

    @field_validator("box", mode="before")
    @classmethod
    def _box(cls, value: Any) -> Any:
        return as_box(value) if isinstance(value, (list, tuple)) else value


class SceneNavigation(SchemaModel):
    next_btn: Optional[SceneButton] = None
    prev_btn: Optional[SceneButton] = None
    submit_btn: Optional[SceneButton] = None
    progress_text: Optional[str] = None
    progress_box: Optional[Box] = None
    progress_current: Optional[int] = None
    progress_total: Optional[int] = None
    auto_advance: bool = False

    @field_validator("progress_box", mode="before")
    @classmethod
    def _box(cls, value: Any) -> Any:
        if value is None:
            return None
        return as_box(value) if isinstance(value, (list, tuple)) else value


class SceneOverlay(SchemaModel):
    text: str = ""
    box: Box
    kind: OverlayKind = OverlayKind.UNKNOWN
    dismissible: Optional[bool] = None
    close_btn: Optional[SceneButton] = None
    font_px: int = Field(14, ge=6, le=96)
    background: Color = (40, 40, 40)
    color: Color = (245, 245, 245)

    @field_validator("box", mode="before")
    @classmethod
    def _box(cls, value: Any) -> Any:
        return as_box(value) if isinstance(value, (list, tuple)) else value

    @field_validator("background", "color", mode="before")
    @classmethod
    def _color(cls, value: Any) -> Any:
        if isinstance(value, str):
            return _hex_to_rgb(value)
        return value


class Scene(SchemaModel):
    """One screen.  ``screen_role`` lets fixtures inject non-quiz states."""

    name: str
    size_px: Tuple[int, int] = (1280, 800)
    theme: Literal["light", "dark"] = "light"
    zoom: float = Field(1.0, gt=0.5, le=2.5)
    dpi_scale: float = Field(1.0, gt=0)
    layout_type: LayoutType = LayoutType.VERTICAL_OPTIONS
    question_type: QuestionType = QuestionType.SINGLE_CHOICE
    screen_role: Literal["quiz", "blank", "lock_screen", "desktop", "end_state", "loading"] = "quiz"
    background: Color = (250, 250, 252)
    question_text: str = ""
    question_region: Optional[Box] = None
    question_font_px: int = Field(22, ge=6, le=96)
    question_confidence: float = Field(0.97, ge=0.0, le=1.0)
    question_contrast: float = Field(1.0, ge=0.0, le=1.0)
    texts: List[SceneText] = Field(default_factory=list)
    options: List[SceneOption] = Field(default_factory=list)
    navigation: SceneNavigation = Field(default_factory=SceneNavigation)
    overlays: List[SceneOverlay] = Field(default_factory=list)
    #: Rectangles with no text: images, charts, tables, card borders.
    rects: List[Dict[str, Any]] = Field(default_factory=list)
    flags: Dict[str, bool] = Field(default_factory=dict)
    #: Chaos hooks consumed by the replay backend / QuizForge parity tests.
    delay_ms: float = Field(0.0, ge=0.0)
    shift_after_load: bool = False
    expected_click_handle: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("size_px", mode="before")
    @classmethod
    def _size(cls, value: Any) -> Any:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return (int(value[0]), int(value[1]))
        return value

    @field_validator("background", mode="before")
    @classmethod
    def _color(cls, value: Any) -> Any:
        if isinstance(value, str):
            return _hex_to_rgb(value)
        return value

    @field_validator("question_region", mode="before")
    @classmethod
    def _box(cls, value: Any) -> Any:
        if value is None:
            return None
        return as_box(value) if isinstance(value, (list, tuple)) else value

    # -- ground truth ------------------------------------------------------ #
    @property
    def width(self) -> int:
        return int(self.size_px[0])

    @property
    def height(self) -> int:
        return int(self.size_px[1])

    def annotation(self) -> Dict[str, Any]:
        """Hand-labeled ground truth in :class:`PerceptionResult` wire shape."""
        return {
            "scene": self.name,
            "screen_role": self.screen_role,
            "layout_type": self.layout_type.value,
            "question_type": self.question_type.value,
            "question_region": list(self.question_region) if self.question_region else None,
            "question_text": self.question_text,
            "question_confidence": self.question_confidence,
            "options": [
                {
                    "index": option.index,
                    "handle": f"opt_{option.index}",
                    "text": option.text,
                    "hit_box": list(option.hit_box),
                    "text_box": list(option.text_box) if option.text_box else None,
                    "text_conf": option.confidence,
                    "selected_marker": option.selected_marker.value,
                    # ADDITIVE ground truth: lets verification expect the marker a
                    # given style actually paints (radio -> dot, card -> highlight).
                    "style": option.style,
                }
                for option in self.options
            ],
            "navigation": {
                "next_btn": _button_annotation(self.navigation.next_btn, "nav_next"),
                "prev_btn": _button_annotation(self.navigation.prev_btn, "nav_prev"),
                "submit_btn": _button_annotation(self.navigation.submit_btn, "nav_submit"),
                "progress_text": self.navigation.progress_text,
                "progress_current": self.navigation.progress_current,
                "progress_total": self.navigation.progress_total,
                "auto_advance": self.navigation.auto_advance,
            },
            "overlays": [
                {
                    "handle": f"overlay_{i}",
                    "text": overlay.text,
                    "box": list(overlay.box),
                    "kind": overlay.kind.value,
                    "dismissible": overlay.dismissible,
                    "close_btn": _button_annotation(overlay.close_btn, f"overlay_{i}_close"),
                }
                for i, overlay in enumerate(self.overlays)
            ],
            "text_blocks": self._text_block_annotations(),
            "flags": dict(self.flags),
            "meta": dict(self.meta),
        }

    def _text_block_annotations(self) -> List[Dict[str, Any]]:
        blocks: List[Dict[str, Any]] = []
        for text in self.texts:
            blocks.append(
                {
                    "text": text.text,
                    "box": list(text.box),
                    "confidence": text.confidence,
                    "role": text.role,
                }
            )
        if self.question_text:
            region = self.question_region or _default_question_region(self)
            blocks.append(
                {
                    "text": self.question_text,
                    "box": list(_question_text_box(self, region)),
                    "confidence": self.question_confidence,
                    "role": "question",
                }
            )
        for option in self.options:
            box = option.text_box or _option_text_box(option)
            blocks.append(
                {
                    "text": option.text,
                    "box": list(box),
                    "confidence": option.confidence,
                    "role": "option",
                    "option_index": option.index,
                }
            )
        for handle, button in (
            ("nav_next", self.navigation.next_btn),
            ("nav_prev", self.navigation.prev_btn),
            ("nav_submit", self.navigation.submit_btn),
        ):
            if button is not None:
                blocks.append(
                    {"text": button.text, "box": list(button.box), "confidence": 0.96, "role": "nav", "handle": handle}
                )
        if self.navigation.progress_text and self.navigation.progress_box:
            blocks.append(
                {
                    "text": self.navigation.progress_text,
                    "box": list(self.navigation.progress_box),
                    "confidence": 0.95,
                    "role": "progress",
                }
            )
        for index, overlay in enumerate(self.overlays):
            if overlay.text:
                blocks.append(
                    {"text": overlay.text, "box": list(overlay.box), "confidence": 0.93, "role": "overlay"}
                )
            if overlay.close_btn is not None and overlay.close_btn.text:
                # A real screen shows the dismiss label, so OCR would read it;
                # without this block the overlay looks undismissable.
                blocks.append(
                    {
                        "text": overlay.close_btn.text,
                        "box": list(overlay.close_btn.box),
                        "confidence": 0.95,
                        "role": "overlay",
                        "handle": f"overlay_{index}_close",
                    }
                )
        return blocks

    # -- (de)serialization ------------------------------------------------- #
    @classmethod
    def load(cls, path: str | Path) -> "Scene":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_wire(), indent=2), encoding="utf-8")
        return path


class SceneSequence(SchemaModel):
    """An ordered run of screens (the replay backend walks this)."""

    name: str
    scenes: List[Scene] = Field(default_factory=list)
    answer_key: Dict[str, int] = Field(default_factory=dict, description="question text -> correct option index")
    meta: Dict[str, Any] = Field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.scenes)

    @classmethod
    def load(cls, path: str | Path) -> "SceneSequence":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_wire(), indent=2), encoding="utf-8")
        return path


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _hex_to_rgb(value: str) -> Color:
    text = value.strip().lstrip("#")
    if len(text) == 3:
        text = "".join(c * 2 for c in text)
    if len(text) != 6:
        raise ValueError(f"color must be #rrggbb, got {value!r}")
    return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))


def _default_question_region(scene: Scene) -> Box:
    if scene.options:
        top = min(option.hit_box[1] for option in scene.options)
        return (40, max(20, top - 90), scene.width - 80, 70)
    return (40, 60, scene.width - 80, 80)


#: Average glyph advance as a fraction of the font size for the renderer's face.
_TEXT_ADVANCE = 0.55


#: How each option style paints its selected state.  One source of truth for the
#: renderer, the simulated world, the fixture annotations and the intents the
#: orchestrator declares -- if these disagree, verification looks for a marker the
#: screen was never going to show (FR-7.9).
STYLE_MARKER: Dict[str, SelectedMarker] = {
    "radio": SelectedMarker.DOT,
    "checkbox": SelectedMarker.CHECK,
    "card": SelectedMarker.HIGHLIGHT,
    "tile": SelectedMarker.HIGHLIGHT,
    "button": SelectedMarker.HIGHLIGHT,
    "text_only": SelectedMarker.HIGHLIGHT,
}


def marker_for_style(style: Optional[str]) -> SelectedMarker:
    """The selected marker a given option style actually paints."""
    return STYLE_MARKER.get(str(style or ""), SelectedMarker.HIGHLIGHT)


def _option_text_box(option: SceneOption) -> Box:
    """Tight box around an option's glyphs, inside its (larger) hit area.

    A real OCR engine reports the *text* extent, not the clickable row.  Emitting
    row-wide boxes would make an option look obscured by anything overlapping the
    row -- a modal dialog, say -- and the option would then be dropped.
    """
    x, y, w, h = option.hit_box
    inset_x = max(12, int(h * 0.62)) if option.style in {"radio", "checkbox"} else max(8, int(w * 0.04))
    # Height follows the *font*, not the card: OCR boxes hug the glyphs, so a
    # 120px-tall card holding 16px text yields a ~22px-tall run.  Sizing it from
    # the card instead made option text outrank the question heading.
    text_h = max(10, int(option.font_px * 1.35))
    text_y = y + max(2, (h - text_h) // 2)
    room = max(20, w - inset_x - 8)
    text_w = int(len(option.text) * max(6, option.font_px) * _TEXT_ADVANCE) + 6
    return (x + inset_x, text_y, max(20, min(text_w, room)), text_h)


def _question_text_box(scene: "Scene", region: Box) -> Box:
    """Tight box around the question text, honouring wrapping inside ``region``."""
    x, y, w, h = region
    font_px = max(8, int(scene.question_font_px))
    text_w = int(len(scene.question_text) * font_px * (_TEXT_ADVANCE - 0.03)) + 8
    lines = max(1, -(-text_w // max(1, w)))
    line_h = int(font_px * 1.35)
    width = w if lines > 1 else max(20, min(text_w, w))
    return (x, y, int(width), max(font_px + 4, min(h, lines * line_h)))


def _button_annotation(button: Optional[SceneButton], handle: str) -> Optional[Dict[str, Any]]:
    if button is None:
        return None
    return {"handle": handle, "text": button.text, "box": list(button.box), "enabled": button.enabled}


def blend(color: Color, background: Color, factor: float) -> Color:
    """Wash a glyph colour towards the background (low-contrast fixture tier)."""
    factor = max(0.0, min(1.0, factor))
    return tuple(
        int(round(c * factor + b * (1 - factor))) for c, b in zip(color, background)
    )  # type: ignore[return-value]
