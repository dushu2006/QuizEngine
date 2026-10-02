"""Fixture scene catalog: the layouts QuizForge renders, as engine fixtures.

One builder, ``build_scene()``, turns a :class:`LayoutVariant` plus a question
into a fully annotated :class:`~quizengine.scenes.Scene`.  The same variant keys
are used by the QuizForge Flask app (section 13.2) so a layout that passes here
is the layout the browser renders there.

Everything is deterministic and offline: no network, no browser, no OCR binary.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .contracts import LayoutType, OverlayKind, QuestionType, SelectedMarker
from .render import render_scene
from .scenes import (
    STYLE_MARKER,
    Scene,
    SceneButton,
    SceneNavigation,
    SceneOption,
    SceneOverlay,
    SceneSequence,
    SceneText,
)

OptionStyle = str

#: Re-exported from :mod:`quizengine.scenes`: one mapping used by the renderer, the
#: simulated world, the fixture annotations and the intents the orchestrator
#: declares.  They must agree or verification looks for a marker the screen never
#: shows (FR-7.9).


@dataclass(frozen=True)
class LayoutVariant:
    """One of the section 13.2 layout variants."""

    key: str
    label: str
    style: OptionStyle
    size: Tuple[int, int] = (1280, 800)
    theme: str = "light"
    zoom: float = 1.0
    columns: int = 1
    layout_type: LayoutType = LayoutType.VERTICAL_OPTIONS
    auto_advance: bool = False
    #: The navigation control reads "Submit" even mid-sequence.
    final_submit: bool = False
    screen_role: str = "quiz"
    sidebar: bool = False
    low_contrast: bool = False
    overlay: Optional[str] = None
    description: str = ""

    @property
    def width(self) -> int:
        return int(self.size[0])

    @property
    def height(self) -> int:
        return int(self.size[1])


LAYOUT_VARIANTS: Dict[str, LayoutVariant] = {
    variant.key: variant
    for variant in (
        LayoutVariant("radio_vertical", "Radio list (classic)", "radio",
                      description="Single-choice vertical list with radio glyphs and a Next button."),
        LayoutVariant("checkbox_vertical", "Checkbox list", "checkbox",
                      description="Checkbox markers; the engine must still pick exactly one option."),
        LayoutVariant("card_grid_2x2", "Card grid 2x2", "card", layout_type=LayoutType.CARD_GRID, columns=2,
                      size=(1280, 800), description="Two-by-two card grid, selection shown as a filled card."),
        LayoutVariant("tile_grid_3x2", "Tile grid 3x2", "tile", layout_type=LayoutType.CARD_GRID, columns=3,
                      size=(1366, 768), description="Three-by-two tile grid with six options."),
        LayoutVariant("button_row", "Button row (horizontal)", "button", layout_type=LayoutType.HORIZONTAL_OPTIONS,
                      columns=4, description="Options rendered as side-by-side buttons."),
        LayoutVariant("text_only_list", "Text-only list", "text_only",
                      description="No marker glyphs at all: selection is an accent bar and text colour."),
        LayoutVariant("radio_horizontal", "Radio row", "radio", layout_type=LayoutType.HORIZONTAL_OPTIONS, columns=4,
                      description="Radio options laid out in a single row."),
        LayoutVariant("compact_dark", "Compact dark (1366x768)", "radio", size=(1366, 768), theme="dark",
                      description="Dark theme at laptop resolution with tighter spacing."),
        LayoutVariant("wide_hd", "Wide HD (1920x1080)", "radio", size=(1920, 1080),
                      description="Full-HD canvas: everything is further apart and larger."),
        LayoutVariant("zoom_150", "Browser zoom 150%", "radio", size=(1600, 1000), zoom=1.5,
                      description="Fractional scaling stress test (FR-7.2.3 zoom invariance). The canvas "
                                  "is the device-pixel size of a 150%%-zoomed window so the whole question "
                                  "stays on screen (the simulated world has no viewport scroll)."),
        LayoutVariant("zoom_200", "Browser zoom 200%", "checkbox", size=(1920, 1200), zoom=2.0,
                      description="Maximum zoom level in the QuizEngine UI, at the device-pixel canvas a "
                                  "200%%-zoomed window occupies."),
        LayoutVariant("two_options", "True/False (2 options)", "radio",
                      description="Minimum option count allowed by section 3.1."),
        LayoutVariant("six_options_long", "Six options (long list)", "radio",
                      description="Maximum single-choice option count; tests rhythm grouping."),
        LayoutVariant("sidebar_progress", "Sidebar progress", "radio", sidebar=True,
                      description="Progress and navigation live in a left sidebar."),
        LayoutVariant("auto_advance", "Auto-advance (no Next)", "card", layout_type=LayoutType.CARD_GRID, columns=2,
                      auto_advance=True, description="Platform advances by itself once an option is chosen."),
        LayoutVariant("submit_final", "Submit button (final screen)", "radio", final_submit=True,
                      description="Last question: the navigation control says Submit."),
        LayoutVariant("popup_modal", "Modal popup (chaos)", "radio", overlay="modal",
                      description="Chaos mode: a dismissible modal covers the question."),
        LayoutVariant("low_contrast", "Low contrast (chaos)", "radio", low_contrast=True,
                      description="Chaos mode: washed-out text that degrades OCR confidence."),
        LayoutVariant("results_screen", "Results / end state", "radio", screen_role="end_state",
                      description="Terminal screen: no question region, score summary only."),
    )
}

DEFAULT_VARIANT_ORDER: Tuple[str, ...] = (
    "radio_vertical",
    "checkbox_vertical",
    "card_grid_2x2",
    "tile_grid_3x2",
    "button_row",
    "text_only_list",
    "radio_horizontal",
    "compact_dark",
    "wide_hd",
    "zoom_150",
    "zoom_200",
    "two_options",
    "six_options_long",
    "sidebar_progress",
    "auto_advance",
    "submit_final",
    "popup_modal",
    "low_contrast",
    "results_screen",
)

#: A bank of distinct questions so a multi-variant run never repeats a question
#: (a repeat would be skipped by the L9 duplicate rule instead of exercised).
QUESTION_BANK: Tuple[Tuple[str, Tuple[str, ...], int], ...] = (
    ("Which city is the capital of France?", ("Berlin", "Paris", "Madrid", "Rome"), 1),
    ("What is 12 * 8?", ("84", "96", "108", "112"), 1),
    ("Which planet is closest to the Sun?", ("Venus", "Mercury", "Earth", "Mars"), 1),
    ("What is 15% of 240?", ("24", "30", "36", "48"), 2),
    ("Which language is primarily used to style web pages?", ("HTML", "CSS", "SQL", "Python"), 1),
    ("How many days are there in a leap year?", ("364", "365", "366", "367"), 2),
    ("How many meters are in 3 km?", ("300", "3000", "30000", "0.3"), 1),
    ("What comes next in the sequence 2, 4, 8, 16?", ("18", "24", "32", "64"), 2),
    ("What is the Roman numeral for 49?", ("XLIX", "IL", "LIX", "XXXXIX"), 0),
    ("Which is the largest ocean on Earth?", ("Atlantic", "Indian", "Pacific", "Arctic"), 2),
    ("How many minutes are in 2 hours?", ("60", "90", "120", "240"), 2),
    ("What is 7 + 5 * 3?", ("36", "22", "27", "15"), 1),
    ("Which element has the chemical symbol O?", ("Gold", "Oxygen", "Osmium", "Silver"), 1),
    ("How many letters are in the word \"banana\"?", ("5", "6", "7", "3"), 1),
    ("Which year did the first Moon landing occur?", ("1965", "1969", "1972", "1959"), 1),
    ("What is 20% of 150?", ("20", "25", "30", "35"), 2),
    ("Which data structure uses FIFO ordering?", ("Stack", "Queue", "Tree", "Graph"), 1),
    ("How many grams are in 2 kg?", ("200", "2000", "20", "20000"), 1),
    ("Which country is home to the kangaroo?", ("New Zealand", "South Africa", "Australia", "Brazil"), 2),
    ("What is the square of 13?", ("149", "159", "169", "179"), 2),
)

#: Demo quiz: mixes knowledge questions (answer key) with arithmetic (local rules).
DEMO_QUESTIONS: Tuple[Tuple[str, Tuple[str, ...], int], ...] = (
    ("Which city is the capital of France?", ("Berlin", "Paris", "Madrid", "Rome"), 1),
    ("What is 12 * 8?", ("84", "96", "108", "112"), 1),
    ("Which planet is closest to the Sun?", ("Venus", "Mercury", "Earth", "Mars"), 1),
    ("What is 15% of 240?", ("24", "30", "36", "48"), 2),
    ("Which language is primarily used to style web pages?", ("HTML", "CSS", "SQL", "Python"), 1),
    ("How many days are there in a leap year?", ("364", "365", "366", "367"), 2),
)


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def _nav_top(variant: LayoutVariant) -> int:
    """Y coordinate of the navigation row; option boxes must stay above it."""
    _width, height = variant.size
    scale = variant.zoom
    return height - int((180 if variant.sidebar else 120) * scale)


def _vertical_fit(
    rows: int, nav_top: int, scale: float, *, first_top: int, box_h: int, gap: int
) -> Tuple[int, int, int]:
    """Pull a stack up / shrink it so its last row clears the navigation row.

    A page never overlaps its own chrome.  A fixture that did would make a
    click meant for ``Next`` land on an option instead -- which is both
    unrealistic and indistinguishable from a genuine navigation failure.
    """
    if rows <= 0:
        return first_top, box_h, gap
    min_h = max(24, int(30 * scale))
    min_gap = max(4, int(6 * scale))

    def needed(height: int, spacing: int) -> int:
        return rows * height + max(0, rows - 1) * spacing

    top = first_top
    if top + needed(box_h, gap) <= nav_top:
        return top, box_h, gap
    top = max(int(80 * scale), nav_top - needed(box_h, gap) - int(10 * scale))
    if top + needed(box_h, gap) <= nav_top:
        return top, box_h, gap
    available = max(rows * min_h, nav_top - top)
    box_h = max(min_h, (available - max(0, rows - 1) * gap) // max(1, rows))
    if rows > 1 and top + needed(box_h, gap) > nav_top:
        gap = max(min_gap, (nav_top - top - rows * box_h) // (rows - 1))
    return top, box_h, gap


def _option_boxes(variant: LayoutVariant, count: int) -> List[Tuple[int, int, int, int]]:
    """Lay ``count`` option hit boxes out according to the variant."""
    width, height = variant.size
    scale = variant.zoom
    margin_x = int(80 * scale)
    top = int(240 * scale)
    if variant.screen_role == "end_state":
        return []
    nav_top = _nav_top(variant)
    if variant.columns >= 2:
        cols = min(variant.columns, count)
        rows = (count + cols - 1) // cols
        gap_x = int(40 * scale)
        gap_y = int(30 * scale)
        usable_w = width - 2 * margin_x - gap_x * (cols - 1)
        box_w = max(120, int(usable_w / cols))
        box_h = max(48, int(min(120, (height - top - int(140 * scale)) / max(1, rows) - gap_y) * scale))
        top, box_h, gap_y = _vertical_fit(rows, nav_top, scale, first_top=top, box_h=box_h, gap=gap_y)
        boxes = []
        for index in range(count):
            col, row = index % cols, index // cols
            boxes.append((margin_x + col * (box_w + gap_x), top + row * (box_h + gap_y), box_w, box_h))
        return boxes
    if variant.layout_type is LayoutType.HORIZONTAL_OPTIONS:
        gap = int(24 * scale)
        usable_w = width - 2 * margin_x - gap * (count - 1)
        box_w = max(110, int(usable_w / count))
        box_h = max(44, int(56 * scale))
        row_top = int(height * 0.52)
        row_top, box_h, _ = _vertical_fit(1, nav_top, scale, first_top=row_top, box_h=box_h, gap=0)
        return [(margin_x + i * (box_w + gap), row_top, box_w, box_h) for i in range(count)]
    gap = int(16 * scale)
    box_w = min(int(720 * scale), width - 2 * margin_x)
    box_h = max(44, int(54 * scale))
    if variant.sidebar:
        margin_x = int(260 * scale)
    top, box_h, gap = _vertical_fit(count, nav_top, scale, first_top=top, box_h=box_h, gap=gap)
    boxes = []
    for index in range(count):
        boxes.append((margin_x, top + index * (box_h + gap), box_w, box_h))
    return boxes


def _question_region(variant: LayoutVariant, boxes: Sequence[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
    """Question header box: below the progress run, above the first option row."""
    width, _height = variant.size
    scale = variant.zoom
    left = int(260 * scale) if variant.sidebar else int(80 * scale)
    region_w = min(int(900 * scale), width - left - int(60 * scale))
    region_h = int(60 * scale)
    first_top = int(boxes[0][1]) if boxes else int(360 * scale)
    top = first_top - int(120 * scale)
    if top < int(40 * scale):
        # A zoomed layout leaves no room for a full header: keep it thin and
        # strictly above the options rather than letting the two overlap.
        top = int(40 * scale)
        region_h = max(int(24 * scale), first_top - top - int(8 * scale))

    progress = _nav_boxes(variant).get("progress")
    if progress is not None:
        progress_bottom = progress[1] + progress[3]
        shares_column = left < progress[0] + progress[2] and left + region_w > progress[0]
        if shares_column and top < progress_bottom:
            if progress_bottom + region_h + int(8 * scale) <= first_top:
                top = progress_bottom + int(8 * scale)
            else:
                region_w = max(int(200 * scale), progress[0] - left - int(20 * scale))
    return (left, int(top), int(region_w), int(region_h))


def _nav_boxes(variant: LayoutVariant) -> Dict[str, Tuple[int, int, int, int]]:
    width, height = variant.size
    scale = variant.zoom
    if variant.sidebar:
        return {
            "next": (int(60 * scale), int(height - 160 * scale), int(160 * scale), int(48 * scale)),
            "progress": (int(60 * scale), int(120 * scale), int(170 * scale), int(28 * scale)),
        }
    return {
        "next": (int(width - 280 * scale), int(height - 100 * scale), int(180 * scale), int(50 * scale)),
        "progress": (int(width - 280 * scale), int(60 * scale), int(220 * scale), int(28 * scale)),
    }


# --------------------------------------------------------------------------- #
# scene building
# --------------------------------------------------------------------------- #
def build_scene(
    variant: LayoutVariant | str,
    question: str,
    options: Sequence[str],
    *,
    correct: Optional[int] = None,
    selected: Optional[int] = None,
    progress: Optional[Tuple[int, int]] = None,
    name: Optional[str] = None,
    final: bool = False,
) -> Scene:
    """Render-ready, fully annotated scene for one layout variant."""
    if isinstance(variant, str):
        variant = LAYOUT_VARIANTS[variant]
    count = len(options)
    boxes = _option_boxes(variant, count)
    if count and len(boxes) != count:  # pragma: no cover - defensive
        raise ValueError(f"layout {variant.key!r} produced {len(boxes)} boxes for {count} options")
    region = _question_region(variant, boxes)
    nav = _nav_boxes(variant)
    marker = STYLE_MARKER.get(variant.style, SelectedMarker.HIGHLIGHT)
    contrast = 0.42 if variant.low_contrast else 1.0
    font_px = max(8, int(16 * variant.zoom))

    scene_options = [
        SceneOption(
            index=index,
            text=text,
            hit_box=boxes[index],
            style=variant.style,  # type: ignore[arg-type]
            font_px=font_px,
            contrast=contrast,
            selected_marker=(marker if selected == index else SelectedMarker.NONE),
        )
        for index, text in enumerate(options)
    ]

    if variant.screen_role == "end_state":
        return Scene(
            name=name or variant.key,
            size_px=variant.size,
            theme=variant.theme,  # type: ignore[arg-type]
            zoom=variant.zoom,
            layout_type=LayoutType.UNKNOWN,
            screen_role="end_state",  # type: ignore[arg-type]
            question_text="",
            question_region=None,
            options=[],
            texts=[
                SceneText(text="Quiz complete", box=(int(region[0]), int(region[1]), 520, 44), font_px=int(30 * variant.zoom), role="header"),
                SceneText(text="Your score: 5 of 6", box=(int(region[0]), int(region[1] + 60 * variant.zoom), 420, 32), font_px=int(20 * variant.zoom), role="result"),
                SceneText(text="Thank you for participating.", box=(int(region[0]), int(region[1] + 110 * variant.zoom), 520, 30), font_px=font_px, role="noise"),
            ],
            navigation=SceneNavigation(
                progress_text="6 of 6",
                # Bottom of the screen: a results page has no question header, and a
                # top-of-page progress run would be mistaken for one.
                progress_box=(int(region[0]), int(variant.size[1] - 70 * variant.zoom), int(180 * variant.zoom), int(28 * variant.zoom)),
                progress_current=6,
                progress_total=6,
            ),
            meta={"variant": variant.key, "correct_index": correct},
        )

    navigation = SceneNavigation(
        next_btn=None if variant.auto_advance else SceneButton(text="Submit" if final else "Next", box=nav["next"]),
        submit_btn=SceneButton(text="Submit", box=nav["next"]) if final and not variant.auto_advance else None,
        progress_text=(f"{progress[0]} of {progress[1]}" if progress else None),
        progress_box=nav["progress"] if progress else None,
        progress_current=progress[0] if progress else None,
        progress_total=progress[1] if progress else None,
        auto_advance=variant.auto_advance,
    )

    overlays: List[SceneOverlay] = []
    if variant.overlay == "modal":
        width, height = variant.size
        box = (int(width * 0.28), int(height * 0.30), int(width * 0.44), int(height * 0.26))
        overlays.append(
            SceneOverlay(
                text="Session notice: your progress has been saved.",
                box=box,
                kind=OverlayKind.MODAL,
                dismissible=True,
                close_btn=SceneButton(text="Close", box=(box[0] + box[2] - 130, box[1] + box[3] - 62, 110, 44)),
                font_px=max(10, int(15 * variant.zoom)),
            )
        )

    return Scene(
        name=name or variant.key,
        size_px=variant.size,
        theme=variant.theme,  # type: ignore[arg-type]
        zoom=variant.zoom,
        layout_type=variant.layout_type,
        question_type=QuestionType.CHECKBOX_TILE if variant.style == "checkbox" else QuestionType.SINGLE_CHOICE,
        screen_role="quiz",  # type: ignore[arg-type]
        question_text=question,
        question_region=region,
        question_font_px=max(10, int(22 * variant.zoom)),
        question_contrast=contrast,
        options=scene_options,
        navigation=navigation,
        overlays=overlays,
        flags={"low_contrast": variant.low_contrast},
        expected_click_handle=(f"opt_{correct}" if correct is not None else None),
        meta={"variant": variant.key, "correct_index": correct, "style": variant.style},
    )


def build_sequence(
    questions: Sequence[Tuple[str, Sequence[str], int]],
    *,
    variants: Optional[Sequence[str]] = None,
    name: str = "fixture-sequence",
    append_end_state: bool = True,
    overlay_on: Optional[int] = None,
) -> SceneSequence:
    """One scene per question, cycling through the requested layout variants."""
    keys = list(variants) if variants else ["radio_vertical"]
    scenes: List[Scene] = []
    answer_key: Dict[str, int] = {}
    total = len(questions)
    for index, (question, options, correct) in enumerate(questions):
        key = keys[index % len(keys)]
        variant = LAYOUT_VARIANTS[key]
        if overlay_on is not None and index == overlay_on:
            variant = LAYOUT_VARIANTS["popup_modal"]
        final = index == total - 1 or variant.final_submit
        scenes.append(
            build_scene(
                variant,
                question,
                options,
                correct=correct,
                progress=(index + 1, total),
                name=f"q{index + 1}-{variant.key}",
                final=final,
            )
        )
        answer_key[question] = int(correct)
    if append_end_state:
        scenes.append(build_scene(LAYOUT_VARIANTS["results_screen"], "", [], name="end-results"))
    return SceneSequence(name=name, scenes=scenes, answer_key=answer_key, meta={"variants": keys})


def demo_sequence(*, variants: Optional[Sequence[str]] = None) -> SceneSequence:
    """The CLI/demo quiz: six questions across several layouts plus an end screen."""
    return build_sequence(
        [(q, opts, correct) for q, opts, correct in DEMO_QUESTIONS],
        variants=variants or ["radio_vertical", "card_grid_2x2", "text_only_list", "button_row", "checkbox_vertical", "submit_final"],
        name="demo",
    )


def variant_matrix_sequence(*, include_chaos: bool = True) -> SceneSequence:
    """Every layout variant applied to its own question -- the generalization sweep.

    ``max_questions`` must be raised to run this to completion (it has one
    question per variant); it is the strongest end-to-end check available offline.
    """
    keys = [
        key
        for key in DEFAULT_VARIANT_ORDER
        if key != "results_screen" and (include_chaos or key not in {"popup_modal", "low_contrast"})
    ]
    bank = list(QUESTION_BANK)
    while len(bank) < len(keys):
        bank.extend(DEMO_QUESTIONS)
    questions = [bank[index] for index in range(len(keys))]
    return build_sequence(questions, variants=keys, name="variant-matrix", append_end_state=True)


# --------------------------------------------------------------------------- #
# fixture frames on disk (replay backend)
# --------------------------------------------------------------------------- #
def write_fixture_frames(sequence: SceneSequence, out_dir: str | Path) -> List[Path]:
    """Render each scene to ``frame_NNN.png`` + ``frame_NNN.json`` ground truth."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for index, scene in enumerate(sequence.scenes):
        pixels = render_scene(scene)
        png = root / f"frame_{index:03d}.png"
        try:
            import numpy as np
            from PIL import Image

            Image.fromarray(np.asarray(pixels)).save(png)
        except Exception:  # pragma: no cover - pillow is a core dependency
            continue
        sidecar = root / f"frame_{index:03d}.json"
        sidecar.write_text(json.dumps(scene.annotation(), indent=2), encoding="utf-8")
        written.append(png)
    (root / "sequence.json").write_text(json.dumps(sequence.to_wire(), indent=2), encoding="utf-8")
    return written


def load_sequence(path: str | Path) -> SceneSequence:
    return SceneSequence.load(path)


def describe_variants() -> str:
    lines = [f"{'KEY':22} {'STYLE':10} {'SIZE':11} {'THEME':6} ZOOM  LAYOUT", "-" * 92]
    for key in DEFAULT_VARIANT_ORDER:
        variant = LAYOUT_VARIANTS[key]
        lines.append(
            f"{variant.key:22} {variant.style:10} {variant.size[0]}x{variant.size[1]:<5} "
            f"{variant.theme:6} {variant.zoom:<5} {variant.layout_type.value}"
        )
    return "\n".join(lines)


__all__ = [
    "DEFAULT_VARIANT_ORDER",
    "DEMO_QUESTIONS",
    "LAYOUT_VARIANTS",
    "LayoutVariant",
    "QUESTION_BANK",
    "STYLE_MARKER",
    "build_scene",
    "build_sequence",
    "demo_sequence",
    "describe_variants",
    "load_sequence",
    "variant_matrix_sequence",
    "write_fixture_frames",
]
