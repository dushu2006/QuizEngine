"""The screen model QuizForge serves -- shared with the engine's fixtures.

Nothing here is HTML-specific: :func:`build_screen` returns a plain description
of one screen (question, options, navigation, overlays, theme, zoom, chaos).  The
Jinja template renders it, ``/api/session`` returns it as JSON, and
``/api/parity/<variant>`` compares it against the fixture annotation the engine
perceives offline.  Keeping one model is what makes "the agent generalizes across
layout variants" a testable claim rather than a promise.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from quizengine.fixtures import (
    DEFAULT_VARIANT_ORDER,
    LAYOUT_VARIANTS,
    QUESTION_BANK,
    LayoutVariant,
)
from quizengine.scenes import STYLE_MARKER

#: Questions served by default.  Distinct per variant so a full run never repeats
#: a question (the engine skips repeats by hash, which would hide layout bugs).
QUESTIONS: Tuple[Tuple[str, Tuple[str, ...], int], ...] = QUESTION_BANK

#: Themes QuizForge exposes (section 13.2).
THEMES: Tuple[str, ...] = ("light", "dark")

#: Supported UI zoom levels: 100%, 125%, 150%, 200%.
ZOOM_LEVELS: Tuple[float, ...] = (1.0, 1.25, 1.5, 2.0)

#: Chaos switches (FR-13.2).  Each one changes what the screen looks like or how
#: it behaves, without changing the correct answer.
CHAOS_MODES: Tuple[str, ...] = (
    "popup",         # a dismissible modal covers the question
    "toast",         # a transient, non-blocking notice
    "layout_shift",  # the layout moves a few px after first paint
    "low_contrast",  # washed-out text and borders
    "delay",         # the server answers slowly
    "reorder",       # option order changes between renders
    "stale",         # the next screen is byte-identical to this one
)

#: Variant keys in the order a full sweep walks them.
VARIANTS: Tuple[str, ...] = DEFAULT_VARIANT_ORDER

#: CSS class per layout family, so the template stays declarative.
_LAYOUT_CLASS = {
    "vertical_options": "layout-vertical",
    "horizontal_options": "layout-horizontal",
    "card_grid": "layout-grid",
    "tile": "layout-grid",
    "text_only_buttons": "layout-vertical",
    "unknown": "layout-vertical",
}


@dataclass
class ScreenOption:
    """One answer choice as QuizForge renders it."""

    index: int
    text: str
    style: str
    selected: bool = False
    letter: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "letter": self.letter or chr(ord("A") + self.index),
            "text": self.text,
            "style": self.style,
            "selected": self.selected,
            "selected_marker": (STYLE_MARKER.get(self.style, "highlight") if self.selected else "none"),
        }


@dataclass
class Screen:
    """One screen: the unit the template renders and the parity check compares."""

    variant: str
    label: str
    layout_type: str
    style: str
    columns: int
    theme: str
    zoom: float
    question_text: str
    options: List[ScreenOption] = field(default_factory=list)
    index: int = 0
    total: int = 0
    next_label: Optional[str] = None
    prev_label: Optional[str] = "Previous"
    auto_advance: bool = False
    sidebar: bool = False
    overlays: List[Dict[str, Any]] = field(default_factory=list)
    chaos: Dict[str, bool] = field(default_factory=dict)
    screen_role: str = "quiz"
    results: Optional[Dict[str, Any]] = None
    correct_index: Optional[int] = None

    # -- rendering helpers -------------------------------------------------- #
    @property
    def progress_text(self) -> str:
        return f"{self.index + 1} of {self.total}" if self.total else ""

    @property
    def layout_class(self) -> str:
        return _LAYOUT_CLASS.get(self.layout_type, "layout-vertical")

    @property
    def style_class(self) -> str:
        """``style-<option style>``: how each choice paints its selected state."""
        return f"style-{self.style}"

    @property
    def grid_columns(self) -> int:
        """CSS grid column count for grid layouts (2x2 cards, 3x2 tiles)."""
        return max(1, int(self.columns)) if self.layout_class == "layout-grid" else 1

    @property
    def is_end_state(self) -> bool:
        return self.screen_role == "end_state"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variant": self.variant,
            "label": self.label,
            "layout_type": self.layout_type,
            "layout_class": self.layout_class,
            "style_class": self.style_class,
            "grid_columns": self.grid_columns,
            "style": self.style,
            "columns": self.columns,
            "theme": self.theme,
            "zoom": self.zoom,
            "screen_role": self.screen_role,
            "question": {
                "text": self.question_text,
                "index": self.index,
                "total": self.total,
                "progress_text": self.progress_text,
            },
            "options": [option.to_dict() for option in self.options],
            "navigation": {
                "next_label": self.next_label,
                "prev_label": self.prev_label,
                "auto_advance": self.auto_advance,
                "sidebar": self.sidebar,
            },
            "overlays": list(self.overlays),
            "chaos": dict(self.chaos),
            "results": self.results,
        }


def variant(key: str) -> LayoutVariant:
    """Look a variant up by key, with an actionable error message."""
    try:
        return LAYOUT_VARIANTS[key]
    except KeyError:
        raise KeyError(f"unknown layout variant {key!r}; try one of: {', '.join(DEFAULT_VARIANT_ORDER)}") from None


def pick_questions(count: int, *, offset: int = 0) -> List[Tuple[str, Tuple[str, ...], int]]:
    """``count`` distinct questions from the bank, cycling if necessary."""
    bank = list(QUESTIONS)
    return [bank[(offset + i) % len(bank)] for i in range(max(0, count))]


def build_screen(
    variant_key: str,
    question: str,
    options: Sequence[str],
    *,
    correct: Optional[int] = None,
    selected: Optional[int] = None,
    index: int = 0,
    total: int = 0,
    theme: str = "light",
    zoom: float = 1.0,
    chaos: Optional[Sequence[str]] = (),
    final: bool = False,
    order: Optional[Sequence[int]] = None,
    overlay_dismissed: bool = False,
) -> Screen:
    """Build one quiz screen for ``variant_key``.

    ``order`` permutes the options (chaos ``reorder``): the served screen and the
    exported answer key both follow the permutation, so the correct answer never
    changes meaning.
    """
    spec = variant(variant_key)
    if spec.screen_role == "end_state":
        return build_results_screen(
            total=total or max(1, index + 1),
            correct=correct or 0,
            theme=theme,
            zoom=zoom,
            variant_key=variant_key,
        )
    flags = {mode: (mode in (chaos or ())) for mode in CHAOS_MODES}
    if spec.low_contrast:
        flags["low_contrast"] = True
    if spec.overlay == "modal" and not overlay_dismissed:
        flags["popup"] = True

    texts = list(options)
    indices = list(range(len(texts)))
    if order and sorted(order) == indices:
        indices = list(order)

    screen_options = [
        ScreenOption(
            index=position,
            text=texts[original],
            style=spec.style,
            selected=(selected == position),
        )
        for position, original in enumerate(indices)
    ]
    mapped_correct: Optional[int] = None
    if correct is not None:
        mapped_correct = indices.index(correct) if correct in indices else correct

    overlays: List[Dict[str, Any]] = []
    if flags["popup"]:
        overlays.append(
            {
                "kind": "modal",
                "handle": "overlay_0",
                "text": "Session notice: your progress has been saved.",
                "dismissible": True,
                "close_label": "Close",
                "close_handle": "overlay_0_close",
            }
        )
    if flags["toast"]:
        overlays.append(
            {
                "kind": "toast",
                "handle": "overlay_1",
                "text": "Saved.",
                "dismissible": True,
                "close_label": "",
                "close_handle": None,
            }
        )

    auto_advance = bool(spec.auto_advance)
    next_label: Optional[str] = None
    if not auto_advance:
        next_label = "Submit" if (final or spec.final_submit) else "Next"

    return Screen(
        variant=spec.key,
        label=spec.label,
        layout_type=spec.layout_type.value,
        style=spec.style,
        columns=int(spec.columns),
        theme=theme if theme in THEMES else "light",
        zoom=float(zoom) if zoom in ZOOM_LEVELS else 1.0,
        question_text=question,
        options=screen_options,
        index=index,
        total=total,
        next_label=next_label,
        prev_label="Previous" if index > 0 else None,
        auto_advance=auto_advance,
        sidebar=bool(spec.sidebar),
        overlays=overlays,
        chaos=flags,
        screen_role=spec.screen_role,
        correct_index=mapped_correct,
    )


def build_results_screen(
    *,
    total: int,
    correct: int,
    theme: str = "light",
    zoom: float = 1.0,
    variant_key: str = "results_screen",
) -> Screen:
    """The terminal screen -- deliberately question-free (FR-7.8.4).

    It carries a score summary and a progress run, and *no* actionable Next or
    Submit control: those three properties are what the engine's end-state rule
    looks for, so this screen must keep them.
    """
    spec = variant(variant_key)
    return Screen(
        variant=spec.key,
        label=spec.label,
        layout_type="unknown",
        style=spec.style,
        columns=1,
        theme=theme,
        zoom=zoom,
        question_text="",
        options=[],
        index=total,
        total=total,
        next_label=None,
        prev_label=None,
        auto_advance=False,
        sidebar=False,
        screen_role="end_state",
        results={
            "headline": "Quiz complete",
            "score": f"Your score: {correct} of {total}",
            "footer": "Thank you for participating.",
            "progress_text": f"{total} of {total}",
            "correct": correct,
            "total": total,
        },
    )


def answer_key(screens: Sequence[Screen]) -> Dict[str, int]:
    """``question text -> correct option index`` for a served sequence."""
    key: Dict[str, int] = {}
    for screen in screens:
        if screen.is_end_state or screen.correct_index is None or not screen.question_text:
            continue
        key[screen.question_text] = int(screen.correct_index)
    return key


def answer_key_csv(key: Dict[str, int], correct_texts: Optional[Dict[str, str]] = None) -> str:
    """The same key as CSV (what a human grader wants)."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["question", "correct_index", "correct_letter", "correct_text"])
    for question, index in key.items():
        correct_text = (correct_texts or {}).get(question, "")
        writer.writerow([question, index, chr(ord("A") + int(index)), correct_text])
    return buffer.getvalue()


def describe_variants() -> List[Dict[str, Any]]:
    """The catalogue, as JSON-friendly rows (the picker page and ``variants`` CLI)."""
    rows: List[Dict[str, Any]] = []
    for key in DEFAULT_VARIANT_ORDER:
        spec = LAYOUT_VARIANTS[key]
        rows.append(
            {
                "key": spec.key,
                "label": spec.label,
                "style": spec.style,
                "size": list(spec.size),
                "theme": spec.theme,
                "zoom": spec.zoom,
                "columns": spec.columns,
                "layout_type": spec.layout_type.value,
                "auto_advance": spec.auto_advance,
                "final_submit": spec.final_submit,
                "sidebar": spec.sidebar,
                "low_contrast": spec.low_contrast,
                "overlay": spec.overlay,
                "screen_role": spec.screen_role,
                "description": spec.description,
            }
        )
    return rows


__all__ = [
    "CHAOS_MODES",
    "QUESTIONS",
    "Screen",
    "ScreenOption",
    "THEMES",
    "VARIANTS",
    "ZOOM_LEVELS",
    "answer_key",
    "answer_key_csv",
    "build_results_screen",
    "build_screen",
    "describe_variants",
    "pick_questions",
    "variant",
]
