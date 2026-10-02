"""Deterministic world simulator for the replay harness (section 14, layer 2).

The simulator owns a :class:`~quizengine.scenes.SceneSequence` and behaves like a
quiz UI: clicking an option marks it selected, clicking *Next* advances, chaos
injections can force blank / stale / lock-screen / toast frames.  Because the
same object feeds both the capture backend (pixels) and the mock actuator
(applied intents), the integration test exercises the **real** closed loop --
perception, extraction, solving, confidence, action, verification, navigation --
with zero display and zero randomness.

It is a test double.  Nothing in a production run may use it: a real run gets
pixels from ``mss`` and has no privileged world state (**L1**).
"""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .contracts import ActionType, Intent, ScreenTransition, SelectedMarker
from .geometry import Box, box_center, box_contains_point
from .render import render_scene, render_uniform
from .scenes import Scene, SceneSequence, marker_for_style

InjectionMode = str  # "blank" | "stale" | "lock_screen" | "toast" | "layout_shift" | "delay" | "reorder"


class SimulatedWorld:
    """A scripted quiz UI with pixel output and action effects."""

    def __init__(self, sequence: SceneSequence, *, clock: Any = time.time, auto_reset: bool = True) -> None:
        self.sequence = sequence
        self._clock = clock
        self._auto_reset = auto_reset
        self.index = 0
        self.current: Scene = copy.deepcopy(sequence.scenes[0]) if sequence.scenes else _empty_scene()
        self.history: List[Dict[str, Any]] = []
        self._injections: List[InjectionMode] = []
        self._last_render: Optional[np.ndarray] = None
        self._stale_armed = False
        self._no_question_frames = 0
        self.applied_intents: List[Dict[str, Any]] = []

    # -- scene access ------------------------------------------------------ #
    @property
    def scene_count(self) -> int:
        return len(self.sequence.scenes)

    @property
    def exhausted(self) -> bool:
        return self.index >= self.scene_count

    def reset(self) -> None:
        self.index = 0
        self._no_question_frames = 0
        self.current = copy.deepcopy(self.sequence.scenes[0]) if self.sequence.scenes else _empty_scene()
        self._last_render = None

    def goto(self, index: int) -> Scene:
        index = max(0, min(index, self.scene_count - 1)) if self.scene_count else 0
        self.index = index
        self.current = copy.deepcopy(self.sequence.scenes[index]) if self.scene_count else _empty_scene()
        self._last_render = None
        return self.current

    def advance(self) -> Optional[Scene]:
        """Move to the next screen (what a Next button / auto-advance does)."""
        if self.index + 1 < self.scene_count:
            return self.goto(self.index + 1)
        if self._auto_reset:
            return None
        return None

    # -- rendering --------------------------------------------------------- #
    def render(self) -> np.ndarray:
        """Produce the pixels the capture backend will hand to the agent."""
        pending = self._consume_injection()
        if pending == "blank":
            return render_uniform((self.current.width, self.current.height), (255, 255, 255))
        if pending == "lock_screen":
            locked = self.current.model_copy(update={"screen_role": "lock_screen"})
            return render_scene(locked)
        if pending == "stale" and self._last_render is not None:
            return self._last_render.copy()
        if pending == "layout_shift":
            shifted = _shifted(self.current)
            pixels = render_scene(shifted)
            self._last_render = pixels
            # The next render is the settled layout (the "1-frame shift" chaos mode).
            self.current = copy.deepcopy(self.current)
            return pixels
        if pending == "toast":
            toasted = copy.deepcopy(self.current)
            from .contracts import OverlayKind
            from .scenes import SceneOverlay

            toasted.overlays.append(
                SceneOverlay(
                    text="Saved.",
                    box=(toasted.width - 260, toasted.height - 90, 220, 52),
                    kind=OverlayKind.TOAST,
                    dismissible=True,
                )
            )
            pixels = render_scene(toasted)
            self._last_render = pixels
            return pixels
        if pending == "reorder":
            self.current = _reordered(self.current)
        pixels = render_scene(self.current)
        self._last_render = pixels
        if self.current.screen_role == "blank":
            self._stale_armed = True
        return pixels

    def annotation(self) -> Dict[str, Any]:
        """Ground truth for the frame just rendered (fixtures/tests only)."""
        return self.current.annotation()

    def text_layer(self) -> List[str]:
        blocks = self.annotation().get("text_blocks", [])
        return [str(b.get("text", "")) for b in blocks]

    # -- chaos ------------------------------------------------------------- #
    def inject(self, mode: InjectionMode, times: int = 1) -> None:
        """Queue adversarial behaviour (FR-13.2 chaos mode parity)."""
        self._injections.extend([mode] * max(1, int(times)))

    def _consume_injection(self) -> Optional[InjectionMode]:
        if not self._injections:
            return None
        return self._injections.pop(0)

    @property
    def pending_injections(self) -> List[InjectionMode]:
        return list(self._injections)

    # -- actions ----------------------------------------------------------- #
    def apply(self, intent: Intent, point: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        """Mutate the world the way the real UI would.

        Returns a record describing the effect, which the replay harness asserts
        on and which mirrors what verification looks for on screen.
        """
        record: Dict[str, Any] = {
            "intent_id": intent.intent_id,
            "action": intent.action.value,
            "handle": intent.handle,
            "point": point,
            "effect": "none",
            "ts": self._clock(),
        }
        if intent.action == ActionType.SCROLL:
            record["effect"] = "scrolled"
            self.applied_intents.append(record)
            return record

        if intent.action in {ActionType.KEY,}:
            key = (intent.key_name or "").lower()
            if key in {"enter", "right", "n"} and self._advance_requested():
                self.advance()
                record["effect"] = "advanced"
            self.applied_intents.append(record)
            return record

        if intent.action != ActionType.CLICK or point is None:
            self.applied_intents.append(record)
            return record

        option = self._option_at(point)
        if option is not None:
            already = option.selected_marker != SelectedMarker.NONE
            for other in self.current.options:  # radio semantics: exclusive
                other.selected_marker = SelectedMarker.NONE
            # The marker the style actually paints (scenes.STYLE_MARKER), so the
            # screen, the annotation and the intent's expected marker agree.
            option.selected_marker = marker_for_style(option.style)
            record["effect"] = "already_selected" if already else "selected"
            record["option_index"] = option.index
            record["option_text"] = option.text
            self.applied_intents.append(record)
            if self.current.navigation.auto_advance and not already:
                self.advance()
                record["effect"] = "selected+advanced"
            return record

        button = self._button_at(point)
        if button is not None:
            record["effect"] = "advanced" if button != "nav_prev" else "retreated"
            if button == "nav_prev" and self.index > 0:
                self.goto(self.index - 1)
            elif button != "nav_prev":
                self.advance()
            self.applied_intents.append(record)
            return record

        overlay_index = self._overlay_close_index(point)
        if overlay_index is not None:
            # Actually take the dialog off the screen.  Recording the effect
            # without mutating the world leaves the overlay in every later frame,
            # and verification (correctly) refuses to believe it was dismissed.
            overlay = self.current.overlays.pop(overlay_index)
            self._last_render = None
            record["effect"] = "overlay_dismissed"
            record["overlay"] = f"overlay_{overlay_index}"
            record["overlay_kind"] = overlay.kind.value
            self.applied_intents.append(record)
            return record

        record["effect"] = "missed"
        self.applied_intents.append(record)
        return record

    def _advance_requested(self) -> bool:
        return self.index + 1 < self.scene_count

    def _option_at(self, point: Tuple[int, int]) -> Optional[Any]:
        for option in self.current.options:
            if box_contains_point(option.hit_box, point):
                return option
        return None

    def _button_at(self, point: Tuple[int, int]) -> Optional[str]:
        for handle, button in (
            ("nav_next", self.current.navigation.next_btn),
            ("nav_prev", self.current.navigation.prev_btn),
            ("nav_submit", self.current.navigation.submit_btn),
        ):
            if button is not None and box_contains_point(button.box, point):
                return handle
        return None

    def _overlay_close_index(self, point: Tuple[int, int]) -> Optional[int]:
        """Index of the overlay whose dismiss control contains ``point``."""
        for index, overlay in enumerate(self.current.overlays):
            if overlay.close_btn is not None and box_contains_point(overlay.close_btn.box, point):
                return index
        return None

    def _overlay_close_at(self, point: Tuple[int, int]) -> Optional[str]:
        index = self._overlay_close_index(point)
        return None if index is None else f"overlay_{index}"

    # -- assertions used by tests ------------------------------------------ #
    def selected_index(self) -> Optional[int]:
        for option in self.current.options:
            if option.selected_marker != SelectedMarker.NONE:
                return option.index
        return None

    def transition_vs(self, previous_annotation: Dict[str, Any]) -> ScreenTransition:
        current = self.annotation()
        if current.get("overlays") and not previous_annotation.get("overlays"):
            return ScreenTransition.POPUP
        if current.get("question_text") == previous_annotation.get("question_text") and current.get(
            "options"
        ) == previous_annotation.get("options"):
            return ScreenTransition.SAME_QUESTION
        if current.get("screen_role") == "end_state":
            return ScreenTransition.END_STATE
        return ScreenTransition.NEW_QUESTION


def _empty_scene() -> Scene:
    return Scene(name="empty", screen_role="blank", question_text="", options=[])


def _shifted(scene: Scene, dx: int = 6, dy: int = 4) -> Scene:
    """The 1-frame layout shift chaos behaviour (FR-13.2)."""
    shifted = copy.deepcopy(scene)

    def move(box: Box) -> Box:
        return (box[0] + dx, box[1] + dy, box[2], box[3])

    for text in shifted.texts:
        text.box = move(text.box)
    for option in shifted.options:
        option.hit_box = move(option.hit_box)
        if option.text_box is not None:
            option.text_box = move(option.text_box)
        if option.marker_box is not None:
            option.marker_box = move(option.marker_box)
    if shifted.question_region is not None:
        shifted.question_region = move(shifted.question_region)
    for button in (shifted.navigation.next_btn, shifted.navigation.prev_btn, shifted.navigation.submit_btn):
        if button is not None:
            button.box = move(button.box)
    return shifted


def _reordered(scene: Scene) -> Scene:
    """Option re-ordering between renders (FR-13.2).  Texts follow their boxes."""
    reordered = copy.deepcopy(scene)
    if len(reordered.options) < 2:
        return reordered
    boxes = [option.hit_box for option in reordered.options]
    text_boxes = [option.text_box for option in reordered.options]
    markers = [option.marker_box for option in reordered.options]
    rotated = reordered.options[1:] + reordered.options[:1]
    for option, hit, text_box, marker in zip(rotated, boxes, text_boxes, markers):
        option.hit_box = hit
        option.text_box = text_box
        option.marker_box = marker
    reordered.options = rotated
    for new_index, option in enumerate(reordered.options):
        option.index = new_index
    return reordered
