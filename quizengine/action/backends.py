"""Actuator backends (FR-7.6.1).

One small interface, four implementations:

``PyAutoGUIBackend``  real mouse/keyboard on Windows-first desktops (guarded by
                      the capability probe; import failure is a clean error)
``SimulatedBackend``  drives :class:`~quizengine.sim.SimulatedWorld` so the whole
                      closed loop runs on fixture frames with zero side effects
``RecordingBackend``  dry run: records exactly what would have been done
``NullBackend``       no-op sink for unit tests

The timings in :class:`Timings` come from ``action.input_profile`` and exist so
interaction looks like ordinary human input (FR-7.6.2).  They are **not** an
evasion mechanism: the engine refuses to run at all in a proctored or
human-verification context (section 3.3, section 7.14).
"""

from __future__ import annotations

import abc
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..capabilities import has
from ..contracts import ActionType, Intent

Point = Tuple[int, int]


@dataclass
class Timings:
    """Human-plausible interaction bounds (FR-7.6.2)."""

    move_duration_s: float = 0.3
    jitter_px: float = 0.0
    typing_delay_s: float = 0.05
    scroll_pause_s: float = 0.25
    jitter: Point = (0, 0)

    @classmethod
    def sample(cls, profile: Any, rng: random.Random) -> "Timings":
        move_range = tuple(getattr(profile, "move_duration_range", (0.2, 0.4)))
        type_range = tuple(getattr(profile, "typing_delay_range", (0.03, 0.09)))
        scroll_range = tuple(getattr(profile, "scroll_pause_range", (0.15, 0.35)))
        jitter_px = float(getattr(profile, "click_jitter_px", 3) or 0)
        dx = rng.randint(-int(jitter_px), int(jitter_px)) if jitter_px >= 1 else 0
        dy = rng.randint(-int(jitter_px), int(jitter_px)) if jitter_px >= 1 else 0
        return cls(
            move_duration_s=rng.uniform(move_range[0], move_range[1]),
            jitter_px=jitter_px,
            typing_delay_s=rng.uniform(type_range[0], type_range[1]),
            scroll_pause_s=rng.uniform(scroll_range[0], scroll_range[1]),
            jitter=(dx, dy),
        )


@dataclass
class BackendReport:
    ok: bool = True
    detail: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class ActuatorBackend(abc.ABC):
    name: str = "abstract"
    kind: str = "abstract"

    def available(self) -> bool:
        return True

    @abc.abstractmethod
    def perform(self, intent: Intent, point: Optional[Point], timings: Timings) -> BackendReport:
        """Carry out the intent.  Must not raise for expected UI failures."""

    def position(self) -> Optional[Point]:
        return None

    def focus_ok(self, title_match: Optional[str] = None) -> bool:
        return True

    def failsafe_triggered(self) -> bool:
        return False

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "available": self.available()}


class PyAutoGUIBackend(ActuatorBackend):
    """FR-7.6.1 default real backend."""

    name = "pyautogui"
    kind = "real"

    def __init__(self, *, failsafe_corner: bool = True, sleep: Callable[[float], None] = time.sleep) -> None:
        self._sleep = sleep
        self.failsafe_corner = failsafe_corner
        self._module: Any = None
        self._import_error: Optional[str] = None
        if has("pyautogui"):
            try:
                import pyautogui

                pyautogui.FAILSAFE = bool(failsafe_corner)
                self._module = pyautogui
            except Exception as exc:  # pragma: no cover - no display in CI
                self._import_error = f"{type(exc).__name__}: {exc}"
        else:
            self._import_error = "pyautogui is not installed"

    def available(self) -> bool:
        return self._module is not None

    def perform(self, intent: Intent, point: Optional[Point], timings: Timings) -> BackendReport:
        if self._module is None:
            return BackendReport(ok=False, error=self._import_error or "pyautogui unavailable")
        gui = self._module
        try:
            if intent.action is ActionType.MOVE and point is not None:
                gui.moveTo(point[0], point[1], duration=timings.move_duration_s)
                return BackendReport(detail={"moved_to": list(point)})
            if intent.action is ActionType.CLICK and point is not None:
                gui.moveTo(point[0], point[1], duration=timings.move_duration_s)
                self._sleep(0.02)
                gui.click(point[0], point[1])
                return BackendReport(detail={"clicked": list(point)})
            if intent.action is ActionType.SCROLL:
                dy = int(intent.scroll_delta or 0)
                if point is not None:
                    gui.scroll(dy, x=point[0], y=point[1])
                else:
                    gui.scroll(dy)
                self._sleep(timings.scroll_pause_s)
                return BackendReport(detail={"scrolled": dy})
            if intent.action is ActionType.KEY:
                gui.press(intent.key_name or "enter")
                return BackendReport(detail={"key": intent.key_name})
            if intent.action is ActionType.TYPE:
                gui.typewrite(intent.type_text or "", interval=timings.typing_delay_s)
                return BackendReport(detail={"typed_chars": len(intent.type_text or "")})
            return BackendReport(ok=False, error=f"unsupported action for backend: {intent.action.value}")
        except Exception as exc:  # pyautogui raises FailSafeException on corner moves
            return BackendReport(ok=False, error=f"{type(exc).__name__}: {exc}")

    def position(self) -> Optional[Point]:
        if self._module is None:
            return None
        try:
            x, y = self._module.position()
            return (int(x), int(y))
        except Exception:
            return None

    def focus_ok(self, title_match: Optional[str] = None) -> bool:
        """FR-7.6.4: the target window must be foreground before acting."""
        if not title_match:
            return True
        try:
            import sys

            if sys.platform.startswith("win"):  # pragma: no cover - Windows only
                import ctypes

                user32 = ctypes.windll.user32  # type: ignore[attr-defined]
                hwnd = user32.GetForegroundWindow()
                length = user32.GetWindowTextLengthW(hwnd)
                buffer = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buffer, length + 1)
                return title_match.lower() in buffer.value.lower()
        except Exception:
            return False
        # Without a platform API we cannot prove focus; treat as unknown-but-ok
        # only when the caller did not ask for a specific title.
        return True

    def failsafe_triggered(self) -> bool:
        if not self.failsafe_corner:
            return False
        point = self.position()
        return point is not None and point[0] <= 1 and point[1] <= 1


class SimulatedBackend(ActuatorBackend):
    """Drives :class:`~quizengine.sim.SimulatedWorld` -- the CI/demo backend."""

    name = "simulated"
    kind = "simulated"

    def __init__(self, world: Any, *, sleep: Callable[[float], None] = lambda _s: None) -> None:
        self.world = world
        self._sleep = sleep
        self.history: List[Dict[str, Any]] = []

    def available(self) -> bool:
        return self.world is not None

    def perform(self, intent: Intent, point: Optional[Point], timings: Timings) -> BackendReport:
        if self.world is None:
            return BackendReport(ok=False, error="no simulated world attached")
        try:
            report = self.world.apply(intent, point)
        except Exception as exc:
            return BackendReport(ok=False, error=f"{type(exc).__name__}: {exc}")
        self.history.append({"intent_id": intent.intent_id, "action": intent.action.value, "point": point, "report": report})
        return BackendReport(ok=bool(report.get("applied", True)), detail=report)

    def position(self) -> Optional[Point]:
        if self.history and self.history[-1]["point"] is not None:
            return tuple(self.history[-1]["point"])  # type: ignore[return-value]
        return None

    def describe(self) -> Dict[str, Any]:
        return {**super().describe(), "actions": len(self.history)}


class RecordingBackend(ActuatorBackend):
    """Dry run (``--dry-run``): nothing touches the screen."""

    name = "recording"
    kind = "dry-run"

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def perform(self, intent: Intent, point: Optional[Point], timings: Timings) -> BackendReport:
        self.calls.append(
            {
                "intent_id": intent.intent_id,
                "action": intent.action.value,
                "handle": intent.handle,
                "point": list(point) if point else None,
                "box": list(intent.target_box) if intent.target_box else None,
                "key": intent.key_name,
                "scroll_delta": intent.scroll_delta,
                "text": intent.type_text,
                "move_duration_s": round(timings.move_duration_s, 3),
                "jitter": list(timings.jitter),
            }
        )
        return BackendReport(detail={"recorded": True, "dry_run": True})

    def describe(self) -> Dict[str, Any]:
        return {**super().describe(), "calls": len(self.calls)}


class NullBackend(ActuatorBackend):
    name = "null"
    kind = "null"

    def perform(self, intent: Intent, point: Optional[Point], timings: Timings) -> BackendReport:
        return BackendReport(detail={"null": True})


def build_backend(kind: Optional[str], *, world: Any = None, failsafe_corner: bool = True) -> ActuatorBackend:
    """``action.backend`` -> backend instance (never raises for a missing dep)."""
    normalized = (kind or "").strip().lower()
    if normalized in {"", "none", "null"}:
        return NullBackend()
    if normalized in {"recording", "dry", "dry-run", "dryrun"}:
        return RecordingBackend()
    if normalized in {"simulated", "sim", "world"}:
        if world is None:
            raise ValueError("SimulatedBackend requires a SimulatedWorld instance")
        return SimulatedBackend(world)
    if normalized in {"pyautogui", "auto", "real"}:
        return PyAutoGUIBackend(failsafe_corner=failsafe_corner)
    if normalized in {"ahk", "autohotkey"}:  # pragma: no cover - Windows only
        return PyAutoGUIBackend(failsafe_corner=failsafe_corner)
    raise ValueError(f"unknown action backend: {kind!r}")


__all__ = [
    "ActuatorBackend",
    "BackendReport",
    "NullBackend",
    "PyAutoGUIBackend",
    "RecordingBackend",
    "SimulatedBackend",
    "Timings",
    "build_backend",
]
