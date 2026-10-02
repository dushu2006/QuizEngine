"""Capture backends (FR-7.1.1).

``MssBackend``      real screens, per-monitor, Windows/Linux/macOS.
``SyntheticBackend``renders :class:`~quizengine.scenes.Scene` fixtures -- the
                    headless CI path and the replay-harness world.
``ReplayBackend``   replays recorded PNG frames (+ JSON sidecars) from disk.

Backends are dumb: they return pixels and optional hints.  All validation,
hashing, freshness and buffering live in :class:`~quizengine.capture.module.CaptureModule`.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import CaptureConfig
from ..failures import CapabilityError
from ..geometry import Box, box_clip
from ..scenes import Scene, SceneSequence
from ..sim import SimulatedWorld
from .base import CaptureBackend, MonitorInfo


# --------------------------------------------------------------------------- #
# mss -- real screens
# --------------------------------------------------------------------------- #
class MssBackend(CaptureBackend):
    """Real screen capture via ``mss``.

    Monitor indices follow the ``mss`` convention: ``0`` is the virtual screen
    spanning every display, ``1..N`` are the physical monitors.
    """

    name = "mss"

    def __init__(self) -> None:
        try:
            import mss  # noqa: F401
        except ImportError as exc:  # pragma: no cover - capability guarded earlier
            raise CapabilityError(
                "capture.backend='mss' needs the 'mss' package: pip install 'quizengine[screen]'",
                detail={"capability": "mss"},
            ) from exc
        import mss

        self._mss = mss
        self._lock = threading.Lock()
        self._instance: Any = None

    def _sct(self) -> Any:
        if self._instance is None:
            self._instance = self._mss.mss()
        return self._instance

    def monitors(self) -> List[MonitorInfo]:
        with self._lock:
            raw = list(self._sct().monitors)
        infos: List[MonitorInfo] = []
        for index, monitor in enumerate(raw):
            infos.append(
                MonitorInfo(
                    index=index,
                    name="virtual-all" if index == 0 else f"monitor-{index}",
                    size_px=(int(monitor["width"]), int(monitor["height"])),
                    origin_px=(int(monitor["left"]), int(monitor["top"])),
                    dpi_scale=_windows_scale_factor(int(monitor["left"]), int(monitor["top"])),
                    is_primary=index == 1,
                    extra={"raw": dict(monitor)},
                )
            )
        return infos

    def _monitor_dict(self, monitor: int, region: Optional[Box]) -> Dict[str, int]:
        with self._lock:
            raw = list(self._sct().monitors)
        if not raw:  # pragma: no cover - no displays at all
            raise CapabilityError("no monitors reported by mss", detail={"capability": "mss"})
        index = monitor if 0 <= monitor < len(raw) else len(raw) - 1
        base = dict(raw[index])
        if region is not None:
            x, y, w, h = region
            return {
                "left": base["left"] + int(x),
                "top": base["top"] + int(y),
                "width": max(1, int(w)),
                "height": max(1, int(h)),
                "mon": index,
            }
        return base

    def grab(self, monitor: int = 0, region: Optional[Box] = None) -> np.ndarray:
        target = self._monitor_dict(monitor, region)
        with self._lock:
            shot = self._sct().grab(target)
        buffer = bytes(shot.rgb)
        array = np.frombuffer(buffer, dtype=np.uint8).reshape(int(shot.height), int(shot.width), 3)
        return np.ascontiguousarray(array)

    def display_state(self) -> Optional[int]:
        from .validation import LockScreenDetector

        return LockScreenDetector.windows_display_state()

    def foreground_window(self) -> Optional[str]:
        return _windows_foreground_title()

    def close(self) -> None:
        with self._lock:
            if self._instance is not None:
                try:
                    self._instance.close()
                finally:
                    self._instance = None


def _windows_scale_factor(x: int, y: int) -> float:
    """Per-monitor DPI scale on Windows; ``1.0`` elsewhere (L6: detect, never assume)."""
    import sys

    if not sys.platform.startswith("win"):
        return 1.0
    try:  # pragma: no cover - Windows only
        import ctypes

        class _POINT(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

        handle = ctypes.windll.user32.MonitorFromPoint(_POINT(x, y), 1)
        scale = ctypes.c_uint(100)
        result = ctypes.windll.shcore.GetScaleFactorForMonitor(handle, ctypes.byref(scale))
        return float(scale.value) / 100.0 if int(result) == 0 else 1.0
    except Exception:
        return 1.0


def _windows_foreground_title() -> Optional[str]:
    import sys

    if not sys.platform.startswith("win"):
        return None
    try:  # pragma: no cover - Windows only
        import ctypes

        hwnd = ctypes.windll.user32.GetForegroundWindow()
        if not hwnd:
            return None
        length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
        buffer = ctypes.create_unicode_buffer(length + 1)
        ctypes.windll.user32.GetWindowTextW(hwnd, buffer, length + 1)
        return buffer.value or None
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# synthetic -- rendered fixtures / replay harness world
# --------------------------------------------------------------------------- #
class SyntheticBackend(CaptureBackend):
    """Renders scenes to pixels.  Optionally backed by a live :class:`SimulatedWorld`."""

    name = "synthetic"

    def __init__(
        self,
        world: Optional[SimulatedWorld] = None,
        scenes: Optional[Sequence[Scene]] = None,
        *,
        monitor_name: str = "SIM-0",
        size_px: Optional[Tuple[int, int]] = None,
    ) -> None:
        if world is None:
            scene_list = list(scenes or [])
            if not scene_list:
                raise ValueError("SyntheticBackend needs either a SimulatedWorld or at least one Scene")
            world = SimulatedWorld(SceneSequence(name="inline", scenes=scene_list))
        self.world = world
        self._monitor_name = monitor_name
        first = world.current
        self._size = size_px or (first.width, first.height)

    def monitors(self) -> List[MonitorInfo]:
        # Report the geometry of the scene currently on screen: a real display can
        # change resolution mid-run and the capture module must be able to notice.
        current = getattr(self.world, "current", None)
        size = (int(current.width), int(current.height)) if current is not None else self._size
        return [
            MonitorInfo(index=0, name="virtual-all", size_px=size, is_primary=False),
            MonitorInfo(index=1, name=self._monitor_name, size_px=size, is_primary=True),
        ]

    def grab(self, monitor: int = 0, region: Optional[Box] = None) -> np.ndarray:
        pixels = self.world.render()
        if region is not None:
            from ..render import crop

            height, width = pixels.shape[:2]
            clipped = box_clip(region, width, height)
            pixels = crop(pixels, clipped)
        return pixels

    def annotation(self) -> Optional[Dict[str, Any]]:
        return self.world.annotation()

    def text_layer_hint(self) -> Optional[List[str]]:
        return self.world.text_layer()

    def describe(self) -> Dict[str, Any]:
        base = super().describe()
        base.update({"scenes": self.world.scene_count, "index": self.world.index})
        return base


# --------------------------------------------------------------------------- #
# replay -- recorded frames from disk
# --------------------------------------------------------------------------- #
class ReplayBackend(CaptureBackend):
    """Replays recorded PNG frames with optional ``.json`` annotation sidecars."""

    name = "replay"

    def __init__(self, frames_dir: str | Path, *, loop: bool = False) -> None:
        self.frames_dir = Path(frames_dir)
        if not self.frames_dir.exists():
            raise CapabilityError(
                f"capture.fixture_dir does not exist: {self.frames_dir}",
                detail={"path": str(self.frames_dir)},
            )
        self.paths: List[Path] = sorted(p for p in self.frames_dir.glob("*.png"))
        if not self.paths:
            raise CapabilityError(
                f"no recorded frames (*.png) in {self.frames_dir}", detail={"path": str(self.frames_dir)}
            )
        self.loop = loop
        self.cursor = 0
        self._cache: Dict[str, np.ndarray] = {}
        self._annotations: Dict[str, Optional[Dict[str, Any]]] = {}

    def _load(self, path: Path) -> np.ndarray:
        key = str(path)
        if key not in self._cache:
            from PIL import Image

            with Image.open(path) as image:
                self._cache[key] = np.asarray(image.convert("RGB"), dtype=np.uint8)
        return self._cache[key]

    def _load_annotation(self, path: Path) -> Optional[Dict[str, Any]]:
        key = str(path)
        if key not in self._annotations:
            sidecar = path.with_suffix(".json")
            if sidecar.exists():
                import json

                self._annotations[key] = json.loads(sidecar.read_text(encoding="utf-8"))
            else:
                self._annotations[key] = None
        return self._annotations[key]

    def monitors(self) -> List[MonitorInfo]:
        array = self._load(self.paths[0])
        height, width = array.shape[:2]
        return [
            MonitorInfo(index=0, name="replay-all", size_px=(width, height)),
            MonitorInfo(index=1, name="replay-0", size_px=(width, height), is_primary=True),
        ]

    def grab(self, monitor: int = 0, region: Optional[Box] = None) -> np.ndarray:
        if self.cursor >= len(self.paths):
            if not self.loop:
                return self._load(self.paths[-1])
            self.cursor = 0
        path = self.paths[self.cursor]
        self.cursor += 1
        pixels = self._load(path)
        if region is not None:
            from ..render import crop

            height, width = pixels.shape[:2]
            pixels = crop(pixels, box_clip(region, width, height))
        return pixels

    def annotation(self) -> Optional[Dict[str, Any]]:
        index = max(0, self.cursor - 1)
        if index >= len(self.paths):
            return None
        return self._load_annotation(self.paths[index])

    def text_layer_hint(self) -> Optional[List[str]]:
        annotation = self.annotation()
        if not annotation:
            return None
        return [str(b.get("text", "")) for b in annotation.get("text_blocks", [])]

    def reset(self) -> None:
        self.cursor = 0


# --------------------------------------------------------------------------- #
# factory
# --------------------------------------------------------------------------- #
def build_backend(config: CaptureConfig) -> CaptureBackend:
    """Instantiate the configured backend (the only legal way to get frames)."""
    if config.backend == "mss":
        from .. import capabilities

        capabilities.require("mss", "Install with: pip install 'quizengine[screen]'")
        return MssBackend()

    if not config.fixture_dir:
        raise CapabilityError(
            f"capture.backend='{config.backend}' requires capture.fixture_dir",
            detail={"backend": config.backend},
        )
    fixture_dir = Path(config.fixture_dir)

    if config.backend == "synthetic":
        return SyntheticBackend(scenes=load_scenes(fixture_dir))

    return ReplayBackend(fixture_dir)


def load_scenes(fixture_dir: str | Path) -> List[Scene]:
    """Load a scene sequence (``sequence.json``) or loose ``*.scene.json`` files."""
    root = Path(fixture_dir)
    if not root.exists():
        raise CapabilityError(f"fixture_dir does not exist: {root}", detail={"path": str(root)})
    sequence_path = root / "sequence.json"
    if sequence_path.exists():
        sequence = SceneSequence.load(sequence_path)
        if sequence.scenes:
            return list(sequence.scenes)
    scenes: List[Scene] = []
    for path in sorted(root.glob("*.json")):
        if path.name == "sequence.json":
            continue
        scenes.append(Scene.load(path))
    if not scenes:
        raise CapabilityError(f"no scene fixtures (*.json) found in {root}", detail={"path": str(root)})
    return scenes


def load_sequence(fixture_dir: str | Path) -> SceneSequence:
    root = Path(fixture_dir)
    sequence_path = root / "sequence.json"
    if sequence_path.exists():
        return SceneSequence.load(sequence_path)
    scenes = load_scenes(root)
    return SceneSequence(name=root.name, scenes=scenes)


__all__ = [
    "MssBackend",
    "SyntheticBackend",
    "ReplayBackend",
    "build_backend",
    "load_scenes",
    "load_sequence",
]
