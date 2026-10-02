"""Capture backend interface (FR-7.1.1, AC-7.1.2).

**[AC-7.1.2]**: zero modules may access raw screen-capture APIs directly.  Every
frame in the system comes from an implementation of :class:`CaptureBackend`,
obtained through :func:`quizengine.capture.build_backend`.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..geometry import Box


@dataclass(frozen=True)
class MonitorInfo:
    """One physical display as reported by the backend."""

    index: int
    name: str
    size_px: Tuple[int, int]
    origin_px: Tuple[int, int] = (0, 0)
    dpi_scale: float = 1.0
    is_primary: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> int:
        return int(self.size_px[0])

    @property
    def height(self) -> int:
        return int(self.size_px[1])


class CaptureBackend(abc.ABC):
    """Produces raw pixel buffers.  Validation lives in the CaptureModule."""

    #: Stable identifier recorded in ``Frame.backend``.
    name: str = "abstract"

    @abc.abstractmethod
    def monitors(self) -> List[MonitorInfo]:
        """Enumerate displays (single or multi-monitor, section 3.1)."""

    @abc.abstractmethod
    def grab(self, monitor: int = 0, region: Optional[Box] = None) -> np.ndarray:
        """Return an ``H x W x 3`` uint8 RGB array for the monitor/region."""

    # -- optional hints ---------------------------------------------------- #
    def display_state(self) -> Optional[int]:
        """OS display/user-notification state, if the platform exposes it.

        ``None`` means "unknown".  On Windows this maps to
        ``SHQueryUserNotificationState`` (see :mod:`quizengine.capture.validation`).
        """
        return None

    def text_layer_hint(self) -> Optional[List[str]]:
        """Text the backend already knows is on screen.

        Only ever non-empty for synthetic/replay backends, which ship ground
        truth with the fixture.  A real backend returns ``None`` and lock-screen
        detection falls back to the OS display state + pixel signature.
        """
        return None

    def annotation(self) -> Optional[Dict[str, Any]]:
        """Ground-truth annotation for the current screen (fixtures only)."""
        return None

    def foreground_window(self) -> Optional[str]:
        """Title of the foreground window, when the platform exposes it (FR-7.6.4)."""
        return None

    def close(self) -> None:  # pragma: no cover - default no-op
        return None

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "monitors": [m.__dict__ for m in self.monitors()]}
