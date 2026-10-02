"""Action Module (PRD section 7.6): actuator backends and guarded execution."""

from __future__ import annotations

from .backends import (
    ActuatorBackend,
    BackendReport,
    NullBackend,
    PyAutoGUIBackend,
    RecordingBackend,
    SimulatedBackend,
    Timings,
    build_backend,
)
from .module import ActionModule, ActionOutcome

__all__ = [
    "ActionModule",
    "ActionOutcome",
    "ActuatorBackend",
    "BackendReport",
    "NullBackend",
    "PyAutoGUIBackend",
    "RecordingBackend",
    "SimulatedBackend",
    "Timings",
    "build_backend",
]
