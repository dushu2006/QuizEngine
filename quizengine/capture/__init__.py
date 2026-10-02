"""Capture Module (PRD section 7.1)."""

from __future__ import annotations

from .backends import MssBackend, ReplayBackend, SyntheticBackend, build_backend, load_scenes, load_sequence
from .base import CaptureBackend, MonitorInfo
from .module import CaptureModule
from .ring_buffer import FrameRingBuffer, frame_memory_mb
from .validation import FrameValidator, LockScreenDetector, ValidityResult, frame_hash, luminance_stats, pixel_std

__all__ = [
    "CaptureBackend",
    "CaptureModule",
    "FrameRingBuffer",
    "FrameValidator",
    "LockScreenDetector",
    "MonitorInfo",
    "MssBackend",
    "ReplayBackend",
    "SyntheticBackend",
    "ValidityResult",
    "build_backend",
    "frame_hash",
    "frame_memory_mb",
    "load_scenes",
    "load_sequence",
    "luminance_stats",
    "pixel_std",
]
