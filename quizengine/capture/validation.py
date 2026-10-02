"""Capture validation (FR-7.1.2) and frame hashing (FR-7.1.4).

Every frame passes through :meth:`FrameValidator.validate` before anything else
in the system is allowed to look at it.  Five rejection modes exist:

``null``         backend returned nothing / an empty buffer
``dimensions``   size outside tolerance of the monitor config (L6: never assume)
``blank``        uniform frame -- std-dev of pixels below threshold
``stale``        byte-identical to the previous frame *when change was expected*
``lock_screen``  lock screen / screensaver / secure desktop signature -> treated
                 as an environment event, not a capture retry (section 7.14)
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..config import CaptureConfig, SafetyConfig
from ..geometry import Box

ValidityReason = Optional[str]


def frame_hash(pixels: Optional[np.ndarray]) -> str:
    """Content digest formatted ``sha1:<hex>`` (FR-7.1.4)."""
    if pixels is None:
        return "sha1:" + "0" * 40
    array = np.ascontiguousarray(pixels)
    digest = hashlib.sha1(array.tobytes()).hexdigest()
    return f"sha1:{digest}"


def pixel_std(pixels: np.ndarray) -> float:
    """Mean per-channel standard deviation -- the uniform-blank detector."""
    if pixels is None or pixels.size == 0:
        return 0.0
    array = pixels.astype(np.float32)
    if array.ndim == 2:
        return float(array.std())
    return float(np.mean([array[:, :, c].std() for c in range(array.shape[2])]))


def luminance_stats(pixels: np.ndarray) -> Dict[str, float]:
    if pixels is None or pixels.size == 0:
        return {"mean": 0.0, "std": 0.0, "bright_ratio": 0.0}
    array = pixels.astype(np.float32)
    if array.ndim == 3:
        lum = array @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    else:
        lum = array
    return {
        "mean": float(lum.mean()),
        "std": float(lum.std()),
        "bright_ratio": float((lum > 170).mean()),
    }


@dataclass
class ValidityResult:
    valid: bool
    reason: ValidityReason = None
    detail: str = ""
    std_dev: float = 0.0
    stats: Dict[str, float] = field(default_factory=dict)
    hash: str = ""

    def as_event_detail(self) -> Dict[str, Any]:
        return {
            "reason": self.reason,
            "detail": self.detail,
            "std_dev": round(self.std_dev, 3),
            "hash": self.hash[:16],
        }


class LockScreenDetector:
    """Detects a lock screen / screensaver / secure desktop (FR-7.1.2).

    Three independent signals, strongest first:

    1. **OS display state** -- on Windows, ``SHQueryUserNotificationState``.
       ``QUNS_NOT_PRESENT`` (1) means the user session is not interactive
       (locked / secure desktop / fast-user-switch).
    2. **Text signature** -- configured strings matched against a text layer the
       backend can supply (fixtures) or against OCR results attached later.
    3. **Pixel signature** -- a near-black, low-structure frame with a small
       bright region (the clock).  Deliberately conservative: it must be *very*
       dark, or a dark-themed quiz would false-positive.
    """

    QUNS_NOT_PRESENT = 1
    QUNS_BUSY = 2

    def __init__(self, signatures: List[str], halt_enabled: bool = True) -> None:
        self.signatures = [s.strip().lower() for s in signatures if s.strip()]
        self.halt_enabled = halt_enabled

    # -- signals ----------------------------------------------------------- #
    @staticmethod
    def windows_display_state() -> Optional[int]:
        """``SHQueryUserNotificationState`` on Windows, else ``None``."""
        if not sys.platform.startswith("win"):  # pragma: no cover - platform specific
            return None
        try:  # pragma: no cover - exercised only on Windows
            import ctypes

            state = ctypes.c_int(0)
            result = ctypes.windll.shell32.SHQueryUserNotificationState(ctypes.byref(state))
            return int(state.value) if int(result) == 0 else None
        except Exception:
            return None

    def match_text(self, texts: Optional[List[str]]) -> Optional[str]:
        if not texts:
            return None
        haystack = " | ".join(str(t).lower() for t in texts)
        for signature in self.signatures:
            if signature and signature in haystack:
                return signature
        return None

    @staticmethod
    def pixel_signature(pixels: np.ndarray) -> Tuple[bool, Dict[str, float]]:
        stats = luminance_stats(pixels)
        strong = stats["mean"] < 28.0 and stats["std"] < 34.0 and 0.0005 < stats["bright_ratio"] < 0.10
        return strong, stats

    # -- verdict ----------------------------------------------------------- #
    def detect(
        self,
        pixels: np.ndarray,
        *,
        text_layer: Optional[List[str]] = None,
        display_state: Optional[int] = None,
    ) -> Optional[str]:
        """Return the matched signature/description, or ``None`` when clear."""
        if not self.halt_enabled:
            return None
        state = display_state if display_state is not None else self.windows_display_state()
        if state == self.QUNS_NOT_PRESENT:
            return "display_state=QUNS_NOT_PRESENT (session locked / secure desktop)"
        matched = self.match_text(text_layer)
        if matched:
            return f"lock-screen signature '{matched}'"
        strong, _stats = self.pixel_signature(pixels)
        if strong:
            return "lock-screen pixel signature (near-black frame with isolated bright region)"
        return None


class FrameValidator:
    """FR-7.1.2 validity checks + FR-7.1.5 freshness bookkeeping."""

    def __init__(self, capture_config: CaptureConfig, safety_config: Optional[SafetyConfig] = None) -> None:
        self.config = capture_config
        self.safety = safety_config or SafetyConfig()
        self.lock_detector = LockScreenDetector(
            list(capture_config.lock_screen_signatures),
            halt_enabled=bool(self.safety.halt_on_lock_screen),
        )
        self.stats: Dict[str, int] = {
            "accepted": 0,
            "null": 0,
            "dimensions": 0,
            "blank": 0,
            "stale": 0,
            "lock_screen": 0,
        }

    # -- main entry -------------------------------------------------------- #
    def validate(
        self,
        pixels: Optional[np.ndarray],
        expected_size: Optional[Tuple[int, int]] = None,
        *,
        previous_hash: Optional[str] = None,
        expect_change: bool = False,
        text_layer: Optional[List[str]] = None,
        display_state: Optional[int] = None,
    ) -> ValidityResult:
        # 1. non-null
        if pixels is None:
            return self._reject("null", "backend returned no pixel buffer")
        array = np.asarray(pixels)
        if array.size == 0:
            return self._reject("null", "backend returned an empty buffer")
        if array.ndim not in (2, 3):
            return self._reject("null", f"unexpected buffer shape {array.shape}")

        digest = frame_hash(array)
        std = pixel_std(array)
        stats = luminance_stats(array)

        # 2. expected dimensions within tolerance of the monitor config
        if expected_size is not None:
            height, width = array.shape[:2]
            tol = int(self.config.dimension_tolerance_px)
            if abs(width - int(expected_size[0])) > tol or abs(height - int(expected_size[1])) > tol:
                return self._fail(
                    "dimensions",
                    f"got {width}x{height}, expected {expected_size[0]}x{expected_size[1]} (+-{tol}px)",
                    std,
                    stats,
                    digest,
                )

        # 3. not uniformly blank
        if std <= float(self.config.blank_std_threshold):
            return self._fail(
                "blank",
                f"pixel std-dev {std:.3f} <= {self.config.blank_std_threshold} (uniform frame)",
                std,
                stats,
                digest,
            )

        # 4. lock screen / screensaver / secure desktop
        signature = self.lock_detector.detect(array, text_layer=text_layer, display_state=display_state)
        if signature is not None:
            return self._fail("lock_screen", signature, std, stats, digest)

        # 5. staleness -- only when a change was expected (section 10)
        if self.config.staleness_check and expect_change and previous_hash and digest == previous_hash:
            return self._fail(
                "stale",
                f"frame identical to previous ({digest[:16]}) although a change was expected",
                std,
                stats,
                digest,
            )

        self.stats["accepted"] += 1
        return ValidityResult(valid=True, reason=None, detail="ok", std_dev=std, stats=stats, hash=digest)

    def _reject(self, reason: str, detail: str) -> ValidityResult:
        return self._fail(reason, detail, 0.0, {}, "")

    def _fail(
        self,
        reason: str,
        detail: str,
        std_dev: float,
        stats: Dict[str, float],
        digest: str,
    ) -> ValidityResult:
        """Single rejection path so ``self.stats`` always matches reality."""
        self.stats[reason] = self.stats.get(reason, 0) + 1
        return ValidityResult(
            valid=False, reason=reason, detail=detail, std_dev=std_dev, stats=stats, hash=digest
        )

    # -- freshness (FR-7.1.5) ---------------------------------------------- #
    def is_fresh(self, ts: float, now: float) -> bool:
        age_ms = max(0.0, (now - ts) * 1000.0)
        return age_ms <= self.config.max_frame_age_ms

    def age_ms(self, ts: float, now: float) -> float:
        return max(0.0, (now - ts) * 1000.0)

    # -- zoom crop (FR-7.1.6) ---------------------------------------------- #
    def zoom_box(self, box: Box, frame_size: Tuple[int, int], scale: Optional[float] = None) -> Box:
        """Expand ``box`` about its centre by the zoom factor, clipped to frame."""
        from ..geometry import box_clip, box_pad_relative

        factor = (scale if scale is not None else self.config.zoom_crop_scale) - 1.0
        padded = box_pad_relative(box, factor / 2.0, factor / 2.0)
        return box_clip(padded, int(frame_size[0]), int(frame_size[1]))


__all__ = [
    "FrameValidator",
    "LockScreenDetector",
    "ValidityResult",
    "frame_hash",
    "pixel_std",
    "luminance_stats",
]
