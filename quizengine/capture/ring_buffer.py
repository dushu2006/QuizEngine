"""Frame ring buffer for post-failure forensics (FR-7.1.7, FR-15.2).

Memory-capped: default 20 frames with a ~200 MB ceiling.  When a single frame is
larger than the per-frame share of the ceiling, the *buffered copy* is downscaled
(adaptive resolution scaling) while the live frame keeps full resolution.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from ..contracts import Frame


def frame_memory_mb(frame: Frame) -> float:
    if frame.pixels is None:
        return 0.0
    array = np.asarray(frame.pixels)
    return (array.nbytes / (1024.0 * 1024.0)) if array.size else 0.0


class FrameRingBuffer:
    def __init__(self, max_frames: int = 20, max_memory_mb: float = 200.0) -> None:
        self.max_frames = max(1, int(max_frames))
        self.max_memory_mb = float(max_memory_mb)
        self._frames: Deque[Frame] = deque(maxlen=self.max_frames)
        self._evicted = 0
        self._downscaled = 0

    # -- writes ------------------------------------------------------------ #
    def push(self, frame: Frame) -> Optional[Frame]:
        """Add a frame; returns the frame evicted to stay within caps (if any)."""
        evicted: Optional[Frame] = self._frames[0] if len(self._frames) == self.max_frames else None
        stored = self._fit_memory(frame)
        self._frames.append(stored)
        if evicted is not None:
            self._evicted += 1
        self._enforce_caps()
        return evicted

    def _enforce_caps(self) -> None:
        while len(self._frames) > 1 and self.memory_mb > self.max_memory_mb:
            self._frames.popleft()
            self._evicted += 1

    def _fit_memory(self, frame: Frame) -> Frame:
        """Downscale the buffered copy when a frame would blow the ceiling."""
        if frame.pixels is None:
            return frame
        share_mb = self.max_memory_mb / self.max_frames
        size_mb = frame_memory_mb(frame)
        if size_mb <= share_mb * 2.0 or size_mb <= self.max_memory_mb:
            return frame
        factor = max(0.1, min(1.0, (share_mb / max(size_mb, 1e-6)) ** 0.5))
        try:
            from PIL import Image

            array = np.asarray(frame.pixels)
            height, width = array.shape[:2]
            image = Image.fromarray(array)
            small = image.resize((max(1, int(width * factor)), max(1, int(height * factor))), Image.Resampling.BILINEAR)
            clone = frame.model_copy(update={"pixels": np.asarray(small, dtype=np.uint8)})
            meta: Dict[str, Any] = dict(clone.backend_meta)
            meta["buffer_downscale"] = round(factor, 4)
            clone.backend_meta = meta
            self._downscaled += 1
            return clone
        except Exception:
            return frame

    # -- reads ------------------------------------------------------------- #
    def last(self, count: Optional[int] = None) -> List[Frame]:
        frames = list(self._frames)
        return frames if count is None else frames[-int(count) :]

    def latest(self) -> Optional[Frame]:
        return self._frames[-1] if self._frames else None

    def previous(self) -> Optional[Frame]:
        """The frame before the most recent one (used for staleness checks)."""
        return self._frames[-2] if len(self._frames) >= 2 else None

    def by_seq(self, seq: int) -> Optional[Frame]:
        for frame in self._frames:
            if frame.seq == seq:
                return frame
        return None

    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self) -> Any:
        return iter(list(self._frames))

    @property
    def memory_mb(self) -> float:
        return sum(frame_memory_mb(frame) for frame in self._frames)

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "frames": len(self._frames),
            "memory_mb": round(self.memory_mb, 2),
            "max_frames": self.max_frames,
            "max_memory_mb": self.max_memory_mb,
            "evicted": self._evicted,
            "downscaled": self._downscaled,
        }

    def clear(self) -> None:
        self._frames.clear()
