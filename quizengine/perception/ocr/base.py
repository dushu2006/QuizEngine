"""OCR engine interface (FR-7.2.1: implementations MUST be swappable).

Contract: an engine is handed a *preprocessed* image plus the geometry mapping
back to the source frame, and returns :class:`~quizengine.contracts.TextBlock`
objects **in frame coordinates**.  Engines never see raw captures -- the
FR-7.2.2 preprocessing pipeline runs first, in Tier 1.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ...contracts import Frame, TextBlock
from ...geometry import Box
from ..preprocessing import PreprocessResult


@dataclass
class OCRRequest:
    """Everything an OCR engine needs, and nothing it should not have."""

    frame: Frame
    image: np.ndarray
    preprocess: Optional[PreprocessResult] = None
    lang: str = "eng"
    region: Optional[Box] = None  # frame-space ROI, informational
    hints: Dict[str, Any] = field(default_factory=dict)

    def unmap(self, box: Box) -> Box:
        """Map a box from preprocessed pixels back to frame pixels."""
        if self.preprocess is None:
            return box
        return self.preprocess.unmap_box(box)

    @property
    def annotation(self) -> Optional[Dict[str, Any]]:
        """Fixture ground truth, present only for synthetic/replay backends."""
        return self.frame.backend_meta.get("fixture_annotation")


class OCREngine(abc.ABC):
    """Swappable OCR backend."""

    name: str = "abstract"
    #: Whether this engine can read text from *pixels alone* (production capable).
    pixel_based: bool = True

    @abc.abstractmethod
    def read(self, request: OCRRequest) -> List[TextBlock]:
        """Return text blocks in **frame coordinates**."""

    @classmethod
    @abc.abstractmethod
    def available(cls) -> bool:
        """True when the engine can actually run on this machine."""

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "available": self.available(), "pixel_based": self.pixel_based}


class NullOCR(OCREngine):
    """Returns nothing.  Used when Tier 2 alone provides structure."""

    name = "none"
    pixel_based = False

    def read(self, request: OCRRequest) -> List[TextBlock]:
        return []

    @classmethod
    def available(cls) -> bool:
        return True


def filter_blocks(blocks: List[TextBlock], min_confidence: float) -> List[TextBlock]:
    """Drop low-confidence noise, keeping the frame-space ordering stable."""
    return [b for b in blocks if b.confidence >= min_confidence and b.text.strip()]
