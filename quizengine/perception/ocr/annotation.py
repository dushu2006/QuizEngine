"""Fixture-backed OCR adapter -- the deterministic CI path (section 14, layer 2).

This engine reads the ground-truth text layer that ships with synthetic/replay
fixtures instead of running OCR on pixels.  Two honesty rules keep it from
becoming a cheat vector:

1. It exposes **only** ``text``, ``box`` and ``confidence``.  Fixture roles,
   handles and option indices are deliberately dropped, so region segmentation
   and option localization still have to be inferred from pixels exactly as they
   would be in production (FR-7.2.3, FR-7.2.4).
2. :func:`quizengine.perception.ocr.build_ocr_engine` refuses to hand this engine
   a real (``mss``) capture backend, and ``pixel_based`` is ``False`` so reports
   can always tell which path produced a result.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ...contracts import TextBlock
from ...geometry import as_box
from .base import OCREngine, OCRRequest


class AnnotationOCR(OCREngine):
    name = "annotation"
    pixel_based = False

    def __init__(self, default_confidence: float = 0.97) -> None:
        self.default_confidence = float(default_confidence)
        self.missing_annotations = 0

    def read(self, request: OCRRequest) -> List[TextBlock]:
        annotation = request.annotation
        if not annotation:
            self.missing_annotations += 1
            return []
        blocks: List[TextBlock] = []
        for entry in annotation.get("text_blocks", []) or []:
            text = str(entry.get("text", "")).strip()
            if not text:
                continue
            try:
                box = as_box(entry.get("box", (0, 0, 0, 0)))
            except ValueError:
                continue
            blocks.append(
                TextBlock(
                    text=text,
                    box=box,
                    confidence=float(entry.get("confidence", self.default_confidence)),
                    source=self.name,
                )
            )
        return blocks

    @classmethod
    def available(cls) -> bool:
        return True

    def describe(self) -> Dict[str, Any]:
        base = super().describe()
        base["missing_annotations"] = self.missing_annotations
        return base
