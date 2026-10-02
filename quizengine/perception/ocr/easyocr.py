"""EasyOCR fallback adapter (FR-7.2.1: implementations MUST be swappable).

EasyOCR pulls in PyTorch, so it is an optional extra.  The reader is constructed
lazily and cached process-wide: model load dominates the first call and must not
be paid per frame.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ...contracts import TextBlock
from ...geometry import Box
from .base import OCREngine, OCRRequest

_reader_cache: Dict[str, Any] = {}


class EasyOCR(OCREngine):
    name = "easyocr"
    pixel_based = True

    def __init__(self, langs: Tuple[str, ...] = ("en",), gpu: bool = False, min_confidence: float = 0.1) -> None:
        self.langs = tuple(langs)
        self.gpu = bool(gpu)
        self.min_confidence = float(min_confidence)

    @classmethod
    def available(cls) -> bool:
        try:
            import easyocr  # noqa: F401

            return True
        except Exception:
            return False

    def _reader(self) -> Any:
        key = ",".join(self.langs) + ("|gpu" if self.gpu else "|cpu")
        reader = _reader_cache.get(key)
        if reader is None:
            import easyocr

            reader = easyocr.Reader(list(self.langs), gpu=self.gpu, verbose=False)
            _reader_cache[key] = reader
        return reader

    def read(self, request: OCRRequest) -> List[TextBlock]:
        import numpy as np

        image = np.asarray(request.image)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        results = self._reader().readtext(image, detail=1, paragraph=False)
        blocks: List[TextBlock] = []
        for entry in results:
            try:
                quad, text, confidence = entry[0], str(entry[1]).strip(), float(entry[2])
            except (IndexError, TypeError, ValueError):
                continue
            if not text or confidence < self.min_confidence:
                continue
            blocks.append(
                TextBlock(
                    text=text,
                    box=request.unmap(_box_from_quad(quad)),
                    confidence=max(0.0, min(1.0, confidence)),
                    source=self.name,
                )
            )
        return blocks

    def describe(self) -> Dict[str, Any]:
        base = super().describe()
        base.update({"langs": list(self.langs), "gpu": self.gpu})
        return base


def _box_from_quad(quad: Any) -> Box:
    points: List[Tuple[float, float]] = [(float(p[0]), float(p[1])) for p in quad]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    left, top = int(round(min(xs))), int(round(min(ys)))
    return (left, top, int(round(max(xs))) - left, int(round(max(ys))) - top)


def clear_reader_cache() -> None:  # pragma: no cover - test hygiene
    _reader_cache.clear()


__all__ = ["EasyOCR", "clear_reader_cache"]
