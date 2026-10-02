"""Tesseract OCR adapter -- the FR-7.2.1 default (PSM 6).

Word boxes are grouped into text lines, then mapped back into frame coordinates
through the preprocessing affine, so hit areas line up with what is on screen.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...contracts import TextBlock
from ...geometry import Box, box_union
from .base import OCREngine, OCRRequest


class TesseractOCR(OCREngine):
    name = "tesseract"
    pixel_based = True

    def __init__(self, lang: str = "eng", psm: int = 6, min_word_confidence: float = 0.0) -> None:
        self.lang = lang
        self.psm = int(psm)
        self.min_word_confidence = float(min_word_confidence)
        self._version: Optional[str] = None

    @classmethod
    def available(cls) -> bool:
        try:
            import pytesseract

            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    def version(self) -> Optional[str]:
        if self._version is None:
            try:
                import pytesseract

                self._version = str(pytesseract.get_tesseract_version())
            except Exception:
                self._version = None
        return self._version

    # -- reading ------------------------------------------------------------ #
    def read(self, request: OCRRequest) -> List[TextBlock]:
        import pytesseract
        from PIL import Image

        image = _as_grayscale_pil(request.image)
        lang = request.lang or self.lang
        try:
            data = pytesseract.image_to_data(
                image, lang=lang, config=f"--psm {self.psm}", output_type=pytesseract.Output.DICT
            )
        except pytesseract.TesseractError as exc:
            raise RuntimeError(f"tesseract failed: {exc}") from exc

        lines: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
        count = len(data.get("text", []))
        for index in range(count):
            word = str(data["text"][index]).strip()
            if not word:
                continue
            try:
                confidence = float(data["conf"][index])
            except (TypeError, ValueError):
                confidence = -1.0
            if confidence < 0 or confidence < self.min_word_confidence:
                continue
            key = (int(data["block_num"][index]), int(data["par_num"][index]), int(data["line_num"][index]))
            entry = lines.setdefault(key, {"words": [], "box": None, "confs": []})
            box = (
                int(data["left"][index]),
                int(data["top"][index]),
                int(data["width"][index]),
                int(data["height"][index]),
            )
            entry["words"].append(word)
            entry["confs"].append(confidence / 100.0)
            entry["box"] = box if entry["box"] is None else box_union(entry["box"], box)

        blocks: List[TextBlock] = []
        for key in sorted(lines):
            entry = lines[key]
            box = entry["box"]
            if box is None:
                continue
            text = " ".join(entry["words"]).strip()
            if not text:
                continue
            confidences = entry["confs"] or [0.0]
            blocks.append(
                TextBlock(
                    text=text,
                    box=request.unmap(box),
                    confidence=float(max(0.0, min(1.0, sum(confidences) / len(confidences)))),
                    source=self.name,
                )
            )
        return blocks

    def describe(self) -> Dict[str, Any]:
        base = super().describe()
        base.update({"lang": self.lang, "psm": self.psm, "version": self.version()})
        return base


def _as_grayscale_pil(image: np.ndarray) -> Any:
    from PIL import Image

    array = np.asarray(image)
    if array.ndim == 3:
        array = array[:, :, 0] if array.shape[2] == 1 else array
    if array.ndim == 3:
        return Image.fromarray(array.astype(np.uint8), mode="RGB").convert("L")
    return Image.fromarray(array.astype(np.uint8), mode="L")


def _box_from_points(points: List[Tuple[float, float]]) -> Box:  # pragma: no cover - shared helper
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    left, top = int(round(min(xs))), int(round(min(ys)))
    return (left, top, int(round(max(xs))) - left, int(round(max(ys))) - top)
