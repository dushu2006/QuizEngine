"""OCR engine registry and factory (FR-7.2.1)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type

from ...config import PerceptionConfig
from ...failures import CapabilityError
from .annotation import AnnotationOCR
from .base import NullOCR, OCREngine, OCRRequest, filter_blocks
from .easyocr import EasyOCR
from .tesseract import TesseractOCR

#: Fixture backends are the only ones allowed to use the annotation engine.
FIXTURE_BACKENDS = frozenset({"synthetic", "replay"})

_REGISTRY: Dict[str, Type[OCREngine]] = {
    TesseractOCR.name: TesseractOCR,
    EasyOCR.name: EasyOCR,
    AnnotationOCR.name: AnnotationOCR,
    NullOCR.name: NullOCR,
}

_INSTALL_HINTS = {
    "tesseract": "pip install 'quizengine[vision]' and install the tesseract-ocr binary",
    "easyocr": "pip install 'quizengine[easyocr]' (pulls PyTorch)",
    "annotation": "only valid with capture.backend=synthetic|replay",
    "none": "always available",
}


def register(engine_class: Type[OCREngine]) -> Type[OCREngine]:
    """Extension point: third parties may add OCR adapters without editing Tier 1."""
    _REGISTRY[engine_class.name] = engine_class
    return engine_class


def available_engines() -> Dict[str, bool]:
    return {name: cls.available() for name, cls in _REGISTRY.items()}


def _instantiate(name: str, config: PerceptionConfig) -> OCREngine:
    if name == TesseractOCR.name:
        return TesseractOCR(lang=config.ocr_lang)
    if name == EasyOCR.name:
        return EasyOCR(langs=(config.ocr_lang or "en",))
    if name == AnnotationOCR.name:
        return AnnotationOCR()
    return NullOCR()


def resolve_chain(config: PerceptionConfig, capture_backend: str) -> List[str]:
    """Ordered candidate engines: configured primary, then the fallback."""
    chain = [config.ocr_engine]
    if config.ocr_fallback_engine and config.ocr_fallback_engine not in chain:
        chain.append(config.ocr_fallback_engine)
    return chain


def build_ocr_engine(
    config: PerceptionConfig,
    capture_backend: str,
    *,
    notes: Optional[List[str]] = None,
) -> Tuple[OCREngine, List[str]]:
    """Pick the first usable engine, or raise an actionable CapabilityError.

    ``notes`` records *why* each candidate was skipped -- surfaced in the run
    report and in ``quizengine doctor`` so a silent degradation never happens.
    """
    trail: List[str] = []
    for name in resolve_chain(config, capture_backend):
        engine_class = _REGISTRY.get(name)
        if engine_class is None:
            trail.append(f"{name}: unknown engine")
            continue
        if name == AnnotationOCR.name and capture_backend not in FIXTURE_BACKENDS:
            trail.append(
                f"{name}: refused -- ground-truth OCR is only valid for fixture backends "
                f"(capture.backend='{capture_backend}')"
            )
            continue
        if not engine_class.available():
            trail.append(f"{name}: unavailable ({_INSTALL_HINTS.get(name, '')})")
            continue
        engine = _instantiate(name, config)
        trail.append(f"{name}: selected")
        if notes is not None:
            notes.extend(trail)
        return engine, trail

    trail.append(
        "no OCR engine available; Tier-1 perception will be structure-only and Tier-2 must carry content"
    )
    if config.ocr_engine != "none":
        raise CapabilityError(
            "no usable OCR engine: " + "; ".join(trail),
            detail={"tried": trail, "capture_backend": capture_backend},
        )
    if notes is not None:
        notes.extend(trail)
    return NullOCR(), trail


def describe_engines() -> List[Dict[str, Any]]:
    return [
        {"name": name, "class": cls.__name__, "available": cls.available(), "install_hint": _INSTALL_HINTS.get(name, "")}
        for name, cls in _REGISTRY.items()
    ]


__all__ = [
    "AnnotationOCR",
    "EasyOCR",
    "NullOCR",
    "OCREngine",
    "OCRRequest",
    "TesseractOCR",
    "available_engines",
    "build_ocr_engine",
    "describe_engines",
    "filter_blocks",
    "register",
    "resolve_chain",
    "FIXTURE_BACKENDS",
]
