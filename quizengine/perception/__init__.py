"""Perception Module (PRD section 7.2)."""

from __future__ import annotations

from .localization import LocalizationResult, OptionLocator, classify_layout
from .module import PerceptionModule, PerceptionOutcome
from .ocr import (
    AnnotationOCR,
    EasyOCR,
    NullOCR,
    OCREngine,
    OCRRequest,
    TesseractOCR,
    available_engines,
    build_ocr_engine,
    describe_engines,
)
from .pixels import diff_ratio, dominant_color, infer_selected_markers, mean_shift_vector
from .preprocessing import PreprocessResult, preprocess, prepare_for_ocr
from .reconcile import Reconciler, ReconciliationResult, normalize
from .segmentation import RegionSegmenter, SegmentationContext, font_size_clusters, whitespace_bands
from .tier1 import Tier1Perception, Tier1Result
from .tier2 import Tier2Analysis, Tier2Outcome, Tier2Perception

__all__ = [
    "AnnotationOCR",
    "EasyOCR",
    "LocalizationResult",
    "NullOCR",
    "OCREngine",
    "OCRRequest",
    "OptionLocator",
    "PerceptionModule",
    "PerceptionOutcome",
    "PreprocessResult",
    "Reconciler",
    "ReconciliationResult",
    "RegionSegmenter",
    "SegmentationContext",
    "Tier1Perception",
    "Tier1Result",
    "Tier2Analysis",
    "Tier2Outcome",
    "Tier2Perception",
    "TesseractOCR",
    "available_engines",
    "build_ocr_engine",
    "classify_layout",
    "describe_engines",
    "diff_ratio",
    "dominant_color",
    "font_size_clusters",
    "infer_selected_markers",
    "mean_shift_vector",
    "normalize",
    "preprocess",
    "prepare_for_ocr",
    "whitespace_bands",
]
