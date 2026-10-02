"""Extraction & Validation Module (PRD section 7.3)."""

from __future__ import annotations

from .module import (
    ExtractionModule,
    ExtractionOutcome,
    content_hash,
    normalize_text,
    question_hash,
)

__all__ = ["ExtractionModule", "ExtractionOutcome", "content_hash", "normalize_text", "question_hash"]
