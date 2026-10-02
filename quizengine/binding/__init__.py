"""Element Binding & Re-resolution Module (PRD section 7.7)."""

from __future__ import annotations

from .module import (
    NAV_HANDLES,
    OPTION_HANDLE_PREFIX,
    OVERLAY_CLOSE_HANDLE,
    BindingResult,
    ElementResolver,
    handle_index,
    option_handle,
)

__all__ = [
    "NAV_HANDLES",
    "OPTION_HANDLE_PREFIX",
    "OVERLAY_CLOSE_HANDLE",
    "BindingResult",
    "ElementResolver",
    "handle_index",
    "option_handle",
]
