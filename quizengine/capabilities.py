"""Optional-capability detection.

QuizEngine targets Windows 10/11 with real screens, real OCR engines and real
actuators.  It must also run -- and be *tested* -- on a headless CI box.  Rather
 than sprinkling ``try: import x`` through the modules, capability probing lives
here so every degradation is explicit, logged once, and visible in
``quizengine doctor``.
"""

from __future__ import annotations

import importlib.util
import shutil
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Capability:
    name: str
    available: bool
    detail: str = ""
    #: Which PRD requirement degrades when this is missing.
    affects: str = ""


@lru_cache(maxsize=None)
def _module_present(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


@lru_cache(maxsize=None)
def _binary_present(binary: str) -> Optional[str]:
    return shutil.which(binary)


def _check(name: str, module: str, affects: str, binary: Optional[str] = None) -> Capability:
    if not _module_present(module):
        return Capability(name, False, f"python package '{module}' not installed", affects)
    if binary is not None:
        path = _binary_present(binary)
        if path is None:
            return Capability(name, False, f"'{module}' installed but '{binary}' binary not on PATH", affects)
        return Capability(name, True, f"{module} + {path}", affects)
    try:
        mod = importlib.import_module(module)
        version = getattr(mod, "__version__", "")
    except Exception as exc:  # pragma: no cover - defensive
        return Capability(name, False, f"import failed: {exc}", affects)
    return Capability(name, True, f"{module} {version}".strip(), affects)


def probe() -> Dict[str, Capability]:
    """Probe every optional backend once."""
    caps: Dict[str, Capability] = {}
    caps["mss"] = _check("mss", "mss", "FR-7.1.1 real screen capture")
    caps["opencv"] = _check("opencv", "cv2", "FR-7.2.2 preprocessing + FR-7.7.2 template match")
    caps["tesseract"] = _check("tesseract", "pytesseract", "FR-7.2.1 default OCR engine", binary="tesseract")
    caps["easyocr"] = _check("easyocr", "easyocr", "FR-7.2.1 fallback OCR adapter")
    caps["pyautogui"] = _check("pyautogui", "pyautogui", "FR-7.6.1 default actuator backend")
    caps["httpx"] = _check("httpx", "httpx", "FR-6.2 cloud/local model provider transport")
    caps["flask"] = _check("flask", "flask", "section 13 QuizForge mock platform")
    caps["psutil"] = _check("psutil", "psutil", "FR-7.14.2 environment scan")
    caps["pillow"] = _check("pillow", "PIL", "frame rasterization / fixture rendering")
    caps["yaml"] = _check("yaml", "yaml", "section 12 YAML config")
    caps["display"] = Capability(
        "display",
        bool(shutil.which("xdpyinfo") or _has_windows_display()),
        "an interactive display is required for real runs (section 3.2: no headless operation)",
    )
    return caps


def _has_windows_display() -> bool:
    import sys

    return sys.platform.startswith(("win", "darwin", "cygwin"))


@lru_cache(maxsize=None)
def capabilities() -> Dict[str, Capability]:
    return probe()


def has(name: str) -> bool:
    return capabilities().get(name, Capability(name, False)).available


def missing(names: List[str]) -> List[str]:
    return [n for n in names if not has(n)]


def require(name: str, hint: str = "") -> None:
    """Raise a CapabilityError with an actionable install hint."""
    from .failures import CapabilityError

    cap = capabilities().get(name)
    if cap is not None and cap.available:
        return
    detail = cap.detail if cap else "unknown capability"
    raise CapabilityError(
        f"capability '{name}' unavailable: {detail}. {hint or cap.affects if cap else ''}".strip(),
        detail={"capability": name, "reason": detail},
    )


def report() -> str:
    lines = ["CAPABILITY REPORT", "-" * 96]
    for name, cap in sorted(capabilities().items()):
        mark = "ok  " if cap.available else "MISS"
        lines.append(f"[{mark}] {name:12} {cap.detail}")
        if not cap.available and cap.affects:
            lines.append(f"{'':19}-> degrades: {cap.affects}")
    return "\n".join(lines)


@dataclass
class CapabilitySnapshot:
    """Serializable snapshot stored in run reports / session files."""

    entries: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def capture(cls) -> "CapabilitySnapshot":
        return cls(entries={name: ("ok" if cap.available else f"missing: {cap.detail}") for name, cap in capabilities().items()})

    def to_dict(self) -> Dict[str, str]:
        return dict(self.entries)
