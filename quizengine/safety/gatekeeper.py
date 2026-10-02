"""Safety Gatekeeper (PRD section 7.14, section 3.3 prohibitions).

Three jobs, none of them optional (**L7** -- safety cannot be disabled by
config, CLI flag or code path):

1. **Attestation gate** (FR-7.14.1): no state-changing action happens before the
   operator has recorded an explicit authorization for *this* run.
2. **Restricted-environment scan** (FR-7.14.2): proctoring software, secure
   exam browsers, CAPTCHA / human-verification overlays and a locked display all
   mean HALT + report.  The engine never attempts to solve, click, bypass or
   otherwise interact with a CAPTCHA (section 3.3.2) and contains no evasion
   logic of any kind (section 3.3.1).
3. **Continuous re-check** (FR-7.14.3): the scan repeats on an interval and on
   every state-changing intent, so a proctoring tool launched mid-run still
   stops the engine.

Probes are injectable so the gate is unit-testable without psutil or a window
server (NFR-17.2).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from ..capabilities import has
from ..config import EngineConfig
from ..contracts import (
    RESTRICTED_OVERLAY_KINDS,
    Attestation,
    EnvIndicator,
    EnvScanResult,
    FailureCode,
    Intent,
    PerceptionResult,
    RunEventName,
    State,
)
from ..failures import FailureSignal

ATTESTATION_TEXT = (
    "I confirm that I am authorized to automate this quiz: it is my own practice "
    "environment or I have explicit permission from its owner, it is not a graded, "
    "proctored or credentialing assessment, and no anti-bot, CAPTCHA or proctoring "
    "control is being bypassed."
)


def default_process_names() -> List[str]:
    """Running process names (best effort, never raises)."""
    if not has("psutil"):
        return []
    try:
        import psutil

        names: List[str] = []
        for process in psutil.process_iter(attrs=["name", "exe"]):
            info = process.info or {}
            for value in (info.get("name"), info.get("exe")):
                if value:
                    names.append(str(value))
        return names
    except Exception:
        return []


def default_window_titles() -> List[str]:
    """Foreground/visible window titles (best effort, platform specific)."""
    try:
        import sys

        if sys.platform.startswith("win"):
            return _windows_titles()
        if sys.platform.startswith("linux"):
            return _x11_titles()
    except Exception:
        return []
    return []


def _windows_titles() -> List[str]:  # pragma: no cover - Windows only
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    titles: List[str] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)  # type: ignore[attr-defined]
    def _collect(hwnd: int, _lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length:
            buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buffer, length + 1)
            if buffer.value.strip():
                titles.append(buffer.value)
        return True

    user32.EnumWindows(_collect, 0)
    return titles


def _x11_titles() -> List[str]:  # pragma: no cover - X11 only
    import shutil
    import subprocess

    for tool, args in (("wmctrl", ["-l"]), ("xdotool", ["search", "--name", ""])):
        if not shutil.which(tool):
            continue
        try:
            out = subprocess.run([tool, *args], capture_output=True, text=True, timeout=2.0)
        except Exception:
            continue
        if out.returncode == 0:
            return [line.strip() for line in out.stdout.splitlines() if line.strip()]
    return []


def display_locked() -> bool:
    """True when the session appears to be locked (best effort)."""
    try:
        import sys

        if sys.platform.startswith("win"):  # pragma: no cover - Windows only
            import ctypes

            user32 = ctypes.windll.user32  # type: ignore[attr-defined]
            # A locked workstation has no foreground window.
            return user32.GetForegroundWindow() == 0
    except Exception:
        return False
    return False


@dataclass
class SafetyStatus:
    attested: bool = False
    attestation: Optional[Attestation] = None
    last_scan: Optional[EnvScanResult] = None
    restricted: bool = False
    halt_reason: Optional[str] = None
    scans: int = 0
    guarded_intents: int = 0
    history: List[EnvScanResult] = field(default_factory=list)


class SafetyGatekeeper:
    def __init__(
        self,
        config: EngineConfig,
        *,
        telemetry: Any = None,
        clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
        process_names: Optional[Callable[[], Iterable[str]]] = None,
        window_titles: Optional[Callable[[], Iterable[str]]] = None,
        lock_probe: Optional[Callable[[], bool]] = None,
    ) -> None:
        self.config = config
        self.safety = config.safety
        self.telemetry = telemetry
        self._clock = clock
        self._monotonic = monotonic
        self._process_names = process_names or default_process_names
        self._window_titles = window_titles or default_window_titles
        self._lock_probe = lock_probe or display_locked
        self.status = SafetyStatus()
        self._last_scan_at: float = 0.0

    # -- attestation (FR-7.14.1) ------------------------------------------- #
    def attest(self, text: str, *, operator: str = "local", method: str = "cli") -> Attestation:
        cleaned = (text or "").strip()
        if not cleaned:
            raise FailureSignal(
                FailureCode.ATTESTATION_MISSING,
                "operator attestation is required before any state-changing action (FR-7.14.1); "
                "re-run with --i-am-authorized or set run.attestation",
                origin_state=State.GATE_CHECK,
            )
        attestation = Attestation(text=cleaned, operator=operator, method=method)  # type: ignore[arg-type]
        self.status.attested = True
        self.status.attestation = attestation
        if self.telemetry is not None:
            self.telemetry.set_gate_status("attested", attestation)
            self.telemetry.event(
                RunEventName.GATE_PASSED,
                state=State.GATE_CHECK,
                module="safety",
                gate="attestation",
                operator=operator,
                method=method,
            )
        return attestation

    def require_attestation(self, state: State = State.ACTING) -> Attestation:
        if not self.status.attested or self.status.attestation is None:
            raise FailureSignal(
                FailureCode.ATTESTATION_MISSING,
                "no recorded operator attestation; refusing to change on-screen state",
                origin_state=state,
            )
        return self.status.attestation

    # -- scanning (FR-7.14.2) ---------------------------------------------- #
    def scan(self, *, perception: Optional[PerceptionResult] = None, state: State = State.GATE_CHECK) -> EnvScanResult:
        indicators: List[EnvIndicator] = []
        scanned_processes = 0

        if self.safety.scan_processes:
            names = list(self._process_names() or [])
            scanned_processes = len(names)
            for name in names:
                lowered = name.lower()
                for signature in self.safety.restricted_process_signatures:
                    if signature.lower() in lowered:
                        indicators.append(
                            EnvIndicator(kind="process", name=name, detail=f"matched signature {signature!r}")
                        )
                        break

        for title in self._window_titles() or []:
            lowered = title.lower()
            for signature in self.safety.restricted_window_signatures:
                if signature.lower() in lowered:
                    indicators.append(
                        EnvIndicator(kind="window_title", name=title[:120], detail=f"matched {signature!r}")
                    )
                    break

        if perception is not None:
            for overlay in perception.overlays:
                if overlay.kind in RESTRICTED_OVERLAY_KINDS:
                    indicators.append(
                        EnvIndicator(
                            kind="overlay",
                            name=overlay.kind.value,
                            detail=(overlay.text or "")[:120],
                        )
                    )

        if self.safety.halt_on_lock_screen and self._lock_probe():
            indicators.append(EnvIndicator(kind="display_state", name="locked", detail="no foreground window"))

        restricted = any(indicator.restricted for indicator in indicators)
        result = EnvScanResult(
            ts=self._clock(),
            restricted=restricted,
            indicators=indicators,
            scanned_processes=scanned_processes,
            halt_reason=_halt_reason(indicators) if restricted else None,
        )
        self.status.last_scan = result
        self.status.scans += 1
        self.status.history.append(result)
        if len(self.status.history) > 50:
            self.status.history = self.status.history[-50:]
        self._last_scan_at = self._monotonic()

        if self.telemetry is not None:
            self.telemetry.set_gate_status("restricted" if restricted else "clear", self.status.attestation)
            self.telemetry.event(
                RunEventName.UNSUPPORTED_ENVIRONMENT if restricted else RunEventName.GATE_PASSED,
                state=state,
                module="safety",
                code=FailureCode.RESTRICTED_ENVIRONMENT if restricted else None,
                gate="environment",
                restricted=restricted,
                indicators=[i.name for i in indicators][:8],
                scanned_processes=scanned_processes,
            )
        if restricted:
            self.status.restricted = True
            self.status.halt_reason = result.halt_reason
        return result

    def continuous_check(self, perception: Optional[PerceptionResult] = None, *, state: State = State.ACTING) -> EnvScanResult:
        """FR-7.14.3: throttled re-scan.  Overlays are always checked."""
        interval = float(self.safety.restricted_scan_interval_s)
        overlay_only = perception is not None and any(
            o.kind in RESTRICTED_OVERLAY_KINDS for o in perception.overlays
        )
        if not overlay_only and interval > 0 and (self._monotonic() - self._last_scan_at) < interval:
            return self.status.last_scan or self.scan(perception=perception, state=state)
        return self.scan(perception=perception, state=state)

    # -- enforcement -------------------------------------------------------- #
    def assert_clear(self, perception: Optional[PerceptionResult] = None, *, state: State = State.ACTING) -> EnvScanResult:
        result = self.continuous_check(perception, state=state)
        if result.restricted:
            raise FailureSignal(
                FailureCode.RESTRICTED_ENVIRONMENT,
                f"restricted environment detected: {result.halt_reason}. Halting without interaction "
                "(section 3.3: no CAPTCHA solving, no proctoring bypass, no evasion).",
                origin_state=state,
                detail={"indicators": [i.model_dump(mode="json") for i in result.indicators]},
            )
        return result

    def guard_intent(
        self, intent: Intent, *, perception: Optional[PerceptionResult] = None, state: State = State.ACTING
    ) -> EnvScanResult:
        """The single choke point before any actuator call."""
        if intent.is_state_changing:
            self.require_attestation(state=state)
            self.status.guarded_intents += 1
        return self.assert_clear(perception, state=state)

    def describe(self) -> Dict[str, Any]:
        return {
            "attested": self.status.attested,
            "attestation": self.status.attestation.model_dump(mode="json") if self.status.attestation else None,
            "scans": self.status.scans,
            "restricted": self.status.restricted,
            "halt_reason": self.status.halt_reason,
            "guarded_intents": self.status.guarded_intents,
            "last_scan": self.status.last_scan.summary if self.status.last_scan else None,
            "capabilities": {
                "process_probe": "psutil" if has("psutil") else "unavailable",
                "window_probe": _window_probe_name(),
                "lock_probe": bool(self._lock_probe()),
            },
        }


def _window_probe_name() -> str:
    """Which window-title probe this platform can offer (diagnostics only)."""
    import shutil
    import sys

    if sys.platform.startswith("win"):
        return "win32-enumwindows"
    if sys.platform.startswith("linux"):
        for tool in ("wmctrl", "xdotool"):
            if shutil.which(tool):
                return f"x11-{tool}"
        return "unavailable (no wmctrl/xdotool)"
    return "unavailable"


def _halt_reason(indicators: Sequence[EnvIndicator]) -> str:
    kinds = {indicator.kind for indicator in indicators}
    names = ", ".join(sorted({indicator.name for indicator in indicators})[:5])
    if "overlay" in kinds:
        return f"human-verification/CAPTCHA overlay on screen ({names})"
    if "process" in kinds:
        return f"proctoring/secure-exam software running ({names})"
    if "window_title" in kinds:
        return f"secure/proctored browser window detected ({names})"
    if "display_state" in kinds:
        return "display appears locked"
    return f"restricted indicator(s): {names}"


__all__ = [
    "ATTESTATION_TEXT",
    "SafetyGatekeeper",
    "SafetyStatus",
    "default_process_names",
    "default_window_titles",
    "display_locked",
]
