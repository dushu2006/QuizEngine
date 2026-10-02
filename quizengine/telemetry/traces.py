"""Trace screenshots and failure artifact bundles (FR-7.13.3, FR-7.11.3).

Artifacts are local-only (FR-16.3): nothing here ever opens a network socket.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..contracts import DecisionTrace, FailureBundleRef, Frame, FailureCode, State


def _save_png(pixels: Any, path: Path) -> bool:
    """Write an ``H x W x 3`` uint8 array to PNG.  Returns False if impossible."""
    if pixels is None:
        return False
    try:
        import numpy as np
        from PIL import Image
    except ImportError:  # pragma: no cover - pillow is a core dependency
        return False
    try:
        array = np.asarray(pixels)
        if array.ndim == 2:
            mode = "L"
        elif array.shape[2] == 4:
            mode = "RGBA"
        else:
            mode = "RGB"
            array = array[:, :, :3]
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(array.astype("uint8"), mode=mode).save(path)
        return True
    except Exception:
        return False


class TraceStore:
    """One screenshot per state transition, referenced from logs."""

    def __init__(self, run_dir: str | Path, enabled: bool = True) -> None:
        self.run_dir = Path(run_dir)
        self.traces_dir = self.run_dir / "traces"
        self.bundles_dir = self.run_dir / "failures"
        self.enabled = enabled
        self._written: List[str] = []
        if enabled:
            self.traces_dir.mkdir(parents=True, exist_ok=True)

    # -- traces ------------------------------------------------------------ #
    def save_frame(self, frame: Frame, *, tag: str = "") -> Optional[str]:
        """Persist a frame; returns the run-relative path stored in RunEvent.trace_img."""
        if not self.enabled or frame.pixels is None:
            return None
        suffix = f"_{tag}" if tag else ""
        name = f"f{frame.seq:05d}{suffix}.png"
        path = self.traces_dir / name
        if _save_png(frame.pixels, path):
            relative = f"traces/{name}"
            self._written.append(relative)
            return relative
        return None

    @property
    def written(self) -> List[str]:
        return list(self._written)

    def count(self) -> int:
        return len(self._written)

    # -- failure bundles --------------------------------------------------- #
    def write_failure_bundle(
        self,
        code: FailureCode,
        state: State,
        *,
        frames: Iterable[Frame],
        perception_json: Optional[Dict[str, Any]] = None,
        intent_ledger: Optional[List[Dict[str, Any]]] = None,
        decision_trace: Optional[DecisionTrace] = None,
        log_tail: Optional[List[Dict[str, Any]]] = None,
        detail: Optional[Dict[str, Any]] = None,
        max_frames: int = 5,
    ) -> FailureBundleRef:
        """FR-7.11.3: last N ring-buffer frames + perception JSON + intent ledger
        + decision trace + logs.  This is the primary debugging artifact."""
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        bundle_dir = self.bundles_dir / f"{stamp}_{code.value}"
        bundle_dir.mkdir(parents=True, exist_ok=True)

        frame_paths: List[str] = []
        for frame in list(frames)[-max_frames:]:
            name = f"frame_{frame.seq:05d}.png"
            if _save_png(frame.pixels, bundle_dir / name):
                frame_paths.append(name)

        manifest: Dict[str, Any] = {
            "code": code.value,
            "state": state.value,
            "created_ts": time.time(),
            "frames": frame_paths,
            "detail": detail or {},
        }
        _write_json(bundle_dir / "manifest.json", manifest)
        if perception_json is not None:
            _write_json(bundle_dir / "perception.json", perception_json)
        if intent_ledger is not None:
            _write_json(bundle_dir / "intent_ledger.json", intent_ledger)
        if decision_trace is not None:
            _write_json(bundle_dir / "decision_trace.json", decision_trace.to_wire())
        if log_tail is not None:
            _write_json(bundle_dir / "log_tail.json", log_tail)

        return FailureBundleRef(
            code=code,
            path=str(bundle_dir),
            created_ts=time.time(),
            state=state,
            frames=frame_paths,
        )

    # -- decision traces --------------------------------------------------- #
    def append_decision_trace(self, trace: DecisionTrace) -> Path:
        path = self.run_dir / "decision_traces.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trace.to_wire(), default=str) + "\n")
        return path

    def write_decision_traces(self, traces: Iterable[DecisionTrace]) -> Path:
        path = self.run_dir / "decision_traces.json"
        _write_json(path, [t.to_wire() for t in traces])
        return path

    def write_report(self, report: Dict[str, Any]) -> Path:
        path = self.run_dir / "report.json"
        _write_json(path, report)
        return path


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str), encoding="utf-8")
    tmp.replace(path)  # atomic write (FR-7.12.1)
