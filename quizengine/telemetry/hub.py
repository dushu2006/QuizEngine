"""Telemetry facade: one object the orchestrator talks to (section 7.13).

Combines structured logs (FR-7.13.1), metrics (FR-7.13.2), trace screenshots
(FR-7.13.3), decision traces (FR-7.13.4) and the operator console (FR-7.13.5).
"""

from __future__ import annotations

import itertools
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..config import EngineConfig
from ..contracts import (
    Attestation,
    DecisionTrace,
    FailureBundleRef,
    FailureCode,
    Frame,
    RunEvent,
    RunEventName,
    State,
)
from .console import OperatorChannel, OperatorConsole, StdinOperator
from .logging import StructuredLogger
from .metrics import MetricsRegistry
from .traces import TraceStore

_correlation_counter = itertools.count(1)


class Telemetry:
    """Run-scoped observability hub."""

    def __init__(
        self,
        config: EngineConfig,
        run_id: str,
        *,
        run_dir: Optional[Path] = None,
        operator: Optional[OperatorChannel] = None,
        console_stream: Any = None,
        echo_logs: bool = True,
    ) -> None:
        self.config = config
        self.run_id = run_id
        self.run_dir = Path(run_dir) if run_dir is not None else Path(config.paths.runs_dir) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        log_file = Path(config.telemetry.log_file) if config.telemetry.log_file else self.run_dir / "engine.log"
        self.logger = StructuredLogger(
            run_id,
            log_file=log_file,
            level=config.telemetry.log_level,
            echo=echo_logs,
            echo_stream=console_stream,
        )
        self.metrics = MetricsRegistry()
        self.traces = TraceStore(self.run_dir, enabled=config.telemetry.trace_screenshots)
        self.console = OperatorConsole(stream=console_stream, enabled=config.telemetry.console)
        self.operator: OperatorChannel = operator if operator is not None else StdinOperator(stream=console_stream)
        self.events: List[RunEvent] = []
        self.intent_ledger: List[Dict[str, Any]] = []
        self.decision_traces: List[DecisionTrace] = []
        self.failure_bundles: List[FailureBundleRef] = []
        self._current_correlation: Optional[str] = None
        self._current_state: State = State.IDLE
        self._attestation: Optional[Attestation] = None
        self._gate_status = "not-scanned"

    # -- identity ---------------------------------------------------------- #
    @staticmethod
    def new_run_id(prefix: str = "run") -> str:
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        return f"{prefix}-{stamp}-{uuid.uuid4().hex[:6]}"

    def new_correlation_id(self, ordinal: Optional[int] = None) -> str:
        """One correlation id per question cycle (FR-7.13.1)."""
        seq = next(_correlation_counter)
        cid = f"{self.run_id}-q{ordinal if ordinal is not None else 0:03d}-c{seq:05d}"
        self._current_correlation = cid
        return cid

    @property
    def correlation_id(self) -> Optional[str]:
        return self._current_correlation

    # -- state / safety ---------------------------------------------------- #
    def set_state(self, state: State) -> None:
        self._current_state = state

    @property
    def state(self) -> State:
        return self._current_state

    def set_gate_status(self, status: str, attestation: Optional[Attestation] = None) -> None:
        self._gate_status = status
        if attestation is not None:
            self._attestation = attestation

    @property
    def gate_status(self) -> str:
        return self._gate_status

    @property
    def attestation(self) -> Optional[Attestation]:
        return self._attestation

    def banner(self, config_fingerprint: str) -> None:
        self.console.banner(self.run_id, self._attestation, self._gate_status, config_fingerprint)

    #: keyword arguments ``OperatorConsole.status`` understands.
    _STATUS_KEYS = frozenset({"state", "question_no", "total", "confidence", "gate_status", "message", "cycle"})

    def status(self, **kwargs: Any) -> None:
        """Update the one-line operator status.

        Unknown keys are folded into ``message`` instead of raising: a console
        formatting mismatch must never take a run down (**L4**).
        """
        kwargs.setdefault("state", self._current_state)
        kwargs.setdefault("gate_status", self._gate_status)
        extra = {key: value for key, value in kwargs.items() if key not in self._STATUS_KEYS}
        kwargs = {key: value for key, value in kwargs.items() if key in self._STATUS_KEYS}
        if extra:
            rendered = " ".join(f"{key}={value}" for key, value in extra.items() if value is not None)
            if rendered:
                kwargs["message"] = f"{kwargs.get('message') or ''} | {rendered}".strip(" |")
        try:
            self.console.status(**kwargs)
        except TypeError:  # pragma: no cover - console signature drift
            pass

    # -- events ------------------------------------------------------------ #
    def event(
        self,
        name: RunEventName,
        *,
        state: Optional[State] = None,
        code: Optional[FailureCode] = None,
        latency_ms: float = 0.0,
        module: str = "orchestrator",
        correlation_id: Optional[str] = None,
        decision_trace_id: Optional[str] = None,
        confidence: Optional[float] = None,
        trace_frame: Optional[Frame] = None,
        trace_tag: str = "",
        record: bool = True,
        **detail: Any,
    ) -> RunEvent:
        """Emit a RunEvent to logs + metrics + console (FR-7.10.4)."""
        current_state = state if state is not None else self._current_state
        trace_img: Optional[str] = None
        if trace_frame is not None:
            trace_img = self.traces.save_frame(trace_frame, tag=trace_tag or current_state.value.lower())
        event = RunEvent(
            ts=time.time(),
            run_id=self.run_id,
            state=current_state,
            event=name,
            code=code,
            latency_ms=round(latency_ms, 3),
            trace_img=trace_img,
            correlation_id=correlation_id if correlation_id is not None else self._current_correlation,
            detail=_clean_detail(detail),
        )
        if record:
            self.events.append(event)
        self.logger.log_event(event, module=module, decision_trace_id=decision_trace_id)
        return event

    def log(
        self,
        message: str,
        *,
        module: str = "orchestrator",
        event: RunEventName = RunEventName.STATE_TRANSITION,
        **extra: Any,
    ) -> None:
        self.logger.emit(event, state=self._current_state, module=module, **{"message": message, **extra})

    # -- ledger ------------------------------------------------------------ #
    def record_intent(self, entry: Dict[str, Any]) -> None:
        self.intent_ledger.append({"ts": time.time(), **entry})

    def record_decision_trace(self, trace: DecisionTrace) -> None:
        self.decision_traces.append(trace)
        if self.config.telemetry.decision_traces:
            self.traces.append_decision_trace(trace)
        self.console.notice(f"  {trace.human_summary()}")

    def record_failure_bundle(self, bundle: FailureBundleRef) -> None:
        self.failure_bundles.append(bundle)

    # -- metrics shortcuts ------------------------------------------------- #
    def observe_latency(self, name: str, seconds_or_ms: float, *, unit: str = "ms", label: Optional[str] = None) -> None:
        self.metrics.observe(name, seconds_or_ms, unit=unit, label=label)

    def timer(self, name: str, *, unit: str = "ms") -> "_Timer":
        return _Timer(self.metrics, name, unit)

    # -- export ------------------------------------------------------------ #
    def flush_metrics(self) -> Path:
        path = self.run_dir / self.config.telemetry.metrics_file
        return self.metrics.export_file(path)

    def write_prometheus(self) -> Path:
        path = self.run_dir / "metrics.prom"
        path.write_text(self.metrics.prometheus_text(), encoding="utf-8")
        return path

    @property
    def log_file(self) -> Optional[Path]:
        return self.logger.log_file

    def close(self) -> None:
        self.flush_metrics()
        if self.config.telemetry.metrics_export != "none":
            self.write_prometheus()
        self.console.newline()
        self.operator.close()


class _Timer:
    """``with telemetry.timer("solver_latency_ms"):`` -> histogram observation."""

    def __init__(self, metrics: MetricsRegistry, name: str, unit: str) -> None:
        self._metrics = metrics
        self._name = name
        self._unit = unit
        self._start = 0.0
        self.elapsed_ms = 0.0

    def __enter__(self) -> "_Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_exc: Any) -> bool:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0
        value = self.elapsed_ms if self._unit == "ms" else self.elapsed_ms / 1000.0
        self._metrics.observe(self._name, value, unit=self._unit)
        return False


def _clean_detail(detail: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in detail.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            out[key] = value
        elif hasattr(value, "value") and isinstance(value.value, str):
            out[key] = value.value
        elif isinstance(value, dict):
            out[key] = _clean_detail(value)
        elif isinstance(value, (list, tuple)):
            out[key] = [v.value if hasattr(v, "value") and isinstance(v.value, str) else v for v in value]
        else:
            out[key] = str(value)
    return out


def ensure_run_dir(config: EngineConfig, run_id: str) -> Path:
    path = Path(config.paths.runs_dir) / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def prune_old_runs(runs_dir: str | Path, keep: int) -> List[str]:
    """FR-7.12.3: keep the last ``keep`` runs, auto-prune older artifacts."""
    root = Path(runs_dir)
    if not root.exists() or keep <= 0:
        return []
    runs = sorted(
        (p for p in root.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    removed: List[str] = []
    for stale in runs[keep:]:
        # Never delete the directory the current process is writing into.
        if os.path.abspath(stale) == os.path.abspath(root):
            continue
        _rmtree(stale)
        removed.append(str(stale))
    return removed


def _rmtree(path: Path) -> None:
    for child in sorted(path.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        try:
            if child.is_dir():
                child.rmdir()
            else:
                child.unlink()
        except OSError:  # pragma: no cover - best effort
            pass
    try:
        path.rmdir()
    except OSError:  # pragma: no cover
        pass


__all__ = ["Telemetry", "ensure_run_dir", "prune_old_runs"]
