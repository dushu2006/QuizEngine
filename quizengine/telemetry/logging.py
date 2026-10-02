"""Structured JSON logging (FR-7.13.1).

Every record carries ``{ts, state, event, module, latency_ms, confidence,
decision_trace_id}`` plus a correlation id that threads through a full question
cycle.  Logs are local files only (FR-16.3).
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from ..contracts import FailureCode, RunEvent, RunEventName, State

_RECORD_KEYS = ("ts", "state", "event", "module", "latency_ms", "confidence", "decision_trace_id")


class StructuredLogger:
    """JSON-lines logger with a human-readable mirror stream."""

    def __init__(
        self,
        run_id: str,
        log_file: Optional[Path] = None,
        level: str = "INFO",
        echo: bool = True,
        echo_stream: Any = None,
    ) -> None:
        self.run_id = run_id
        self.log_file = Path(log_file) if log_file else None
        self.level = getattr(logging, str(level).upper(), logging.INFO)
        self.echo = echo
        self._stream = echo_stream if echo_stream is not None else sys.stderr
        self._records: list[Dict[str, Any]] = []
        self._keep_in_memory = 2000
        if self.log_file is not None:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

    # -- core -------------------------------------------------------------- #
    def emit(
        self,
        event: RunEventName | str,
        *,
        state: State | str = State.IDLE,
        module: str = "orchestrator",
        latency_ms: float = 0.0,
        confidence: Optional[float] = None,
        decision_trace_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        code: Optional[FailureCode | str] = None,
        level: Optional[int] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "ts": time.time(),
            "run_id": self.run_id,
            "state": state.value if isinstance(state, State) else str(state),
            "event": event.value if isinstance(event, RunEventName) else str(event),
            "module": module,
            "latency_ms": round(float(latency_ms), 3),
            "confidence": None if confidence is None else round(float(confidence), 4),
            "decision_trace_id": decision_trace_id,
            "correlation_id": correlation_id,
            "code": (code.value if isinstance(code, FailureCode) else code) if code is not None else None,
        }
        if extra:
            record["extra"] = _jsonable(extra)
        self._write(record, level)
        return record

    def log_event(self, event: RunEvent, *, module: str = "orchestrator", decision_trace_id: Optional[str] = None) -> Dict[str, Any]:
        return self.emit(
            event.event,
            state=event.state,
            module=module,
            latency_ms=event.latency_ms,
            decision_trace_id=decision_trace_id,
            correlation_id=event.correlation_id,
            code=event.code,
            **event.detail,
        )

    def _write(self, record: Dict[str, Any], level: Optional[int] = None) -> None:
        severity = level if level is not None else _severity_for(record["event"], record.get("code"))
        line = json.dumps(record, sort_keys=False, default=str)
        if self.log_file is not None and severity >= self.level:
            with self.log_file.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        self._records.append(record)
        if len(self._records) > self._keep_in_memory:
            del self._records[: len(self._records) - self._keep_in_memory]
        if self.echo and severity >= self.level:
            self._stream.write(_human_line(record) + "\n")
            try:
                self._stream.flush()
            except Exception:  # pragma: no cover - stream may be closed
                pass

    # -- access ------------------------------------------------------------ #
    @property
    def records(self) -> list[Dict[str, Any]]:
        return list(self._records)

    def tail(self, count: int = 20) -> list[Dict[str, Any]]:
        return self._records[-count:]

    def count_events(self, event: RunEventName | str) -> int:
        name = event.value if isinstance(event, RunEventName) else str(event)
        return sum(1 for record in self._records if record["event"] == name)

    def records_for_correlation(self, correlation_id: str) -> list[Dict[str, Any]]:
        return [r for r in self._records if r.get("correlation_id") == correlation_id]

    @staticmethod
    def key_subset(record: Dict[str, Any]) -> Dict[str, Any]:
        """The exact FR-7.13.1 field set."""
        return {key: record.get(key) for key in _RECORD_KEYS}


def _severity_for(event: str, code: Optional[str]) -> int:
    if code is not None:
        return logging.ERROR if code in {"RESTRICTED_ENVIRONMENT", "POPUP_UNKNOWN", "ILLEGAL_TRANSITION"} else logging.WARNING
    if event in {
        RunEventName.RUN_COMPLETE.value,
        RunEventName.ACTION_VERIFIED.value,
        RunEventName.QUESTION_EXTRACTED.value,
        RunEventName.DECISION_MADE.value,
    }:
        return logging.INFO
    if event in {RunEventName.STATE_TRANSITION.value, RunEventName.FRAME_CAPTURED.value, RunEventName.MODEL_CALL.value}:
        return logging.DEBUG
    return logging.INFO


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "value") and isinstance(getattr(value, "value"), str):
        return value.value
    return str(value)


def _human_line(record: Dict[str, Any]) -> str:
    stamp = time.strftime("%H:%M:%S", time.localtime(record["ts"]))
    millis = int((record["ts"] % 1) * 1000)
    parts = [f"{stamp}.{millis:03d}", f"[{record['state']:>18}]", f"{record['event']:<26}", record["module"]]
    if record.get("latency_ms"):
        parts.append(f"{record['latency_ms']:.0f}ms")
    if record.get("confidence") is not None:
        parts.append(f"conf={record['confidence']:.2f}")
    if record.get("code"):
        parts.append(f"code={record['code']}")
    extra = record.get("extra") or {}
    if extra:
        parts.append(" ".join(f"{k}={v}" for k, v in list(extra.items())[:4]))
    return " ".join(parts)


def read_jsonl(path: str | Path) -> list[Dict[str, Any]]:
    """Read a log file back -- used by the AC-14.7 trace-reconstruction test."""
    out: list[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
