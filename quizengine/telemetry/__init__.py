"""Telemetry, logging and observability (PRD section 7.13)."""

from __future__ import annotations

from .console import (
    ALL_COMMANDS,
    CommandKind,
    NullOperator,
    OperatorChannel,
    OperatorCommand,
    OperatorConsole,
    ScriptedOperator,
    StdinOperator,
)
from .hub import Telemetry, ensure_run_dir, prune_old_runs
from .logging import StructuredLogger, read_jsonl
from .metrics import Counter, Histogram, MetricsRegistry
from .traces import TraceStore

__all__ = [
    "ALL_COMMANDS",
    "CommandKind",
    "Counter",
    "Histogram",
    "MetricsRegistry",
    "NullOperator",
    "OperatorChannel",
    "OperatorCommand",
    "OperatorConsole",
    "ScriptedOperator",
    "StdinOperator",
    "StructuredLogger",
    "Telemetry",
    "TraceStore",
    "ensure_run_dir",
    "prune_old_runs",
    "read_jsonl",
]
