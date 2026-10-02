"""Metrics registry (FR-7.13.2).

Local-only counters and histograms with a file export -- no network, no
telemetry leaves the machine (FR-16.3).  ``p95`` is what the section 15
performance budget is measured against.
"""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Counter:
    name: str
    value: float = 0.0
    labels: Dict[str, float] = field(default_factory=dict)

    def inc(self, amount: float = 1.0, label: Optional[str] = None) -> None:
        self.value += amount
        if label is not None:
            self.labels[label] = self.labels.get(label, 0.0) + amount

    def to_dict(self) -> Dict[str, object]:
        return {"name": self.name, "value": self.value, "labels": dict(self.labels)}


@dataclass
class Histogram:
    """Fixed-reservoir histogram; keeps every sample up to ``max_samples``."""

    name: str
    unit: str = "ms"
    max_samples: int = 5000
    samples: List[float] = field(default_factory=list)
    count: int = 0
    total: float = 0.0
    by_label: Dict[str, List[float]] = field(default_factory=dict)

    def observe(self, value: float, label: Optional[str] = None) -> None:
        value = float(value)
        self.count += 1
        self.total += value
        if len(self.samples) < self.max_samples:
            self.samples.append(value)
        if label is not None:
            bucket = self.by_label.setdefault(label, [])
            if len(bucket) < self.max_samples:
                bucket.append(value)

    @staticmethod
    def _quantile(values: List[float], q: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        if len(ordered) == 1:
            return ordered[0]
        pos = (len(ordered) - 1) * q
        low = math.floor(pos)
        high = math.ceil(pos)
        if low == high:
            return ordered[int(pos)]
        return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)

    def quantile(self, q: float) -> float:
        return self._quantile(self.samples, q)

    @property
    def p50(self) -> float:
        return self.quantile(0.50)

    @property
    def p95(self) -> float:
        return self.quantile(0.95)

    @property
    def p99(self) -> float:
        return self.quantile(0.99)

    @property
    def mean(self) -> float:
        return (self.total / self.count) if self.count else 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "unit": self.unit,
            "count": self.count,
            "mean": round(self.mean, 3),
            "p50": round(self.p50, 3),
            "p95": round(self.p95, 3),
            "p99": round(self.p99, 3),
            "max": round(max(self.samples), 3) if self.samples else 0.0,
            "labels": {
                label: {
                    "count": len(values),
                    "p95": round(self._quantile(values, 0.95), 3),
                }
                for label, values in sorted(self.by_label.items())
            },
        }


class MetricsRegistry:
    """Thread-safe registry (model calls run in an async worker pool, FR-6.1)."""

    #: Counters required by FR-7.13.2.
    DEFAULT_COUNTERS = (
        "questions_answered",
        "questions_correct",
        "questions_incorrect",
        "capture_failures",
        "frames_rejected_blank",
        "frames_rejected_stale",
        "frames_rejected_lock_screen",
        "tier2_invocations",
        "tier2_schema_failures",
        "action_success",
        "action_unverified",
        "stale_coordinate_blocks",
        "illegal_transitions",
        "human_pauses",
        "model_calls",
        "model_timeouts",
        "model_fallbacks",
        "restricted_env_detections",
        "session_saves",
        "resume_events",
    )

    DEFAULT_HISTOGRAMS = {
        "capture_latency_ms": "ms",
        "tier1_latency_ms": "ms",
        "tier2_latency_ms": "ms",
        "extraction_latency_ms": "ms",
        "solver_latency_ms": "ms",
        "verification_latency_ms": "ms",
        "navigation_latency_ms": "ms",
        "per_question_time_s": "s",
        "cycle_time_ms": "ms",
        "confidence_composite": "score",
        "end_to_end_accuracy": "ratio",
    }

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: Dict[str, Counter] = {name: Counter(name) for name in self.DEFAULT_COUNTERS}
        self._histograms: Dict[str, Histogram] = {
            name: Histogram(name, unit) for name, unit in self.DEFAULT_HISTOGRAMS.items()
        }
        self._started = time.time()
        #: "recovery invocations by class" (FR-7.13.2)
        self.recovery_invocations: Dict[str, int] = {}
        self.recovery_successes: Dict[str, int] = {}

    # -- API --------------------------------------------------------------- #
    def inc(self, name: str, amount: float = 1.0, label: Optional[str] = None) -> None:
        with self._lock:
            counter = self._counters.setdefault(name, Counter(name))
            counter.inc(amount, label)

    def observe(self, name: str, value: float, unit: str = "ms", label: Optional[str] = None) -> None:
        with self._lock:
            histogram = self._histograms.get(name)
            if histogram is None:
                histogram = Histogram(name, unit)
                self._histograms[name] = histogram
            histogram.observe(value, label)

    def record_recovery(self, code: str, success: bool) -> None:
        with self._lock:
            self.recovery_invocations[code] = self.recovery_invocations.get(code, 0) + 1
            if success:
                self.recovery_successes[code] = self.recovery_successes.get(code, 0) + 1

    def get(self, name: str) -> float:
        with self._lock:
            counter = self._counters.get(name)
            return counter.value if counter else 0.0

    def histogram(self, name: str) -> Optional[Histogram]:
        with self._lock:
            return self._histograms.get(name)

    # -- export ------------------------------------------------------------ #
    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            return {
                "uptime_s": round(time.time() - self._started, 3),
                "counters": {name: c.to_dict() for name, c in sorted(self._counters.items())},
                "histograms": {name: h.to_dict() for name, h in sorted(self._histograms.items())},
                "recovery": {
                    code: {
                        "invoked": self.recovery_invocations.get(code, 0),
                        "recovered": self.recovery_successes.get(code, 0),
                    }
                    for code in sorted(set(self.recovery_invocations) | set(self.recovery_successes))
                },
            }

    def export_file(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.snapshot(), indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)  # atomic (FR-7.12.1 discipline applied to metrics too)
        return path

    def prometheus_text(self) -> str:
        """Prometheus text-exposition format (local endpoint / file, FR-7.13.2)."""
        lines: List[str] = []
        snapshot = self.snapshot()
        for name, counter in snapshot["counters"].items():  # type: ignore[index]
            lines.append(f"# TYPE quizengine_{name} counter")
            lines.append(f"quizengine_{name} {counter['value']}")
            for label, value in counter["labels"].items():  # type: ignore[union-attr]
                lines.append(f'quizengine_{name}{{code="{label}"}} {value}')
        for name, hist in snapshot["histograms"].items():  # type: ignore[index]
            lines.append(f"# TYPE quizengine_{name} summary")
            lines.append(f'quizengine_{name}{{quantile="0.5"}} {hist["p50"]}')
            lines.append(f'quizengine_{name}{{quantile="0.95"}} {hist["p95"]}')
            lines.append(f'quizengine_{name}{{quantile="0.99"}} {hist["p99"]}')
            lines.append(f"quizengine_{name}_count {hist['count']}")
        for code, stats in snapshot["recovery"].items():  # type: ignore[index,union-attr]
            lines.append(f'quizengine_recovery_invoked{{code="{code}"}} {stats["invoked"]}')
            lines.append(f'quizengine_recovery_recovered{{code="{code}"}} {stats["recovered"]}')
        return "\n".join(lines) + "\n"
