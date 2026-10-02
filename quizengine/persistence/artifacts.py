"""Run artifacts and the terminal report (FR-7.12.2, FR-7.11.3, FR-7.12.3).

Layout of ``runs/{run_id}/``::

    engine.log              structured JSON logs (FR-7.13.1)
    events.jsonl            one RunEvent per line
    metrics.json            counters + latency histograms (FR-7.13.2)
    metrics.prom            the same metrics in Prometheus text format
    decision_traces.jsonl   per-question audit trail (FR-7.13.4)
    traces/*.png            one screenshot per state transition (FR-7.13.3)
    failures/<ts>_<CODE>/   artifact bundles (FR-7.11.3)
    session.json            resume ledger (FR-7.12.1)
    report.json             terminal report (FR-7.12.2)

Everything is local: no artifact leaves the machine (FR-16.3).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..config import EngineConfig
from ..contracts import (
    AnsweredQuestion,
    Attestation,
    BudgetCounters,
    DecisionTrace,
    FailureBundleRef,
    FailureCode,
    Frame,
    RunEvent,
    RunOutcome,
    RunReport,
    SessionState,
    State,
)
from ..telemetry.hub import Telemetry, prune_old_runs


class RunArtifacts:
    def __init__(self, config: EngineConfig, telemetry: Telemetry, *, run_dir: Optional[Path] = None) -> None:
        self.config = config
        self.telemetry = telemetry
        self.run_dir = Path(run_dir) if run_dir is not None else telemetry.run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.bundles: List[FailureBundleRef] = []

    # -- failure bundles (FR-7.11.3) --------------------------------------- #
    def failure_bundle(
        self,
        code: FailureCode,
        state: State,
        *,
        frames: Iterable[Frame],
        perception: Any = None,
        decision_trace: Optional[DecisionTrace] = None,
        detail: Optional[Dict[str, Any]] = None,
        message: str = "",
    ) -> FailureBundleRef:
        perception_json: Optional[Dict[str, Any]] = None
        if perception is not None:
            perception_json = (
                perception.to_wire()
                if hasattr(perception, "to_wire")
                else (perception.describe() if hasattr(perception, "describe") else None)
            )
        bundle = self.telemetry.traces.write_failure_bundle(
            code,
            state,
            frames=frames,
            perception_json=perception_json,
            intent_ledger=list(self.telemetry.intent_ledger),
            decision_trace=decision_trace,
            log_tail=self.telemetry.logger.tail(self.config.telemetry.artifact_bundle_frames * 20),
            detail={**(detail or {}), "message": message},
            max_frames=self.config.telemetry.artifact_bundle_frames,
        )
        self.bundles.append(bundle)
        self.telemetry.record_failure_bundle(bundle)
        return bundle

    # -- report (FR-7.12.2) ------------------------------------------------- #
    def write_report(
        self,
        *,
        run_id: str,
        outcome: RunOutcome,
        started_at: float,
        session: Optional[SessionState] = None,
        traces: Optional[Sequence[DecisionTrace]] = None,
        env_scan_summary: str = "",
        attestation: Optional[Attestation] = None,
        accuracy: Optional[float] = None,
        correct: Optional[int] = None,
        incorrect: Optional[int] = None,
        halted_code: Optional[FailureCode] = None,
        halted_detail: str = "",
        config_fingerprint: Optional[str] = None,
        finished_at: Optional[float] = None,
    ) -> RunReport:
        finished = finished_at if finished_at is not None else time.time()
        questions: List[AnsweredQuestion] = list(session.questions_answered) if session else []
        decision_traces = list(traces) if traces is not None else list(self.telemetry.decision_traces)
        counters = session.budget_counters if session is not None else BudgetCounters()
        unverified = sum(1 for t in decision_traces if t.verification is not None and not t.verification.passed)
        stale = sum(
            1
            for entry in self.telemetry.intent_ledger
            if str(entry.get("refusal_reason") or "").lower().startswith("stale")
        )
        report = RunReport(
            run_id=run_id,
            outcome=outcome,
            started_at=started_at,
            finished_at=finished,
            questions=questions,
            decision_traces=decision_traces,
            total_time_s=round(finished - started_at, 3),
            failure_bundles=list(self.bundles) + [
                b for b in self.telemetry.failure_bundles if b not in self.bundles
            ],
            env_scan_summary=env_scan_summary,
            attestation=attestation,
            budget_counters=counters,
            accuracy=accuracy,
            correct=correct,
            incorrect=incorrect,
            unverified_actions=unverified,
            illegal_transitions=int(self.telemetry.metrics.get("illegal_transitions")),
            stale_coordinate_violations=stale,
            halted_code=halted_code,
            halted_detail=halted_detail[:500],
            config_fingerprint=config_fingerprint or (self.config.fingerprint() if hasattr(self.config, "fingerprint") else None),
        )
        self.telemetry.traces.write_report(report.to_wire())
        self.write_events()
        self.telemetry.close()
        return report

    def write_events(self) -> Path:
        path = self.run_dir / "events.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for event in self.telemetry.events:
                handle.write(json.dumps(event.to_wire(), default=str) + "\n")
        return path

    def prune(self) -> List[str]:
        """FR-7.12.3: keep the last N runs."""
        keep = int(self.config.telemetry.retention_runs)
        if keep <= 0:
            return []
        removed = prune_old_runs(self.config.paths.runs_dir, keep)
        return [r for r in removed if Path(r).name != self.run_dir.name]

    def summary(self, report: RunReport) -> str:
        lines = [
            f"run {report.run_id}: {report.outcome.value} in {report.total_time_s:.1f}s",
            f"  questions answered : {len(report.questions)}",
            f"  verified actions   : {len(report.questions) - report.unverified_actions}/{len(report.questions)}",
            f"  failure bundles    : {len(report.failure_bundles)}",
            f"  env scan           : {report.env_scan_summary or 'n/a'}",
        ]
        if report.accuracy is not None:
            lines.append(f"  accuracy vs key    : {report.accuracy:.1%} ({report.correct} correct, {report.incorrect} wrong)")
        if report.halted_code is not None:
            lines.append(f"  halted             : {report.halted_code.value} -- {report.halted_detail[:100]}")
        lines.append(f"  artifacts          : {self.run_dir}")
        return "\n".join(lines)


__all__ = ["RunArtifacts"]
