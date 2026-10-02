"""Session persistence (FR-7.12.1, **L9** no re-answered questions).

``session.json`` is written atomically after every answered question so a crash
or an operator stop never loses the ledger.  Resume is guarded by the config
fingerprint: changing anything that would alter answers (OCR engine, model,
strategy chain, safety-relevant knobs) invalidates the resume instead of
silently mixing two configurations in one ledger.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ..config import EngineConfig
from ..contracts import (
    AnsweredQuestion,
    BudgetCounters,
    Decision,
    RunEventName,
    SessionState,
    SolverStrategy,
    State,
)


@dataclass
class ResumeCheck:
    allowed: bool
    reason: str
    fingerprint_match: bool = True
    stale_seconds: float = 0.0
    answered: int = 0


class SessionStore:
    """Atomic, local-only session ledger."""

    def __init__(
        self,
        config: EngineConfig,
        *,
        run_id: str,
        path: Optional[str | Path] = None,
        telemetry: Any = None,
        clock: Any = time.time,
    ) -> None:
        self.config = config
        self.run_id = run_id
        self.telemetry = telemetry
        self._clock = clock
        self.path = Path(path) if path is not None else self._default_path(config, run_id)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.state: SessionState = SessionState(run_id=run_id, started_at=self._clock())
        self.state.config_fingerprint = config.fingerprint()
        self.saves = 0
        self._answered_hashes: Set[str] = set()
        self._answered_content: Set[str] = set()

    @staticmethod
    def _default_path(config: EngineConfig, run_id: str) -> Path:
        configured = Path(config.paths.session_file)
        if configured.is_absolute():
            return configured
        return Path(config.paths.runs_dir) / run_id / configured.name

    # -- lifecycle ---------------------------------------------------------- #
    def begin(self, *, platform_profile: Optional[Dict[str, Any]] = None, attestation_text: Optional[str] = None) -> SessionState:
        self.state = SessionState(
            run_id=self.run_id,
            started_at=self._clock(),
            platform_profile=platform_profile or {},
            state=State.IDLE,
            config_fingerprint=self.config.fingerprint(),
            attestation_text=attestation_text,
        )
        self.save()
        return self.state

    def save(self) -> Path:
        """Atomic write: temp file in the same directory, then ``os.replace``."""
        self.state.questions_answered = self._ordered_ledger()
        payload = self.state.to_wire()
        payload.pop("pixels", None)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, self.path)
        self.saves += 1
        if self.telemetry is not None and self.saves % 1 == 0:
            self.telemetry.event(
                RunEventName.SESSION_SAVED,
                module="persistence",
                state=self.state.state,
                path=str(self.path),
                answered=len(self.state.questions_answered),
                record=False,
            )
        return self.path

    def load(self) -> Optional[SessionState]:
        if not self.path.is_file():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            loaded = SessionState.model_validate(payload)
        except Exception:
            return None
        self.state = loaded
        self._answered_hashes = {entry.hash for entry in loaded.questions_answered}
        self._answered_content = {entry.content_hash for entry in loaded.questions_answered if entry.content_hash}
        return loaded

    # -- resume guard ------------------------------------------------------- #
    def check_resume(self, *, max_age_s: Optional[float] = None) -> ResumeCheck:
        loaded = self.load()
        if loaded is None:
            return ResumeCheck(allowed=False, reason="no session file to resume from")
        fingerprint_match = loaded.config_fingerprint == self.config.fingerprint()
        stale = max(0.0, self._clock() - float(loaded.last_verified_at or loaded.started_at))
        if not fingerprint_match:
            return ResumeCheck(
                allowed=False,
                reason=(
                    "config fingerprint changed since the session was written "
                    f"({str(loaded.config_fingerprint)[:12]} -> {self.config.fingerprint()[:12]}); "
                    "resume would mix two configurations in one ledger"
                ),
                fingerprint_match=False,
                stale_seconds=stale,
                answered=len(loaded.questions_answered),
            )
        if max_age_s is not None and stale > max_age_s:
            return ResumeCheck(
                allowed=False,
                reason=f"session is {stale / 3600.0:.1f}h old (max {max_age_s / 3600.0:.1f}h)",
                fingerprint_match=True,
                stale_seconds=stale,
                answered=len(loaded.questions_answered),
            )
        if loaded.state in {State.DONE, State.FAILED_SAFE}:
            return ResumeCheck(
                allowed=False,
                reason=f"previous run already reached terminal state {loaded.state.value}",
                fingerprint_match=True,
                stale_seconds=stale,
                answered=len(loaded.questions_answered),
            )
        return ResumeCheck(
            allowed=True,
            reason=f"resuming with {len(loaded.questions_answered)} answered question(s)",
            fingerprint_match=True,
            stale_seconds=stale,
            answered=len(loaded.questions_answered),
        )

    def resume(self) -> SessionState:
        """Adopt a previously saved session (fingerprint already validated)."""
        loaded = self.load()
        if loaded is None:
            raise FileNotFoundError(f"no session file at {self.path}")
        loaded.resume_count += 1
        loaded.run_id = self.run_id
        self.state = loaded
        self.save()
        if self.telemetry is not None:
            self.telemetry.event(
                RunEventName.SESSION_RESUMED,
                module="persistence",
                state=loaded.state,
                answered=len(loaded.questions_answered),
                resume_count=loaded.resume_count,
            )
        return self.state

    # -- ledger (L9) -------------------------------------------------------- #
    def already_answered(self, question_hash: str, content_hash: str = "") -> bool:
        if question_hash and question_hash in self._answered_hashes:
            return True
        return bool(content_hash) and content_hash in self._answered_content

    def answered_hashes(self) -> Set[str]:
        return set(self._answered_hashes)

    def answered_content_hashes(self) -> Set[str]:
        return set(self._answered_content)

    def record_answer(
        self,
        *,
        question_hash: str,
        ordinal: int,
        decision: Decision,
        content_hash: str = "",
        verified: bool = False,
        decision_trace_id: Optional[str] = None,
    ) -> AnsweredQuestion:
        entry = AnsweredQuestion(
            hash=question_hash,
            ordinal=ordinal,
            chosen_index=decision.option_index,
            confidence=decision.confidence,
            timestamp=self._clock(),
            verified=verified,
            decision_trace_id=decision_trace_id,
            content_hash=content_hash,
            strategy=decision.strategy,
        )
        self.state.questions_answered.append(entry)
        self._answered_hashes.add(question_hash)
        if content_hash:
            self._answered_content.add(content_hash)
        self.state.last_question_hash = question_hash
        self.state.last_verified_at = entry.timestamp
        self.state.budget_counters.questions_answered += 1
        self.save()
        return entry

    def mark_verified(self, question_hash: str, verified: bool) -> None:
        for entry in self.state.questions_answered:
            if entry.hash == question_hash:
                entry.verified = verified
        self.save()

    def _ordered_ledger(self) -> List[AnsweredQuestion]:
        return sorted(self.state.questions_answered, key=lambda e: (e.timestamp, e.ordinal))

    # -- state / budgets ---------------------------------------------------- #
    def set_state(self, state: State) -> None:
        self.state.state = state

    @property
    def counters(self) -> BudgetCounters:
        return self.state.budget_counters

    def snapshot(self) -> Dict[str, Any]:
        return {
            "path": str(self.path),
            "run_id": self.state.run_id,
            "state": self.state.state.value,
            "answered": len(self.state.questions_answered),
            "saves": self.saves,
            "resume_count": self.state.resume_count,
            "config_fingerprint": self.state.config_fingerprint,
            "counters": self.state.budget_counters.model_dump(mode="json"),
        }


__all__ = ["ResumeCheck", "SessionStore"]
