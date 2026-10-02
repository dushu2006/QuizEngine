"""Duplicate-protected one-shot run controller for the global hotkey UI."""

from __future__ import annotations

import threading
from enum import Enum
from typing import Any, Callable, Dict, Optional

from ..runtime_config import provider_status


class AgentStatus(str, Enum):
    IDLE = "IDLE"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"


class AgentController:
    """Start at most one foreground engine run; cancellation is best effort/bounded."""

    def __init__(self, config: Any, runtime_factory: Callable[[], Any]) -> None:
        self.config = config
        self.runtime_factory = runtime_factory
        self._lock = threading.RLock()
        self._state = AgentStatus.IDLE
        self._thread: Optional[threading.Thread] = None
        self._runtime: Any = None
        self._stop_requested = False
        self.duplicate_starts = 0
        self.start_count = 0
        self.stop_count = 0
        self.last_outcome: Optional[str] = None
        self.last_error: Optional[str] = None
        self.last_run_id: Optional[str] = None
        self.last_answered: Optional[int] = None
        self.last_accuracy: Optional[float] = None
        self.last_run_dir: Optional[str] = None

    @property
    def state(self) -> AgentStatus:
        with self._lock:
            return self._state

    def start(self) -> bool:
        """Start one run. Returns False rather than creating overlapping runs."""
        with self._lock:
            if self._state is not AgentStatus.IDLE:
                self.duplicate_starts += 1
                return False
            self._state = AgentStatus.STARTING
            self._stop_requested = False
            self.last_error = None
            self.start_count += 1
            self._thread = threading.Thread(target=self._run, name="quizengine-run", daemon=True)
            self._thread.start()
            return True

    def stop(self) -> bool:
        """Request cancellation; an in-flight HTTP request is bounded by its timeout."""
        with self._lock:
            if self._state is AgentStatus.IDLE:
                return False
            if self._state is not AgentStatus.STOPPING:
                self.stop_count += 1
            self._stop_requested = True
            self._state = AgentStatus.STOPPING
            runtime = self._runtime
        if runtime is not None:
            request_stop = getattr(runtime.orchestrator, "request_stop", None)
            if callable(request_stop):
                request_stop()
        return True

    def wait(self, timeout: Optional[float] = None) -> bool:
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            model_state = provider_status(self.config)
            return {
                "status": self._state.value,
                "provider": model_state["provider"],
                "provider_configured": model_state["configured"],
                "provider_detail": model_state["detail"],
                "start_hotkey": self.config.agent.start_hotkey,
                "stop_hotkey": self.config.agent.stop_hotkey,
                "active_run_id": getattr(self._runtime, "run_id", None),
                "duplicate_starts": self.duplicate_starts,
                "start_count": self.start_count,
                "stop_count": self.stop_count,
                "last_outcome": self.last_outcome,
                "last_error": self.last_error,
                "last_run_id": self.last_run_id,
                "last_answered": self.last_answered,
                "last_accuracy": self.last_accuracy,
                "last_run_dir": self.last_run_dir,
            }

    def _run(self) -> None:
        try:
            runtime = self.runtime_factory()
            with self._lock:
                self._runtime = runtime
                if self._stop_requested:
                    runtime.orchestrator.request_stop()
                else:
                    self._state = AgentStatus.RUNNING
            report = runtime.run()
            with self._lock:
                self.last_outcome = report.outcome.value
                self.last_run_id = report.run_id
                self.last_answered = len(report.questions)
                self.last_accuracy = report.accuracy
                self.last_run_dir = str(runtime.run_dir)
        except Exception as exc:
            # Error class only: prevent provider/network exception payloads from
            # accidentally putting secrets into the visible status UI.
            with self._lock:
                self.last_error = type(exc).__name__
                self.last_outcome = "failed_safe"
        finally:
            with self._lock:
                self._runtime = None
                self._state = AgentStatus.IDLE
                self._stop_requested = False


__all__ = ["AgentController", "AgentStatus"]
