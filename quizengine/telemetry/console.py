"""Live operator console (FR-7.13.5) and the AWAITING_HUMAN input channel (FR-10.3).

The console is deliberately dumb: it renders state and forwards operator
commands.  All policy lives in the orchestrator, so a TUI, a stdin prompt or a
scripted test driver are interchangeable behind :class:`OperatorChannel`.

NFR-17.6: gatekeeper status and the attestation are always visible.
"""

from __future__ import annotations

import abc
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..contracts import Attestation, State, UncertaintyPolicy


class CommandKind:
    RESUME = "resume"
    ANSWER = "answer"
    SKIP = "skip"
    ABORT = "abort"
    PAUSE = "pause"
    STOP = "stop"
    STATUS = "status"
    NONE = "none"


ALL_COMMANDS = (
    CommandKind.RESUME,
    CommandKind.ANSWER,
    CommandKind.SKIP,
    CommandKind.ABORT,
    CommandKind.PAUSE,
    CommandKind.STOP,
    CommandKind.STATUS,
)


@dataclass(frozen=True)
class OperatorCommand:
    kind: str = CommandKind.NONE
    payload: Optional[str] = None
    ts: float = field(default_factory=time.time)

    @classmethod
    def parse(cls, text: str) -> "OperatorCommand":
        """Parse ``resume`` / ``answer B`` / ``B`` / ``skip`` / ``abort`` / ``stop``."""
        raw = (text or "").strip()
        if not raw:
            return cls(CommandKind.NONE)
        parts = raw.split()
        head = parts[0].lower()
        rest = parts[1] if len(parts) > 1 else None
        if head in {"answer", "a", "choose"}:
            if rest is None:
                return cls(CommandKind.NONE)
            return cls(CommandKind.ANSWER, rest.upper())
        if head in {"resume", "continue", "r"}:
            return cls(CommandKind.RESUME)
        if head in {"skip", "s", "next"}:
            return cls(CommandKind.SKIP)
        if head in {"abort", "quit", "q"}:
            return cls(CommandKind.ABORT)
        if head in {"stop", "halt"}:
            return cls(CommandKind.STOP)
        if head in {"pause", "p"}:
            return cls(CommandKind.PAUSE)
        if head in {"status", "?"}:
            return cls(CommandKind.STATUS)
        if len(head) == 1 and "a" <= head <= "z":
            return cls(CommandKind.ANSWER, head.upper())
        return cls(CommandKind.NONE, raw)

    @property
    def is_terminal(self) -> bool:
        return self.kind in {CommandKind.ABORT, CommandKind.STOP}


class OperatorChannel(abc.ABC):
    """Where AWAITING_HUMAN gets its decision from."""

    @abc.abstractmethod
    def request_decision(self, prompt: str, policy: UncertaintyPolicy, context: Dict[str, Any]) -> OperatorCommand:
        """Block until the operator answers (or the channel's policy resolves it)."""

    def poll_command(self) -> OperatorCommand:
        """Non-blocking control command (pause/stop) -- default: nothing."""
        return OperatorCommand()

    def close(self) -> None:  # pragma: no cover - default no-op
        return None


class NullOperator(OperatorChannel):
    """Headless default: never answers, always declines.

    Used by CI and by ``--deterministic`` runs so that a low-confidence question
    produces a *halt* instead of a hang.
    """

    def __init__(self, on_prompt: str = "abort") -> None:
        self.prompts: List[Dict[str, Any]] = []
        self._on_prompt = on_prompt

    def request_decision(self, prompt: str, policy: UncertaintyPolicy, context: Dict[str, Any]) -> OperatorCommand:
        self.prompts.append({"prompt": prompt, "policy": policy.value, "context": context, "ts": time.time()})
        return OperatorCommand.parse(self._on_prompt)


class ScriptedOperator(OperatorChannel):
    """Deterministic test driver: replay a scripted list of decisions."""

    def __init__(self, script: List[str]) -> None:
        self._script = [OperatorCommand.parse(item) for item in script]
        self._index = 0
        self.prompts: List[Dict[str, Any]] = []

    def request_decision(self, prompt: str, policy: UncertaintyPolicy, context: Dict[str, Any]) -> OperatorCommand:
        self.prompts.append({"prompt": prompt, "policy": policy.value, "context": context})
        if self._index >= len(self._script):
            return OperatorCommand(CommandKind.ABORT, "script exhausted")
        command = self._script[self._index]
        self._index += 1
        return command

    @property
    def consumed(self) -> int:
        return self._index


class StdinOperator(OperatorChannel):
    """Interactive operator: prompt on stdin, control keys polled from a thread.

    Pause is honored *between* cycles (FR-7.10.3), never mid-action: this class
    only records the request, the orchestrator decides when to honor it.
    """

    def __init__(self, stream: Any = None, input_stream: Any = None, echo: bool = True) -> None:
        self._out = stream if stream is not None else sys.stdout
        self._in = input_stream if input_stream is not None else sys.stdin
        self._echo = echo
        self._pending: List[OperatorCommand] = []
        self._lock = threading.Lock()
        self._stop_reader = threading.Event()
        self._reader: Optional[threading.Thread] = None
        self.paused = False

    def start_control_reader(self) -> None:
        """Background reader for pause/stop/status typed between prompts."""
        if self._reader is not None or not self._in.isatty():
            return
        self._reader = threading.Thread(target=self._read_loop, daemon=True, name="operator-console")
        self._reader.start()

    def _read_loop(self) -> None:  # pragma: no cover - interactive only
        while not self._stop_reader.is_set():
            try:
                line = self._in.readline()
            except Exception:
                return
            if not line:
                return
            command = OperatorCommand.parse(line)
            if command.kind in {CommandKind.PAUSE, CommandKind.RESUME, CommandKind.STOP, CommandKind.STATUS}:
                with self._lock:
                    self._pending.append(command)

    def poll_command(self) -> OperatorCommand:
        with self._lock:
            return self._pending.pop(0) if self._pending else OperatorCommand()

    def request_decision(self, prompt: str, policy: UncertaintyPolicy, context: Dict[str, Any]) -> OperatorCommand:
        self.write("")
        self.write("=" * 78)
        self.write("HUMAN DECISION REQUIRED  (AWAITING_HUMAN blocks all actions - FR-10.3)")
        self.write("-" * 78)
        self.write(prompt)
        options = context.get("options") or []
        for index, text in enumerate(options):
            marker = " *" if context.get("proposed_index") == index else "  "
            self.write(f" {chr(ord('A') + index)}{marker} {text}")
        if context.get("rationale"):
            self.write(f"  rationale: {context['rationale']}")
        self.write(
            f"  confidence={context.get('confidence', 0.0):.2f} "
            f"threshold={context.get('low_conf', 0.0):.2f} policy={policy.value}"
        )
        self.write("-" * 78)
        self.write("  commands: resume | answer <A-Z> | skip | abort")
        self.write("=" * 78)
        try:
            line = input("quizengine> ")
        except (EOFError, KeyboardInterrupt):
            return OperatorCommand(CommandKind.ABORT, "stdin closed")
        command = OperatorCommand.parse(line)
        if command.kind == CommandKind.NONE:
            self.write(f"  unrecognized command {line!r}; treating as abort (L4: fail safe)")
            return OperatorCommand(CommandKind.ABORT, f"unrecognized: {line!r}")
        return command

    def write(self, text: str) -> None:
        if not self._echo:
            return
        try:
            self._out.write(text + "\n")
            self._out.flush()
        except Exception:  # pragma: no cover
            pass

    def close(self) -> None:
        self._stop_reader.set()


class OperatorConsole:
    """Single-line status renderer (FR-7.13.5) + safety-state banner (NFR-17.6)."""

    def __init__(self, stream: Any = None, enabled: bool = True) -> None:
        self._out = stream if stream is not None else sys.stdout
        self.enabled = enabled
        self._last_line = ""
        self._is_tty = bool(getattr(self._out, "isatty", lambda: False)())

    def banner(self, run_id: str, attestation: Optional[Attestation], gate_status: str, config_fingerprint: str) -> None:
        if not self.enabled:
            return
        attest = "RECORDED" if attestation else "MISSING"
        lines = [
            "=" * 78,
            f" QuizEngine run {run_id}",
            f" gatekeeper : {gate_status}",
            f" attestation: {attest}" + (f"  ({attestation.text[:48]}...)" if attestation and len(attestation.text) > 48 else ""),
            f" config     : {config_fingerprint}",
            "=" * 78,
        ]
        self._print("\n".join(lines))

    def status(
        self,
        *,
        state: State,
        question_no: Optional[int] = None,
        total: Optional[int] = None,
        confidence: Optional[float] = None,
        gate_status: str = "clear",
        message: str = "",
        cycle: int = 0,
    ) -> None:
        if not self.enabled:
            return
        qno = "-" if question_no is None else str(question_no)
        tot = "?" if total is None else str(total)
        conf = "-" if confidence is None else f"{confidence:.2f}"
        line = (
            f"[{time.strftime('%H:%M:%S')}] cycle={cycle:>3} state={state.value:<18} "
            f"q={qno}/{tot} conf={conf} gate={gate_status}"
        )
        if message:
            line += f" | {message}"
        if self._is_tty and line != self._last_line:
            self._out.write("\r\x1b[K" + line)
            self._out.flush()
        elif line != self._last_line:
            self._print(line)
        self._last_line = line

    def notice(self, text: str) -> None:
        if self.enabled:
            self._print(text)

    def newline(self) -> None:
        if self.enabled and self._is_tty:
            self._print("")

    def _print(self, text: str) -> None:
        try:
            self._out.write(text + "\n")
            self._out.flush()
        except Exception:  # pragma: no cover
            pass
