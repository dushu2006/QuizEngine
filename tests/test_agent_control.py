from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from quizengine.agent import AgentController, AgentStatus, HotkeySpec, WindowsHotkeyListener
from quizengine.config import EngineConfig


def test_hotkey_spec_parses_and_rejects_unsafe_or_ambiguous_keys():
    parsed = HotkeySpec.parse("Win+ALT+q")
    assert parsed.source == "win+alt+q"
    assert parsed.virtual_key == ord("Q")
    assert parsed.modifiers != 0
    for invalid in ("q", "ctrl+", "ctrl+q+z", "ctrl+ctrl+q", "ctrl+é"):
        with pytest.raises(ValueError):
            HotkeySpec.parse(invalid)


def test_hotkey_listener_requires_distinct_start_and_stop():
    with pytest.raises(ValueError, match="must differ"):
        WindowsHotkeyListener("win+alt+q", "win+alt+Q", lambda: None, lambda: None)


def test_controller_prevents_parallel_runs_and_delivers_stop_request():
    entered = threading.Event()
    release = threading.Event()
    stop_called = threading.Event()

    class FakeOrchestrator:
        def request_stop(self):
            stop_called.set()

    class FakeRuntime:
        run_id = "test-run"
        orchestrator = FakeOrchestrator()

        def run(self):
            entered.set()
            assert release.wait(2)
            return SimpleNamespace(
                outcome=SimpleNamespace(value="stopped_by_operator"),
                run_id="test-run",
                questions=[],
                accuracy=None,
            )

        @property
        def run_dir(self):
            return "test-artifacts"

    controller = AgentController(EngineConfig.default(), FakeRuntime)
    assert controller.start() is True
    assert entered.wait(1)
    assert controller.state in {AgentStatus.RUNNING, AgentStatus.STOPPING}
    assert controller.start() is False
    assert controller.snapshot()["duplicate_starts"] == 1
    assert controller.stop() is True
    assert stop_called.wait(1)
    release.set()
    assert controller.wait(2)
    assert controller.state is AgentStatus.IDLE
    assert controller.snapshot()["last_outcome"] == "stopped_by_operator"


def test_controller_rejects_stop_when_idle():
    controller = AgentController(EngineConfig.default(), lambda: None)
    assert controller.stop() is False
    assert controller.snapshot()["status"] == "IDLE"


def test_hotkeys_report_platform_capability_without_starting_capture():
    listener = WindowsHotkeyListener("win+alt+q", "win+alt+x", lambda: None, lambda: None)
    if not listener.supported():
        with pytest.raises(RuntimeError, match="require Windows"):
            listener.start_listening()
