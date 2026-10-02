"""Runtime wiring: turn a config plus a *target* into a runnable engine.

Two targets are supported, and they differ in exactly two modules -- capture and
actuation.  Everything else (perception, extraction, solver, confidence, safety,
verification, navigation, recovery, persistence, telemetry, the FSM) is the same
code path in both, which is the whole point of the simulated world: it exercises
the real closed loop, not a parallel toy loop.

``target="sim"``     offline.  Frames come from :mod:`quizengine.sim` rendering
                     annotated fixture scenes; clicks are applied to that world.
                     No API keys, no display, no OS integration required.
``target="screen"``  the real desktop (Windows-first: ``mss`` capture plus a
                     ``pyautogui``/AutoHotkey actuator).  Refuses to run without
                     the section 3.3 attestation and halts in a restricted
                     environment.

Scenarios name a fixture sequence: ``demo``, ``variant-matrix``, ``chaos``,
``end-state``, ``restricted``, ``duplicate``, ``budget``, or ``fixture-dir``
(replay frames written by ``quizengine fixtures write``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .action.backends import SimulatedBackend
from .capture.backends import SyntheticBackend
from .capture.module import CaptureModule
from .config import EngineConfig
from .contracts import RunReport
from .fixtures import (
    DEFAULT_VARIANT_ORDER,
    DEMO_QUESTIONS,
    QUESTION_BANK,
    build_scene,
    build_sequence,
    demo_sequence,
    variant_matrix_sequence,
)
from .orchestrator import Orchestrator
from .perception.module import PerceptionModule
from .perception.ocr import AnnotationOCR, NullOCR, build_ocr_engine
from .safety import ATTESTATION_TEXT
from .scenes import Scene, SceneSequence
from .sim import SimulatedWorld
from .telemetry import NullOperator, ScriptedOperator, StdinOperator, Telemetry

#: Scenarios the harness can build without any external input.
SCENARIOS: Tuple[str, ...] = (
    "demo",
    "variant-matrix",
    "chaos",
    "end-state",
    "restricted",
    "duplicate",
    "budget",
)

#: Chaos injections applied by the ``chaos`` scenario (FR-13.2 parity).
CHAOS_INJECTIONS: Tuple[Tuple[str, int], ...] = (
    ("toast", 1),
    ("layout_shift", 1),
    ("stale", 1),
    ("blank", 1),
)


# --------------------------------------------------------------------------- #
# runtime
# --------------------------------------------------------------------------- #
@dataclass
class Runtime:
    """A fully wired engine plus the handles tests and the CLI need."""

    config: EngineConfig
    orchestrator: Orchestrator
    telemetry: Telemetry
    capture: CaptureModule
    target: str = "sim"
    scenario: str = "demo"
    world: Optional[SimulatedWorld] = None
    sequence: Optional[SceneSequence] = None
    ocr_notes: List[str] = field(default_factory=list)

    # -- convenience ------------------------------------------------------- #
    def run(self) -> RunReport:
        """Run the closed loop to a terminal state and return the report."""
        return self.orchestrator.run()

    @property
    def run_dir(self) -> Path:
        return Path(self.orchestrator.artifacts.run_dir)

    @property
    def run_id(self) -> str:
        return self.orchestrator.run_id

    def describe(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "scenario": self.scenario,
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "config_fingerprint": self.config.fingerprint(),
            "ocr": self.orchestrator.perception.ocr.name if self.orchestrator.perception else None,
            "ocr_notes": list(self.ocr_notes),
            "scenes": len(self.sequence.scenes) if self.sequence is not None else None,
            "world_index": self.world.index if self.world is not None else None,
            "answer_key_size": len(self.sequence.answer_key) if self.sequence is not None else None,
        }


# --------------------------------------------------------------------------- #
# sequences
# --------------------------------------------------------------------------- #
def scenario_sequence(
    scenario: str,
    *,
    fixture_dir: Optional[str | Path] = None,
    questions: Optional[Sequence[Tuple[str, Sequence[str], int]]] = None,
) -> SceneSequence:
    """Build the fixture sequence named by ``scenario``."""
    if scenario == "fixture-dir":
        if not fixture_dir:
            raise ValueError("scenario 'fixture-dir' needs fixture_dir=")
        return sequence_from_dir(fixture_dir)
    if scenario == "demo":
        return demo_sequence()
    if scenario == "variant-matrix":
        return variant_matrix_sequence()
    if scenario == "chaos":
        keys = ["popup_modal", "low_contrast", "auto_advance", "card_grid_2x2", "submit_final"]
        bank = list(questions or QUESTION_BANK)
        return build_sequence(
            [bank[index % len(bank)] for index in range(len(keys))],
            variants=keys,
            name="chaos",
            overlay_on=0,
        )
    if scenario == "end-state":
        # Only the results screen: the loop must detect the end state and stop
        # without ever treating the summary as a question (FR-7.8.4).
        scene = build_scene("results_screen", "", (), name="results")
        return SceneSequence(name="end-state", scenes=[scene], answer_key={})
    if scenario == "restricted":
        # A locked/proctored screen: the gatekeeper must halt, not answer.
        locked = build_scene("radio_vertical", QUESTION_BANK[0][0], QUESTION_BANK[0][1], correct=0, name="locked")
        locked = locked.model_copy(update={"screen_role": "lock_screen"})
        return SceneSequence(name="restricted", scenes=[locked], answer_key={})
    if scenario == "duplicate":
        # The same question twice: L9 forbids answering a hash already answered.
        first = DEMO_QUESTIONS[0]
        return build_sequence(
            [first, first, DEMO_QUESTIONS[1]],
            variants=["radio_vertical", "radio_vertical", "card_grid_2x2"],
            name="duplicate",
        )
    if scenario == "budget":
        # More questions than the budget allows: FR-7.10.2 must abort the run.
        bank = list(QUESTION_BANK)
        return build_sequence(bank, variants=["radio_vertical"], name="budget")
    raise ValueError(f"unknown scenario {scenario!r}; expected one of {', '.join(SCENARIOS)} or 'fixture-dir'")


def sequence_from_dir(fixture_dir: str | Path) -> SceneSequence:
    """Load ``scene-*.json`` frames written by :func:`quizengine.fixtures.write_fixture_frames`."""
    from .capture.backends import load_scenes

    directory = Path(fixture_dir)
    scenes: List[Scene] = list(load_scenes(directory))
    if not scenes:
        raise ValueError(f"no fixture scenes found under {directory}")
    answer_key: Dict[str, int] = {}
    for scene in scenes:
        correct = scene.meta.get("correct_index")
        if scene.question_text and correct is not None:
            answer_key[scene.question_text] = int(correct)
    return SceneSequence(name=f"fixtures:{directory.name}", scenes=scenes, answer_key=answer_key)


# --------------------------------------------------------------------------- #
# config shaping
# --------------------------------------------------------------------------- #
def apply_overrides(config: EngineConfig, overrides: Optional[Sequence[str]]) -> EngineConfig:
    """Apply ``section.field=value`` overrides (CLI ``--set``), typed by pydantic.

    Values are parsed as YAML scalars so ``--set budgets.max_questions=30`` yields
    an int and ``--set run.dev_mode=false`` yields a bool.
    """
    if not overrides:
        return config
    import yaml

    payload = config.to_dict()
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set expects section.field=value, got {item!r}")
        path, _, raw = item.partition("=")
        keys = [part for part in path.strip().split(".") if part]
        if len(keys) < 2:
            raise ValueError(f"--set needs a section and a field: {item!r}")
        cursor: Dict[str, Any] = payload
        for key in keys[:-1]:
            nxt = cursor.get(key)
            if not isinstance(nxt, dict):
                nxt = {}
                cursor[key] = nxt
            cursor = nxt
        cursor[keys[-1]] = yaml.safe_load(raw) if raw.strip() else None
    return EngineConfig.from_dict(payload, source_path=config.source_path)


def sim_config(
    config: Optional[EngineConfig] = None,
    *,
    scenario: str = "demo",
    runs_dir: Optional[str | Path] = None,
    console: bool = False,
    screenshots: bool = False,
) -> EngineConfig:
    """Return a copy of ``config`` shaped for the offline simulated target.

    The overrides are the minimum needed to run with no display, no OCR binary
    and no API key; they never weaken a safety or verification setting.
    """
    cfg = (config or EngineConfig.default()).model_copy(deep=True)
    cfg.capture.backend = "synthetic"
    cfg.capture.monitor = 1
    if not cfg.capture.fixture_dir:
        cfg.capture.fixture_dir = "fixtures"
    cfg.perception.ocr_engine = "annotation"
    # FR-7.2.2 forbids raw OCR input for real engines; the deterministic
    # annotation backend is the one case where it is allowed (config post-validation).
    cfg.perception.preprocess.allow_raw_ocr = True
    cfg.telemetry.console = console
    cfg.telemetry.trace_screenshots = screenshots
    cfg.run.deterministic = True
    if runs_dir is not None:
        cfg.paths.runs_dir = str(runs_dir)
    # Verification polls frames; keep the offline loop snappy but still multi-frame.
    cfg.verification.verify_timeout_ms = min(cfg.verification.verify_timeout_ms, 400.0)
    cfg.verification.verify_recheck_ms = max(
        cfg.verification.verify_timeout_ms, min(cfg.verification.verify_recheck_ms, 900.0)
    )
    cfg.verification.max_post_action_frames = max(2, min(cfg.verification.max_post_action_frames, 3))
    if scenario == "variant-matrix":
        cfg.budgets.max_questions = max(cfg.budgets.max_questions, 40)
        cfg.budgets.max_cycles = max(cfg.budgets.max_cycles, 240)
    if scenario == "budget":
        cfg.budgets.max_questions = min(cfg.budgets.max_questions, 3)
    return cfg


# --------------------------------------------------------------------------- #
# builder
# --------------------------------------------------------------------------- #
def build_runtime(
    *,
    config: Optional[EngineConfig] = None,
    config_path: Optional[str | Path] = None,
    target: str = "sim",
    scenario: str = "demo",
    fixture_dir: Optional[str | Path] = None,
    questions: Optional[Sequence[Tuple[str, Sequence[str], int]]] = None,
    sequence: Optional[SceneSequence] = None,
    world: Optional[SimulatedWorld] = None,
    answer_key: Optional[Dict[str, Any]] = None,
    operator: Any = None,
    operator_script: Optional[Sequence[str]] = None,
    console: bool = False,
    screenshots: bool = False,
    runs_dir: Optional[str | Path] = None,
    attest: Optional[bool] = None,
    attestation: Optional[str] = None,
    overrides: Optional[Sequence[str]] = None,
    injections: Optional[Sequence[Tuple[str, int]]] = None,
) -> Runtime:
    """Wire every module and return a :class:`Runtime` ready to ``run()``.

    ``attest`` supplies the section 3.3 attestation text when the caller has not
    set one; pass ``attest=False`` to exercise the refusal path.
    """
    if target not in {"sim", "screen"}:
        raise ValueError(f"target must be 'sim' or 'screen', got {target!r}")

    from .runtime_config import apply_environment, load_dotenv

    load_dotenv()
    cfg = config or (EngineConfig.load(config_path) if config_path else EngineConfig.default())
    cfg = apply_environment(cfg)
    cfg = apply_overrides(cfg, overrides)
    if target == "sim":
        cfg = sim_config(cfg, scenario=scenario, runs_dir=runs_dir, console=console, screenshots=screenshots)
    elif runs_dir is not None:
        cfg = cfg.model_copy(deep=True)
        cfg.paths.runs_dir = str(runs_dir)
        cfg.telemetry.console = console
        cfg.telemetry.trace_screenshots = screenshots

    # The offline fixture target is pre-authorized test data. A real desktop run
    # is different: default to *no* attestation and require an explicit caller
    # opt-in (CLI: --attest) or an attestation already present in operator config.
    supply_attestation = (target == "sim") if attest is None else bool(attest)
    if attestation is not None:
        cfg.run.attestation = attestation
    elif supply_attestation and not cfg.run.attestation:
        cfg.run.attestation = ATTESTATION_TEXT

    # -- capture / world ---------------------------------------------------- #
    ocr_notes: List[str] = []
    sim_world = world
    sim_sequence = sequence
    if target == "sim":
        if sim_world is None:
            sim_sequence = sim_sequence or scenario_sequence(
                scenario, fixture_dir=fixture_dir, questions=questions
            )
            sim_world = SimulatedWorld(sim_sequence)
        elif sim_sequence is None:
            sim_sequence = getattr(sim_world, "sequence", None)
        capture_backend: Any = SyntheticBackend(world=sim_world)
        capture = CaptureModule(cfg, backend=capture_backend)
        action_backend: Any = SimulatedBackend(sim_world)
    else:
        if cfg.run.attestation:
            # Construct the real capture backend only after explicit consent is
            # configured; constructor probes the display on some platforms.
            capture = CaptureModule(cfg)
        else:
            # A no-attestation run must produce the normal halt/report without
            # even probing a real monitor. The safety gate runs first in run().
            # This inert frame source can never reach the action loop.
            inert_scene = build_scene("radio_vertical", "Authorization required", ["Stop", "Continue"])
            capture = CaptureModule(cfg, backend=SyntheticBackend(scenes=[inert_scene]))
        action_backend = None
        sim_sequence = None

    # -- perception --------------------------------------------------------- #
    ocr = _build_ocr(cfg, ocr_notes, target=target)
    perception = PerceptionModule(cfg, ocr=ocr, capture=capture)

    # -- operator ----------------------------------------------------------- #
    channel = operator
    if channel is None:
        if operator_script:
            channel = ScriptedOperator(list(operator_script))
        elif target == "screen":
            channel = StdinOperator()
        else:
            # Unattended offline runs must not block on a prompt: abort instead,
            # which is the safe answer (FR-7.10.3).
            channel = NullOperator(on_prompt="abort")

    orchestrator = Orchestrator(
        cfg,
        capture=capture,
        perception=perception,
        world=sim_world,
        backend=action_backend,
        operator=channel,
        answer_key=answer_key if answer_key is not None else (sim_sequence.answer_key if sim_sequence else None),
    )

    for mode, times in list(injections or []) + (list(CHAOS_INJECTIONS) if scenario == "chaos" and not injections else []):
        if sim_world is not None:
            sim_world.inject(mode, times)

    return Runtime(
        config=cfg,
        orchestrator=orchestrator,
        telemetry=orchestrator.telemetry,
        capture=capture,
        target=target,
        scenario=scenario,
        world=sim_world,
        sequence=sim_sequence,
        ocr_notes=ocr_notes,
    )


def _build_ocr(cfg: EngineConfig, notes: List[str], *, target: str) -> Any:
    """Pick the OCR engine, recording *why* each candidate was skipped."""
    if target == "sim" and cfg.perception.ocr_engine == "annotation":
        notes.append("annotation: selected (deterministic ground-truth OCR for fixture backends)")
        return AnnotationOCR()
    try:
        engine, trail = build_ocr_engine(cfg.perception, cfg.capture.backend, notes=notes)
    except Exception as exc:  # CapabilityError and friends: report, do not crash here
        notes.append(f"{type(exc).__name__}: {exc}")
        if target == "sim":
            notes.append("annotation: fallback for the simulated target")
            return AnnotationOCR()
        # Construct enough runtime to run the safety gate before frame capture.
        # An authorized real run with no OCR will remain blind and fail closed;
        # lack of a local OCR binary must never bypass the attestation refusal.
        notes.append("none: screen run is content-blind until a real OCR engine is installed; safety gate still runs first")
        return NullOCR()
    return engine


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def summarize(report: RunReport, *, stats: Any = None, lines: Optional[List[str]] = None) -> str:
    """Human-readable run summary (the CLI prints it, tests assert on it)."""
    out = lines if lines is not None else []
    out.append(f"run {report.run_id}  outcome={report.outcome.value}  time={report.total_time_s:.2f}s")
    accuracy = "n/a" if report.accuracy is None else f"{report.accuracy:.2f}"
    out.append(
        f"  answered={len(report.questions)}  correct={report.correct}  incorrect={report.incorrect}  accuracy={accuracy}"
    )
    out.append(
        f"  unverified_actions={report.unverified_actions}  illegal_transitions={report.illegal_transitions}  "
        f"stale_coordinate_violations={report.stale_coordinate_violations}"
    )
    if report.halted_code is not None:
        out.append(f"  halted: {report.halted_code.value} -- {report.halted_detail[:160]}")
    if stats is not None:
        out.append(
            f"  cycles={stats.cycles} answered={stats.answered} retries={stats.retries} "
            f"recoveries={stats.recoveries} verifications={stats.verifications_passed}+"
            f"{stats.verifications_failed} overlays={stats.overlays_dismissed} "
            f"navigations={stats.navigations} scrolls={stats.scrolls} end_state_frames={stats.end_state_frames}"
        )
    if report.decision_traces:
        out.append("  decisions:")
        for trace in report.decision_traces:
            out.append(f"    {trace.human_summary()} | {trace.outcome}")
    if report.failure_bundles:
        out.append("  failure bundles:")
        for bundle in report.failure_bundles:
            out.append(f"    {bundle.code.value}: {bundle.path}")
    return "\n".join(out)


__all__ = [
    "CHAOS_INJECTIONS",
    "DEFAULT_VARIANT_ORDER",
    "Runtime",
    "SCENARIOS",
    "apply_overrides",
    "build_runtime",
    "scenario_sequence",
    "sequence_from_dir",
    "sim_config",
    "summarize",
]
