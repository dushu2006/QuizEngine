"""``quizengine`` -- the command-line interface (section 15).

Everything the engine can do offline is reachable from here without an API key,
a display or a Windows host:

    quizengine demo                       closed loop over the demo quiz
    quizengine run --scenario variant-matrix
                                          every layout variant, one run
    quizengine selftest                   the acceptance-criteria suite
    quizengine run --target screen --attest
                                          the real desktop (Windows-first)
    quizengine doctor                     what this machine can and cannot do
    quizengine fixtures write out/        render frames + ground truth
    quizengine variants                   the layout-variant catalogue
    quizengine probe                      capability table
    quizengine config show|fingerprint    effective configuration
    quizengine report RUN_DIR             re-read a stored run report

Exit codes: 0 completed, 2 halted, 3 aborted, 4 failed safe, 5 stopped by the
operator, 1 usage/configuration error.  A non-zero exit is never a crash: the run
report is written first (FR-7.12).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import PRD_VERSION, __version__
from .harness import SCENARIOS, Runtime, build_runtime, scenario_sequence, summarize

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_HALTED = 2
EXIT_ABORTED = 3
EXIT_FAILED_SAFE = 4
EXIT_STOPPED = 5

#: ``RunOutcome.value`` -> process exit code.
EXIT_BY_OUTCOME = {
    "completed": EXIT_OK,
    "halted": EXIT_HALTED,
    "aborted": EXIT_ABORTED,
    "failed_safe": EXIT_FAILED_SAFE,
    "stopped_by_operator": EXIT_STOPPED,
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _emit(payload: Any, as_json: bool, stream: Any = None) -> None:
    out = stream or sys.stdout
    if as_json:
        out.write(json.dumps(payload, indent=2, default=str) + "\n")
    else:
        out.write(str(payload) + "\n")
    out.flush()


def _config_for(args: argparse.Namespace) -> Optional[Any]:
    from .config import EngineConfig

    if getattr(args, "config", None):
        return EngineConfig.load(args.config)
    return None


def _split_script(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    return [item.strip() for item in raw.split(",") if item.strip()]


def _run_and_report(runtime: Runtime, *, as_json: bool, show: bool = True) -> int:
    report = runtime.run()
    stats = runtime.orchestrator.stats
    if as_json:
        payload: Dict[str, Any] = json.loads(report.model_dump_json())
        payload["stats"] = stats.summary()
        payload["runtime"] = runtime.describe()
        _emit(payload, True)
    elif show:
        _emit(summarize(report, stats=stats), False)
        _emit(f"  artifacts: {runtime.run_dir}", False)
    return EXIT_BY_OUTCOME.get(report.outcome.value, EXIT_FAILED_SAFE)


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_demo(args: argparse.Namespace) -> int:
    """The offline closed loop, end to end, with nothing to install."""
    runtime = build_runtime(
        config=_config_for(args),
        target="sim",
        scenario=args.scenario,
        runs_dir=args.runs_dir,
        console=args.console,
        screenshots=args.screenshots,
        attest=not args.no_attest,
        overrides=args.set,
    )
    if not args.json:
        _emit(f"QuizEngine {__version__} (PRD v{PRD_VERSION}) -- offline demo", False)
        _emit(f"  target=sim scenario={args.scenario} ocr={runtime.describe()['ocr']}", False)
        _emit(f"  scenes={len(runtime.sequence.scenes) if runtime.sequence else 0} "
              f"attestation={'given' if runtime.config.run.attestation else 'MISSING'}", False)
    return _run_and_report(runtime, as_json=args.json)


def cmd_run(args: argparse.Namespace) -> int:
    """Run against the simulated world or the real screen."""
    runtime = build_runtime(
        config=_config_for(args),
        target=args.target,
        scenario=args.scenario,
        fixture_dir=args.fixture_dir,
        runs_dir=args.runs_dir,
        console=args.console,
        screenshots=args.screenshots,
        attest=(args.attest if args.target == "screen" else not args.no_attest),
        attestation=args.attestation,
        operator_script=_split_script(args.operator_script),
        overrides=args.set,
    )
    if args.target == "screen" and not args.json:
        _emit("target=screen: capturing the real desktop. Ctrl-C or 'stop' at a prompt to end the run.", False)
    return _run_and_report(runtime, as_json=args.json)


def cmd_agent(args: argparse.Namespace) -> int:
    """Wait for explicit Windows start/stop hotkeys; never starts on launch."""
    import threading
    import time

    from .agent import AgentController, WindowsHotkeyListener
    from .runtime_config import apply_environment, load_dotenv, provider_status

    if not WindowsHotkeyListener.supported():
        _emit("global hotkeys are Windows-only; no runtime was started", False, sys.stderr)
        return EXIT_USAGE
    if not args.attest:
        _emit("refusing to arm real-screen hotkeys without explicit --attest authorization", False, sys.stderr)
        return EXIT_USAGE

    load_dotenv()
    config = apply_environment(_config_for(args))
    # Apply --set before runtime construction as the regular run command does.
    from .harness import apply_overrides
    config = apply_overrides(config, args.set)
    info = provider_status(config)

    def make_runtime() -> Runtime:
        return build_runtime(
            config=config,
            target="screen",
            runs_dir=args.runs_dir,
            attest=True,
            overrides=args.set,
        )

    controller = AgentController(config, make_runtime)
    output_lock = threading.Lock()

    def say(text: str) -> None:
        with output_lock:
            _emit(text, False)

    def on_start() -> None:
        accepted = controller.start()
        say("hotkey start accepted; run is starting" if accepted else "start ignored: a run is already active")

    def on_stop() -> None:
        requested = controller.stop()
        say("stop requested; waiting for the current bounded operation" if requested else "stop ignored: no run is active")

    listener = WindowsHotkeyListener(
        config.agent.start_hotkey,
        config.agent.stop_hotkey,
        on_start,
        on_stop,
    )
    try:
        listener.start_listening()
        say("QuizEngine agent armed; no screen capture or action occurs until the start hotkey.")
        say(f"provider={info['provider']} configured={info['configured']} ({info['detail']})")
        say(f"start={config.agent.start_hotkey} | stop={config.agent.stop_hotkey} | Ctrl+C exits")
        previous = controller.state
        previous_result: Optional[tuple[int, Optional[str]]] = None
        while True:
            time.sleep(0.25)
            current = controller.state
            if current != previous:
                say(f"agent status: {current.value}")
                previous = current
            snapshot = controller.snapshot()
            outcome = snapshot["last_outcome"]
            result_key = (snapshot["start_count"], outcome)
            if outcome is not None and result_key != previous_result:
                if snapshot["last_error"]:
                    say(f"last run failed safely ({snapshot['last_error']})")
                else:
                    accuracy = snapshot["last_accuracy"]
                    accuracy_text = "n/a" if accuracy is None else f"{accuracy:.2f}"
                    say(f"last run: {outcome}; answered={snapshot['last_answered']} accuracy={accuracy_text}; "
                        f"artifacts={snapshot['last_run_dir']}")
                previous_result = result_key
    except KeyboardInterrupt:
        say("exiting; requesting stop and waiting for the current run")
        controller.stop()
        controller.wait(timeout=config.models.request_timeout_s + 5.0)
        return EXIT_STOPPED
    except RuntimeError as exc:
        say(f"agent could not be armed or continued: {exc}")
        return EXIT_USAGE
    finally:
        listener.stop_listening()


#: ``selftest`` cases: (name, scenario, expected outcome, extra assertions).
SELFTEST_CASES: Sequence[Dict[str, Any]] = (
    {
        "name": "closed loop answers every question",
        "scenario": "demo",
        "outcome": "completed",
        "expect": lambda rep, rt: len(rep.questions) == 6 and rep.accuracy == 1.0,
        "law": "L1/L2 observe-then-act, every action verified",
    },
    {
        "name": "generalizes across all layout variants",
        "scenario": "variant-matrix",
        "outcome": "completed",
        "expect": lambda rep, rt: len(rep.questions) >= 18 and rep.accuracy == 1.0 and rt.orchestrator.stats.overlays_dismissed >= 1,
        "law": "FR-7.2.3 zoom/layout invariance, FR-13.2 variants",
    },
    {
        "name": "no unverified state-changing action (AC-14.3)",
        "scenario": "variant-matrix",
        "outcome": "completed",
        "expect": lambda rep, rt: rep.unverified_actions == 0,
        "law": "L2 / AC-14.3",
    },
    {
        "name": "no stale coordinates (AC-14.2)",
        "scenario": "variant-matrix",
        "outcome": "completed",
        "expect": lambda rep, rt: rep.stale_coordinate_violations == 0,
        "law": "L1 / AC-14.2",
    },
    {
        "name": "no illegal state transition",
        "scenario": "variant-matrix",
        "outcome": "completed",
        "expect": lambda rep, rt: rep.illegal_transitions == 0,
        "law": "L4 explicit FSM",
    },
    {
        "name": "end state is confirmed, never answered (FR-7.8.4)",
        "scenario": "end-state",
        "outcome": "completed",
        "expect": lambda rep, rt: len(rep.questions) == 0 and rt.orchestrator.stats.end_state_frames >= 2,
        "law": "FR-7.8.4 two-frame end-state rule",
    },
    {
        "name": "an answered question is never re-answered (AC-14.6)",
        "scenario": "duplicate",
        "outcome": "completed",
        "expect": lambda rep, rt: rt.orchestrator.stats.skipped_duplicate >= 1
        and len({q.hash for q in rep.questions}) == len(rep.questions),
        "law": "L9 idempotence",
    },
    {
        "name": "budget exhaustion aborts the run (FR-7.10.2)",
        "scenario": "budget",
        "outcome": "aborted",
        "expect": lambda rep, rt: rep.halted_code is not None and rep.halted_code.value == "BUDGET_EXCEEDED",
        "law": "L3 bounded everything",
    },
    {
        "name": "restricted environment halts before acting (section 3.3)",
        "scenario": "restricted",
        "outcome": "halted",
        "expect": lambda rep, rt: len(rep.questions) == 0
        and rep.halted_code is not None
        and rep.halted_code.value == "RESTRICTED_ENVIRONMENT",
        "law": "section 3.3 / FR-7.14",
    },
    {
        "name": "chaos: overlays, shifts, stale and blank frames survive",
        "scenario": "chaos",
        "outcome": "completed",
        "expect": lambda rep, rt: rep.accuracy == 1.0 and rep.unverified_actions == 0,
        "law": "FR-13.2 chaos mode, FR-7.11 recovery",
    },
    {
        "name": "missing attestation refuses to act (section 3.3)",
        "scenario": "demo",
        "outcome": "halted",
        "attest": False,
        "expect": lambda rep, rt: len(rep.questions) == 0,
        "law": "FR-7.14 attestation gate",
    },
)


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Repeat deterministic offline replay runs and summarize measured latency."""
    import statistics

    elapsed_ms: List[float] = []
    iteration_p95_ms: List[float] = []
    runs: List[Dict[str, Any]] = []
    for index in range(args.repeat):
        runtime = build_runtime(
            config=_config_for(args),
            target="sim",
            scenario=args.scenario,
            fixture_dir=args.fixture_dir,
            runs_dir=args.runs_dir,
            attest=True,
            overrides=args.set,
        )
        report = runtime.run()
        stats = runtime.orchestrator.stats.summary()
        elapsed = round(report.total_time_s * 1000.0, 2)
        elapsed_ms.append(elapsed)
        p95 = float(stats.get("iteration_latency_ms", {}).get("p95", 0.0))
        iteration_p95_ms.append(p95)
        runs.append({
            "iteration": index + 1,
            "run_id": report.run_id,
            "outcome": report.outcome.value,
            "answered": len(report.questions),
            "accuracy": report.accuracy,
            "total_ms": elapsed,
            "iteration_latency_ms": stats.get("iteration_latency_ms", {}),
            "latency_ms": stats.get("latency_ms", {}),
            "unverified_actions": report.unverified_actions,
            "illegal_transitions": report.illegal_transitions,
            "stale_coordinate_violations": report.stale_coordinate_violations,
        })
    ordered = sorted(elapsed_ms)
    p95_position = (len(ordered) - 1) * 0.95
    p95_low = int(p95_position)
    p95_high = min(p95_low + 1, len(ordered) - 1)
    p95_fraction = p95_position - p95_low
    p95_total = ordered[p95_low] * (1.0 - p95_fraction) + ordered[p95_high] * p95_fraction
    payload = {
        "mode": "offline_replay",
        "scenario": args.scenario,
        "repeat": args.repeat,
        "total_ms": {
            "mean": round(statistics.fmean(elapsed_ms), 2),
            "median": round(statistics.median(elapsed_ms), 2),
            "p95": round(p95_total, 2),
            "min": round(min(elapsed_ms), 2),
            "max": round(max(elapsed_ms), 2),
        },
        "iteration_p95_ms_mean": round(statistics.fmean(iteration_p95_ms), 2),
        "safety_invariants_passed": all(
            not run["unverified_actions"]
            and not run["illegal_transitions"]
            and not run["stale_coordinate_violations"]
            for run in runs
        ),
        "runs": runs,
    }
    if args.json:
        _emit(payload, True)
    else:
        _emit(f"offline replay benchmark: scenario={args.scenario} repeat={args.repeat}", False)
        _emit(f"  total ms: mean={payload['total_ms']['mean']} median={payload['total_ms']['median']} "
              f"p95={payload['total_ms']['p95']} range={payload['total_ms']['min']}..{payload['total_ms']['max']}", False)
        _emit(f"  mean iteration p95={payload['iteration_p95_ms_mean']}ms | "
              f"safety invariants={'PASS' if payload['safety_invariants_passed'] else 'FAIL'}", False)
        for run in runs:
            accuracy = "n/a" if run["accuracy"] is None else f"{run['accuracy']:.2f}"
            _emit(f"  #{run['iteration']} {run['outcome']} answered={run['answered']} accuracy={accuracy} "
                  f"total={run['total_ms']}ms run={run['run_id']}", False)
    if not payload["safety_invariants_passed"] or any(run["outcome"] != "completed" for run in runs):
        return EXIT_FAILED_SAFE
    return EXIT_OK


def cmd_selftest(args: argparse.Namespace) -> int:
    """Run the acceptance-criteria suite offline and print a pass/fail table."""
    cases = [c for c in SELFTEST_CASES if not args.quick or c["scenario"] in {"demo", "end-state", "restricted"}]
    width = max(len(case["name"]) for case in cases)
    failures: List[str] = []
    _emit(f"QuizEngine {__version__} selftest -- PRD v{PRD_VERSION}, {len(cases)} case(s)\n", False)
    for case in cases:
        runtime = build_runtime(
            target="sim",
            scenario=case["scenario"],
            runs_dir=args.runs_dir,
            attest=case.get("attest", True),
        )
        report = runtime.run()
        ok_outcome = report.outcome.value == case["outcome"]
        try:
            ok_expect = bool(case["expect"](report, runtime))
        except Exception as exc:  # a broken assertion is a failure, not a crash
            ok_expect = False
            failures.append(f"{case['name']}: assertion raised {type(exc).__name__}: {exc}")
        passed = ok_outcome and ok_expect
        detail = f"outcome={report.outcome.value}"
        if not ok_outcome:
            detail += f" (expected {case['outcome']})"
        detail += f" answered={len(report.questions)}"
        if report.accuracy is not None:
            detail += f" accuracy={report.accuracy:.2f}"
        if report.halted_code is not None:
            detail += f" halted={report.halted_code.value}"
        _emit(f"[{'PASS' if passed else 'FAIL'}] {case['name']:<{width}}  {detail}", False)
        if not passed and not any(f.startswith(case["name"]) for f in failures):
            failures.append(f"{case['name']}: {detail} [{case['law']}]")
    _emit("", False)
    if failures:
        _emit(f"{len(failures)} case(s) failed:", False)
        for line in failures:
            _emit(f"  - {line}", False)
        return EXIT_FAILED_SAFE
    _emit(f"all {len(cases)} case(s) passed", False)
    return EXIT_OK


def cmd_variants(args: argparse.Namespace) -> int:
    from .fixtures import DEFAULT_VARIANT_ORDER, LAYOUT_VARIANTS, describe_variants

    if args.json:
        payload = [
            {
                "key": LAYOUT_VARIANTS[key].key,
                "label": LAYOUT_VARIANTS[key].label,
                "style": LAYOUT_VARIANTS[key].style,
                "size": list(LAYOUT_VARIANTS[key].size),
                "theme": LAYOUT_VARIANTS[key].theme,
                "zoom": LAYOUT_VARIANTS[key].zoom,
                "columns": LAYOUT_VARIANTS[key].columns,
                "layout_type": LAYOUT_VARIANTS[key].layout_type.value,
                "auto_advance": LAYOUT_VARIANTS[key].auto_advance,
                "sidebar": LAYOUT_VARIANTS[key].sidebar,
                "low_contrast": LAYOUT_VARIANTS[key].low_contrast,
                "overlay": LAYOUT_VARIANTS[key].overlay,
                "screen_role": LAYOUT_VARIANTS[key].screen_role,
                "description": LAYOUT_VARIANTS[key].description,
            }
            for key in DEFAULT_VARIANT_ORDER
        ]
        _emit(payload, True)
    else:
        _emit(describe_variants(), False)
        _emit(f"\n{len(DEFAULT_VARIANT_ORDER)} variants (section 13.2 requires at least 12)", False)
    return EXIT_OK


def cmd_fixtures(args: argparse.Namespace) -> int:
    from .fixtures import write_fixture_frames

    sequence = scenario_sequence(args.scenario, fixture_dir=args.fixture_dir)
    out = Path(args.out_dir)
    written = write_fixture_frames(sequence, out)
    _emit(
        f"wrote {len(written)} frame(s) + ground truth for scenario '{args.scenario}' to {out}\n"
        f"  sequence.json holds the answer key ({len(sequence.answer_key)} question(s))",
        False,
    )
    return EXIT_OK


def cmd_probe(args: argparse.Namespace) -> int:
    from .capabilities import probe

    caps = probe()
    if args.json:
        _emit({name: vars(cap) for name, cap in caps.items()}, True)
        return EXIT_OK
    width = max(len(name) for name in caps)
    _emit(f"{'CAPABILITY':<{width}}  STATE      DETAIL", False)
    _emit("-" * (width + 60), False)
    for name, cap in caps.items():
        _emit(f"{name:<{width}}  {'available' if cap.available else 'MISSING':10} {cap.detail}", False)
        if not cap.available and cap.affects:
            _emit(f"{'':<{width}}  {'':10} degrades: {cap.affects}", False)
    return EXIT_OK


def cmd_doctor(args: argparse.Namespace) -> int:
    """Capability + configuration + perception health, in one readable report."""
    from .capabilities import has, probe
    from .config import ConfigError, EngineConfig
    from .perception.ocr import describe_engines

    problems: List[str] = []
    try:
        cfg = EngineConfig.load(args.config) if args.config else EngineConfig.default()
    except ConfigError as exc:
        _emit(f"CONFIG: invalid -- {exc}", False)
        return EXIT_USAGE
    _emit(f"QuizEngine {__version__} doctor (PRD v{PRD_VERSION})", False)
    _emit(f"  config          : {cfg.source_path or 'built-in defaults'}", False)
    _emit(f"  fingerprint     : {cfg.fingerprint()}", False)
    _emit(f"  capture backend : {cfg.capture.backend}", False)
    _emit(f"  action backend  : {cfg.action.backend}", False)
    _emit(f"  ocr engine      : {cfg.perception.ocr_engine}", False)
    _emit(f"  model provider  : {cfg.models.primary.kind}", False)
    _emit(f"  attestation     : {'present' if cfg.run.attestation else 'absent (a real run will refuse to act)'}", False)

    caps = probe()
    missing = [name for name, cap in caps.items() if not cap.available]
    _emit(f"\ncapabilities: {len(caps) - len(missing)}/{len(caps)} available", False)
    for name in missing:
        _emit(f"  MISSING {name}: {caps[name].detail} (affects {caps[name].affects or 'optional feature'})", False)

    _emit("\nocr engines:", False)
    for engine in describe_engines():
        _emit(f"  {'ok     ' if engine['available'] else 'missing'} {engine['name']:12} {engine['install_hint']}", False)

    if args.target == "screen":
        for required in ("mss", "pyautogui"):
            if not has(required):
                problems.append(f"target=screen needs '{required}': {caps[required].detail if required in caps else 'not probed'}")
        if not cfg.run.attestation:
            problems.append("target=screen refuses to act without --attest (section 3.3)")
    else:
        _emit("\noffline perception self-check:", False)
        try:
            runtime = build_runtime(config=cfg, target="sim", scenario="demo", runs_dir=args.runs_dir)
            report = runtime.run()
            _emit(
                f"  scenario=demo outcome={report.outcome.value} answered={len(report.questions)} "
                f"accuracy={report.accuracy if report.accuracy is None else round(report.accuracy, 3)} "
                f"unverified={report.unverified_actions} illegal={report.illegal_transitions} stale={report.stale_coordinate_violations}",
                False,
            )
            if report.outcome.value != "completed" or report.unverified_actions or report.illegal_transitions:
                problems.append("offline self-check did not complete cleanly")
            _emit(f"  artifacts: {runtime.run_dir}", False)
        except Exception as exc:
            problems.append(f"offline self-check raised {type(exc).__name__}: {exc}")

    if problems:
        _emit("\nproblems:", False)
        for problem in problems:
            _emit(f"  - {problem}", False)
        return EXIT_FAILED_SAFE
    _emit("\nno problems found", False)
    return EXIT_OK


def cmd_config(args: argparse.Namespace) -> int:
    from .config import EngineConfig

    cfg = EngineConfig.load(args.config) if args.config else EngineConfig.default()
    if args.what == "fingerprint":
        _emit(cfg.fingerprint(), False)
        return EXIT_OK
    if args.json:
        _emit(cfg.to_dict(), True)
    else:
        _emit(cfg.to_yaml(), False)
    return EXIT_OK


def cmd_report(args: argparse.Namespace) -> int:
    from .contracts import RunReport

    run_dir = Path(args.run_dir)
    path = run_dir / "report.json" if run_dir.is_dir() else run_dir
    if not path.exists():
        _emit(f"no report at {path}", False)
        return EXIT_USAGE
    report = RunReport.model_validate_json(path.read_text(encoding="utf-8"))
    if args.json:
        _emit(json.loads(report.model_dump_json()), True)
    else:
        _emit(summarize(report), False)
        _emit(f"  artifacts: {run_dir if run_dir.is_dir() else run_dir.parent}", False)
    return EXIT_OK


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #
def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="YAML config file (defaults are built in)")
    parser.add_argument("--runs-dir", help="where run artifacts are written (default: config paths.runs_dir)")
    parser.add_argument("--set", action="append", metavar="SECTION.FIELD=VALUE",
                        help="override one config value; repeatable (value parsed as YAML)")
    parser.add_argument("--json", action="store_true", help="machine-readable output")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quizengine",
        description="QuizEngine -- a perception-driven, closed-loop quiz automation agent (PRD v%s)." % PRD_VERSION,
        epilog="exit codes: 0 completed, 1 usage, 2 halted, 3 aborted, 4 failed safe, 5 stopped by operator",
    )
    parser.add_argument("--version", action="version", version=f"quizengine {__version__} (PRD v{PRD_VERSION})")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("demo", help="run the offline closed loop over the demo quiz")
    p.add_argument("--scenario", default="demo", choices=(*SCENARIOS, "fixture-dir"))
    p.add_argument("--console", action="store_true", help="interactive operator console")
    p.add_argument("--screenshots", action="store_true", help="attach trace screenshots to events")
    p.add_argument("--no-attest", action="store_true", help="withhold the attestation (exercises the refusal gate)")
    _add_common(p)
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("run", help="run against the simulated world or the real screen")
    p.add_argument("--target", default="sim", choices=("sim", "screen"))
    p.add_argument("--scenario", default="demo", choices=(*SCENARIOS, "fixture-dir"))
    p.add_argument("--fixture-dir", help="directory of frame_NNN.png + frame_NNN.json (scenario=fixture-dir)")
    p.add_argument("--console", action="store_true", help="interactive operator console")
    p.add_argument("--screenshots", action="store_true", help="attach trace screenshots to events")
    attest_group = p.add_mutually_exclusive_group()
    attest_group.add_argument("--attest", action="store_true",
                              help="explicitly authorize the real desktop run (required for --target screen)")
    attest_group.add_argument("--no-attest", action="store_true",
                              help="withhold attestation and exercise the safety refusal path")
    p.add_argument("--attestation", help="custom attestation text (supplying it is explicit opt-in)")
    p.add_argument("--operator-script", help="comma-separated operator replies, e.g. 'answer B,skip,abort'")
    _add_common(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("agent", help="arm manual Windows start/stop hotkeys for the real screen")
    p.add_argument("--attest", action="store_true", help="explicitly authorize this real-screen session")
    p.add_argument("--config", help="YAML config file (defaults are built in)")
    p.add_argument("--runs-dir", help="where run artifacts are written")
    p.add_argument("--set", action="append", metavar="SECTION.FIELD=VALUE", help="typed config override; repeatable")
    p.set_defaults(func=cmd_agent)

    p = sub.add_parser("benchmark", help="repeat offline replay runs and report latency/safety metrics")
    p.add_argument("--scenario", default="demo", choices=("demo", "variant-matrix", "chaos", "fixture-dir"))
    p.add_argument("--fixture-dir", help="fixture directory when scenario=fixture-dir")
    p.add_argument("--repeat", type=int, default=3, choices=range(1, 21), metavar="1..20")
    p.add_argument("--config", help="YAML config file")
    p.add_argument("--runs-dir", help="where run artifacts are written")
    p.add_argument("--set", action="append", metavar="SECTION.FIELD=VALUE", help="typed config override; repeatable")
    p.add_argument("--json", action="store_true", help="machine-readable report")
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("selftest", help="run the acceptance-criteria suite offline")
    p.add_argument("--quick", action="store_true", help="only the fast cases")
    p.add_argument("--runs-dir", help="where run artifacts are written")
    p.set_defaults(func=cmd_selftest)

    p = sub.add_parser("variants", help="list the layout-variant catalogue")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_variants)

    p = sub.add_parser("fixtures", help="write rendered fixture frames + ground truth")
    p.add_argument("action", choices=("write",))
    p.add_argument("out_dir")
    p.add_argument("--scenario", default="demo", choices=(*SCENARIOS, "fixture-dir"))
    p.add_argument("--fixture-dir", help="source directory when scenario=fixture-dir")
    p.set_defaults(func=cmd_fixtures)

    p = sub.add_parser("probe", help="capability table (what this machine can do)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_probe)

    p = sub.add_parser("doctor", help="capabilities + config + an offline perception self-check")
    p.add_argument("--config")
    p.add_argument("--target", default="sim", choices=("sim", "screen"))
    p.add_argument("--runs-dir")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("config", help="show the effective configuration or its fingerprint")
    p.add_argument("what", nargs="?", default="show", choices=("show", "fingerprint"))
    p.add_argument("--config")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_config)

    p = sub.add_parser("report", help="re-read a stored run report")
    p.add_argument("run_dir")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_report)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:  # pragma: no cover - interactive
        _emit("interrupted", False, sys.stderr)
        return EXIT_STOPPED
    except Exception as exc:
        from .failures import QuizEngineError

        if isinstance(exc, QuizEngineError):
            _emit(f"{type(exc).__name__}: {exc}", False, sys.stderr)
            return EXIT_USAGE
        raise


__all__ = ["EXIT_BY_OUTCOME", "SELFTEST_CASES", "build_parser", "main"]
