# QuizEngine

**QuizEngine** is a local-first, perception-driven quiz automation agent. It observes a screen, infers question and answer choices, solves supported question types, declares a bounded intent, acts, then verifies the result from a fresh frame. It fails closed when evidence, authorization, or verification is insufficient.

The **QuizEngine** browser UI is a local practice and test surface backed by the legacy `quizforge` Python package. It serves real HTML/CSS across the same 19 layout fixtures used by offline tests, with light/dark themes, 100%/125%/150%/200% zoom, chaos conditions, answer-key export, and fixture-parity checks. “QuizForge” remains an internal package/API identifier; user-facing UI branding is QuizEngine.

> **Use only where you are authorized.** The agent is for practice, accessibility, and testing. It must not be used to evade proctoring, defeat CAPTCHA, bypass a secure/restricted environment, or automate an assessment without authorization. A real-screen run requires an explicit attestation and halts on detected restricted environments. No stealth or evasion features are provided.

## Quick start

Python 3.10+ is supported. Create an environment and install the development extras:

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Run QuizEngine's deterministic offline demonstration (no API key, OCR executable, display, or desktop actuator required):

```bash
python -m quizengine demo
```

Run the acceptance suite, including all fixture layouts, verification invariants, duplicate-question handling, budget exhaustion, restricted-environment halt, and chaos recovery:

```bash
python -m quizengine selftest
```

Expected high-level results: demo **6/6**, variant sweep **18/18**, no unverified actions, illegal transitions, or stale-coordinate violations; all self-test cases pass.

## QuizEngine browser UI and practice test surface

The supported launcher is `quizengine-ui`; the historical `quizforge` module and command remain as compatibility aliases. By default, the UI server binds to loopback only:

```bash
quizengine-ui serve
# open http://127.0.0.1:5050
```

It has no accounts, telemetry service, remote fonts, CDN assets, or outbound browser requests. Server sessions are held in bounded process memory and isolated by a random browser session cookie; the active tab keeps a validated `sessionStorage` recovery snapshot so navigation survives a server restart or in-memory eviction without writing answers to server disk. The answer key is not returned in normal screen state; export it explicitly from **Export answer key** (CSV) or `/api/answer-key?format=json`.

The server refuses a non-loopback bind unless you deliberately opt in:

```bash
# Only do this on a network you control:
QUIZFORGE_ALLOW_REMOTE=1 quizengine-ui serve --host 0.0.0.0 --port 5050
# alternatively add --allow-remote
```

Available in the app: **19 layout variants** (18 answerable layouts plus a terminal results fixture), light/dark, 100%/125%/150%/200% zoom, popup/toast/layout-shift/low-contrast/delay/reorder/stale chaos modes, previous/next navigation, restart, a full 18-layout sweep, CSV/JSON key export, and a gallery with per-variant fixture parity checks. The compact default practice run has 12 questions.

## QuizEngine CLI

```bash
python -m quizengine --help
python -m quizengine demo                         # deterministic offline loop
python -m quizengine run --target sim --scenario variant-matrix
python -m quizengine selftest                     # acceptance criteria
python -m quizengine doctor                       # capabilities + offline health check
python -m quizengine probe                        # optional capability table
python -m quizengine variants                     # fixture catalogue
python -m quizengine fixtures write ./frames      # render PNGs + annotations
python -m quizengine config show
python -m quizengine report ./runs/<run-id>
```

The `sim` target uses annotated fixture frames and the deterministic offline provider. It runs the real orchestrator and perception/solver/confidence/action/verification/navigation/recovery modules; only desktop capture and input are simulated.

### Authorized real-screen operation

A real-screen run is Windows-first and needs screen capture, OCR, and visible input dependencies. Install `.[screen,vision,actuate,models]`, install the Tesseract executable separately, and verify with `python -m quizengine doctor --target screen`. Configure OCR/actuation for the target computer before authorizing any run. It refuses to act without explicit consent and halts in restricted/locked environments; never use it to bypass proctoring, lockdown, CAPTCHA, or monitoring.

A direct one-shot invocation is available:

```bash
python -m quizengine run --target screen --attest
```

For manual hotkey control, copy `.env.example` to `.env` and then launch:

```bash
python run.py --attest
```

The process arms but does **not** capture or act on launch. Press the configured start hotkey to start exactly one run; another start while active is ignored. Press the configured stop hotkey to request cancellation; an in-flight provider/capture action can take up to its configured timeout to return. `Ctrl+C` exits the controller. Defaults are **Win+Alt+Q** (start) and **Win+Alt+X** (stop), configured by `AGENT_START_HOTKEY` and `AGENT_STOP_HOTKEY`. Supported syntax is modifiers (`ctrl`, `alt`, `shift`, `win`) plus one ASCII letter or digit, such as `ctrl+shift+q`.

**IDE instructions (VS Code or another Python IDE):** open this repository as the workspace, select the Python 3.10+ interpreter in `.venv`, install the extras above in that interpreter, and create a root `.env` from `.env.example`. Add a run configuration whose script is the repository's `run.py` and whose arguments include `--attest`; start it from the IDE terminal so status output and operator prompts remain visible. The separate start hotkey is still required. Do not put API keys in launch arguments or checked-in configuration. The `.env` file is ignored by Git.

A direct screen invocation or the hotkey controller without explicit authorization is refused. `--no-attest` on the CLI run command exercises the refusal path. Never disable the attestation requirement.

Common configuration overrides can be supplied repeatedly:

```bash
python -m quizengine run --target sim --set budgets.max_questions=30 --set run.deterministic=true
```

## Model providers

The provider is selected explicitly; there is no automatic switching between Gemini and NVIDIA:

- `MODEL_PROVIDER=offline` (default): deterministic, key-free fixture and test mode; makes no external model calls.
- `MODEL_PROVIDER=gemini`: requires `GEMINI_API_KEY`; configure `GEMINI_PRIMARY_MODEL` and, optionally, `GEMINI_FAST_MODEL`, `GEMINI_VISION_MODEL`, `GEMINI_VERIFIER_MODEL`, and `GEMINI_BASE_URL`.
- `MODEL_PROVIDER=nvidia`: requires `NVIDIA_NIM_API_KEY`; configure `NVIDIA_NIM_PRIMARY_MODEL` and optional `NVIDIA_NIM_FAST_MODEL`, `NVIDIA_NIM_VISION_MODEL`, `NVIDIA_NIM_VERIFIER_MODEL`, and `NVIDIA_NIM_BASE_URL`.

Provider-level runtime controls include `MODEL_REQUEST_TIMEOUT_SECONDS`, `MODEL_MAX_RETRIES`, `MODEL_TEMPERATURE`, `MODEL_MAX_TOKENS`, `MODEL_PRIMARY_CONFIDENCE_THRESHOLD`, `MODEL_VERIFICATION_CONFIDENCE_THRESHOLD`, `ENABLE_PARALLEL_MODEL_CALLS`, `ENABLE_FAST_PATH`, `ENABLE_TIER2_PERCEPTION`, `ENABLE_RESULT_CACHE`, `ENABLE_SCREEN_CACHE`, and `MAX_PARALLEL_MODELS`. See `.env.example` for the full set and example endpoint/model values. Use a model identifier enabled for your vendor account. Credentials are loaded from the process environment or local `.env`; they are never sent to the browser or written into config fingerprints/logs. The test suite uses fake HTTP responses, not external accounts.

`python -m quizengine benchmark --scenario demo --repeat 3` performs repeated offline replay and reports measured total/iteration latency, outcome, accuracy, and safety invariants. Use `--scenario variant-matrix` for the broader layout sweep; `--json` emits machine-readable metrics.

## Fixture/catalogue API

The QuizEngine browser UI provides:

- `GET /health` — local service health.
- `GET /api/session` — current public screen state (does not reveal the key).
- `POST /api/answer`, `/api/next`, `/api/restart` — local quiz actions.
- `GET /api/variants` and `/api/parity/<variant>` — shared catalogue and fixture consistency check.
- `GET /api/answer-key?format=csv|json` — explicitly requested answer key.

The parity check uses the same `quizengine.fixtures.build_scene()` catalogue as the offline renderer and compares layout type, option style/text/count, and screen role.

## Tests

The suite is deterministic and headless. A real monitor, OCR executable, external model provider, or OS input backend is not needed for CI; HTTP adapters are exercised with fake clients. Optional backend installation hints are listed by `python -m quizengine probe` / `python -m quizengine doctor`.

```bash
python -m pytest
python -m quizengine selftest
python -m quizengine benchmark --scenario demo --repeat 3
```

## Project status

The repository retains the M1–M3 architecture and adds explicit Gemini/NVIDIA model-provider selection, offline replay, latency reporting, and a Windows manual-hotkey controller. Offline/simulated workflows are CI-testable. Real-screen execution is still platform/dependency dependent and must be validated on the authorized Windows target before use. QuizForge remains the fixture-driven practice/test application described above.

**QuizForge is the test environment. The runtime agent is target-agnostic and is not hard-coded to QuizForge.**
