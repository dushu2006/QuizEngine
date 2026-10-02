"""QuizEngine -- a modular, vision-driven, closed-loop computer-use agent.

See ``README.md`` for the operating manual and ``docs/STATUS.md`` for the
PRD-section -> implementation map.

Quick orientation (PRD section 7 module order)::

    quizengine.contracts     section 8   data contracts (the only shared language)
    quizengine.config        section 12  schema-validated configuration
    quizengine.failures      section 11  failure catalog + recovery matrix
    quizengine.capture       section 7.1 validated, fresh, cropped frames
    quizengine.perception    section 7.2 pixels -> structured screen description
    quizengine.extraction    section 7.3 perception -> validated Question
    quizengine.solver        section 7.4 Question -> Decision (pure function)
    quizengine.confidence    section 7.5 calibrated score + uncertainty policy
    quizengine.models        FR-6.2      provider abstraction (retry/fallback)
    quizengine.action        section 7.6 actuator abstraction
    quizengine.binding       section 7.7 element handle -> live coordinates
    quizengine.navigation    section 7.8 question N -> question N+1
    quizengine.verification  section 7.9 closed-loop verification controller
    quizengine.recovery      section 7.11 bounded, classified recovery
    quizengine.persistence   section 7.12 session file + run reports
    quizengine.telemetry     section 7.13 logs, metrics, traces, console
    quizengine.safety        section 7.14 environment gatekeeper (mandatory)
    quizengine.orchestrator  section 7.10 the FSM that sequences everything

Design laws L1-L10 (section 5) are enforced in code, not just documented; each
law's enforcement point is named in the module docstring that implements it.
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__", "PRD_VERSION", "DESIGN_LAWS"]

#: The PRD revision this tree implements.
PRD_VERSION = "1.0"

#: Design laws from section 5, quoted so code review can check against them.
DESIGN_LAWS = {
    "L1": "Pixels are the only truth; no cached coordinates older than one frame.",
    "L2": "Closed loop mandatory; every state-changing action is verified.",
    "L3": "Bounded everything; every wait/retry has a timeout and max attempts.",
    "L4": "Fail safe, never fail blind.",
    "L5": "Modularity by interface; modules talk only through section 8 contracts.",
    "L6": "Never assume resolution, DPI, zoom, theme, font or window position.",
    "L7": "Confidence gates action; below threshold -> uncertainty policy.",
    "L8": "One verified step at a time.",
    "L9": "Idempotent re-entry; resume without double-answering.",
    "L10": "Compliance is architectural; the environment gate is a pipeline stage.",
}


def __getattr__(name: str):  # pragma: no cover - convenience lazy imports
    """Expose the orchestrator entry points without importing the world."""
    if name == "EngineConfig":
        from .config import EngineConfig

        return EngineConfig
    if name == "run":
        from .orchestrator import run

        return run
    if name == "Engine":
        from .orchestrator import Engine

        return Engine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
