"""QuizForge -- the local practice-quiz platform QuizEngine is built against.

QuizForge exists so the agent has something honest to look at: a real HTTP app
rendering real HTML/CSS, with the same layout variants, themes, zoom levels and
chaos switches the engine's offline fixtures model.  Variant keys, option styles,
navigation labels and the results screen are shared with
:mod:`quizengine.fixtures`, so a screen served here and a fixture frame rendered
there describe the *same* layout (``/api/parity/<variant>`` proves it).

It is a local tool: the server binds loopback by default and refuses to expose
itself further without an explicit override (section 16, local-first).
"""

from __future__ import annotations

from .app import create_app
from .catalog import CHAOS_MODES, QUESTIONS, THEMES, VARIANTS, ZOOM_LEVELS, build_screen

__version__ = "0.1.0"

__all__ = [
    "CHAOS_MODES",
    "QUESTIONS",
    "THEMES",
    "VARIANTS",
    "ZOOM_LEVELS",
    "__version__",
    "build_screen",
    "create_app",
]
