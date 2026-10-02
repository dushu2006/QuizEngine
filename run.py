"""Convenience launcher for the manually hotkey-controlled real-screen agent.

Examples:
    python run.py --attest
    python run.py --attest --config config/production.yaml --runs-dir runs

The process only registers hotkeys on launch. A separate explicit start-key
press is required before capture or desktop interaction begins.
"""

from __future__ import annotations

import sys

from quizengine.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["agent", *sys.argv[1:]]))
