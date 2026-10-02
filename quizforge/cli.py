"""QuizEngine command line for its local UI and test target."""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import sys
from typing import Optional, Sequence

from . import __version__
from .catalog import describe_variants


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quizengine-ui",
        description="QuizEngine local UI and practice-test server (no account, cloud, or external assets).",
    )
    parser.add_argument("--version", action="version", version=f"QuizEngine {__version__}")
    sub = parser.add_subparsers(dest="command")
    serve = sub.add_parser("serve", help="serve the QuizEngine local UI (default command)")
    serve.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    serve.add_argument("--port", type=int, default=5050)
    serve.add_argument("--debug", action="store_true", help="enable Flask debug mode (local development only)")
    serve.add_argument("--allow-remote", action="store_true", help="explicitly allow non-loopback access; local-only is safer")
    variants = sub.add_parser("variants", help="print QuizEngine's shared layout catalogue")
    variants.add_argument("--json", action="store_true")
    return parser


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    raw = list(argv) if argv is not None else sys.argv[1:]
    # No command means `serve`, including flags such as --port.
    if raw and raw[0] not in {"serve", "variants", "-h", "--help", "--version"}:
        raw = ["serve", *raw]
    args = parser.parse_args(raw)
    if args.command == "variants":
        rows = describe_variants()
        print(json.dumps(rows, indent=2) if args.json else "\n".join(f"{row['key']:<22} {row['label']}" for row in rows))
        return 0
    if args.command not in {None, "serve"}:
        parser.error(f"unknown command {args.command}")
    # With no subcommand, accept global-style serve flags by parsing the default
    # serve namespace. This keeps the natural `quizforge --port 5050` ergonomic.
    if args.command is None:
        args = build_parser().parse_args(["serve", *(list(argv) if argv is not None else sys.argv[1:])])
    opted_in = bool(args.allow_remote or os.environ.get("QUIZFORGE_ALLOW_REMOTE") == "1")
    if not _is_loopback(args.host) and not opted_in:
        parser.error("non-loopback bind refused; pass --allow-remote (or set QUIZFORGE_ALLOW_REMOTE=1) explicitly")
    if args.debug and not _is_loopback(args.host):
        parser.error("Flask debug mode may only bind to loopback")
    from .app import create_app

    app = create_app({"QUIZFORGE_ALLOW_REMOTE": opted_in})
    print(f"QuizEngine {__version__} UI listening at http://{args.host}:{args.port} (local-only={not opted_in})")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True, use_reloader=False)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
