"""Command line: python -I -m rc_runtime [flags]. Exit codes: see loader.py."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
from pathlib import Path
from typing import Optional

from rc_sdk.contract import CONTRACT, SCHEMA_VERSION
from rc_sdk.transport import tls_server_context

from . import loader
from .loader import RuntimeStartupError
from .server import RuntimeServer

DEFAULT_SUBMISSION = "/submission"
DEFAULT_WEIGHTS = "/weights"
DEFAULT_SCRATCH = "/scratch"
DEFAULT_RELEASE_JSON = "/opt/rc/release.json"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rc_runtime")
    p.add_argument("--submission", default=None, help="submission dir (default $RC_SUBMISSION_DIR or /submission)")
    p.add_argument("--entry-point", default=None, help="entry_point override (MOD:ATTR, dotted module, or relative .py path; default: manifest.json)")
    p.add_argument("--port", type=int, default=None, help="default $RC_PORT or 8080")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--reserved-modules", default=None, help="release.json or JSON list of reserved names")
    p.add_argument("--selfcheck", action="store_true",
                   help="resolve entry point and shadow checks without serving; prints one JSON line")
    return p


def _selfcheck(submission: Path, entry_point: str, reserved: frozenset) -> int:
    try:
        spec = loader.parse_entry(entry_point)
        if spec.is_program:
            loader.prepare_program(submission, spec, reserved)
            loader.check_program_resolves(spec)
        else:
            loader.check_class_schema(submission)
            loader.resolve_entry(submission, entry_point, reserved)
    except RuntimeStartupError as exc:
        print(json.dumps({"ok": False, "error_code": exc.code, "error": exc.detail}), flush=True)
        if exc.code in ("reserved_module_shadow", "sdk_shadowed"):
            return loader.EXIT_RESERVED_SHADOW
        return exc.exit_code if exc.exit_code != loader.EXIT_INTERNAL else loader.EXIT_SELFCHECK_RESOLVE
    print(json.dumps({"ok": True, "entry_point": entry_point}), flush=True)
    return 0


def _run_program(submission: Path, spec, reserved: frozenset, port: int, env: dict) -> int:
    """Program form: the team program owns the HTTP server and runs as __main__ right here."""
    # One name for "the port to listen on"; RC_PORT is kept for the class form and older docs.
    os.environ["RC_POLICY_PORT"] = str(port)
    os.environ["RC_PORT"] = str(port)
    try:
        script = loader.prepare_program(submission, spec, reserved)
    except RuntimeStartupError as exc:
        print(f"[rc_runtime] startup error {exc.code}: {exc.detail}", flush=True)
        return exc.exit_code
    loader.start_shadow_watchdog(submission, reserved)
    loader.start_port_watchdog(port, deadline=loader.listen_deadline_seconds(env))
    print(f"[rc_runtime] program form: your program must listen on 0.0.0.0:{port} "
          f"(RC_POLICY_PORT={port})", flush=True)
    return loader.run_program(submission, spec, script)


def run(argv: Optional[list] = None, env: Optional[dict] = None) -> int:
    env = os.environ if env is None else env
    args = build_parser().parse_args(argv)  # argparse exits 2 on usage errors
    submission = Path(args.submission or env.get("RC_SUBMISSION_DIR") or DEFAULT_SUBMISSION)
    scratch = Path(env.get("RC_SCRATCH") or DEFAULT_SCRATCH)
    try:
        port = args.port if args.port is not None else int(env.get("RC_PORT") or 8080)
    except ValueError:
        print("[rc_runtime] invalid RC_PORT", flush=True)
        return loader.EXIT_INTERNAL
    if env is os.environ:
        os.environ.setdefault("RC_WEIGHTS_DIR", DEFAULT_WEIGHTS)

    try:
        loader.create_scratch(scratch)
        entry_point = args.entry_point or loader.read_manifest_entry_point(submission)
        spec = loader.parse_entry(entry_point)
        reserved = loader.load_reserved_modules(
            args.reserved_modules, default_release=DEFAULT_RELEASE_JSON, env=env)
        if spec.top_name:
            loader.check_static_shadow(spec.top_name, reserved)
    except RuntimeStartupError as exc:
        print(f"[rc_runtime] startup error {exc.code}: {exc.detail}", flush=True)
        return exc.exit_code
    except Exception as exc:  # defensive: organizer-side bug
        print(f"[rc_runtime] internal error: {exc!r}", flush=True)
        return loader.EXIT_INTERNAL

    if args.selfcheck:
        return _selfcheck(submission, entry_point, reserved)

    if spec.is_program:
        return _run_program(submission, spec, reserved, port, env)

    def load_policy():
        loader.check_class_schema(submission)
        policy = loader.resolve_entry(submission, entry_point, reserved)
        loader.run_load(policy)
        return policy

    try:
        tls_context = tls_server_context(env)
    except ValueError as exc:
        print(f"[rc_runtime] policy TLS configuration error: {exc}", flush=True)
        return loader.EXIT_INTERNAL
    try:
        server = RuntimeServer(args.host, port, load_policy, tls_context=tls_context)
    except OSError as exc:
        print(f"[rc_runtime] cannot bind port {port}: {exc}", flush=True)
        return loader.EXIT_INTERNAL
    server.start_load()
    transport = "HTTPS" if tls_context else "HTTP"
    print(f"[rc_runtime] serving {transport} contract {CONTRACT}, schema {SCHEMA_VERSION}, on port {port}",
          flush=True)

    def _stop(signum, frame):
        threading.Thread(target=server.shutdown, daemon=True).start()

    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, _stop)
        signal.signal(signal.SIGINT, _stop)
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


def main(argv: Optional[list] = None) -> None:
    sys.exit(run(argv))
