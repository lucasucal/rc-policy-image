"""HTTP server for contract v0. Wire behavior matches rc_sdk.server byte for byte.

Differences from rc_sdk.server (all additive): the policy is produced by a loader
callable in the background thread (so /healthz answers before the entry module is
even imported), and /readyz carries ``error_code`` next to ``error`` on failure.
"""

from __future__ import annotations

import json
import threading
import traceback
from http.server import BaseHTTPRequestHandler
from typing import Any, Callable, Dict, Optional

from rc_sdk.contract import CAPABILITIES, CONTRACT, SCHEMA_VERSION, observation_error
from rc_sdk.policy import Policy
from rc_sdk.transport import PolicyHTTPServer

from .loader import RuntimeStartupError

class _Handler(BaseHTTPRequestHandler):
    server: "RuntimeServer"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[rc_runtime] {self.address_string()} {fmt % args}", flush=True)

    def _send_json(self, status: int, body: Dict[str, Any],
                   extra_headers: Optional[Dict[str, str]] = None) -> None:
        payload = json.dumps(body).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            for k, v in (extra_headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError) as exc:
            # The runner closed the connection (budget cut, step limit) before the response was
            # written. Expected, not an error: one line instead of a socketserver traceback.
            self.close_connection = True
            print(f"[rc_runtime] client closed the connection before the {status} response "
                  f"to {self.command} {self.path} ({type(exc).__name__})", flush=True)

    def do_GET(self) -> None:
        if self.path == "/healthz":
            self._send_json(200, {})
        elif self.path == "/readyz":
            body: Dict[str, Any] = {
                "ready": self.server.ready.is_set(),
                "contract": CONTRACT,
                "schema_version": SCHEMA_VERSION,
                "capabilities": dict(CAPABILITIES),
            }
            err = self.server.load_error
            if err is not None:
                body["error"] = err.message
                body["error_code"] = err.code
            self._send_json(200, body)
        else:
            self._send_json(404, {"error": f"no route {self.path}"})

    def do_POST(self) -> None:
        if self.path != "/act":
            self._send_json(404, {"error": f"no route {self.path}"})
            return
        if not self.server.ready.is_set():
            self._send_json(503, {"error": "not ready; weights still loading"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0:
                raise ValueError("Content-Length must be nonnegative")
            envelope = json.loads(self.rfile.read(length) or b"{}")
        except (BrokenPipeError, ConnectionResetError) as exc:
            self.close_connection = True
            print(f"[rc_runtime] client closed the connection while sending the request body "
                  f"of {self.command} {self.path} ({type(exc).__name__})", flush=True)
            return
        except ValueError as exc:
            self._send_json(400, {"error": f"invalid JSON body: {exc}"})
            return
        error = observation_error(envelope)
        if error:
            self._send_json(400, {"error": error})
            return
        episode_id, step = envelope["episode_id"], envelope["step"]
        try:
            action = self.server.policy.act(envelope)
        except Exception as exc:
            self._send_json(500, {"error": f"policy act() failed: {exc}"})
            return
        try:
            json.dumps(action)
        except (TypeError, ValueError) as exc:
            print(f"[rc_runtime] /act step {step}: action is not JSON-serializable: {exc}", flush=True)
            self._send_json(500, {"error": f"action is not JSON-serializable: {exc}"})
            return
        self._send_json(200, {"episode_id": episode_id, "step": step, "action": action},
                        {"X-RC-Episode-Id": str(episode_id), "X-RC-Step": str(step)})


class RuntimeServer(PolicyHTTPServer):
    """ThreadingHTTPServer + background load thread.

    ``loader`` returns a fully loaded Policy or raises RuntimeStartupError. Any other
    exception from the loader is classified ``load_failed``.
    """

    daemon_threads = True

    def __init__(self, host: str, port: int, loader: Callable[[], Policy], *, tls_context=None) -> None:
        super().__init__((host, port), _Handler, tls_context=tls_context)
        self.loader = loader
        self.policy: Optional[Policy] = None
        self.ready = threading.Event()
        self.load_error: Optional[RuntimeStartupError] = None
        self._load_thread = threading.Thread(target=self._load, daemon=True)

    def _load(self) -> None:
        try:
            policy = self.loader()
        except Exception as exc:
            err = exc if isinstance(exc, RuntimeStartupError) else RuntimeStartupError(
                "load_failed", type(exc).__name__)
            print(f"[rc_runtime] startup FAILED ({err.code}); full diagnostic follows:", flush=True)
            cause = err.__cause__ or err
            traceback.print_exception(type(cause), cause, cause.__traceback__)
            if getattr(err, "hint", ""):
                print(f"[rc_runtime] hint: {err.hint}", flush=True)
            self.load_error = err
            return
        self.policy = policy
        self.ready.set()
        print("[rc_runtime] ready", flush=True)

    def start_load(self) -> None:
        self._load_thread.start()

    def wait_loaded(self, timeout: Optional[float] = None) -> None:
        self._load_thread.join(timeout)
