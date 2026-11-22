"""Entry-point parsing, import, Policy resolution and shadow verification.

Spec: docs/policy-runtime-pin.md section 4.4, docs/runtime-interfaces.md section 5.
Nothing here reads team-controlled environment variables.
"""

from __future__ import annotations

import ast
import errno
import importlib
import importlib.machinery
import importlib.util
import json
import os
import re
import runpy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from rc_sdk.policy import Policy

ENTRY_POINT_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*:[A-Za-z_][A-Za-z0-9_]*$"
)

MODULE_PATH_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")
PATH_PART_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]*$")

# Always reserved, regardless of what the release list says.
ALWAYS_RESERVED = frozenset({"rc_sdk", "rc_runtime"})

# Exit codes for failures before the server binds (contract section 5).
EXIT_ENTRY_POINT_INVALID = 70
EXIT_SELFCHECK_RESOLVE = 71  # selfcheck only: import/resolve/load-free failures
EXIT_RESERVED_SHADOW = 72
EXIT_SCRATCH_UNWRITABLE = 74
EXIT_INTERNAL = 75
EXIT_PROGRAM_EXITED = 76  # program form: the team program returned / called sys.exit
EXIT_PROGRAM_FAILED = 77  # program form: the team program raised (incl. import errors)
EXIT_NOT_LISTENING = 78  # program form: nothing listened on the policy port within the listen deadline
DEFAULT_LISTEN_DEADLINE_SECONDS = 600.0

# /readyz error_code values (contract section 5) -> short fixed /readyz.error string.
ERROR_MESSAGES = {
    "entry_import_failed": "entry point import failed",
    "entry_not_policy": "entry point is not a Policy",
    "entry_factory_failed": "entry point factory failed",
    "reserved_module_shadow": "submission shadows an image module",
    "sdk_shadowed": "submission shadows rc_sdk",
    "load_failed": "policy failed to load",
}


class RuntimeStartupError(Exception):
    """A failure with a stable machine code (and, pre-serve, an exit code)."""

    def __init__(self, code: str, detail: str = "", exit_code: int = EXIT_INTERNAL,
                 hint: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail
        self.exit_code = exit_code
        # Log-only advice for common sandbox mistakes; never part of /readyz.
        self.hint = hint

    @property
    def message(self) -> str:
        return ERROR_MESSAGES.get(self.code, self.code)


def parse_entry_point(value: Any) -> tuple[str, str]:
    """'pkg.mod:Attr' -> ('pkg.mod', 'Attr'); raises entry_point_invalid (exit 70)."""
    if not isinstance(value, str) or not ENTRY_POINT_RE.fullmatch(value):
        raise RuntimeStartupError(
            "entry_point_invalid", f"entry_point {value!r} must look like 'package.module:Attr'",
            EXIT_ENTRY_POINT_INVALID)
    module, _, attr = value.partition(":")
    return module, attr


@dataclass(frozen=True)
class EntrySpec:
    """A parsed entry_point. kind: 'class' (module:attr), 'module' (dotted) or 'file' (.py path)."""
    kind: str
    module: str = ""      # class/module forms: dotted module path
    attr: str = ""        # class form only
    path: str = ""        # file form: posix path relative to the submission root

    @property
    def top_name(self) -> str:
        """Top-level name that must not collide with an image module ('' when run by path)."""
        if self.kind == "file":
            return ""
        return self.module.split(".", 1)[0]

    @property
    def is_program(self) -> bool:
        return self.kind != "class"


ENTRY_POINT_HELP = ("entry_point must be 'package.module:Attr' (a Policy class), or a program: a dotted "
                    "module path ('mypkg.server') or a relative .py path inside the repository ('serve.py')")


def parse_entry(value: Any) -> EntrySpec:
    """Parse either entry_point form; raises entry_point_invalid (exit 70)."""
    def bad(why: str) -> RuntimeStartupError:
        return RuntimeStartupError("entry_point_invalid", f"entry_point {value!r}: {why}",
                                   EXIT_ENTRY_POINT_INVALID)
    if not isinstance(value, str) or not value:
        raise bad(ENTRY_POINT_HELP)
    if ":" in value:
        module, attr = parse_entry_point(value)
        return EntrySpec("class", module=module, attr=attr)
    if "/" in value or value.endswith(".py") or "\\" in value:
        if value.startswith("/") or re.match(r"^[A-Za-z]:", value):
            raise bad("absolute paths are not allowed; use a path relative to the repository root")
        if "\\" in value:
            raise bad("use '/' as the path separator")
        parts = value.split("/")
        if any(p in ("", ".", "..") for p in parts) or not all(PATH_PART_RE.match(p) for p in parts):
            raise bad("path must be a clean relative path inside the repository (no '..', '.', empty or odd segments)")
        if not value.endswith(".py"):
            raise bad("a program path must end in .py")
        return EntrySpec("file", path=value)
    if not MODULE_PATH_RE.fullmatch(value):
        raise bad(ENTRY_POINT_HELP)
    return EntrySpec("module", module=value)


def read_manifest_entry_point(submission: Path) -> str:
    path = Path(submission) / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeStartupError(
            "entry_point_invalid", f"cannot read {path}: {type(exc).__name__}",
            EXIT_ENTRY_POINT_INVALID) from exc
    if not isinstance(manifest, dict) or "entry_point" not in manifest:
        raise RuntimeStartupError(
            "entry_point_invalid", f"{path} has no entry_point", EXIT_ENTRY_POINT_INVALID)
    return manifest["entry_point"]


def load_reserved_modules(
    explicit_file: Optional[str] = None,
    default_release: str = "/opt/rc/release.json",
    env: Optional[dict] = None,
) -> frozenset[str]:
    """Reserved top-level module names.

    Source order: --reserved-modules FILE, else default_release (key
    reserved_modules), else the file named by RC_RESERVED_MODULES. A file may be a
    release.json object (reserved_modules and stdlib_module_names are both used) or a
    plain JSON list. If nothing is configured the set is only rc_sdk/rc_runtime plus
    this interpreter's stdlib names (sys.path ordering remains the primary guard).
    """
    env = os.environ if env is None else env
    names: set[str] = set(ALWAYS_RESERVED) | set(sys.stdlib_module_names)
    candidates = [explicit_file] if explicit_file else [default_release, env.get("RC_RESERVED_MODULES")]
    for cand in candidates:
        if not cand or not os.path.exists(cand):
            if explicit_file:
                raise RuntimeStartupError("reserved_list_unreadable", f"{cand} not found")
            continue
        try:
            data = json.loads(Path(cand).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeStartupError("reserved_list_unreadable", f"{cand}: {exc}") from exc
        if isinstance(data, dict):
            names.update(data.get("reserved_modules") or [])
            names.update(data.get("stdlib_module_names") or [])
        elif isinstance(data, list):
            names.update(data)
        else:
            raise RuntimeStartupError("reserved_list_unreadable", f"{cand}: unexpected JSON")
        break
    return frozenset(n for n in names if isinstance(n, str))


def check_static_shadow(module_name: str, reserved: Iterable[str]) -> None:
    top = module_name.split(".", 1)[0]
    if top in reserved:
        raise RuntimeStartupError(
            "reserved_module_shadow",
            f"entry point top-level package {top!r} collides with an image module",
            EXIT_RESERVED_SHADOW)


def _under(path: Any, root: str) -> bool:
    """True when ``path`` is an ABSOLUTE path that resolves inside ``root``.

    Relative paths are ignored on purpose: they are resolved against the process cwd
    (which is the submission root), so e.g. torch's ``torch.ops.__file__ == "_ops.py"``
    would otherwise look like a submission file. Independent of the cwd.
    """
    if not isinstance(path, str) or not path or not os.path.isabs(path):
        return False
    try:
        real = os.path.realpath(path)
    except (OSError, ValueError):
        return False
    return real == root or real.startswith(root.rstrip(os.sep) + os.sep)


def _module_locations(mod: Any) -> list[str]:
    """Absolute file/package locations of a module; builtin, frozen, namespace and
    relative-path modules contribute nothing."""
    locs: list[str] = []
    try:
        f = getattr(mod, "__file__", None)
        if isinstance(f, str) and os.path.isabs(f):
            locs.append(f)
        spec = getattr(mod, "__spec__", None)
        origin = getattr(spec, "origin", None)
        if (isinstance(origin, str) and os.path.isabs(origin)
                and getattr(spec, "has_location", True)):
            locs.append(origin)
        if f is not None or origin not in (None, "namespace"):  # not a namespace package
            for p in list(getattr(mod, "__path__", None) or []):
                if isinstance(p, str) and os.path.isabs(p):
                    locs.append(p)
    except Exception:  # exotic module objects
        pass
    return locs


def verify_no_shadow(submission: Path, reserved: Iterable[str]) -> None:
    """Post-import check (spec 4.4): raises sdk_shadowed / reserved_module_shadow."""
    root = os.path.realpath(str(submission))
    reserved = set(reserved)
    sdk = sys.modules.get("rc_sdk")
    if sdk is None or any(_under(p, root) for p in _module_locations(sdk)):
        raise RuntimeStartupError("sdk_shadowed", "rc_sdk is loaded from the submission tree")
    for name, mod in list(sys.modules.items()):
        top = name.split(".", 1)[0]
        if top not in reserved or mod is None:
            continue
        if any(_under(p, root) for p in _module_locations(mod)):
            if top in ALWAYS_RESERVED and top == "rc_sdk":
                raise RuntimeStartupError("sdk_shadowed", f"{name} is loaded from the submission tree")
            raise RuntimeStartupError(
                "reserved_module_shadow", f"module {name!r} is loaded from the submission tree")


def append_submission_path(submission: Path, module_name: Optional[str] = None) -> list[str]:
    """Append (never insert) so image packages always win name collisions.

    Returns the directories appended. ``/submission/src`` is appended after
    ``/submission`` only when it exists and the entry module's top-level package is not
    importable from ``/submission`` itself (a ``src/`` layout).
    """
    added = []
    p = str(submission)
    if p not in sys.path:
        sys.path.append(p)
        added.append(p)
    src = Path(submission) / "src"
    if module_name and src.is_dir():
        top = module_name.split(".", 1)[0]
        if importlib.machinery.PathFinder.find_spec(top, [p]) is None:
            sp = str(src)
            if sp not in sys.path:
                sys.path.append(sp)
                added.append(sp)
    importlib.invalidate_caches()
    return added


def top_level_names(directory: Path) -> set[str]:
    """Importable top-level module/package names directly inside ``directory``."""
    names: set[str] = set()
    try:
        entries = list(Path(directory).iterdir())
    except OSError:
        return names
    for e in entries:
        n = e.name
        if n.startswith("__"):
            continue
        if e.is_dir() and n.isidentifier():
            names.add(n)
        elif e.suffix == ".py" and e.stem.isidentifier():
            names.add(e.stem)
    return names


# Same rule as checks/rc_checks (imports.NON_RUNTIME_DIRS): these directories collide with stray image
# modules all the time and are never run by a policy, so a shadow is only worth a warning when other
# code in the submission actually imports the name.
NON_RUNTIME_DIRS = frozenset({"tests", "test", "examples", "docs", "scripts", "cuda", "benchmarks",
                              "notebooks", "assets"})
_NON_RUNTIME_FILES = frozenset({"conftest.py", "setup.py", "noxfile.py", "fabfile.py"})
_SCAN_LIMIT = 3000  # .py files; bounded so a huge vendored tree cannot slow startup


def _imported_by_code(root: Path, name: str, entry_module: Optional[str]) -> bool:
    """True when a runtime .py file under ``root`` (not ``name`` itself) imports top-level ``name``."""
    scanned = 0
    for current, dirs, files in os.walk(root):
        rel = Path(current).relative_to(root).parts
        dirs[:] = sorted(x for x in dirs if not x.startswith(".") and x != "__pycache__"
                         and x not in NON_RUNTIME_DIRS and not (not rel and x == name))
        for f in sorted(files):
            if not f.endswith(".py") or f in _NON_RUNTIME_FILES or f.startswith("test_") \
                    or f.endswith("_test.py") or (not rel and f == name + ".py"):
                continue
            scanned += 1
            if scanned > _SCAN_LIMIT:
                return True  # cannot tell cheaply: keep the warning
            try:
                tree = ast.parse((Path(current) / f).read_text(encoding="utf-8", errors="replace"))
            except (SyntaxError, ValueError, OSError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    if any(a.name.split(".", 1)[0] == name for a in node.names):
                        return True
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    if node.module.split(".", 1)[0] == name:
                        return True
    return False


def warn_image_collisions(directories: Iterable[str], reserved: Iterable[str],
                          entry_module: Optional[str] = None) -> list[str]:
    """Print a warning for each submission top-level module that the image shadows.

    The image's copy wins (the submission is appended to sys.path), so a vendored
    ``lerobot/`` or ``numpy/`` is silently ignored; this makes that visible in the logs.
    """
    reserved = set(reserved)
    entry_top = entry_module.split(".", 1)[0] if entry_module else None
    hits: list[str] = []
    for d in directories:
        for name in sorted(top_level_names(Path(d))):
            if name in reserved and name != entry_top and name not in hits:
                if name in NON_RUNTIME_DIRS and not _imported_by_code(Path(d), name, entry_module):
                    continue
                hits.append(name)
                print(f"[rc_runtime] WARNING: {d}/{name} has the same name as a module in the "
                      f"pinned image; the image's {name!r} is used and your copy is ignored. "
                      "Remove it or rename it.", flush=True)
    return hits


_NET_MARKERS = ("huggingface.co", "name resolution", "name or service not known",
                "network is unreachable", "failed to resolve", "couldn't connect",
                "could not connect", "max retries exceeded", "offline mode",
                "outgoing traffic has been disabled", "connection refused",
                "temporary failure", "connection error")
_NET_HINT = ("no network at runtime; load files from /weights (your snapshot) or /hf-base "
             "(curated base models)")
_ROFS_HINT = "the filesystem is read-only; write under /scratch (TMPDIR is /scratch/tmp)"


def hint_for(exc: BaseException) -> str:
    """Short advice for the usual sandbox mistakes, from an exception and its causes."""
    seen = set()
    cur: Optional[BaseException] = exc
    rofs = net = False
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        text = f"{type(cur).__name__}: {cur}".lower()
        if getattr(cur, "errno", None) == errno.EROFS or "read-only file system" in text:
            rofs = True
        if isinstance(cur, ConnectionError) or any(m in text for m in _NET_MARKERS):
            net = True
        cur = cur.__cause__ or cur.__context__
    if rofs:
        return _ROFS_HINT
    if net:
        return _NET_HINT
    return ""


def resolve_policy(module_name: str, attr: str) -> Policy:
    """Import the module and turn Attr into a Policy instance (class or factory)."""
    try:
        module = importlib.import_module(module_name)
    except BaseException as exc:  # SystemExit etc. in team code must not kill the loader
        if isinstance(exc, KeyboardInterrupt):
            raise
        raise RuntimeStartupError("entry_import_failed", f"{type(exc).__name__}: {exc}",
                                  hint=hint_for(exc)) from exc
    try:
        target = getattr(module, attr)
    except AttributeError as exc:
        raise RuntimeStartupError(
            "entry_not_policy", f"module {module_name!r} has no attribute {attr!r}") from exc
    is_class = isinstance(target, type)
    if is_class and not issubclass(target, Policy):
        raise RuntimeStartupError("entry_not_policy", f"{attr} is a class but not a Policy subclass")
    if not is_class and not callable(target):
        raise RuntimeStartupError("entry_not_policy", f"{attr} is neither a Policy subclass nor callable")
    try:
        policy = target()
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            raise
        raise RuntimeStartupError("entry_factory_failed", f"{type(exc).__name__}: {exc}",
                                  hint=hint_for(exc)) from exc
    if not isinstance(policy, Policy):
        raise RuntimeStartupError(
            "entry_not_policy", f"{attr}() returned {type(policy).__name__}, not a Policy")
    return policy


def resolve_program_file(submission: Path, spec: EntrySpec) -> Path:
    """Absolute path of a file-form entry; must exist inside the submission tree."""
    root = os.path.realpath(str(submission))
    target = os.path.realpath(os.path.join(root, spec.path))
    if not (target == root or target.startswith(root.rstrip(os.sep) + os.sep)):
        raise RuntimeStartupError("entry_point_invalid",
                                  f"entry_point {spec.path!r} resolves outside the repository",
                                  EXIT_ENTRY_POINT_INVALID)
    if not os.path.isfile(target):
        raise RuntimeStartupError("entry_point_invalid",
                                  f"entry_point file {spec.path!r} not found in the repository",
                                  EXIT_ENTRY_POINT_INVALID)
    return Path(target)


def prepare_program(submission: Path, spec: EntrySpec, reserved: Iterable[str]) -> Optional[Path]:
    """Sandbox setup shared by serving and selfcheck for program-form entries.

    Same steps as the class form: static reserved check, sys.path (submission last, src/
    fallback), collision warnings, chdir. Returns the resolved file for the file form.
    """
    if spec.kind == "module":
        check_static_shadow(spec.module, reserved)
    script = resolve_program_file(submission, spec) if spec.kind == "file" else None
    added = append_submission_path(Path(submission), spec.module or None)
    if script is not None:
        # Like `python path/to/script.py`, the script's own directory is importable (last).
        d = str(script.parent)
        if d not in sys.path:
            sys.path.append(d)
            added.append(d)
        importlib.invalidate_caches()
    warn_image_collisions(added, reserved, spec.module or None)
    try:
        os.chdir(str(submission))
    except OSError:
        pass
    return script


def check_program_resolves(spec: EntrySpec) -> None:
    """Selfcheck: the module form must be findable (no program code is executed)."""
    if spec.kind != "module":
        return
    try:
        found = importlib.util.find_spec(spec.module)
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            raise
        raise RuntimeStartupError("entry_import_failed", f"{type(exc).__name__}: {exc}",
                                  hint=hint_for(exc)) from exc
    if found is None:
        raise RuntimeStartupError("entry_import_failed", f"No module named {spec.module!r}")


def run_program(submission: Path, spec: EntrySpec, script: Optional[Path]) -> int:
    """Run the team program as __main__ in this interpreter. Returns an exit code (never 0)."""
    import traceback
    name = spec.module or spec.path
    sys.argv = [str(script) if script else name]
    print(f"[rc_runtime] running entry program {name!r} as __main__", flush=True)
    try:
        if script is not None:
            runpy.run_path(str(script), run_name="__main__")
        else:
            runpy.run_module(spec.module, run_name="__main__", alter_sys=True)
    except SystemExit as exc:
        print(f"[rc_runtime] entry program exited (sys.exit({exc.code!r})); a policy server must keep "
              "running and listen on 0.0.0.0:$RC_POLICY_PORT", flush=True)
        return EXIT_PROGRAM_EXITED
    except KeyboardInterrupt:
        print("[rc_runtime] entry program exited (interrupted)", flush=True)
        return EXIT_PROGRAM_EXITED
    except BaseException as exc:
        print(f"[rc_runtime] entry program exited: raised {type(exc).__name__}; full diagnostic follows:",
              flush=True)
        traceback.print_exception(type(exc), exc, exc.__traceback__)
        hint = hint_for(exc)
        if isinstance(exc, ImportError):
            hint = hint or ("that module is neither in the image nor in your repository; nothing is "
                            "installed at judging")
        if hint:
            print(f"[rc_runtime] hint: {hint}", flush=True)
        return EXIT_PROGRAM_FAILED
    print("[rc_runtime] entry program exited (returned); a policy server must keep running and listen "
          "on 0.0.0.0:$RC_POLICY_PORT", flush=True)
    return EXIT_PROGRAM_EXITED


def _decode_proc_addr(hexaddr: str):
    """/proc/net/tcp{,6} address field -> ipaddress object (kernel prints each 32-bit word in host order)."""
    import ipaddress
    import sys as _sys
    raw = bytes.fromhex(hexaddr)
    if len(raw) % 4:
        raise ValueError(hexaddr)
    order = "little" if _sys.byteorder == "little" else "big"
    packed = b"".join(int.from_bytes(raw[i:i + 4], order).to_bytes(4, "big") for i in range(0, len(raw), 4))
    addr = ipaddress.ip_address(packed)
    mapped = getattr(addr, "ipv4_mapped", None)
    return mapped or addr


def listening_sockets(proc_dir: str = "/proc/net") -> "Optional[dict]":
    """{port: [ip, ...]} for TCP LISTEN sockets in this network namespace, from /proc/net/tcp{,6}.
    None when /proc/net/tcp is unreadable (the caller then cannot tell loopback from reachable)."""
    out: dict = {}
    readable = False
    for name in ("tcp", "tcp6"):
        try:
            with open(os.path.join(proc_dir, name), encoding="ascii", errors="replace") as fh:
                readable = True
                next(fh, None)
                for line in fh:
                    parts = line.split()
                    if len(parts) > 3 and parts[3] == "0A":  # TCP_LISTEN
                        try:
                            host, port = parts[1].rsplit(":", 1)
                            out.setdefault(int(port, 16), []).append(_decode_proc_addr(host))
                        except (IndexError, ValueError):
                            pass
        except OSError:
            continue
    return out if readable else None


def listening_ports(proc_dir: str = "/proc/net") -> "set[int]":
    """TCP ports with a LISTEN socket in this network namespace (any address), from /proc/net/tcp{,6}."""
    return set(listening_sockets(proc_dir) or ())


def port_reachability(port: int, sockets: "Optional[dict]") -> str:
    """'reachable' (a wildcard or non-loopback listener: the runner connects via the container IP),
    'loopback' (only 127.0.0.0/8 or ::1), 'none', or 'unknown' (/proc unreadable)."""
    if sockets is None:
        return "unknown"
    addrs = sockets.get(port)
    if not addrs:
        return "none"
    return "reachable" if any(not a.is_loopback for a in addrs) else "loopback"


def listen_deadline_seconds(env) -> float:
    """RC_POLICY_LISTEN_DEADLINE_SECONDS (default 600); <= 0 disables the exit."""
    raw = (env.get("RC_POLICY_LISTEN_DEADLINE_SECONDS") or "").strip()
    if not raw:
        return DEFAULT_LISTEN_DEADLINE_SECONDS
    try:
        return float(raw)
    except ValueError:
        return DEFAULT_LISTEN_DEADLINE_SECONDS


def not_listening_message(port: int, deadline: float, others: "Iterable[int]" = ()) -> str:
    msg = f"nothing is listening on port {port} after {deadline:g} s"
    others = sorted(set(others) - {port})
    # ephemeral ports (library-internal listeners) are only worth naming when nothing else listens
    named = [p for p in others if p < 32768]
    others = named or others
    if others:
        msg += " (found a listener on port " + ", ".join(str(p) for p in others) + ")"
    return (msg + f"; your program must serve /healthz, /readyz and /act on $RC_POLICY_PORT ({port}) "
            "within the deadline. Start the server first and load slow models afterwards, reporting "
            "not-ready via /readyz")


def loopback_only_message(port: int, deadline: Optional[float] = None) -> str:
    when = f" after {deadline:g} s" if deadline is not None else " yet"
    return (f"your server listens on 127.0.0.1:{port} only{when}; the judge connects through the "
            f"container's network address, so bind 0.0.0.0:{port} (read $RC_POLICY_PORT; "
            f"host=\"0.0.0.0\", not \"localhost\" or \"127.0.0.1\")")


def start_port_watchdog(port: int, *, deadline: float = DEFAULT_LISTEN_DEADLINE_SECONDS,
                        warn_after: float = 60.0, warn_every: float = 120.0, poll: float = 1.0,
                        exit_fn=None, ports_fn=None, sockets_fn=None) -> "threading.Thread":
    """Program form: if nothing reachable listens on the contract port within `deadline` seconds, say
    why and exit EXIT_NOT_LISTENING (the runner would otherwise wait out the whole warmup deadline,
    again for every restart). A loopback-only listener does NOT satisfy the watchdog: the runner
    connects via the container IP. Warns earlier at `warn_after`. deadline <= 0 only warns.
    `ports_fn` (tests) supplies ports treated as reachable; `sockets_fn` supplies {port: [ip, ...]}."""
    import socket
    import threading
    import time
    exit_fn = exit_fn or os._exit
    if sockets_fn is None:
        sockets_fn = (lambda: {p: [] for p in ports_fn()}) if ports_fn else listening_sockets

    def state() -> str:
        sockets = sockets_fn()
        if ports_fn is not None:
            return "reachable" if port in ports_fn() else "none"
        status = port_reachability(port, sockets)
        if status == "unknown":  # cannot read /proc: a loopback probe is the best available signal
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    return "reachable"
            except OSError:
                return "none"
        return status

    def watch() -> None:
        start = time.monotonic()
        next_warn = warn_after
        while True:
            status = state()
            if status == "reachable":
                return
            elapsed = time.monotonic() - start
            if deadline > 0 and elapsed >= deadline:
                if status == "loopback":
                    detail = loopback_only_message(port, deadline)
                else:
                    detail = not_listening_message(port, deadline, (ports_fn() if ports_fn else sockets_fn() or {}))
                print(f"[rc_runtime] startup error program_not_listening: {detail}", flush=True)
                exit_fn(EXIT_NOT_LISTENING)
                return
            if elapsed >= next_warn:
                if status == "loopback":
                    print(f"[rc_runtime] WARNING: {loopback_only_message(port)}", flush=True)
                else:
                    print(f"[rc_runtime] WARNING: nothing is listening on port {port} yet. Your program must "
                          f"listen on 0.0.0.0:{port} (read $RC_POLICY_PORT); a different port is never probed.",
                          flush=True)
                next_warn += warn_every
            if deadline <= 0 and elapsed > 1800:
                return
            time.sleep(poll)

    t = threading.Thread(target=watch, daemon=True, name="rc-listen-watchdog")
    t.start()
    return t


def start_shadow_watchdog(submission: Path, reserved: Iterable[str], *, interval: float = 3.0,
                          duration: float = 180.0) -> "threading.Thread":
    """Program form blocks in team code, so re-run the post-import shadow check periodically."""
    import threading
    import time
    reserved = frozenset(reserved)

    def watch() -> None:
        end = time.monotonic() + duration
        while time.monotonic() < end:
            time.sleep(interval)
            try:
                verify_no_shadow(submission, reserved)
            except RuntimeStartupError as exc:
                print(f"[rc_runtime] startup error {exc.code}: {exc.detail}", flush=True)
                os._exit(EXIT_RESERVED_SHADOW)
            except Exception:
                return

    t = threading.Thread(target=watch, daemon=True)
    t.start()
    return t


def resolve_entry(
    submission: Path, entry_point: str, reserved: Iterable[str]
) -> Policy:
    """Full pre-load resolution: static check, path append, import, verification."""
    module_name, attr = parse_entry_point(entry_point)
    check_static_shadow(module_name, reserved)
    added = append_submission_path(Path(submission), module_name)
    warn_image_collisions(added, reserved, module_name)
    # Relative asset paths in team code resolve against the submission root. The tree is
    # read-only: writes must go to /scratch.
    try:
        os.chdir(str(submission))
    except OSError:
        pass
    policy = resolve_policy(module_name, attr)
    verify_no_shadow(Path(submission), reserved)
    return policy


SCRATCH_SUBDIRS = ("tmp", "home", "hf", "xdg", "triton", "inductor", "jax", "cuda",
                   "numba", "mpl", "pycache")


def create_scratch(scratch: Path) -> None:
    try:
        scratch = Path(scratch)
        scratch.mkdir(parents=True, exist_ok=True)
        for name in SCRATCH_SUBDIRS:
            d = scratch / name
            d.mkdir(mode=0o700, exist_ok=True)
            if not os.access(d, os.W_OK | os.X_OK):
                raise PermissionError(str(d))
    except OSError as exc:
        raise RuntimeStartupError(
            "scratch_unwritable", f"{scratch}: {exc}", EXIT_SCRATCH_UNWRITABLE) from exc


def run_load(policy: Policy) -> None:
    try:
        policy.load()
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            raise
        raise RuntimeStartupError("load_failed", f"{type(exc).__name__}",
                                  hint=hint_for(exc)) from exc


LoadFn = Callable[[], Policy]
