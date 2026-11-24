#!/usr/bin/env python3
"""Release metadata tooling: reserved-modules list, package inventory, release.json.

Subcommands (run `reserved` and `packages` inside the image, once per architecture):

  rc_release.py reserved [--site DIR ...] [-o reserved.json]
  rc_release.py packages [-o packages.<arch>.json]
  rc_release.py write -o release.json --release-id ID --repository REPO --tag TAG \
        --manifest-digest sha256:.. --platform-digest linux/arm64=sha256:.. \
        --platform-digest linux/amd64=sha256:.. [--config-digest PLAT=sha256:..] \
        --packages linux/arm64=packages.arm64.json --packages linux/amd64=packages.amd64.json \
        [--reserved reserved.json] [--base-cache CACHE_MANIFEST.json] [--partial]
  rc_release.py validate release.json

Schema: docs/runtime-interfaces.md section 2 (version 1). Stdlib only.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata as md
import importlib.util
import json
import os
import re
import site
import sys
import sysconfig
from pathlib import Path
from typing import Iterable, Optional

HERE = Path(__file__).resolve().parent
DEFAULT_NUMERIC_FORMATS = HERE.parent / "numeric_formats.json"

PLATFORMS = ("linux/arm64", "linux/amd64")
STATUSES = {"untested", "installed", "import-tested", "operation-tested",
            "end-to-end-tested", "unsupported"}
DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
CREATED_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
MODULE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# Names that are never import packages even though they sit in site-packages.
SKIP_NAMES = {"__pycache__", "__editable__", "_distutils_hack"}


# --- reserved modules -------------------------------------------------------

def site_dirs() -> list[str]:
    dirs = {sysconfig.get_paths()["purelib"], sysconfig.get_paths()["platlib"]}
    try:
        dirs.update(site.getsitepackages())
    except AttributeError:
        pass
    return sorted(d for d in dirs if os.path.isdir(d))


def _top_name(entry: str) -> Optional[str]:
    if entry in SKIP_NAMES or entry.startswith(("_virtualenv", "__editable__")):
        return None
    if entry.endswith((".dist-info", ".egg-info", ".data", ".pth", ".egg-link", ".pyc")):
        return None
    name = entry.split(".", 1)[0]  # foo.py, foo.cpython-312-x86_64-linux-gnu.so, foo (dir)
    return name if MODULE_RE.match(name) else None


# Top-level files some wheels ship by mistake (nvidia-ml-py's example.py). Not reserved unless the
# distribution declares them in top_level.txt, so a team module with such a name is not rejected.
STRAY_NAMES = {"example", "examples", "test", "tests", "conftest"}


def reserved_modules(dirs: Optional[Iterable[str]] = None) -> list[str]:
    """Top-level importable names found in site-packages (dirs, modules, extensions).

    Also includes top_level.txt names from installed distributions, which catches
    packages that install into other path entries.
    """
    names: set[str] = set()
    for d in (dirs if dirs is not None else site_dirs()):
        for entry in os.listdir(d):
            full = os.path.join(d, entry)
            if os.path.isdir(full) and not os.path.exists(os.path.join(full, "__init__.py")):
                # namespace packages or data dirs: reserve only if they contain python
                has_py = any(f.endswith((".py", ".so")) for f in os.listdir(full)) or \
                    any(os.path.isdir(os.path.join(full, s)) for s in os.listdir(full))
                if not has_py:
                    continue
            n = _top_name(entry)
            if n and n not in STRAY_NAMES:
                names.add(n)
    if dirs is None:
        for dist in md.distributions():
            try:
                top = dist.read_text("top_level.txt") or ""
            except Exception:
                top = ""
            for t in (t.strip() for t in top.splitlines()):
                if not MODULE_RE.match(t):
                    continue
                # A stray name declared in top_level.txt whose file was removed (nvidia-ml-py's
                # example.py) is not importable, so it must not block a team module of that name.
                if t in STRAY_NAMES and importlib.util.find_spec(t) is None:
                    continue
                names.add(t)
    names.update({"rc_sdk", "rc_runtime"})
    return sorted(names)


# --- package inventory ------------------------------------------------------

def installed_packages() -> dict[str, str]:
    out: dict[str, str] = {}
    for dist in md.distributions():
        name = dist.metadata["Name"]
        if name:
            out[re.sub(r"[-_.]+", "-", name).lower()] = dist.version
    return dict(sorted(out.items()))


# --- release.json -----------------------------------------------------------

def _load_json(path) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _kv(items: Iterable[str], what: str) -> dict[str, str]:
    out = {}
    for item in items or []:
        k, sep, v = item.partition("=")
        if not sep or not k or not v:
            raise SystemExit(f"{what} must look like PLATFORM=VALUE, got {item!r}")
        out[k] = v
    return out


def build_release(*, release_id: str, repository: str, tag: str,
                  manifest_digest: Optional[str], platform_digests: dict, config_digests: dict,
                  packages_by_platform: dict, reserved: list, numeric_formats: list,
                  base_cache: Optional[dict], docs_url: str = "", created_at: Optional[str] = None,
                  python_version: Optional[str] = None,
                  package_status: Optional[dict] = None) -> dict:
    created_at = created_at or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    package_status = package_status or {}
    all_names = sorted({n for pk in packages_by_platform.values() for n in pk})
    packages = []
    for name in all_names:
        per = {p: pk[name] for p, pk in sorted(packages_by_platform.items()) if name in pk}
        canonical = per.get("linux/arm64") or next(iter(per.values()))
        packages.append({
            "name": name, "version": canonical, "platforms": per,
            "status": package_status.get(name, {"spark": "installed", "modal": "installed"}),
        })
    release = {
        "schema_version": 1,
        "release_id": release_id,
        "created_at": created_at,
        "image": {
            "repository": repository, "tag": tag, "manifest_digest": manifest_digest,
            "platform_digests": dict(platform_digests), "config_digests": dict(config_digests),
        },
        "python": {"version": python_version or ".".join(map(str, sys.version_info[:3])),
                   "implementation": sys.implementation.name},
        "stdlib_module_names": sorted(sys.stdlib_module_names),
        "reserved_modules": sorted(set(reserved)),
        "packages": packages,
        "numeric_formats": numeric_formats,
        "base_cache": base_cache,
        "docs_url": docs_url,
    }
    return release


def validate_release(r: object, partial: bool = False) -> list[str]:
    """Return a list of problems (empty = valid). partial=True allows missing digests."""
    errs: list[str] = []

    def need(cond: bool, msg: str) -> bool:
        if not cond:
            errs.append(msg)
        return cond

    if not need(isinstance(r, dict), "release must be an object"):
        return errs
    need(r.get("schema_version") == 1, "schema_version must be 1")
    need(isinstance(r.get("release_id"), str) and r["release_id"], "release_id must be a non-empty string")
    need(isinstance(r.get("created_at"), str) and CREATED_RE.match(r["created_at"] or ""),
         "created_at must be UTC like 2026-11-21T12:00:00Z")
    img = r.get("image")
    if need(isinstance(img, dict), "image must be an object"):
        need(isinstance(img.get("repository"), str) and img["repository"], "image.repository missing")
        need(isinstance(img.get("tag"), str) and img["tag"], "image.tag missing")
        md_ = img.get("manifest_digest")
        need((partial and md_ is None) or (isinstance(md_, str) and DIGEST_RE.match(md_)),
             "image.manifest_digest must be sha256:<64 hex>")
        for key in ("platform_digests", "config_digests"):
            d = img.get(key)
            if need(isinstance(d, dict), f"image.{key} must be an object"):
                for plat, dig in d.items():
                    need(plat in PLATFORMS, f"image.{key}: unknown platform {plat!r}")
                    need(isinstance(dig, str) and DIGEST_RE.match(dig), f"image.{key}[{plat}] bad digest")
                if key == "platform_digests" and not partial:
                    need(set(d) == set(PLATFORMS), "image.platform_digests needs linux/arm64 and linux/amd64")
    py = r.get("python")
    need(isinstance(py, dict) and isinstance(py.get("version"), str), "python.version missing")
    for key in ("stdlib_module_names", "reserved_modules"):
        v = r.get(key)
        need(isinstance(v, list) and v and all(isinstance(x, str) for x in v), f"{key} must be a non-empty string list")
    if isinstance(r.get("reserved_modules"), list):
        need({"rc_sdk", "rc_runtime"} <= set(r["reserved_modules"]), "reserved_modules must include rc_sdk and rc_runtime")
    pk = r.get("packages")
    if need(isinstance(pk, list), "packages must be a list"):
        for p in pk:
            ok = isinstance(p, dict) and isinstance(p.get("name"), str) and isinstance(p.get("version"), str) \
                and isinstance(p.get("platforms"), dict) and isinstance(p.get("status"), dict)
            if not need(ok, f"bad package entry {p!r}"):
                continue
            for plat in p["platforms"]:
                need(plat in PLATFORMS, f"package {p['name']}: unknown platform {plat!r}")
            for cell in p["status"].values():
                need(cell in STATUSES, f"package {p['name']}: bad status {cell!r}")
    nf = r.get("numeric_formats")
    if need(isinstance(nf, list) and nf, "numeric_formats must be a non-empty list"):
        for row in nf:
            if not need(isinstance(row, dict), "numeric_formats row must be an object"):
                continue
            for k in ("format", "dtype_codes", "display", "baseline", "spark", "modal", "libraries"):
                need(k in row, f"numeric_formats[{row.get('format')}] missing {k}")
            need(row.get("spark") in STATUSES, f"numeric_formats[{row.get('format')}] bad spark status")
            modal = row.get("modal")
            if need(isinstance(modal, dict), f"numeric_formats[{row.get('format')}] modal must be an object"):
                for gpu, cell in modal.items():
                    need(cell in STATUSES, f"numeric_formats[{row.get('format')}] bad modal status {gpu}={cell!r}")
    bc = r.get("base_cache")
    if bc is not None and need(isinstance(bc, dict), "base_cache must be an object or null"):
        need(bc.get("version") is None or isinstance(bc["version"], str), "base_cache.version bad")
        need(bc.get("sha256") is None or DIGEST_RE.match(str(bc["sha256"])), "base_cache.sha256 bad")
        ents = bc.get("entries")
        if need(isinstance(ents, list), "base_cache.entries must be a list"):
            for e in ents:
                need(isinstance(e, dict) and isinstance(e.get("repo_id"), str)
                     and re.fullmatch(r"[0-9a-f]{40}", str(e.get("sha", ""))), f"bad base_cache entry {e!r}")
    return errs


def example_release() -> dict:
    """Deterministic stand-in committed as checks/rc_checks/tests/fixtures/release.example.json."""
    arm = {"numpy": "2.2.6", "torch": "2.14.1+cu130", "torchvision": "0.29.1+cu130",
           "transformers": "5.22.2", "huggingface-hub": "2.4.0", "safetensors": "0.8.0",
           "lerobot": "0.7.0", "jax": "0.12.1", "pin": "4.1.0", "trimesh": "5.2.1",
           "rc-sdk": "0.1.0", "rc-runtime": "0.1.0"}
    amd = dict(arm)
    reserved = ["numpy", "torch", "torchvision", "transformers", "huggingface_hub", "safetensors",
                "lerobot", "jax", "jaxlib", "pinocchio", "trimesh", "scipy", "cv2", "PIL",
                "rc_sdk", "rc_runtime"]
    return build_release(
        release_id="rc-policy-2026.11.21-1", created_at="2026-11-21T12:00:00Z",
        repository="ghcr.io/example/rc-policy", tag="2026.11.21-1",
        manifest_digest="sha256:" + "a" * 64,
        platform_digests={"linux/arm64": "sha256:" + "b" * 64, "linux/amd64": "sha256:" + "c" * 64},
        config_digests={"linux/arm64": "sha256:" + "d" * 64, "linux/amd64": "sha256:" + "e" * 64},
        packages_by_platform={"linux/arm64": arm, "linux/amd64": amd}, reserved=reserved,
        numeric_formats=_load_json(DEFAULT_NUMERIC_FORMATS),
        base_cache={"version": "bc-2026.11.21-1", "sha256": "sha256:" + "f" * 64,
                    "entries": [{"repo_id": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
                                 "sha": "0123456789abcdef0123456789abcdef01234567"}]},
        docs_url="https://dashboard.example/docs", python_version="3.12.8")


# --- CLI --------------------------------------------------------------------

def _emit(obj, out: Optional[str]) -> None:
    text = json.dumps(obj, indent=2, sort_keys=False) + "\n"
    if out and out != "-":
        Path(out).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("reserved")
    p.add_argument("--site", action="append", default=None)
    p.add_argument("-o", "--out")
    p = sub.add_parser("packages")
    p.add_argument("-o", "--out")
    p = sub.add_parser("example")
    p.add_argument("-o", "--out")
    p = sub.add_parser("validate")
    p.add_argument("file")
    p.add_argument("--partial", action="store_true")
    w = sub.add_parser("write")
    w.add_argument("-o", "--out", required=True)
    w.add_argument("--release-id", required=True)
    w.add_argument("--repository", required=True)
    w.add_argument("--tag", required=True)
    w.add_argument("--manifest-digest")
    w.add_argument("--platform-digest", action="append", default=[])
    w.add_argument("--config-digest", action="append", default=[])
    w.add_argument("--packages", action="append", default=[], help="PLATFORM=packages.json (repeat)")
    w.add_argument("--package-status", help="JSON {name: {spark:..., modal:...}} overrides")
    w.add_argument("--reserved", help="reserved list JSON (default: generate from this environment)")
    w.add_argument("--numeric-formats", default=str(DEFAULT_NUMERIC_FORMATS))
    w.add_argument("--base-cache", help="CACHE_MANIFEST.json from build_base_cache.py")
    w.add_argument("--docs-url", default="")
    w.add_argument("--created-at")
    w.add_argument("--python-version")
    w.add_argument("--partial", action="store_true",
                   help="allow missing digests (the copy baked into the image cannot contain its own digest)")
    a = ap.parse_args(argv)

    if a.cmd == "reserved":
        _emit(reserved_modules(a.site), a.out)
    elif a.cmd == "packages":
        _emit(installed_packages(), a.out)
    elif a.cmd == "example":
        _emit(example_release(), a.out)
    elif a.cmd == "validate":
        errs = validate_release(_load_json(a.file), partial=a.partial)
        for e in errs:
            print(f"invalid: {e}", file=sys.stderr)
        return 1 if errs else 0
    else:
        pk = {plat: _load_json(path) for plat, path in _kv(a.packages, "--packages").items()}
        if not pk:
            pk = {"linux/arm64" if os.uname().machine in ("aarch64", "arm64") else "linux/amd64": installed_packages()}
        bc = None
        if a.base_cache:
            m = _load_json(a.base_cache)
            bc = {"version": m.get("version"), "sha256": m.get("sha256"), "entries": m.get("entries", [])}
        release = build_release(
            release_id=a.release_id, repository=a.repository, tag=a.tag,
            manifest_digest=a.manifest_digest, platform_digests=_kv(a.platform_digest, "--platform-digest"),
            config_digests=_kv(a.config_digest, "--config-digest"), packages_by_platform=pk,
            reserved=_load_json(a.reserved) if a.reserved else reserved_modules(),
            numeric_formats=_load_json(a.numeric_formats), base_cache=bc, docs_url=a.docs_url,
            created_at=a.created_at, python_version=a.python_version,
            package_status=_load_json(a.package_status) if a.package_status else None)
        errs = validate_release(release, partial=a.partial)
        if errs:
            for e in errs:
                print(f"invalid: {e}", file=sys.stderr)
            return 1
        _emit(release, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
