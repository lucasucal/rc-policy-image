#!/usr/bin/env python3
"""Build the curated base cache: an offline Hugging Face hub cache of pinned commits.

  build_base_cache.py <entries.json> <out_dir> [--version bc-2026.11.21-1] [--dry-run]
  build_base_cache.py verify <cache_dir> [--no-hub]

entries.json: a list (or {"entries": [...]}) of {"repo_id": "org/name", "sha": "<40 hex>"}.
Optional per entry: "license_override": "<reason>" to accept a license the tool would
otherwise refuse (organizer decision, recorded in CACHE_DETAILS.json).

File selection (so a 0.5B model is not 6.7 GB of ONNX/GGUF variants): per entry,
  "allow_patterns": [glob, ...]   download only matching files (fnmatch on the repo path)
  "ignore_patterns": [glob, ...]  then drop matching files
If NEITHER key is present the default filter applies: always skip onnx/ and *.onnx, *.onnx_data,
*.gguf, *.tflite, *.mlmodel, *.mlpackage/; and, only when the repo has *.safetensors, also skip the
duplicate-format weights (tf_model.h5, *.msgpack (flax), rust_model.ot, model.ckpt*,
pytorch_model*.bin and its index). Giving "ignore_patterns": [] (empty) disables filtering for that
entry. A selection that leaves no files is an error. The patterns used are recorded in
CACHE_DETAILS.json. Patterns are part of the cache you build, not of the manifest sha; the manifest
hash covers the files actually present.

Output layout (docs/runtime-interfaces.md 4.3, spec 5.5):
  CACHE_VERSION, CACHE_MANIFEST.json {"version","sha256","entries"}, CACHE_DETAILS.json,
  models--<org>--<name>/{blobs/<sha256>, refs/main, snapshots/<sha>/<path> -> ../../blobs/..}

Refuses gated, private, or non-redistributable (license missing/other/non-commercial)
entries and any entry whose hub commit does not equal the requested sha. The cache is
made read-only. Requires huggingface_hub only for the real downloader and for the
optional offline-resolution check.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol

REPO_ID_RE = re.compile(r"^(?:[A-Za-z0-9][A-Za-z0-9._-]{0,95}/)?[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
BAD_LICENSES = {"", "other", "unknown", "none"}

# Default file filter (see module docstring). Patterns use fnmatch on the path inside the repo;
# "*" matches "/" too, so "*.onnx" also matches "sub/dir/model.onnx".
DEFAULT_IGNORE_ALWAYS = ["onnx/*", "*/onnx/*", "*.onnx", "*.onnx_data", "*.gguf", "*.tflite",
                         "*.mlmodel", "*.mlpackage/*"]
DEFAULT_IGNORE_IF_SAFETENSORS = ["tf_model*.h5", "*.msgpack", "rust_model.ot", "model.ckpt*",
                                 "pytorch_model*.bin", "pytorch_model*.bin.index.json"]


class CacheBuildError(Exception):
    pass


@dataclass
class RepoInfo:
    sha: str
    gated: object = False       # False | "auto" | "manual" | True
    private: bool = False
    license: Optional[str] = None


class Fetcher(Protocol):
    def info(self, repo_id: str, sha: str) -> RepoInfo: ...
    def fetch(self, repo_id: str, sha: str, dest: Path, select=None) -> None: ...


class HubFetcher:
    """Real downloader (huggingface_hub). No token is ever sent: gated/private repos fail."""

    def __init__(self) -> None:
        from huggingface_hub import HfApi
        self.api = HfApi(token=False)

    def info(self, repo_id: str, sha: str) -> RepoInfo:
        i = self.api.model_info(repo_id, revision=sha)
        lic = None
        card = getattr(i, "card_data", None) or getattr(i, "cardData", None)
        if card is not None:
            lic = card.get("license") if hasattr(card, "get") else getattr(card, "license", None)
        if not lic:
            lic = next((t.split(":", 1)[1] for t in (i.tags or []) if t.startswith("license:")), None)
        if isinstance(lic, list):
            lic = lic[0] if lic else None
        return RepoInfo(sha=i.sha, gated=i.gated, private=bool(i.private), license=lic)

    def fetch(self, repo_id: str, sha: str, dest: Path, select=None) -> None:
        """Download the repo files chosen by select(all_paths) -> paths (all files if None)."""
        from huggingface_hub import hf_hub_download
        files = sorted(self.api.list_repo_files(repo_id, revision=sha))
        for rel in (select(files) if select else files):
            hf_hub_download(repo_id, rel, revision=sha, local_dir=str(dest), token=False)
        shutil.rmtree(dest / ".cache", ignore_errors=True)


def load_entries(path: Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("entries")
    if not isinstance(data, list) or not data:
        raise CacheBuildError("entries.json must be a non-empty list (or {'entries': [...]})")
    seen, out = set(), []
    for e in data:
        if not isinstance(e, dict):
            raise CacheBuildError(f"bad entry {e!r}")
        repo_id, sha = e.get("repo_id"), e.get("sha")
        if not isinstance(repo_id, str) or not REPO_ID_RE.match(repo_id) or ".." in repo_id:
            raise CacheBuildError(f"bad repo_id {repo_id!r}")
        if not isinstance(sha, str) or not SHA_RE.match(sha):
            raise CacheBuildError(f"{repo_id}: sha must be a full 40-hex commit, got {sha!r}")
        if repo_id in seen:
            raise CacheBuildError(f"duplicate repo_id {repo_id}")
        seen.add(repo_id)
        item = {"repo_id": repo_id, "sha": sha, "license_override": e.get("license_override")}
        for key in ("allow_patterns", "ignore_patterns"):
            v = e.get(key)
            if v is not None:
                if not isinstance(v, list) or not all(isinstance(x, str) and x for x in v):
                    raise CacheBuildError(f"{repo_id}: {key} must be a list of non-empty glob strings")
                item[key] = v
        out.append(item)
    return out


def select_files(entry: dict, files: list[str]) -> list[str]:
    """Apply the entry's allow/ignore patterns (or the default filter) to repo file paths."""
    allow, ignore = entry.get("allow_patterns"), entry.get("ignore_patterns")
    keep = list(files)
    if allow is not None:
        keep = [f for f in keep if any(fnmatch.fnmatchcase(f, g) for g in allow)]
    if ignore is not None:
        drop = ignore
    elif allow is not None:
        drop = []
    else:
        drop = list(DEFAULT_IGNORE_ALWAYS)
        if any(f.endswith(".safetensors") for f in files):
            drop += DEFAULT_IGNORE_IF_SAFETENSORS
    keep = [f for f in keep if not any(fnmatch.fnmatchcase(f, g) for g in drop)]
    if not keep:
        raise CacheBuildError(f"{entry['repo_id']}: the file selection leaves no files")
    return keep


def check_redistributable(entry: dict, info: RepoInfo) -> str:
    """Return the license string recorded for the entry or raise CacheBuildError."""
    rid = entry["repo_id"]
    if info.gated:
        raise CacheBuildError(f"{rid}: refused, repository is gated ({info.gated!r})")
    if info.private:
        raise CacheBuildError(f"{rid}: refused, repository is private")
    if info.sha != entry["sha"]:
        raise CacheBuildError(f"{rid}: hub resolved {entry['sha']} to {info.sha}; pin the full commit sha")
    lic = (info.license or "").strip().lower()
    bad = lic in BAD_LICENSES or "-nc" in lic or lic.startswith("cc-by-nd")
    if bad:
        if entry.get("license_override"):
            return f"{lic or 'unspecified'} (override: {entry['license_override']})"
        raise CacheBuildError(f"{rid}: refused, license {lic or 'missing'!r} is not known to be redistributable "
                              "(set license_override with a reason to accept)")
    return lic


def repo_dir_name(repo_id: str) -> str:
    return "models--" + repo_id.replace("/", "--")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _lay_out(entry: dict, src: Path, out: Path) -> list[tuple[str, str, int]]:
    """Move files from src into hub layout. Returns [(relpath, sha256, size)]."""
    repo = out / repo_dir_name(entry["repo_id"])
    blobs, snap = repo / "blobs", repo / "snapshots" / entry["sha"]
    (repo / "refs").mkdir(parents=True)
    blobs.mkdir()
    snap.mkdir(parents=True)
    files = []
    for f in sorted(p for p in src.rglob("*") if p.is_file() and not p.is_symlink()):
        rel = f.relative_to(src).as_posix()
        digest = _sha256_file(f)
        blob = blobs / digest
        if blob.exists():
            f.unlink()
        else:
            shutil.move(str(f), blob)
        link = snap / rel
        link.parent.mkdir(parents=True, exist_ok=True)
        depth = len(Path(rel).parts)  # snapshots/<sha>/<dirs..> -> ../ * (depth+1) to repo root
        link.symlink_to(os.path.join(*([".."] * (depth + 1)), "blobs", digest))
        files.append((rel, digest, blob.stat().st_size))
    if not files:
        raise CacheBuildError(f"{entry['repo_id']}: fetcher produced no files")
    (repo / "refs" / "main").write_text(entry["sha"], encoding="utf-8")
    return files


def content_hash(rows: list[tuple[str, str, str, str]]) -> str:
    h = hashlib.sha256()
    for repo_id, sha, rel, digest in sorted(rows):
        h.update(f"{repo_id}\t{sha}\t{rel}\t{digest}\n".encode())
    return "sha256:" + h.hexdigest()


def _make_readonly(root: Path) -> None:
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        for n in filenames:
            p = Path(dirpath, n)
            if not p.is_symlink():
                p.chmod(0o444)
        Path(dirpath).chmod(0o555)


def make_writable(root: Path) -> None:
    """Undo _make_readonly (for deleting a cache or rebuilding)."""
    for dirpath, dirnames, filenames in os.walk(root):
        Path(dirpath).chmod(0o755)
        for n in filenames:
            p = Path(dirpath, n)
            if not p.is_symlink():
                p.chmod(0o644)


def build_cache(entries: list[dict], out_dir: Path, fetcher: Fetcher, version: Optional[str] = None,
                dry_run: bool = False) -> dict:
    out_dir = Path(out_dir)
    version = version or "bc-" + dt.datetime.now(dt.timezone.utc).strftime("%Y.%m.%d") + "-1"
    # Pre-flight every entry before any download so a refusal wastes nothing.
    licenses = {e["repo_id"]: check_redistributable(e, fetcher.info(e["repo_id"], e["sha"])) for e in entries}
    if dry_run:
        return {"version": version, "entries": [{"repo_id": e["repo_id"], "sha": e["sha"]} for e in entries]}
    if out_dir.exists() and any(out_dir.iterdir()):
        raise CacheBuildError(f"{out_dir} is not empty")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, details = [], []
    try:
        for e in entries:
            with tempfile.TemporaryDirectory(prefix="rc-cache-") as tmp:
                fetcher.fetch(e["repo_id"], e["sha"], Path(tmp), select=lambda fs, e=e: select_files(e, fs))
                files = _lay_out(e, Path(tmp), out_dir)
            rows += [(e["repo_id"], e["sha"], rel, d) for rel, d, _ in files]
            details.append({"repo_id": e["repo_id"], "sha": e["sha"], "license": licenses[e["repo_id"]],
                            "files": len(files), "size_bytes": sum(s for _, _, s in files),
                            "allow_patterns": e.get("allow_patterns"),
                            "ignore_patterns": e.get("ignore_patterns", "default" if "allow_patterns" not in e else [])})
    except BaseException:
        make_writable(out_dir)
        shutil.rmtree(out_dir, ignore_errors=True)
        raise
    manifest = {"version": version, "sha256": content_hash(rows),
                "entries": [{"repo_id": e["repo_id"], "sha": e["sha"]} for e in entries]}
    (out_dir / "CACHE_VERSION").write_text(version + "\n", encoding="utf-8")
    (out_dir / "CACHE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (out_dir / "CACHE_DETAILS.json").write_text(
        json.dumps({"entries": details,
                    "created_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}, indent=2) + "\n",
        encoding="utf-8")
    _make_readonly(out_dir)
    return manifest


def verify_cache(cache_dir: Path) -> list[str]:
    """Structural verification. Returns problems (empty = ok)."""
    cache_dir = Path(cache_dir)
    errs: list[str] = []
    try:
        manifest = json.loads((cache_dir / "CACHE_MANIFEST.json").read_text())
        version = (cache_dir / "CACHE_VERSION").read_text().strip()
    except (OSError, ValueError) as exc:
        return [f"cannot read CACHE_MANIFEST.json/CACHE_VERSION: {exc}"]
    if manifest.get("version") != version:
        errs.append("CACHE_VERSION differs from CACHE_MANIFEST.json version")
    rows = []
    for e in manifest.get("entries", []):
        repo = cache_dir / repo_dir_name(e["repo_id"])
        ref = repo / "refs" / "main"
        if not ref.is_file() or ref.read_text().strip() != e["sha"]:
            errs.append(f"{e['repo_id']}: refs/main missing or not {e['sha']}")
            continue
        snap = repo / "snapshots" / e["sha"]
        if not snap.is_dir():
            errs.append(f"{e['repo_id']}: snapshot dir missing")
            continue
        found = 0
        for f in sorted(snap.rglob("*")):
            if f.is_dir():
                continue
            found += 1
            real = f.resolve()
            if not f.is_symlink() or (repo / "blobs") not in real.parents:
                errs.append(f"{e['repo_id']}: {f.relative_to(snap)} is not a symlink into blobs/")
                continue
            digest = _sha256_file(real)
            if digest != real.name:
                errs.append(f"{e['repo_id']}: blob {real.name} content hash mismatch")
            rows.append((e["repo_id"], e["sha"], f.relative_to(snap).as_posix(), digest))
        if not found:
            errs.append(f"{e['repo_id']}: empty snapshot")
    if not errs and content_hash(rows) != manifest.get("sha256"):
        errs.append("content hash differs from CACHE_MANIFEST.json sha256")
    return errs


def _tree_state(root: Path) -> dict:
    state = {}
    for dirpath, dirnames, filenames in os.walk(root):
        for n in dirnames + filenames:
            p = Path(dirpath, n)
            st = p.lstat()
            state[str(p.relative_to(root))] = (st.st_mode, st.st_size, st.st_mtime_ns)
    return state


_RESOLVE_SNIPPET = r"""
import json, sys
from huggingface_hub import snapshot_download, hf_hub_download
repo, rel = sys.argv[1], sys.argv[2]
snap = snapshot_download(repo)                      # by hub id, uses HF_HUB_CACHE, offline
file = hf_hub_download(repo, rel)
print(json.dumps({"snapshot": snap, "file": file}))
"""


def check_hub_offline_resolution(cache_dir: Path) -> list[str]:
    """Resolve every entry by hub id with huggingface_hub offline from the (read-only)
    cache and confirm the cache tree is byte-for-byte untouched afterwards.
    Skipped (with a note) if huggingface_hub is not importable."""
    cache_dir = Path(cache_dir)
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        return ["huggingface_hub not importable; offline resolution not checked"]
    manifest = json.loads((cache_dir / "CACHE_MANIFEST.json").read_text())
    before = _tree_state(cache_dir)
    errs = []
    env = {k: v for k, v in os.environ.items() if not k.startswith(("HF_", "HUGGINGFACE_"))}
    with tempfile.TemporaryDirectory(prefix="rc-hf-home-") as home:
        env.update(HF_HUB_CACHE=str(cache_dir), HF_HUB_OFFLINE="1", HF_HOME=home, HOME=home)
        for e in manifest["entries"]:
            snap = cache_dir / repo_dir_name(e["repo_id"]) / "snapshots" / e["sha"]
            rel = sorted(p.relative_to(snap).as_posix() for p in snap.rglob("*") if p.is_symlink())[0]
            r = subprocess.run([sys.executable, "-c", _RESOLVE_SNIPPET, e["repo_id"], rel],
                               env=env, capture_output=True, text=True)
            if r.returncode != 0:
                errs.append(f"{e['repo_id']}: offline resolution failed: {r.stderr.strip().splitlines()[-1:]}")
                continue
            got = json.loads(r.stdout.strip().splitlines()[-1])
            if Path(got["snapshot"]).resolve() != snap.resolve():
                errs.append(f"{e['repo_id']}: resolved to {got['snapshot']}, expected {snap}")
    if _tree_state(cache_dir) != before:
        errs.append("offline resolution modified the cache directory")
    return errs


def main(argv: Optional[list] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "verify":
        ap = argparse.ArgumentParser(prog="build_base_cache.py verify")
        ap.add_argument("cache_dir")
        ap.add_argument("--no-hub", action="store_true")
        a = ap.parse_args(argv[1:])
        errs = verify_cache(Path(a.cache_dir))
        if not errs and not a.no_hub:
            errs = check_hub_offline_resolution(Path(a.cache_dir))
        for e in errs:
            print(e, file=sys.stderr)
        return 1 if errs else 0
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("entries")
    ap.add_argument("out_dir")
    ap.add_argument("--version")
    ap.add_argument("--dry-run", action="store_true", help="check gating/license/sha only; download nothing")
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args(argv)
    try:
        manifest = build_cache(load_entries(Path(a.entries)), Path(a.out_dir), HubFetcher(), a.version, a.dry_run)
        if not a.dry_run and not a.no_verify:
            errs = verify_cache(Path(a.out_dir)) or check_hub_offline_resolution(Path(a.out_dir))
            if errs:
                raise CacheBuildError("; ".join(errs))
    except CacheBuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
