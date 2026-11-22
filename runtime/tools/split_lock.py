#!/usr/bin/env python3
"""Split a hashed lock into per-layer slices (Dockerfile layer order, see locks/slices.txt).

  split_lock.py locks/requirements.arm64.txt locks/slices.txt --out-dir locks   # writes files
  split_lock.py ... --check                                                     # up to date?

Blocks (a "name==version \\" line, its --hash lines and comments) are copied verbatim, so the
hashes are exactly the ones in the full lock; every slice starts with the lock's header lines.
Deterministic: slice order from slices.txt, block order from the lock. A package that matches
no slice goes to the final "*" slice; one that matches an earlier slice never reaches a later one.
"""
import argparse
import fnmatch
import re
import sys
from pathlib import Path

PIN = re.compile(r"^([A-Za-z0-9_.-]+)(\[[^\]]*\])?==")


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_slices(text: str):
    slices = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        name, _, pats = line.partition(":")
        slices.append((name.strip(), [norm(p) if "*" not in p else p.lower() for p in pats.split()]))
    if not slices or slices[-1][1] != ["*"]:
        raise ValueError("the last slice must be '*'")
    return slices


def parse_lock(text: str):
    """Return (header_lines, [(normalised_name, block_text)])."""
    header, blocks, cur = [], [], None
    for line in text.splitlines(keepends=True):
        m = PIN.match(line)
        if m:
            cur = [norm(m.group(1)), line]
            blocks.append(cur)
        elif cur is None:
            header.append(line)
        else:
            cur[1] += line
    return "".join(header), [(n, b) for n, b in blocks]


def split(lock_text: str, slices_text: str):
    slices = parse_slices(slices_text)
    header, blocks = parse_lock(lock_text)
    out = {name: [] for name, _ in slices}
    for pkg, block in blocks:
        for name, pats in slices:
            if any(fnmatch.fnmatchcase(pkg, p) for p in pats):
                out[name].append(block)
                break
    return {name: header + "".join(bs) for name, bs in out.items()}


def slice_path(out_dir: Path, lock: Path, name: str) -> Path:
    return out_dir / (lock.stem + "." + name + lock.suffix)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("lock", type=Path)
    ap.add_argument("slices", type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    out_dir = a.out_dir or a.lock.parent
    parts = split(a.lock.read_text(), a.slices.read_text())
    bad = 0
    for name, text in parts.items():
        p = slice_path(out_dir, a.lock, name)
        if a.check:
            if not p.exists() or p.read_text() != text:
                print(f"stale or missing: {p}", file=sys.stderr)
                bad += 1
        else:
            p.write_text(text)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
