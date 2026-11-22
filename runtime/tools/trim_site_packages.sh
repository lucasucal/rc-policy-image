#!/bin/sh
# Image-size trims, run in the SAME RUN as the pip install that created the files (so the layer
# shrinks). Each mode is idempotent and a no-op when its target is absent. Reasons: runtime/README.md.
#   triton-cupti      triton's bundled CUPTI copies + static .a (profiler only)
#   tests PKG...      "tests" directories inside the named top-level site-packages packages
#   pycache           __pycache__ dirs (PYTHONPYCACHEPREFIX makes Python ignore them anyway)
set -eu
SP=$(python -c 'import site; print(site.getsitepackages()[0])')
mode=$1; shift
case "$mode" in
  triton-cupti)
    rm -rf "$SP"/triton/backends/nvidia/lib/cupti "$SP"/triton/backends/nvidia/lib/cupti-blackwell ;;
  tests)
    for p in "$@"; do
      [ -d "$SP/$p" ] && find "$SP/$p" -type d -name tests -prune -exec rm -rf {} + || true
    done ;;
  pycache)
    find "$SP" -type d -name __pycache__ -prune -exec rm -rf {} + ;;
  *) echo "unknown trim mode: $mode" >&2; exit 2 ;;
esac
rm -rf /root/.cache /tmp/pip-*
