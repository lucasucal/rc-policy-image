# Pinned policy runtime (work item A)

Spec: `docs/policy-runtime-pin.md`; contracts: `docs/runtime-interfaces.md` sections 2, 4.3, 5.

| Path | What |
|---|---|
| `rc_runtime/` | In-image loader and HTTP server. Entrypoint `python -I -m rc_runtime` |
| `Dockerfile`, `Dockerfile.dockerignore` | Multi-arch image (build context = repo root) |
| `requirements.in`, `locks/requirements.{amd64,arm64}.txt`, `make-locks.sh` | Hashed per-arch locks (wheels only) |
| `tools/rc_release.py` | reserved-modules list, package inventory, `release.json` writer/validator |
| `tools/build_base_cache.py` | Curated base cache builder and offline-resolution verifier |
| `tools/bake_weights.py` | Bakes torchvision ResNet checkpoints into `TORCH_HOME=/opt/rc/torch` |
| `numeric_formats.json` | Numeric format support table (every cell `untested` until tested) |
| `tests/` | Unit tests (no GPU, no Docker); `tests/docker/` CPU smoke image |

## Tests

```
python -m pip install pytest huggingface_hub   # hub tests skip without it
python -m pytest runtime/tests -q
```

The `-I` test needs `rc_sdk` and `rc_runtime` installed (`pip install ./starter/rc_sdk ./runtime`); it skips otherwise.

## Update the locks

`./runtime/make-locks.sh` (needs `uv`). It resolves for `x86_64` and `aarch64` (manylinux_2_28, Python 3.12) and fails if the pinned versions differ between architectures. Commit the result. It also writes `locks/requirements.sdist.txt` (the one sdist-only pin, `antlr4-python3-runtime`, which the Dockerfile builds from the hash-checked sdist) and strips `docopt` and `embreex`. The wheels' existence is checked by the resolver; it says nothing about GB10 behavior.

## Build and publish

1. Build the curated base cache on a networked machine (separate from the image, versioned independently):
   `python runtime/tools/build_base_cache.py entries.json /srv/rc/hf-base --version bc-2026.11.21-1`
   Entries may set `allow_patterns` / `ignore_patterns`; by default onnx, gguf, tflite and duplicate-format weights are skipped when safetensors exist (see the tool's docstring). It refuses gated, private and non-redistributable entries, builds the hub layout with `refs/main`, makes it read-only and runs `huggingface_hub` offline resolution against it. Distribute the directory to every Spark (`RC_BASE_CACHE_DIR`).
2. Build each architecture on a host of that architecture (arm64 on a Spark) and push by digest:
   `docker buildx build --platform linux/arm64 -f runtime/Dockerfile --build-arg BASE_IMAGE=nvidia/cuda:13.0.3-base-ubuntu24.04@sha256:<pin> --build-arg RELEASE_ID=... --build-arg IMAGE_TAG=... --push .`
   then the same for `linux/amd64`, then `docker buildx imagetools create` for the manifest list. Pin the base by digest.
3. Per-architecture inventory, from each image (tools are kept in `/opt/rc/tools`):
   `docker run --rm --entrypoint python <img>@sha256:<platform digest> -I /opt/rc/tools/rc_release.py packages > packages.arm64.json` (same for amd64), and `... rc_release.py reserved > reserved.json`.
4. Write the published release (digests are only known after push):
   `python runtime/tools/rc_release.py write -o runtime/release.json --release-id ... --repository ... --tag ... --manifest-digest sha256:... --platform-digest linux/arm64=sha256:... --platform-digest linux/amd64=sha256:... --config-digest linux/arm64=sha256:... --config-digest linux/amd64=sha256:... --packages linux/arm64=packages.arm64.json --packages linux/amd64=packages.amd64.json --reserved reserved.json --base-cache /srv/rc/hf-base/CACHE_MANIFEST.json`
   Commit `runtime/release.json`.
   `/opt/rc/release.json` inside the image is the `--partial` build-time copy (no digests); the loader reads only `reserved_modules` and `stdlib_module_names` from it.
5. Update the numeric format and package status cells in `numeric_formats.json` / `--package-status` only from real test results.

## Verified here (x86-64 CPU, no Docker daemon, no GPU)

Loader/server unit and contract tests, release schema validation, cache layout and `huggingface_hub` 2.4.0 offline resolution from a read-only directory without writes, `uv` resolution of the full stack for both architectures with identical pins.

## Verification checklist: Spark (arm64, GB10) and amd64 GPU host

This is the original checklist, not a list of exclusively outstanding items.
The [Spark image report](../docs/verification/spark-image-arm64.md),
[end-to-end report](../docs/verification/spark-e2e-pinned.md), and
[amd64 emulation report](../docs/verification/image-amd64.md) record what was
actually exercised and what remains untested. Results for a local image do not
establish that a published image or the current working tree has been verified.

- The image builds from the lock on both architectures, reproducibly; digests recorded.
- `torch` 2.14.1 runs CUDA ops on sm_121 with the pip-provided CUDA 13.0 libraries; `torch.compile` and Triton (`ptxas` for sm_121a; research flags `TRITON_PTXAS_PATH`).
- JAX 0.12.1 `jax[cuda13]`: device visible, a jitted computation runs, `XLA_PYTHON_CLIENT_PREALLOCATE=false` is enough on unified memory, torch and JAX in one process. The lock resolves `nvidia-cuda-nvcc 13.5.37` (JAX ptxas); check it against driver 580.
- `torchcodec` 0.17.0 against torch 2.14.1 and the Ubuntu ffmpeg libs; `lerobot[smolvla,pi,diffusion]` import and one policy forward.
- Hardened run: `--read-only --user 10001 --cap-drop ALL --security-opt no-new-privileges` with `--gpus all`/CDI; `/scratch` tmpfs `exec`; `/tmp` an ordinary 1777 directory in the image (a symlink broke Modal) with a runner tmpfs on top, `TMPDIR=/scratch/tmp`; scratch sizing and compile caches. `tests/docker/smoke.sh` covers the non-GPU part on any Docker host (not run here).
- `from_pretrained("org/name")` by hub id offline against the read-only `/hf-base` mount with the exact image versions (the unit check only used `snapshot_download` / `hf_hub_download`), including SmolVLA's SmolVLM backbone.
- Baked torchvision checkpoints load from `TORCH_HOME=/opt/rc/torch` read-only; `pin`, `casadi`, `shapely` and the other geometry wheels import on arm64.
- Fill the numeric format table per GPU class from real operation tests.

## Verified on a Spark (2026-11-21)

See `docs/verification/spark-image-arm64.md`: the arm64 image builds from the lock on a GB10 Spark (after dropping the sdist-only `docopt` from the lock and adding `gcc`/`python3.12-dev` for Triton) and passes the hardened smoke matrix.

## Deviations from the research / contracts

- `huggingface_hub` is `>=1.32,<2` (locked 1.33.0), not 2.4.0: LeRobot 0.7.0 caps `diffusers<0.40`, and diffusers 0.38/0.39 require `huggingface-hub<2`. The base cache tool tests pass with both 1.33.0 (locked) and 2.4.0.
- `rc_runtime --selfcheck` exit codes: `71` (new) for import/resolve failures; 70/72 as in the contract.
