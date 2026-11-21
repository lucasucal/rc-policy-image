# syntax=docker/dockerfile:1.7
#
# Pinned policy image (docs/policy-runtime-pin.md sections 2, 4.4, 5.5, 7).
# Build context is the REPO ROOT (it needs starter/rc_sdk and runtime/):
#
#   docker buildx build --platform linux/arm64,linux/amd64 -f runtime/Dockerfile \
#     --build-arg BASE_IMAGE=nvidia/cuda:13.0.3-base-ubuntu24.04@sha256:<digest> \
#     --build-arg RELEASE_ID=rc-policy-2026.11.21-1 -t ghcr.io/<org>/rc-policy:<tag> --push .
#
# One architecture at a time on the matching host (a Spark for arm64) is the supported
# way to verify; see runtime/README.md. Python packages come only from the hashed locks
# (locks/requirements.<arch>.txt, wheels only). TensorFlow GPU is intentionally omitted.

ARG BASE_IMAGE=nvidia/cuda:13.0.3-base-ubuntu24.04
FROM ${BASE_IMAGE}

ARG TARGETARCH
ARG RELEASE_ID=dev
ARG IMAGE_REPOSITORY=ghcr.io/example/rc-policy
ARG IMAGE_TAG=dev

ENV DEBIAN_FRONTEND=noninteractive
# Ubuntu 24.04 ships Python 3.12. libgomp (torch/numba), libglib (opencv), ffmpeg libs
# (torchcodec/PyAV video decode), libgl-free opencv-headless needs nothing else.
# gcc + libc6-dev + python3.12-dev (Python.h): Triton compiles a small C launcher stub at first
# kernel use, so torch.compile / Triton fail with "Failed to find C compiler" without them
# (found on a real Spark, docs/verification/spark-image-arm64.md).
# g++ + ninja-build: torch.compile on CPU (Inductor's C++ backend) and torch.utils.cpp_extension
# (ninja is an apt package, not in the wheels-only lock; ninja-build is ~0.3 MB). No nvcc: custom
# C/C++/CUDA extensions are not supported by the rules, GPU kernels go through Triton.
# libegl1 libgl1 libosmesa6 libglvnd0: headless MuJoCo / dm_control rendering (MUJOCO_GL=egl or
# osmesa); all from the signed Ubuntu archive.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates python3.12 python3.12-venv \
        libgomp1 libglib2.0-0 ffmpeg \
        gcc g++ libc6-dev python3.12-dev ninja-build \
        libegl1 libgl1 libosmesa6 libglvnd0 \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m venv /opt/venv

# "graphics" makes the NVIDIA container toolkit inject the EGL vendor ICD (libEGL_nvidia) with
# --gpus all, which MuJoCo's EGL backend needs; the base image defaults to compute,utility.
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

ENV PATH=/opt/venv/bin:$PATH \
    VIRTUAL_ENV=/opt/venv \
    PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# 1. The stack: exact hashed pins, binary wheels only, no dependency resolution at build.
COPY runtime/locks/requirements.${TARGETARCH}.txt /tmp/requirements.txt
RUN pip install --only-binary=:all: --no-deps --require-hashes -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# 1b. The one sdist-only pin (antlr4-python3-runtime 4.9.3, required by omegaconf/hydra; pure
#     python, no compiler involved): built from the hash-checked sdist. Everything else above is
#     wheels-only. Build isolation fetches setuptools/wheel from PyPI (they do not stay installed).
COPY runtime/locks/requirements.sdist.txt /tmp/requirements.sdist.txt
RUN pip install --no-deps --require-hashes --no-binary=antlr4-python3-runtime -r /tmp/requirements.sdist.txt \
    && rm /tmp/requirements.sdist.txt

# 2. SDK and loader into site-packages (not editable, not importable from /submission).
COPY starter/rc_sdk /tmp/build/rc_sdk
COPY runtime/pyproject.toml /tmp/build/runtime/pyproject.toml
COPY runtime/rc_runtime /tmp/build/runtime/rc_runtime
RUN pip install --no-deps /tmp/build/rc_sdk /tmp/build/runtime && rm -rf /tmp/build

# 3. Baked backbone checkpoints (read-only at run time): TORCH_HOME=/opt/rc/torch.
COPY runtime/tools/bake_weights.py runtime/tools/rc_release.py runtime/numeric_formats.json /opt/rc/tools/
RUN mkdir -p /opt/rc/torch && TORCH_HOME=/opt/rc/torch python /opt/rc/tools/bake_weights.py \
    && chmod -R a+rX /opt/rc

# 4. Build-time release.json (reserved modules + inventory for THIS arch). It cannot contain
#    its own image digest, so it is written --partial; the published copy adds the digests.
RUN python /opt/rc/tools/rc_release.py packages -o /tmp/packages.json \
    && python /opt/rc/tools/rc_release.py write --partial -o /opt/rc/release.json \
        --release-id "${RELEASE_ID}" --repository "${IMAGE_REPOSITORY}" --tag "${IMAGE_TAG}" \
        --packages "linux/${TARGETARCH}=/tmp/packages.json" \
        --numeric-formats /opt/rc/tools/numeric_formats.json \
    && chmod -R a+rX /opt/rc && rm -f /tmp/packages.json

# 5. Non-root user and empty mount points / scratch. The root filesystem is read-only at
#    run time. /tmp stays an ordinary 1777 directory (a symlink breaks platforms that mount their
#    own /tmp, e.g. Modal); the runner mounts a tmpfs there, TMPDIR points into scratch, which the
#    loader creates at startup.
RUN useradd --uid 10001 --user-group --no-create-home --shell /usr/sbin/nologin rc \
    && mkdir -p /submission /weights /hf-base /scratch \
    && chmod 1777 /scratch \
    && mkdir -p /tmp && chmod 1777 /tmp

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPYCACHEPREFIX=/scratch/pycache \
    PYTHONUNBUFFERED=1 \
    HOME=/scratch/home \
    HF_HOME=/scratch/hf \
    HF_HUB_CACHE=/hf-base \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TORCH_HOME=/opt/rc/torch \
    XDG_CACHE_HOME=/scratch/xdg \
    TRITON_CACHE_DIR=/scratch/triton \
    TORCHINDUCTOR_CACHE_DIR=/scratch/inductor \
    JAX_COMPILATION_CACHE_DIR=/scratch/jax \
    CUDA_CACHE_PATH=/scratch/cuda \
    NUMBA_CACHE_DIR=/scratch/numba \
    MPLCONFIGDIR=/scratch/mpl \
    TMPDIR=/scratch/tmp \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    KERAS_BACKEND=jax \
    NO_ALBUMENTATIONS_UPDATE=1 \
    MUJOCO_GL=egl \
    RC_SUBMISSION_DIR=/submission \
    RC_WEIGHTS_DIR=/weights \
    RC_SCRATCH=/scratch \
    RC_PORT=8080

LABEL org.opencontainers.image.title="rc-policy" \
      org.opencontainers.image.description="Reality-check pinned policy runtime" \
      rc.release_id="${RELEASE_ID}"

USER 10001:10001
WORKDIR /
EXPOSE 8080
# Fixed command; the runner supplies no CMD or arguments.
ENTRYPOINT ["python", "-I", "-m", "rc_runtime"]
