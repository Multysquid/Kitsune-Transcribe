# Kitsune-Transcribe training image: dependencies only.
#
# No code, data, weights or secrets go in here. vast/onstart.sh clones the repo at a pinned commit (KITSUNE_SHA) and
# vast/bootstrap.sh pulls the data from HF at boot, so the image only changes when requirements-train.txt does.
# Built and pushed by .github/workflows/image.yml (GitHub Actions -> ghcr.io/multysquid/kitsune-train); the build
# context is the repo root.
#
# Base: vastai/pytorch mini, torch 2.14.0+cu130 in the conda env /venv/main, vast's portal/TensorBoard/Jupyter tooling,
# pinned by the digest of its multi-arch index (Docker Hub registry API, 2026-09-23) so a re-pushed tag cannot change
# the build. Hosts cache these base layers, so a cold pull only fetches our few hundred MB on top.
FROM vastai/pytorch:2.14.0-cu130-cuda-13.2-mini-py312-2026-09-08@sha256:450710d89268fcd55632f28fdd80508ee920a53075b3f8e9f828bdc154e3d856

LABEL org.opencontainers.image.source="https://github.com/Multysquid/Kitsune-Transcribe" \
      org.opencontainers.image.description="Kitsune-Transcribe training dependencies (torch from vastai/pytorch, pinned pip deps)" \
      org.opencontainers.image.licenses="MIT"

# The base image's PATH does not include /venv/main/bin (its shells activate the env from .bashrc); the non-interactive
# on-start script and `docker run` need the env's python first on PATH.
ENV PYTHONUNBUFFERED=1 TZ=UTC HF_XET_HIGH_PERFORMANCE=1 \
    VIRTUAL_ENV=/venv/main \
    PATH=/venv/main/bin:/opt/instance-tools/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# tmux: vast drops SSH logins into tmux when it exists; the run itself is detached with nohup and does not need it.
RUN if ! command -v tmux >/dev/null; then \
        apt-get update && apt-get install -y --no-install-recommends tmux && \
        apt-get clean && rm -rf /var/lib/apt/lists/*; \
    fi

COPY requirements-train.txt docker/smoke_import.py /opt/kitsune/

# Hold torch and its CUDA companions at the base image's exact builds: a transitive dependency must never swap the
# cu130 torch for a PyPI build. uv is what the base image itself installs with.
RUN set -eux; \
    uv pip freeze --python /venv/main/bin/python \
        | { grep -iE '^(torch|torchvision|torchaudio|triton|nvidia-[a-z0-9-]+|cuda-[a-z0-9-]+)==' || true; } \
        > /opt/kitsune/constraints-base.txt; \
    grep -q '^torch==' /opt/kitsune/constraints-base.txt; \
    uv pip install --python /venv/main/bin/python --no-cache \
        -c /opt/kitsune/constraints-base.txt -r /opt/kitsune/requirements-train.txt; \
    /venv/main/bin/python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)"; \
    if command -v env-hash >/dev/null; then env-hash > /.env_hash; fi

WORKDIR /workspace
