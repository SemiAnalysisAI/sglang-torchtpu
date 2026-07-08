# Dockerfile for sglang on Google Cloud TPUs (torch_tpu backend).
#
# Structure mirrors torchtpu-vllm/docker/Dockerfile: a `base` with the pinned TPU
# dependency stack, a `ci` variant that adds CI/build tooling, a `dev` variant
# with source (sglang + torch_tpu) and test extras, and a lean `prod` variant.
#
# The sglang and torch_tpu sources are CLONED FRESH AT `main` during the build
# (over HTTPS, authenticated with the `github_token` BuildKit secret), so the
# image bakes in canonical main -- not whatever local branch the build was
# invoked from. Override the refs with --build-arg SGLANG_REF=... /
# TORCH_TPU_REF=... if needed.
# NOTE: for PR CI (build+test the commit under test) switch the sglang app source
# to `COPY .` so the image is the checked-out ref rather than main.
#
# torch_tpu the *package* is pinned in python/pyproject_tpu.toml and installed
# from the private torch-tpu virtual registry (BuildKit secret `gcloud_token`);
# torch/jax/jaxlib come warm from torch-tpu-base. The torch_tpu *source* is a
# checkout under /workspace/torch_tpu for reference/patching (Option A) -- the
# installed package stays the pinned wheel.

# torch-tpu-base ships torch/jax/jaxlib/libtpu warm (torch_tpu is not installed;
# we add the pinned wheel on top). Debian bookworm / Python 3.12.
ARG BASE_IMAGE="us-docker.pkg.dev/ml-oss-artifacts-transient/torch-tpu-docker-container/torch-tpu-base:nightly-latest"

# Sources cloned at build time (main). Override refs if needed.
ARG SGLANG_REPO="https://github.com/sgl-project/sglang-torchtpu.git"
ARG SGLANG_REF="main"
ARG TORCH_TPU_REPO="https://github.com/google-pytorch/torch_tpu.git"
ARG TORCH_TPU_REF="main"

########################################################################
# Variant 1: base -- pinned TPU dependency stack (no sglang, no source)
########################################################################
FROM ${BASE_IMAGE} AS base

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    SGLANG_PLATFORM=tpu \
    HF_HOME=/cache/hf \
    XLA_PYTHON_CLIENT_PREALLOCATE=false

WORKDIR /workspace

# apt's http method drops to the unprivileged _apt sandbox user, which cannot
# reach the network in some restricted container-build environments (e.g. behind
# an authenticating egress proxy). Disable the sandbox so apt uses the build's
# network. Harmless for normal builds; inherited by the ci stage.
RUN echo 'APT::Sandbox::User "root";' > /etc/apt/apt.conf.d/99-disable-sandbox

# System deps needed to resolve/build the dependency wheels + clone sources.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      clang \
      ccache \
      cmake \
      curl \
      git \
      libgl1 \
      libglib2.0-0 \
      lld \
      locales \
      pkg-config \
      unzip \
      wget \
      zlib1g-dev \
    && update-ca-certificates \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip uv

# Rust toolchain (mirrors scripts/ci/tpu/tpu_ci_install_dependency.sh, which
# installs rustup before pip installs; some dep/sdist builds via setuptools-rust).
ENV PATH="/root/.cargo/bin:${PATH}"
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --no-modify-path --profile minimal \
    && rustc --version && cargo --version

# Install the pinned deps from pyproject_tpu.toml (only the dep spec is needed
# here, not the sglang source). torch/jax/jaxlib are already warm from
# torch-tpu-base; uv installs the pinned torch_tpu wheel + remaining deps. The
# gcloud_token secret authenticates the torch-tpu registry (embedded in the
# index URL); other packages fall back to PyPI. Token is a BuildKit secret so it
# never lands in a layer.
COPY python/pyproject_tpu.toml /tmp/deps/pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=secret,id=gcloud_token \
    set -eu; \
    TOKEN=""; \
    if [ -f /run/secrets/gcloud_token ]; then TOKEN="$(cat /run/secrets/gcloud_token)"; fi; \
    if [ -z "${TOKEN}" ]; then echo "Missing BuildKit secret gcloud_token for registry mode" >&2; exit 1; fi; \
    uv pip install --system \
      --index "https://oauth2accesstoken:${TOKEN}@us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/simple/" \
      --extra-index-url https://pypi.org/simple \
      --index-strategy unsafe-best-match \
      -r /tmp/deps/pyproject.toml


########################################################################
# Variant 2: ci -- base + CI tooling (no sglang install)
########################################################################
FROM base AS ci

RUN apt-get update && apt-get install -y --no-install-recommends \
      apt-transport-https gnupg lsb-release lsof psmisc \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt gcsfuse-$(lsb_release -c -s) main" | tee -a /etc/apt/sources.list.d/gcsfuse.list \
    && apt-get update && apt-get install -y --no-install-recommends google-cloud-cli gcsfuse \
    && rm -rf /var/lib/apt/lists/*

RUN git config --global --add safe.directory '*'


########################################################################
# Variant 3: dev -- ci + source (sglang + torch_tpu @ main) + test extras
########################################################################
FROM ci AS dev
ARG SGLANG_REPO SGLANG_REF TORCH_TPU_REPO TORCH_TPU_REF

# Developer quality-of-life tooling.
RUN apt-get update && apt-get install -y --no-install-recommends \
      vim tmux htop less tree silversearcher-ag zsh git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Clone sources at main. torch_tpu SOURCE checkout (Option A) is for
# reference/patching/building later -- the *installed* torch_tpu stays the pinned
# wheel from base. The github_token secret authenticates the clones; we scrub it
# from the .git remote afterwards so it never lands in a layer.
RUN --mount=type=secret,id=github_token \
    set -eu; \
    TOK="$(cat /run/secrets/github_token)"; \
    git clone --depth 1 -b "${TORCH_TPU_REF}" "https://x-access-token:${TOK}@${TORCH_TPU_REPO#https://}" /workspace/torch_tpu; \
    git -C /workspace/torch_tpu remote set-url origin "${TORCH_TPU_REPO}"; \
    git clone --depth 1 -b "${SGLANG_REF}" "https://x-access-token:${TOK}@${SGLANG_REPO#https://}" /workspace/sglang; \
    git -C /workspace/sglang remote set-url origin "${SGLANG_REPO}"

# Overlay our pinned TPU pyproject so the editable install keeps torch_tpu at the
# version already installed in base (avoids a re-resolve/downgrade).
COPY python/pyproject_tpu.toml /workspace/sglang/python/pyproject.toml
WORKDIR /workspace/sglang
RUN --mount=type=cache,target=/root/.cache/uv \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 \
    uv pip install --system -e "python[dev_tpu]"

CMD ["/bin/bash"]


########################################################################
# Variant 4: prod -- base + source @ main, regular install (lean, no dev tools)
########################################################################
FROM base AS prod
ARG SGLANG_REPO SGLANG_REF TORCH_TPU_REPO TORCH_TPU_REF

RUN git config --global --add safe.directory '*'

# Clone sources at main (torch_tpu source = Option A). Token scrubbed after.
RUN --mount=type=secret,id=github_token \
    set -eu; \
    TOK="$(cat /run/secrets/github_token)"; \
    git clone --depth 1 -b "${TORCH_TPU_REF}" "https://x-access-token:${TOK}@${TORCH_TPU_REPO#https://}" /workspace/torch_tpu; \
    git -C /workspace/torch_tpu remote set-url origin "${TORCH_TPU_REPO}"; \
    git clone --depth 1 -b "${SGLANG_REF}" "https://x-access-token:${TOK}@${SGLANG_REPO#https://}" /workspace/sglang; \
    git -C /workspace/sglang remote set-url origin "${SGLANG_REPO}"
COPY python/pyproject_tpu.toml /workspace/sglang/python/pyproject.toml
WORKDIR /workspace/sglang
# "./python[srt_tpu]" (leading ./) forces uv to treat this as the local project
# path rather than a PyPI package named "python".
RUN --mount=type=cache,target=/root/.cache/uv \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 \
    uv pip install --system "./python[srt_tpu]"

CMD ["/bin/bash"]
