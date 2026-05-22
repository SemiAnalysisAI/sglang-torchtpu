# TPU container image for sglang-torchtpu CI + dev.
#
# Base image is the same one torch_tpu uses for its presubmit (see
# torch_tpu/.github/workflows/presubmit.yml). Staying on that base
# means torch_tpu wheels we build are guaranteed to be ABI-compatible
# with this runtime.
#
# Auth model for the private torch_tpu wheel server: the workflow that
# invokes `docker build` is expected to inject a short-lived OAuth
# bearer token via `--secret id=torch_tpu_token,...`. We mount it as
# a build-time secret rather than baking it into a layer; this keeps
# the token out of the published image (and out of `docker history`).
ARG BASE_IMAGE="us-docker.pkg.dev/ml-oss-artifacts-published/ml-public-container/ml-build:latest"
FROM ${BASE_IMAGE}

ARG PYTORCH_VERSION="2.10.0"
# Private wheel-server simple-index base URL, e.g.
#   https://<region>-python.pkg.dev/<project>/<repo>/simple/
# Workflow passes via --build-arg TORCH_TPU_INDEX_URL=...
ARG TORCH_TPU_INDEX_URL=""
ARG PIP_INSTALL="python3 -m pip install --no-cache-dir"

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    SGLANG_PLATFORM=tpu \
    HF_HOME=/cache/hf \
    XLA_PYTHON_CLIENT_PREALLOCATE=false

WORKDIR /sgl-workspace

# Base tools. The ml-build base already ships Python + git; we only
# need the extras sglang pulls in.
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        clang \
        ccache \
        cmake \
        curl \
        libgl1 \
        libglib2.0-0 \
        libgl1-mesa-glx \
        libgl1-mesa-dri \
        lld \
        locales \
        net-tools \
        pkg-config \
        unzip \
        wget \
        zlib1g-dev \
    && update-ca-certificates \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/*

RUN ${PIP_INSTALL} --upgrade pip uv \
    && ${PIP_INSTALL} wheel==0.45.1 pybind11 pyyaml decorator scipy attrs psutil

# Torch CPU build — torch_tpu provides the TPU dispatch on top.
RUN ${PIP_INSTALL} torch==${PYTORCH_VERSION} \
    --index-url https://download.pytorch.org/whl/cpu

# torch_tpu / libtpu / jax[tpu] from the private wheel server.
# RUN --mount=type=secret,... so the token never lands in any layer.
RUN --mount=type=secret,id=torch_tpu_token \
    if [ -z "${TORCH_TPU_INDEX_URL}" ]; then \
        echo "TORCH_TPU_INDEX_URL build-arg is required" >&2; exit 1; \
    fi \
    && TOKEN="$(cat /run/secrets/torch_tpu_token)" \
    && INDEX="https://oauth2accesstoken:${TOKEN}@${TORCH_TPU_INDEX_URL#https://}" \
    && ${PIP_INSTALL} --index-url "${INDEX}" \
        --extra-index-url https://pypi.org/simple/ \
        torch_tpu libtpu "jax[tpu]"

# Install sglang from the TPU pyproject. Persistent HF cache mount point
# is created in the same layer so the workflow can volume-mount it.
COPY . /sgl-workspace/sglang
WORKDIR /sgl-workspace/sglang
RUN rm -f python/pyproject.toml \
    && cp python/pyproject_tpu.toml python/pyproject.toml \
    && ${PIP_INSTALL} -v -e "python[dev_tpu]" \
    && mkdir -p /cache/hf

CMD ["/bin/bash"]
