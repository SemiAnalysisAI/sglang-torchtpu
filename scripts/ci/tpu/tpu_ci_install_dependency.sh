#!/bin/bash
# Installs runtime dependencies for the TPU CI on a GKE-hosted runner.
#
# Auth model:
#   - Wheel server (private GAR / IAP-fronted PyPI) is read-only via
#     workload identity. Caller (either the CI workflow or this script
#     in dev mode) is expected to have a gcloud ADC that can mint an
#     access token via `gcloud auth print-access-token`.
#   - Token lifetime is 1h. The wheel is downloaded in the first minute,
#     so jobs longer than 1h are fine.
#
# Required env (set by the CI workflow or by the dev caller):
#   TORCH_TPU_INDEX_URL   - base URL of the private wheel server, e.g.
#                           https://<region>-python.pkg.dev/<project>/<repo>/simple/
#   PYTORCH_VERSION       - torch pin (default 2.10.0)
#   JAX_TPU_VERSION       - jax[tpu] pin (default left unpinned for first cut)
#   LIBTPU_VERSION        - libtpu pin (default left unpinned for first cut)
#
# Failure-mode notes:
#   - Wheel-server fetch is retried 3x with exponential backoff so one
#     network blip doesn't burn a 4-chip v7 runner-hour.
#   - The script ends with `verify_tpu_visible.py`, which asserts
#     `torch.tpu.is_available()`. Fail fast if the runner lost its TPU.

set -euo pipefail

PIP_INSTALL="python3 -m pip install --no-cache-dir"
UV_PIP_INSTALL="uv pip install"

PYTORCH_VERSION="${PYTORCH_VERSION:-2.10.0}"
TORCH_TPU_INDEX_URL="${TORCH_TPU_INDEX_URL:-}"

# Install base apt deps. Same set as scripts/ci/npu/npu_ci_install_dependency.sh
# minus aarch64-only bits.
apt update -y && apt install -y \
    unzip \
    build-essential \
    cmake \
    wget \
    curl \
    net-tools \
    zlib1g-dev \
    lld \
    clang \
    locales \
    ccache \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    ca-certificates \
    libgl1 \
    libglib2.0-0
update-ca-certificates
${PIP_INSTALL} --upgrade pip
${PIP_INSTALL} uv
export UV_NO_CACHE=true
export UV_SYSTEM_PYTHON=true
export UV_INDEX_STRATEGY=unsafe-best-match

# Rust toolchain — sglang's gRPC extension is built via setuptools-rust.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/../utils/install_rustup.sh"
export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:${PATH}"

# Pin wheel to 0.45.1, REF: https://github.com/pypa/wheel/issues/662
${UV_PIP_INSTALL} wheel==0.45.1 pybind11 pyyaml decorator scipy attrs psutil

### Install PyTorch (CPU build; torch_tpu provides the TPU dispatch).
${UV_PIP_INSTALL} torch=="${PYTORCH_VERSION}" \
    --index-url "${TORCH_CACHE_URL:-https://download.pytorch.org/whl/cpu}" \
    --extra-index-url "${PYPI_CACHE_URL:-https://pypi.org/simple/}"

### Install torch_tpu + libtpu + jax[tpu] from the private wheel server.
#
# Auth: short-lived OAuth bearer fetched via gcloud. Fails closed if the
# caller didn't provide credentials — do NOT fall back to anonymous fetch
# against a private repo because that yields a misleading 403.
if [ -z "${TORCH_TPU_INDEX_URL}" ]; then
    echo "TORCH_TPU_INDEX_URL is unset. Set it to the private wheel-server URL." >&2
    echo "  e.g. https://<region>-python.pkg.dev/<project>/<repo>/simple/" >&2
    exit 1
fi

if ! command -v gcloud >/dev/null 2>&1; then
    echo "gcloud is required to mint a short-lived wheel-server token." >&2
    echo "On GKE-hosted runners it should be present in the runner image." >&2
    exit 1
fi

# Retry 3x with exponential backoff (2s, 4s, 8s) to absorb transient
# wheel-server blips before burning the rest of the job.
fetch_wheels_with_retry() {
    local attempt
    for attempt in 1 2 3; do
        local token
        token="$(gcloud auth print-access-token)"
        local index="https://oauth2accesstoken:${token}@${TORCH_TPU_INDEX_URL#https://}"
        if ${UV_PIP_INSTALL} \
            --index-url "${index}" \
            --extra-index-url "${PYPI_CACHE_URL:-https://pypi.org/simple/}" \
            torch_tpu libtpu "jax[tpu]"; then
            return 0
        fi
        echo "wheel-server install attempt ${attempt} failed; backing off..." >&2
        sleep "$((2 ** attempt))"
    done
    return 1
}
fetch_wheels_with_retry

### Install SGLang in editable mode.
rm -f python/pyproject.toml
cp python/pyproject_tpu.toml python/pyproject.toml
${UV_PIP_INSTALL} -v -e "python[dev_tpu]"

### Verify TPU is actually visible (fail fast if the runner lost its chip).
python3 "${SCRIPT_DIR}/verify_tpu_visible.py"
