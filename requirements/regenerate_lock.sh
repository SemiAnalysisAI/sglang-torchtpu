#!/usr/bin/env bash
# Regenerate requirements_lock_3_12.txt from python/pyproject_tpu.toml.
#
# Run it through Bazel so the uv binary is the hermetic one pinned in
# MODULE.bazel (rules_uv), identical on every machine:
#
#   bazel run //requirements:regenerate_lock
#
# The lock is the Python environment the Bazel build resolves
# (MODULE.bazel's pip.parse). Its source of truth is pyproject_tpu.toml --
# the same file the TPU container images install from -- plus the `test`
# extra, so test-only deps (lm-eval, pytest, ...) are in the Bazel closure
# too. The TPU stack itself (torch-tpu/libtpu/jax/jaxlib/tpu-raiden-torch) is
# excluded: Bazel builds torch_tpu from the pinned source module and takes
# libtpu/jax from torch_tpu's own pip hub, so resolving them here would only
# fight the private-registry credentials they require.
#
# pyproject_tpu.toml is copied to a scratch dir because `uv pip compile`
# insists on the exact filename `pyproject.toml`.
set -euo pipefail

if [[ -n "${BUILD_WORKSPACE_DIRECTORY:-}" ]]; then
  # `bazel run`: inputs come from runfiles, the lock goes to the workspace.
  REPO_ROOT="$BUILD_WORKSPACE_DIRECTORY"
  PYPROJECT="python/pyproject_tpu.toml"
  EXCLUDES="requirements/tpu_stack_excludes.txt"
  UV="$(command -v uv)"  # @multitool//tools/uv data dep lands its dir on PATH
else
  # Plain `bash regenerate_lock.sh` from anywhere: use whatever uv is
  # installed. Prefer the Bazel path above for reproducibility.
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  PYPROJECT="$REPO_ROOT/python/pyproject_tpu.toml"
  EXCLUDES="$REPO_ROOT/requirements/tpu_stack_excludes.txt"
  UV="uv"
fi

SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT
cp "$PYPROJECT" "$SCRATCH/pyproject.toml"

"$UV" pip compile "$SCRATCH/pyproject.toml" \
  --extra test \
  --excludes "$EXCLUDES" \
  --python-version 3.12 \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match \
  --output-file "$REPO_ROOT/requirements/requirements_lock_3_12.txt"
