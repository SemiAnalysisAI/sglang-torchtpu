#!/usr/bin/env bash
# Shared environment for the MiMo-V2.5 DP-attention TPU experiments.
# Source this from the launch/bench scripts. Override any var by exporting it
# before sourcing (e.g. `MODEL=/data/MiMo-V2.5 source common_env.sh`).
#
# The three hosts must all run the SAME:
#   - sglang-torchtpu branch/commit  (validated on main @ 11059a1d3, which
#     contains the DP-attention fix 312cb2bf5 "[TPU/DPA] Do not present
#     zero-length sync-padding rows to the RPA kernel as active")
#   - torch_tpu pinned wheel 0.1.1.dev20260707090224  (topk sorted=False)
#   - torch 2.11.0+cpu, libtpu 0.0.40
# Verify on each host:  python -c "import torch_tpu, torch; print(torch_tpu.__version__, torch.__version__)"
#                       git -C "$SGLANG_TPU_ROOT" merge-base --is-ancestor 312cb2bf5 HEAD && echo "fix present"

# --- paths (override per host if different) ---
export SGLANG_TPU_ROOT=${SGLANG_TPU_ROOT:-/cache/work/sglang-torchtpu}
export LAUNCHER_DIR=${LAUNCHER_DIR:-$SGLANG_TPU_ROOT/benchmark/tpu}
export PYTHON=${PYTHON:-$SGLANG_TPU_ROOT/.venv/bin/python}
export MODEL=${MODEL:-/cache/models/MiMo-V2.5}      # local path — NOT /gcs (GCS-FUSE mmap SIGBUS)

# --- HF: fully offline, cache on local disk (never /gcs) ---
export HF_HOME=${HF_HOME:-/cache/hf}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

# --- sglang / TPU knobs ---
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=${SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION:-0}
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=${SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN:-1}
# 0 = fresh compile every launch (safe). Set 1 once the branch/flags are stable
# to reuse cached TPU binaries (per-shape compile ~30s -> ~1s).
export ENABLE_COMPILE_CACHE=${ENABLE_COMPILE_CACHE:-0}

# --- XLA / libtpu: TP/EP serving tuning + SparseCore-offload disables ---
# The 4 SC-offload=false flags are the DP-attention hardening set. Overriding
# this string invalidates the compile cache.
export LIBTPU_INIT_ARGS=${LIBTPU_INIT_ARGS:-"--xla_tpu_scoped_vmem_limit_kib=65536 \
--xla_tpu_enable_async_collective_fusion=true \
--xla_tpu_overlap_compute_collective_tc=true \
--xla_tpu_enable_async_collective_fusion_multiple_steps=true \
--xla_tpu_enable_latency_hiding_scheduler=true \
--xla_latency_hiding_scheduler_rerun=10 \
--xla_tpu_all_gather_collective_matmul_mode=post_spmd_conservative \
--xla_tpu_reduce_scatter_collective_matmul_mode=post_spmd_conservative \
--xla_tpu_enable_sparse_core_collective_offload_all_reduce=false \
--xla_tpu_offload_gather_to_sparsecore=false \
--xla_tpu_enable_sparse_core_collective_offload_all_gather=false \
--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false"}
