#!/usr/bin/env bash
# Multi-host CI recipe: Qwen3-Coder-480B-A35B-Instruct-FP8 colocated agentic
# benchmark server -- tp=16, ep=16, dp=2 -> attention_tp = 16/2 = 8, across 2 hosts.
# (dp is config.sh's COLOCATED_DP, which #574 moved 4 -> 2.)
#
# Declared by test/registered/tpu/e2e/test_tpu_qwen3_coder_480b_multihost_agentic.py via
# MULTIHOST_LAUNCH; the TPU CI JobSet runs this on EVERY host (one global TP group)
# and rank-0 then runs that test file against the resulting server. GKE injects
# TPU_WORKER_ID / TPU_WORKER_HOSTNAMES, so launch_server.sh auto-derives NNODES,
# node rank and the rendezvous address -- nothing to plumb here.
#
# This is a thin shim on purpose: every server knob lives in the benchmark's
# config.sh, so the CI server and a hand-run server are the same server. It only
# supplies what the CI environment owns:
#   * MODEL defaults to the ReadOnlyMany Hyperdisk mount (MULTIHOST_LAUNCH's
#     weights_pvc), not the HF repo id -- a 480 GB pull per run is not viable.
#   * PROFILE=ci sizes the sweep to the JobSet's deadline (see config.sh).
# HOST / PORT / ENABLE_PROFILER_ENDPOINT / ENABLE_COMPILE_CACHE come from the
# JobSet template.
#
# WEIGHTS: /weights/Qwen3-Coder-480B-A35B-Instruct-FP8 is already on the shared
# model-weights-rox PVC (verified on the live disk: 449GiB, 49 safetensors
# shards, config + tokenizer + index), alongside the EAGLE3 draft the combo test
# uses. No hydration pass is needed. The disk's contents list lives in
# .github/tpu-ci/pvc-model-weights-rox.yaml.tmpl.
set -uo pipefail
cd "$(dirname "$0")/.."   # recipe/ -> benchmark/tpu

# The 449 GiB target checkpoint otherwise stays in the host page cache after
# loading. Together with the pinned HiCache host pool -- one per DP group -- and
# compiler state, that exhausts host memory while HBM still has headroom, and
# the kernel OOM-kills the pod mid-startup (the container requests 32Gi and sets
# no memory limit, so this lands as node-level pressure, not a cgroup cap).
# Same flag and same reason as the EAGLE3 sibling recipe and deepseek_r1_0528_mh.
EXTRA_ARGS="--weight-loader-drop-cache-after-load${EXTRA_ARGS:+ $EXTRA_ARGS}"

PROFILE=${PROFILE:-ci}

# Since libtpu 0.0.47 (#727) the resident compiled executables no longer fit in
# host RAM beside raiden's host KV pool, and the pod is OOM-killed during warmup
# with HBM still ~39 GB free. The node has 963 GB allocatable (config.sh:220).
#
# The compile ladder is NOT the lever. Every reduction of it pads negative under
# DP attention -- prepare_mlp_sync_batch sizes each rank from the all-gathered
# global_num_tokens and a non-default ladder lets a rank's own padded batch
# exceed that target, so _pad_tensor_to_size is asked to shrink
# (forward_batch_info.py:1302). Four shapes, four failures:
#
#   decode [1,8]    extend [4096]            -> -123  at c1  (34903370712)
#   decode [1,8]    extend derived           -> -1017 at c1  (34908974337)
#   decode [1,8,16] extend derived           -> -1352 at c16 (34914967096)
#   decode derived  extend [1024,2048,4096]  -> -253  at c16 (34919948429)
#
# So both ladders stay derived, exactly as the last green run had them, and the
# host RAM comes from raiden instead. Its host pool is pinned eagerly at
# HICACHE_RATIO x the ~26 GB device pool x 8 ranks -- 208 GB at ratio 1 -- and
# this benchmark never touches it: kv_offload_written and kv_offload_reloaded
# are 0.0 on all four legs of every run observed, the 09-11 green one included
# (the known gap in the recipe README). Halving it frees ~104 GB of pinned host
# memory and cannot change a measurement that is identically zero.
if [[ "$PROFILE" == "ci" ]]; then
  export HICACHE_RATIO="${HICACHE_RATIO:-0.5}"
fi

exec env \
  MODEL="${MODEL:-/weights/Qwen3-Coder-480B-A35B-Instruct-FP8}" \
  PROFILE="$PROFILE" \
  EXTRA_ARGS="$EXTRA_ARGS" \
  ROLE=colocated \
  bash recipe/qwen3_coder_480b/nightly/serve.sh
