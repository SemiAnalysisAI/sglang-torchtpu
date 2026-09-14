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

# Compile only the buckets the ci sweep can reach. config.sh derives MAX_RUNNING
# and TORCH_COMPILE_MAX_BS from max(SWEEP_BS)=16, but that is the *client's
# global* concurrency while both knobs are *per DP rank* -- at dp=2 no rank ever
# sees more than ~8 in flight, so half the decode ladder is compiled for batches
# that cannot occur. config.sh:347 already warns that sizing above the sweep
# "only buys extra torch.compile buckets, which is the most expensive thing in a
# CI bring-up"; since libtpu 0.0.47 (#727) that surplus no longer fits in host
# RAM beside the pinned HiCache pool, and the pod is OOM-killed during warmup.
# Takes the ladder from 5 decode + 7 extend variants to 2 + 1, matching the
# EAGLE3 sibling, which runs the same sweep on the same topology and is green.
if [[ "$PROFILE" == "ci" ]]; then
  export MAX_RUNNING="${MAX_RUNNING:-8}"
  export TORCH_COMPILE_MAX_BS="${TORCH_COMPILE_MAX_BS:-8}"
  export TPU_COMPILE_TOKENS="${TPU_COMPILE_TOKENS:-4096}"
  export TPU_COMPILE_BS="${TPU_COMPILE_BS:-1 8}"
fi

exec env \
  MODEL="${MODEL:-/weights/Qwen3-Coder-480B-A35B-Instruct-FP8}" \
  PROFILE="$PROFILE" \
  EXTRA_ARGS="$EXTRA_ARGS" \
  ROLE=colocated \
  bash recipe/qwen3_coder_480b/nightly/serve.sh
