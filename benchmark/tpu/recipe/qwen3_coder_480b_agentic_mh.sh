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
# host RAM beside the pinned HiCache pool, and the pod is OOM-killed during
# warmup with HBM still ~39 GB free. config.sh:347 already warns that extra
# torch.compile buckets are "the most expensive thing in a CI bring-up"; 12
# graph variants now OOM-kill the pod and 9 do not, so the ladder has to lose a
# few. Which few is the whole question.
#
# Thin the EXTEND ladder, and leave the decode ladder and MAX_RUNNING exactly
# as the last green run had them. Every negative pad seen while fixing this has
# coincided with a decode-side change, never with the extend side alone:
#
#   bs=[1,8]    + tokens=[4096] + MAX_RUNNING=8   -> pad -123  at c1  (34903370712)
#   bs=[1,8]    + derived       + MAX_RUNNING=8   -> pad -1017 at c1  (34908974337)
#   bs=[1,8,16] + derived       + MAX_RUNNING=16  -> pad -1352 at c16 (34914967096)
#
# The mechanism is DP attention: prepare_mlp_sync_batch sizes every rank from
# the all-gathered global_num_tokens, and a decode ladder missing rungs lets a
# rank's own padded batch exceed that target, so the pad goes negative
# (forward_batch_info.py:1302). The last run got both c1 legs green and matched
# the 09-11 baseline on hit rate and throughput, so only the c16 legs, where
# per-rank batch size varies most, still trip it.
#
# Cutting extends instead keeps 4096 as the maximum -- nothing can overflow the
# ladder that could not before -- and only costs padding on chunks under 1024.
# 5 decode + 3 extend = 8 variants, below the 9 that ran OOM-free.
if [[ "$PROFILE" == "ci" ]]; then
  export TPU_COMPILE_TOKENS="${TPU_COMPILE_TOKENS:-1024 2048 4096}"
fi

exec env \
  MODEL="${MODEL:-/weights/Qwen3-Coder-480B-A35B-Instruct-FP8}" \
  PROFILE="$PROFILE" \
  EXTRA_ARGS="$EXTRA_ARGS" \
  ROLE=colocated \
  bash recipe/qwen3_coder_480b/nightly/serve.sh
