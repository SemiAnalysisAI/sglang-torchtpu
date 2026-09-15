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
# Drops the intermediate decode rungs only. The ladder's MAXIMUM stays at
# max(SWEEP_BS)=16 so it still covers max_running_requests, and MAX_RUNNING is
# left derived -- both of those matter:
#
#   * Lowering MAX_RUNNING to 8 is what config.sh:337 warns about (a leg gets
#     admission-capped below the concurrency it reports). It also breaks DP
#     attention outright: prepare_mlp_sync_batch pads every rank up to the
#     global token count, and with the cap in place a rank's own batch can
#     exceed it, so the pad goes negative --
#       forward_batch_info.py:1302 _pad_tensor_to_size
#       RuntimeError: empty(): expected dimension sizes to be >= 0, got [-1017]
#     seen in run 34908974337; the 09-11 green run, identical but for
#     max_running_requests=16, has none.
#   * Pinning the extend ladder to a single 4096 bucket fails the same way
#     (run 34903370712, got [-123]), so it stays derived.
#
# bs 2..7 and 9..15 now round up, which the ci sweep barely visits: SWEEP_BS is
# "1 16", so per rank at dp=2 the live sizes are ~1 and ~8, both exact rungs.
# 12 graph variants -> 10. Run 34908974337 showed 9 clears the OOM and 12 does
# not, so this should sit inside the margin.
if [[ "$PROFILE" == "ci" ]]; then
  export TPU_COMPILE_BS="${TPU_COMPILE_BS:-1 8 16}"
fi

exec env \
  MODEL="${MODEL:-/weights/Qwen3-Coder-480B-A35B-Instruct-FP8}" \
  PROFILE="$PROFILE" \
  EXTRA_ARGS="$EXTRA_ARGS" \
  ROLE=colocated \
  bash recipe/qwen3_coder_480b/nightly/serve.sh
