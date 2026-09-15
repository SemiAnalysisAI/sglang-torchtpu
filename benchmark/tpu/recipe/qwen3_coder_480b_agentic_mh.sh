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

# HOST-RAM REGRESSION, UNFIXED HERE. Since the libtpu 0.0.44.1 -> 0.0.47 pin
# bump (#727, merged 09-11 18:32 UTC) this server is OOM-killed during
# _execute_server_warmup, with HBM still ~39 GB free. The 09-11 green run (job
# 103217467337) was configuration-identical, so nothing in this recipe caused
# it. The node has 963 GB allocatable; the consumers are raiden's eagerly
# pinned host pool (~208 GB) and the resident compiled executables.
#
# Every lever below this recipe owns has been tried and is closed. Do not
# retry them:
#
#   * Compile ladder. Any reduction pads negative under DP attention --
#     prepare_mlp_sync_batch sizes each rank from the all-gathered
#     global_num_tokens and a non-default ladder lets a rank's own padded batch
#     exceed it, so _pad_tensor_to_size is asked to shrink
#     (forward_batch_info.py:1302). Four shapes, four failures:
#       decode [1,8]    extend [4096]           -> -123  at c1  (34903370712)
#       decode [1,8]    extend derived          -> -1017 at c1  (34908974337)
#       decode [1,8,16] extend derived          -> -1352 at c16 (34914967096)
#       decode derived  extend [1024,2048,4096] -> -253  at c16 (34919948429)
#     Worth its own bug: a pad must never be negative.
#   * HICACHE_RATIO below 1. Structurally illegal --
#     kv_pool_host.py:97 asserts the host pool strictly exceeds the device
#     pool, so 0.5 dies at bring-up with
#     "Host KV pool must be larger than the device pool (host=438784 <=
#     device=877440)" (34924965546). Ratio 1 only clears it on page rounding.
#   * Shrinking max_total_num_tokens would shrink the host pool with it, but it
#     is what sizes the sweep's reachable prefix cache, so it changes the
#     number this benchmark exists to report.
#
# What remains is the libtpu pin itself. The page-cache flag above is kept
# because it is correct hygiene and matches both sibling recipes, but on its
# own it is not enough (34895558228 still OOM-killed with it applied).

exec env \
  MODEL="${MODEL:-/weights/Qwen3-Coder-480B-A35B-Instruct-FP8}" \
  PROFILE="$PROFILE" \
  EXTRA_ARGS="$EXTRA_ARGS" \
  ROLE=colocated \
  bash recipe/qwen3_coder_480b/nightly/serve.sh
