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

# Right-size the KV pool. Since the libtpu 0.0.44.1 -> 0.0.47 bump (#727) the
# resident compiled executables no longer fit in host RAM beside raiden's
# eagerly pinned host pool, and the pod is OOM-killed during warmup with HBM
# still ~39 GB free. The 09-11 green run (job 103217467337) was
# configuration-identical, so nothing here caused it; this buys the headroom
# back from the one consumer that is demonstrably oversized.
#
# config.sh pins 877440 tokens, a number inherited from an old dp=4 derivation
# rather than from this workload. The ci sweep is 16 sessions of 30720-token
# context; at dp=2 that is 8 per rank, so the live-plus-retained working set is
# ~246k tokens -- and the pool peaks at 0.27 in every run. 524288 keeps it at
# ~0.47, and raiden's host pool, sized from the device pool, drops 208 GB ->
# ~124 GB per host.
#
# 524288 specifically, not lower. Issue #363 deadlocks the extend precompile
# (silently -- no crash, no watchdog) when a large program meets a small pool,
# and #395's automatic mitigation triggers on tokens alone
# (env_override.py: max_total_tokens < 4096*64) while the measured boundary was
# 4096 PAGES. At this recipe's page_size=128 those disagree: 438720 tokens is
# 3427 pages, inside the danger zone, yet above the token trigger so the
# mitigation would not fire. 524288 = 4096 pages AND >= 262144 tokens, so it is
# safe whichever unit the threshold really is. Qwen3-Coder-480B has 62 layers,
# well above #395's 35-layer floor, so it is in the affected class.
#
# Keeping KV offload is the point: --enable-hierarchical-cache, raiden and
# ratio 1 are untouched. A smaller pool should also make the feature engage for
# the first time -- kv_offload_written is 0.0 and #host-hit-token is 0 on every
# prefill of every run, green included, because nothing is ever evicted.
#
# Not lowering the pool further, and not touching these, all proven broken:
#   * Compile ladder. Any reduction pads negative under DP attention
#     (forward_batch_info.py:1302): decode [1,8] + extend [4096] -> -123 c1
#     (34903370712); decode [1,8] + derived -> -1017 c1 (34908974337);
#     decode [1,8,16] + derived -> -1352 c16 (34914967096); decode derived +
#     extend [1024,2048,4096] -> -253 c16 (34919948429). Worth its own bug.
#   * HICACHE_RATIO below 1. kv_pool_host.py:97 asserts the host pool strictly
#     exceeds the device pool; 0.5 dies at bring-up (34924965546).
if [[ "$PROFILE" == "ci" ]]; then
  export MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-524288}"
fi

exec env \
  MODEL="${MODEL:-/weights/Qwen3-Coder-480B-A35B-Instruct-FP8}" \
  PROFILE="$PROFILE" \
  EXTRA_ARGS="$EXTRA_ARGS" \
  ROLE=colocated \
  bash recipe/qwen3_coder_480b/nightly/serve.sh
