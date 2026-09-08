"""Qwen3-Coder-480B single-host prefix caching + KV offload (raiden) gate.

Covers the P1 `prefix_caching_kv_offload` requirement (issue #372): one server with
the radix cache on and HiCache mirroring it into host RAM over the raiden DMA
backend must show a cache hit, show offloading actually happening, and still clear
the GSM8K quality bar.

Qwen3-Coder is the uniform-attention half of that requirement, and the larger one:
it is where the host pool has to absorb a 480B model's KV, and it is the only row
that exercises raiden's single-pool controller, since GPT-OSS being hybrid-SWA
routes to the two-pool one instead.

The prefix-cache + KV-offload gate runs first, on a cache it flushes and fully
controls. GSM8K then scores the same boot, with prefix caching still on.
"""

import os

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main
from sglang.test.tpu.kv_offload_probe import kv_offload_runner

# Bring-up plus the offload probe's ~26k tokens of filler prefill plus 200 GSM8K
# examples. Above the 3600 the spec-decoding lane uses, by the probe's cost.
register_tpu_ci(
    est_time=4200,
    suite="nightly-tpu-4-v7",
    nightly=True,
    labels=["singlehost", "qwen3-coder-480b", "prefix-cache", "kv-offload"],
)

SERVED_MODEL_NAME = "qwen3-coder-480b"
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "3600"))

# Same floor as the spec-decoding lane: offloading changes where KV lives, not what
# the model computes, so a drop here is a corrupted host round-trip.
GSM8K_ACCURACY = 0.65

# Per DP worker. Small enough that a handful of filler prefills evict the probe to
# host, large enough that GSM8K's share of CONCUR=8 still fits on a worker (short
# prompts plus 512 output tokens each). Both hold at 8k once dp splits the traffic.
DEVICE_POOL_TOKENS = 8192
HICACHE_RATIO = 3.0
PROBE_TOKENS = 2048
# recipe/qwen3_coder_480b/server.sh exports PAGE_SIZE unconditionally, so this is
# not overridable from `env` -- it is restated because the gate's "at least one
# page" checks are in terms of it, and a prefix match is floored to a page multiple.
PAGE_SIZE = 256
# Matching the one 480B lane that is green, singlehost_spec_dpa_gsm8k. dp=1 has
# no working setting on this model: 512 boots but decodes garbage (GSM8K 0.0,
# measured with HiCache OFF too, so the cache is not involved), and 1024 and
# 4096 both abort every rank in the extend precompile on
# absl::raw_hash_map<>::at. Under DP attention the same model serves correctly.
#
# dp=2, not the 4 the spec_dpa lane uses: HiCache allocates a host pool PER DP
# GROUP, so at ratio 3.0 dp=4 asks for 4 x 3 x 8192 = 98304 host tokens, double
# the dp=1 total, and the OS killed a scheduler mid weight-load (run 34252903494,
# server exited with code -9). spec_dpa gets away with dp=4 because it runs no
# HiCache at all. dp=2 puts the host total back at the dp=1 level.
DP_SIZE = 2

LAUNCH_ENV = {
    "MODEL": "/weights/Qwen3-Coder-480B-A35B-Instruct-FP8",
    "ISL": "4096",
    "OSL": "512",
    "CONCUR": "8",
    "CONTEXT_LEN": "8192",
    "MAX_TOTAL_TOKENS": str(DEVICE_POOL_TOKENS),
    # DP attention divides --chunked-prefill-size by dp_size, so 2048 at dp=2
    # compiles the 1024 extend graph the green spec_dpa lane compiles.
    "CHUNK": "2048",
    "DP_SIZE": str(DP_SIZE),
    # Only the tiers the DP-divided chunk can actually reach. Eight ranks
    # compile extend graphs concurrently and the host cannot hold three tiers
    # worth at once: run 34258644286 died in _precompile_extend with "server
    # exited with code -9" (host OOM) after clearing weight load, KV allocation
    # and decode capture. The shipped dp=1 config compiled one tier; this is two.
    "TPU_COMPILE_TOKENS": "512 1024",
    "TPU_COMPILE_BS": "1 2 4",
    # The recipe serves this model on pallas_tpu_rpa_batched, which raiden rejects:
    # RPA v3 is the only TPU backend carrying the HiCache reload-commit hook and the
    # fused KV pool. Overridden here, together with the layout it forces.
    "ATTENTION_BACKEND": "pallas_tpu_rpa_v3",
    "HICACHE_MEM_LAYOUT": "page_first_fused",
    # Prefix caching is the substrate HiCache is built on, and launch_server.sh
    # defaults it OFF.
    "DISABLE_RADIX_CACHE": "0",
    "KV_OFFLOAD": "1",
    "HICACHE_IO_BACKEND": "raiden",
    "HICACHE_RATIO": str(HICACHE_RATIO),
    # Write through on every prefill so the probe is already on host by the time the
    # fillers evict it. write_back would race that eviction and drop it instead.
    "HICACHE_WRITE_POLICY": "write_through",
    # cached_tokens_details in meta_info is how the gate tells a device hit from a
    # host reload. Without this flag both read 0.
    "ENABLE_CACHE_REPORT": "1",
    "ENABLE_PROFILER_ENDPOINT": "0",
    "EXTRA_ARGS": f"--served-model-name {SERVED_MODEL_NAME}",
}

MODEL_CONFIG = {
    "model": SERVED_MODEL_NAME,
    "test_name": "qwen3-coder-480b-prefix-cache-kv-offload",
    "variant": "TP8+EP8+DP2+HiCache-raiden",
    "launch": {
        "recipe": "recipe/qwen3_coder_480b/server.sh",
        "env": LAUNCH_ENV,
        "health_timeout": HEALTH_TIMEOUT,
        "tag": "qwen3-coder-480b-prefix-cache-kv-offload-e2e",
    },
    # No dashboard key: this is feature validation, and its GSM8K score is gated at a
    # config the accuracy lane does not serve, so plotting it next to that lane's
    # would compare two different servers.
    "gates": [
        # First, on a cache it flushes and fully owns -- GSM8K would otherwise leave
        # the pool full and the eviction arithmetic would no longer hold.
        {
            "dataset": "prefix_cache_kv_offload",
            "label": "prefix-cache-kv-offload",
            "runner": kv_offload_runner(
                test_name="qwen3-coder-480b-prefix-cache-kv-offload",
                device_pool_tokens=DEVICE_POOL_TOKENS,
                page_size=PAGE_SIZE,
                probe_tokens=PROBE_TOKENS,
                hicache_ratio=HICACHE_RATIO,
                # The probe pins each read to one DP rank and checks the
                # response came back from it, so it runs once per rank.
                dp_size=DP_SIZE,
            ),
        },
        {
            "dataset": "gsm8k",
            "label": "gsm8k",
            "baseline": GSM8K_ACCURACY,
            "num_examples": 200,
            # Matches CONCUR above: the small device pool cannot hold more requests
            # in flight without retracting them.
            "num_threads": 8,
            "max_tokens": 512,
            "temperature": 0.0,
        },
    ],
}


if __name__ == "__main__":
    model_config_main(MODEL_CONFIG, base_url=BASE_URL)
