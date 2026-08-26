"""GPT-OSS-120B single-host prefix caching + KV offload (raiden) + EAGLE3 gate.

Merges two existing single-host lanes -- prefix cache + KV offload, and EAGLE3
GSM8K -- because neither covers their interaction, and there is a real one.
With HiCache on, a speculative boot builds an extra host pool for the draft KV,
so each rank announces three raiden pools instead of two (full attention,
sliding window, draft sidecar), and every page the target reloads has to line up
with a draft pool mirrored alongside it. This is row 2 of the manual
feature-combo matrix, promoted to a registered nightly.

raiden is the backend under test rather than torch_tpu because GPT-OSS is
hybrid-SWA: raiden gives every sub-pool its own host pool, while torch_tpu
mirrors only full attention and is rejected outright for a sliding-window model.

The offload gate runs first, on a cache it flushes and fully owns, and checks
both halves of the combination -- which tier served each read, and that the
draft worker still ran on the read whose prefix came back over the transport.
GSM8K then scores the same boot with all three features still on.
"""

import os

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main
from sglang.test.tpu.kv_offload_probe import kv_offload_runner

# The offload lane this merges, unchanged. Speculation adds the draft model's
# load and one target-verify graph capture, a few hundred seconds, and takes
# most of it back on the GSM8K run: at the measured accept length of 2.77 the
# same 200 examples need roughly a third of the decode steps. Budgeting the two
# separately double-counted and made this the heaviest file in a suite whose
# total already has to fit inside the runner's six-hour cap.
register_tpu_ci(est_time=4200, suite="nightly-tpu-4-v7", nightly=True)

SERVED_MODEL_NAME = "gpt-oss-120b"
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "1800"))

DRAFT_MODEL_PATH = "lmsys/EAGLE3-gpt-oss-120b-bf16"

# Same floor as both lanes this merges. Neither offloading nor speculation is
# supposed to change what the model computes -- one moves where KV lives, the
# other only changes how many tokens a step proposes -- so a drop here is a
# corrupted host round-trip or a draft/target mismatch, not a tuning miss.
GSM8K_ACCURACY = 0.70  # measured 0.83-0.84 over 200 examples, accept length 2.77

# The device pool has to be small enough that a handful of filler prefills evict
# the probe to host, and large enough that GSM8K at CONCUR=8 still fits (short
# prompts plus 512 output tokens each, plus the draft tokens each step holds).
# Both hold at 16k.
DEVICE_POOL_TOKENS = 16384
HICACHE_RATIO = 3.0
PROBE_TOKENS = 2048
# The recipe's default, restated here because the gate's "at least one page"
# checks are in terms of it -- a prefix match is floored to a page multiple.
PAGE_SIZE = 64

# Draft tokens accepted per verify forward, as a strict floor over the probe
# reads. A server that accepts nothing sits at exactly 1.0, so anything above it
# means drafts are landing. Kept at the floor rather than near the measured
# value because the probe reads are only 16 tokens each -- they prove the draft
# worker is alive on a host-reloaded prefix, and GSM8K below is what measures
# speculation at a useful length.
MIN_ACCEPT_LENGTH = 1.0  # measured 2.67 on the probe reads, 2.77 over GSM8K

LAUNCH_ENV = {
    "MODEL": "/weights/gpt-oss-120b",
    "ISL": "4096",
    "OSL": "512",
    # 8, not the spec lane's 16: the offload gate's eviction arithmetic is sized
    # against DEVICE_POOL_TOKENS, and 16 in-flight GSM8K requests plus their
    # draft tokens would spend that pool on retraction instead.
    "CONCUR": "8",
    "CONTEXT_LEN": "8192",
    "MAX_TOTAL_TOKENS": str(DEVICE_POOL_TOKENS),
    "PAGE_SIZE": str(PAGE_SIZE),
    "TPU_COMPILE_TOKENS": "512 2048 4096",
    "TPU_COMPILE_BS": "1 2 4 8",
    # raiden requires the RPA v3 backend and the fused layout, and server_args
    # rejects any other pairing. Both are recipe defaults, pinned here so a
    # recipe change cannot take the offload path away from this gate.
    "ATTENTION_BACKEND": "pallas_tpu_rpa_v3",
    "HICACHE_MEM_LAYOUT": "page_first_fused",
    # Prefix caching is the substrate HiCache is built on, and launch_server.sh
    # defaults it OFF.
    "DISABLE_RADIX_CACHE": "0",
    "KV_OFFLOAD": "1",
    "HICACHE_IO_BACKEND": "raiden",
    "HICACHE_RATIO": str(HICACHE_RATIO),
    # Write through on every prefill so the probe is already on host by the time
    # the fillers evict it. write_back would race that eviction and drop it instead.
    "HICACHE_WRITE_POLICY": "write_through",
    # cached_tokens_details in meta_info is how the gate tells a device hit from a
    # host reload. Without this flag both read 0.
    "ENABLE_CACHE_REPORT": "1",
    "ENABLE_PROFILER_ENDPOINT": "0",
    # TODO(tpu): remove once the TPU distributed top-k path is precompiled for
    # the same token buckets as the decoder. Today it only knows 512 and 4096,
    # so the first request at 2048 trips fail_on_recompile, takes every TP rank's
    # scheduler down, and leaves the eval scoring 0.0 against a dead server. This
    # flag is the escape hatch spec decoding already provides: recompile instead
    # of dying. It costs tens of seconds once (est_time has room) and hides
    # nothing, the recompile is still logged. This gate needs it doubly, since
    # PROBE_TOKENS is exactly the 2048 bucket.
    "SGLANG_TPU_ALLOW_SPEC_RECOMPILE": "1",
    "EXTRA_ARGS": (
        "--reasoning-parser gpt-oss "
        f"--served-model-name {SERVED_MODEL_NAME} "
        "--speculative-algorithm EAGLE3 "
        f"--speculative-draft-model-path {DRAFT_MODEL_PATH} "
        "--speculative-num-steps 3 "
        "--speculative-eagle-topk 1 "
        "--speculative-num-draft-tokens 4"
    ),
}

MODEL_CONFIG = {
    "model": SERVED_MODEL_NAME,
    "test_name": "gpt-oss-120b-prefix-cache-kv-offload-spec",
    "variant": "TP8+EP8+HiCache-raiden+EAGLE3",
    "launch": {
        "recipe": "recipe/gpt_oss_120b/server.sh",
        "env": LAUNCH_ENV,
        "health_timeout": HEALTH_TIMEOUT,
        "tag": "gpt-oss-prefix-cache-kv-offload-spec-e2e",
    },
    # No dashboard key: feature validation on a config no accuracy lane serves,
    # so plotting its score beside one would compare two different servers.
    "gates": [
        # First, on a cache it flushes and fully owns -- GSM8K would otherwise leave
        # the pool full and the eviction arithmetic would no longer hold.
        {
            "dataset": "prefix_cache_kv_offload",
            "label": "prefix-cache-kv-offload",
            "runner": kv_offload_runner(
                test_name="gpt-oss-120b-prefix-cache-kv-offload-spec",
                device_pool_tokens=DEVICE_POOL_TOKENS,
                page_size=PAGE_SIZE,
                probe_tokens=PROBE_TOKENS,
                hicache_ratio=HICACHE_RATIO,
                # Speculation is checked on the same three reads, so this gate
                # fails if the draft worker went missing on the host-reloaded
                # prefix -- the interaction the whole lane exists for. Without it
                # a server that silently fell back to plain decoding would still
                # pass every cache assertion and still score a normal GSM8K.
                min_accept_length=MIN_ACCEPT_LENGTH,
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
