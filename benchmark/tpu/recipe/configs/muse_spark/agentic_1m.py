"""Muse Spark for many agentic sessions per host at up to 1M tokens of context.

The AgentX 1M-context configuration for concurrency 24-32 on one host: fp8 KV in
the SEQ_ALONG_LANE layout (batched RPA prefill, megakernel decode), host KV
offload with write-back, and a sliding-window pool sized for long cold prefills.
This file is the concurrency-32 profile: 8 running requests with 4 DFlash draft
tokens. The other profiles change only these settings:

    concurrency 24: max_running_requests 6, 8 draft tokens,
                    tpu_compile_bs [1, 2, 3, 4, 5, 6], SGLANG_TPU_MUSE_MK_MAX_ROWS=48
    concurrency 32, throughput: max_running_requests 10, prefill_decode_interval 16,
                    tpu_compile_bs [1, 2, 4, 6, 8, 10], SGLANG_TPU_MUSE_MK_MAX_ROWS=48
    concurrency 4-16: agentic.py with bf16 KV, max_running_requests = concurrency / 4,
                    8 draft tokens

The pool settings assume a windowed draft: the DFlash2 draft's layers are all
2048-token sliding-window, and a draft config that declares it (is_hybrid_swa and
an all-SWA hybrid_layer_pattern) keeps the draft's KV in the target's SWA slots.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/muse_spark/agentic_1m.py
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.muse_spark.agentic import CONFIG as AGENTIC_CONFIG

CONFIG = deepcopy(AGENTIC_CONFIG)
# The pool follows the HBM budget; the cc=1 recipe pins a smaller one.
CONFIG["server_args"].pop("max_total_tokens")
CONFIG["server_args"].update(
    {
        "context_length": 1048576,
        "max_running_requests": 8,
        "torch_compile_max_bs": 8,
        # Even buckets keep bucket x 4 draft tokens a whole number of 8-row tiles.
        "tpu_compile_bs": [1, 2, 4, 6, 8],
        # One 4096-token prefill graph: each prefill bucket of a 1M-context model
        # holds its own copy of the page-table-sized metadata in host RAM.
        "tpu_compile_tokens": [4096],
        "speculative_num_draft_tokens": 4,
        "prefill_decode_interval": 128,
        "page_size": 256,
        "kv_cache_dtype": "fp8_e4m3",
        "attention_backend": "pallas_tpu_rpa_batched",
        "speculative_draft_attention_backend": "pallas_tpu_rpa_v3",
        # The windowed draft frees its full-context KV for the target's pool.
        "mem_fraction_static": 0.95,
        "swa_full_tokens_ratio": 0.0625,
        "enable_hierarchical_cache": True,
        "hicache_io_backend": "raiden",
        "hicache_mem_layout": "page_first_fused",
        # Host pool = ratio x device pool per rank; 2.0 keeps a ~4.4M-token device
        # pool's host copy within a 960 GiB node.
        "hicache_ratio": 2.0,
        "hicache_write_policy": "write_back",
    }
)
CONFIG["runtime_env"].update(
    {
        "SGLANG_TPU_USE_SEQ_ALONG_LANE_KV": "1",
        "SGLANG_TPU_MK_Y_EARLY": "1",
        "SGLANG_TPU_FUSED_RS_ETP_LOCAL_ROWS": "1",
        "SGLANG_TPU_FUSED_RS_ETP_TILED_REDUCE": "1",
        "SGLANG_PREFILL_DECODE_INTERVAL_SKIP_IDLE": "1",
        # Checkpoint a long prefill into the radix tree every 128K tokens, not
        # every chunk.
        "SGLANG_OPT_UNIFIED_CACHE_CHUNK_INSERT_STRIDE": "131072",
        # Reserve each request's full max_new_tokens at admission: no retraction.
        "SGLANG_INIT_NEW_TOKEN_RATIO": "1.0",
        "SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR": "1.0",
        "TPU_PREMAPPED_BUFFER_SIZE": "4294967296",
    }
)
