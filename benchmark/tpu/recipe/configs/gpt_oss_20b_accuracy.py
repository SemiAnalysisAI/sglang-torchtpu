"""GPT-OSS-20B nightly accuracy: TP8/EP8, BF16 KV, 64K generation."""

from copy import deepcopy

from benchmark.tpu.recipe.configs.gpt_oss_120b.server import CONFIG as BASE_CONFIG

CONFIG = deepcopy(BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "model_path": "openai/gpt-oss-20b",
        "served_model_name": "gpt-oss-20b",
        "reasoning_parser": "gpt-oss",
        "kv_cache_dtype": "auto",
        "page_size": 128,
        "disable_radix_cache": True,
        "max_running_requests": 16,
        "torch_compile_max_bs": 16,
        "context_length": 69632,
        "max_total_tokens": 1114112,
        "chunked_prefill_size": 4096,
        "max_prefill_tokens": 4096,
        "tpu_compile_tokens": [4096],
        "tpu_compile_bs": [1, 2, 4, 8, 16],
    }
)

# Preserve the shell recipes' long-run diagnostic settings.
CONFIG["runtime_env"].update(
    {
        "TPU_ENABLE_DIAGNOSTIC_COORDINATOR": "0",
        "TPU_DISABLE_DIAGNOSTIC_COORDINATOR": "1",
        "TPU_DIAGNOSTIC_HARVESTER_INTERVAL_SEC": "86400",
        "TORCH_LOGS": "+recompiles",
    }
)
