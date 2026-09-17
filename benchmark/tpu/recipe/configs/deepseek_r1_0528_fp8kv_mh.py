"""DeepSeek-R1-0528 accuracy: two hosts, TP16/EP16, FP8 KV, no DP attention.

Run on each host with TPU_WORKER_ID and TPU_WORKER_HOSTNAMES set:
    python3 -m benchmark.tpu.launch_server --config recipe/configs/deepseek_r1_0528_fp8kv_mh.py
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.base import _FP8_BASE_CONFIG

CONFIG = deepcopy(_FP8_BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "model_path": "/weights/DeepSeek-R1-0528",
        "nnodes": 2,
        "tp_size": 16,
        "ep_size": 16,
        "dp_size": 1,
        "attention_backend": "pallas_tpu_mla",
        "page_size": 64,
        "mem_fraction_static": 0.85,
        "max_running_requests": 128,
        "context_length": 40960,
        "chunked_prefill_size": 8192,
        "max_prefill_tokens": 8192,
        "tpu_compile_tokens": [8192],
        "tpu_compile_bs": [64, 128],
        "torch_compile_max_bs": 128,
        "reasoning_parser": "deepseek-r1",
        "weight_loader_drop_cache_after_load": True,
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
