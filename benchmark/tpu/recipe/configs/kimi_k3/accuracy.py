"""Kimi-K3 (2.7T, MXFP4 routed experts) on one 2x2x4 tpu7x slice: tp32/ep32,
bf16 KV. Serves the GSM8K, GPQA and MMLU nightly.

    python3 -m benchmark.tpu.launch_server --config recipe/configs/kimi_k3/accuracy.py
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.base import _FP8_BASE_CONFIG

CONCURRENCY = 64
# 32k thinking budget + prompt.
CONTEXT_LENGTH = 36864
# One chunk holds a whole 5-shot GSM8K prompt (at most ~850 tokens).
PREFILL_TOKENS = 1024

CONFIG = deepcopy(_FP8_BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "model_path": "/weights/Kimi-K3",
        # The name requests carry; model_path is a mount.
        "served_model_name": "moonshotai/Kimi-K3",
        # node_rank / dist_init_addr come from TPU_WORKER_ID / TPU_WORKER_HOSTNAMES.
        "nnodes": 4,
        "tp_size": 32,
        "ep_size": 32,
        # The checkpoint ships its own MXFP4 config; fp8 would re-quantize it.
        "quantization": None,
        "kv_cache_dtype": "auto",
        "attention_backend": "pallas_tpu_mla",
        "page_size": 128,
        "max_running_requests": CONCURRENCY,
        "torch_compile_max_bs": CONCURRENCY,
        "context_length": CONTEXT_LENGTH,
        "chunked_prefill_size": PREFILL_TOKENS,
        "max_prefill_tokens": PREFILL_TOKENS,
        "tpu_compile_tokens": [PREFILL_TOKENS],
        # Each full-model graph compiles for ~2 min (decode) to ~5 min (extend)
        # at startup; smaller batches pad to 16.
        "tpu_compile_bs": [16, 32, CONCURRENCY],
        "mem_fraction_static": 0.85,
        "trust_remote_code": True,
        "reasoning_parser": "kimi_k3",
        # Each rank reads only its EP slice of the routed experts.
        "weight_loader_disable_mmap": True,
        # Without it the 1.5 TB of checkpoint pages stay in the host page cache.
        "weight_loader_drop_cache_after_load": True,
        # The weight load outlasts the default rendezvous and watchdog timeouts.
        "dist_timeout": 3600,
        "watchdog_timeout": 3600,
        "skip_server_warmup": True,
    }
)
