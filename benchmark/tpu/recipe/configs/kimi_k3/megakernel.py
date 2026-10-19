"""Kimi-K3 (2.7T, MXFP4 experts) decode megakernel on one 2x2x4 tpu7x slice.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/kimi_k3/megakernel.py

MK=0 serves the same config with SGLang's decode, for A/B.
"""

import os
from copy import deepcopy

from benchmark.tpu.recipe.configs.base import _FP8_BASE_CONFIG

ROWS = 8
CONTEXT_LENGTH = 9344
PREFILL_TOKENS = 1024

CONFIG = deepcopy(_FP8_BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "model_path": "/weights/Kimi-K3",
        "host": "0.0.0.0",
        "port": 47903,
        # node_rank / dist_init_addr come from TPU_WORKER_ID / TPU_WORKER_HOSTNAMES.
        "nnodes": 4,
        "tp_size": 32,
        "ep_size": 8,
        # MXFP4 experts (compressed-tensors): no online quantization.
        "quantization": "unquant",
        "kv_cache_dtype": "bfloat16",
        "attention_backend": "pallas_tpu_mla",
        "page_size": 128,
        "max_running_requests": ROWS,
        "torch_compile_max_bs": ROWS,
        "tpu_compile_bs": [1, 2, 4, 8],
        "context_length": CONTEXT_LENGTH,
        "chunked_prefill_size": PREFILL_TOKENS,
        "max_prefill_tokens": PREFILL_TOKENS,
        "tpu_compile_tokens": [PREFILL_TOKENS],
        # Weights plus the megakernel's caches and stacked experts take ~81% of
        # HBM per rank; the rest holds the prefill KV pool and activations.
        "mem_fraction_static": 0.87,
        "trust_remote_code": True,
        "weight_loader_disable_mmap": True,
        "watchdog_timeout": 3600,
        "dist_timeout": 3600,
    }
)
CONFIG["runtime_env"].update(
    {
        "SGLANG_TPU_ENABLE_K3_MEGAKERNEL": os.environ.get("MK", "1"),
        "SGLANG_TPU_CHECK_RECOMPILATION": "1",
    }
)
