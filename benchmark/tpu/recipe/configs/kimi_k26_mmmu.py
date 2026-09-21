"""One TP16/EP16, attention-DP4 instance for half of full MMMU-Pro."""

from copy import deepcopy

from benchmark.tpu.recipe.configs.base import _FP8_BASE_CONFIG

CONFIG = deepcopy(_FP8_BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "model_path": "/weights/Kimi-K2.6",
        "nnodes": 2,
        "tp_size": 16,
        "ep_size": 16,
        "dp_size": 4,
        "enable_dp_attention": True,
        "enable_dp_lm_head": True,
        "quantization": None,
        "kv_cache_dtype": "fp8_e4m3",
        "attention_backend": "pallas_tpu_mla",
        "trust_remote_code": True,
        "page_size": 128,
        "mem_fraction_static": 0.88,
        "max_running_requests": 512,
        "context_length": 106496,
        "chunked_prefill_size": 16384,
        "max_prefill_tokens": 16384,
        "tpu_compile_tokens": [4096],
        "tpu_compile_bs": [1, 16, 32, 64, 128],
        "torch_compile_max_bs": 128,
        "scheduler_recv_interval": 8,
        "skip_server_warmup": True,
        "weight_loader_drop_cache_after_load": True,
    }
)
CONFIG["runtime_env"].update(
    {
        "SGLANG_TPU_MM_EMBED_ON_DEVICE": "1",
        "SGLANG_ENABLE_EMBED_REPLICATION": "1",
        "SGLANG_TPU_INT4_MOE_MODE": "vmem-bf16",
        "SGLANG_TPU_VIT_PRECOMPILE_SIZES": "256,400,576,784,1024,1600,2304,3136,4096,6400,9216,12544,16900",
    }
)
