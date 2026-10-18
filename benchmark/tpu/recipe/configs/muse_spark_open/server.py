"""Muse Spark Open 816B-A42B (NVFP4 MoE) text serving on one tpu7x host, tp8/ep8.

Sample usage (from the repo root):
    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/muse_spark_open/server.py
    # add --dry-run to print the resolved sglang.launch_server command
    # (inherited base args included) without launching.
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.base import _FP8_BASE_CONFIG

INPUT_TOKENS = 1024
OUTPUT_TOKENS = 1024
CONCURRENCY = 64
CONTEXT_LENGTH = INPUT_TOKENS + OUTPUT_TOKENS + 64
PREFILL_TOKENS = 4096

CONFIG = deepcopy(_FP8_BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "model_path": "meta-models/Muse-Spark-Open-816B-A42B-NVFP4",
        # The block-16 NVFP4 experts run only on the fused reduce-scatter MoE
        # kernel, which needs expert parallelism equal to the TP world.
        "tp_size": 8,
        "ep_size": 8,
        # The checkpoint ships its own ModelOpt quantization_config.
        "quantization": None,
        # Vision and audio towers are not served on TPU.
        "language_model_only": True,
        "page_size": 64,
        "context_length": CONTEXT_LENGTH,
        "chunked_prefill_size": PREFILL_TOKENS,
        "max_prefill_tokens": PREFILL_TOKENS,
        "max_running_requests": CONCURRENCY,
        "torch_compile_max_bs": CONCURRENCY,
        # Each compiled bucket of this model takes 5-6 minutes.
        "tpu_compile_tokens": [1024, PREFILL_TOKENS],
        "tpu_compile_bs": [1, 8, 32, CONCURRENCY],
        # The distributed TPU top-k sampler can halt at tp8.
        "sampling_backend": "pytorch",
        "mem_fraction_static": 0.9,
    }
)
# The SparseCore gather halts on the per-slot MoE path of the BF16 expert
# layers (the first and last).
CONFIG["runtime_env"].update({"SGLANG_TPU_FUSED_MOE_SC_GATHER": "0"})
