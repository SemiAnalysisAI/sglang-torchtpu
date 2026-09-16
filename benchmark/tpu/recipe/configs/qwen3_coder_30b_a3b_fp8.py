"""TPU serving configuration. Callers copy CONFIG before overriding it."""

from copy import deepcopy

from benchmark.tpu.recipe.configs.base import _FP8_BASE_CONFIG

CONFIG = deepcopy(_FP8_BASE_CONFIG)
CONFIG["server_args"].update(
    {
        # HF repo id, not a /weights path: this config predates the shared
        # model-weights-rox disk and pulls from the HF cache.
        "model_path": "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
        "tp_size": 8,
        "ep_size": 8,
        # Sized for the lm-eval MMLU sweep at max_running_requests=64.
        "context_length": 8192,
        "max_total_tokens": 262144,
        # Leaves room for that KV pool at tp=8.
        "mem_fraction_static": 0.3,
    }
)
