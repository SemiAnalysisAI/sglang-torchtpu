"""DeepSeek-V3.1 thinking accuracy on two hosts, TP16/EP16, FP8 KV.

Requires the DeepSeek-V3.1 checkpoint on the shared CI weights disk.
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.deepseek_r1_0528_fp8kv_mh import CONFIG as BASE_CONFIG

CONFIG = deepcopy(BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "model_path": "/weights/DeepSeek-V3.1",
        "reasoning_parser": "deepseek-v3",
        "context_length": 73728,
    }
)
