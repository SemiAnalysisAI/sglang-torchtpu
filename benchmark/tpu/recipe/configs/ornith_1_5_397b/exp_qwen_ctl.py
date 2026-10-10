"""Control: Qwen3.5-397B-A17B-FP8 (block FP8 -> channelwise runtime) on the identical Ornith server config."""

from copy import deepcopy

from benchmark.tpu.recipe.configs.ornith_1_5_397b.perf import CONFIG as _BASE

CONFIG = deepcopy(_BASE)
CONFIG["server_args"].update(
    {
        "model_path": "/dev/shm/Qwen3.5-397B-A17B-FP8",
        "served_model_name": "Qwen/Qwen3.5-397B-A17B-FP8",
        "quantization": "fp8",
        "tpu_fp8_runtime": "channelwise",
    }
)
