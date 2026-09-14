"""Single-chip example: load shared settings and override them in Python."""

from copy import deepcopy

from recipe.configs.base import CONFIG as BASE_CONFIG

INPUT_TOKENS = 1024
OUTPUT_TOKENS = 256
CONCURRENCY = 4

CONFIG = deepcopy(BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "model-path": "Qwen/Qwen3-0.6B",
        "tp-size": 1,
        "attention-backend": "pallas_tpu_rpa_v3",
        "page-size": 128,
        "context-length": INPUT_TOKENS + OUTPUT_TOKENS + 20,
        "max-running-requests": CONCURRENCY,
        "torch-compile-max-bs": CONCURRENCY,
        "enable-torch-compile": True,
    }
)
CONFIG["runtime_env"].update(
    {
        "TPU_VISIBLE_CHIPS": "0",
        "TPU_CHIPS_PER_HOST_BOUNDS": "1,1,1",
        "TPU_HOST_BOUNDS": "1,1,1",
    }
)
