from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash_eval.mla_fp8_mtp516 import CONFIG as _BASE

# Diagnostic boot: log the call sites of TPU -> CPU copies (misc sitecustomize).
CONFIG = deepcopy(_BASE)
CONFIG["runtime_env"].update(
    {"SYNCSITES": "1", "PYTHONPATH": "/dev/shm/home/qinghanc/work/syncsites"}
)
