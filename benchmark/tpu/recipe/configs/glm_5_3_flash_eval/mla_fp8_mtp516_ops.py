from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash_eval.mla_fp8_mtp516 import CONFIG as _BASE

# Diagnostic boot: count call sites of small eager TPU ops after warmup (misc sitecustomize).
CONFIG = deepcopy(_BASE)
CONFIG["runtime_env"].update({"OPSITES": "1", "PYTHONPATH": "/dev/shm/home/qinghanc/work/opsites"})
