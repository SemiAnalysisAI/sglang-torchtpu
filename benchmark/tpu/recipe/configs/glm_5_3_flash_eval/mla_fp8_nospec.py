from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash.server import CONFIG as _BASE
from benchmark.tpu.recipe.configs.glm_5_3_flash_eval.mla import apply

CONFIG = apply(deepcopy(_BASE), kv_dtype="fp8_e4m3")
