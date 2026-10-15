from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash.dflash import CONFIG as _BASE
from benchmark.tpu.recipe.configs.glm_5_3_flash_eval.long_ctx import apply

CONFIG = apply(deepcopy(_BASE))
