from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash.dflash import CONFIG as _BASE
from benchmark.tpu.recipe.configs.glm_5_3_flash_eval import cookbook, mla

CONFIG = mla.apply(cookbook.apply(deepcopy(_BASE)))
