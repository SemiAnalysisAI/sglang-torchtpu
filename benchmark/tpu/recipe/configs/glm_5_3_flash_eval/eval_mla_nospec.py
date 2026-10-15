from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash.server import CONFIG as _BASE
from benchmark.tpu.recipe.configs.glm_5_3_flash_eval import long_ctx, mla

CONFIG = long_ctx.apply(mla.apply(deepcopy(_BASE), kv_dtype="bfloat16"))
