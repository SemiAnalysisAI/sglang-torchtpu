from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash.mtp_s5 import CONFIG as _BASE
from benchmark.tpu.recipe.configs.glm_5_3_flash_eval.cookbook import apply

# Full batch list: tests whether the c16 Mosaic crash came from uncompiled bs 2..15.
CONFIG = apply(deepcopy(_BASE))
CONFIG["server_args"]["tpu_compile_bs"] = [1, 2, 4, 8, 16]
CONFIG["server_args"]["tpu_compile_tokens"] = [8192]
