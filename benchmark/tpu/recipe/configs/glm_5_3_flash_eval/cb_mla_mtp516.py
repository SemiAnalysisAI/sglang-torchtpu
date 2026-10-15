from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash.mtp_s5 import CONFIG as _BASE
from benchmark.tpu.recipe.configs.glm_5_3_flash_eval import cookbook, mla

# Full batch list: the MHA MTP c16 run crashed compiling bs 2..15 at runtime.
CONFIG = mla.apply(cookbook.apply(deepcopy(_BASE)))
CONFIG["server_args"]["tpu_compile_bs"] = [1, 2, 4, 8, 16]
