from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash_eval.pc_recipe import CONFIG as _BASE

# The prefix_cache recipe with the radix cache off (accuracy control).
CONFIG = deepcopy(_BASE)
CONFIG["server_args"]["disable_radix_cache"] = True
