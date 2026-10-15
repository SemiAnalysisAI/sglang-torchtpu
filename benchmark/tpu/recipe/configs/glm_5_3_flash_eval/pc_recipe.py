from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash.prefix_cache import CONFIG as _BASE

# The prefix_cache recipe as shipped (radix on); tier-2 compile cache off so it
# stays out of /dev/shm, which counts toward the pod's memory limit.
CONFIG = deepcopy(_BASE)
CONFIG["runtime_env"]["TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE"] = None
