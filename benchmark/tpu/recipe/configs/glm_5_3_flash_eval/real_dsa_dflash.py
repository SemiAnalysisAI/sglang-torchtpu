from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash.dflash import CONFIG as _BASE

# DFlash on the recipe's DSA at real acceptance (no simulated accept length), 8k1k c1.
CONFIG = deepcopy(_BASE)
assert "SGLANG_TPU_GLM_FLASH_DENSE_MAX_CONTEXT" not in CONFIG["runtime_env"]
