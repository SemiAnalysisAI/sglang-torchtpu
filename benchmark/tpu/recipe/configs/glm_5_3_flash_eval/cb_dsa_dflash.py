from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash.dflash import CONFIG as _BASE
from benchmark.tpu.recipe.configs.glm_5_3_flash_eval import cookbook

# 8k1k cookbook parity for DFlash on the recipe's own DSA (no mla.apply: that
# switches back to dense pallas_tpu_mla).
CONFIG = cookbook.apply(deepcopy(_BASE))
assert "SGLANG_TPU_GLM_FLASH_DENSE_MAX_CONTEXT" not in CONFIG["runtime_env"]
