from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash_eval.c1_dsa_dflash_acc3 import CONFIG as _BASE

# 8k1k c1 DFlash on the recipe's DSA at simulated acceptance 3.5.
CONFIG = deepcopy(_BASE)
CONFIG["runtime_env"]["SGLANG_SIMULATE_ACC_LEN"] = "3.5"
