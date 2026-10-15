from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash_eval.c1_dsa_dflash_acc35_scmoe_p8192 import CONFIG as _BASE

# SparseCore-halt test: the halting 8192 config with libtpu's default flags (no async collective
# fusion, no compute/collective overlap, default scheduler).
CONFIG = deepcopy(_BASE)
CONFIG["runtime_env"]["LIBTPU_INIT_ARGS"] = None
