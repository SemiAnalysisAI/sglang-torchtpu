from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash_eval.c1_dsa_dflash_acc3 import CONFIG as _BASE

# Option 2: the same with the fused MoE's SparseCore gathers moved to XLA.
CONFIG = deepcopy(_BASE)
CONFIG["runtime_env"]["SGLANG_TPU_FUSED_MOE_SC_GATHER"] = "0"
