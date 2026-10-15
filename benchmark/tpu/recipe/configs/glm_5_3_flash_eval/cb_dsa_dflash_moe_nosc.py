from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash_eval.cb_dsa_dflash import CONFIG as _BASE

# DFlash + DSA 8k1k with the fused MoE's SparseCore gathers off (option 2: free SparseCore
# capacity for the indexer top-k and the sparse-MLA gather by moving MoE gathers to XLA).
CONFIG = deepcopy(_BASE)
CONFIG["runtime_env"]["SGLANG_TPU_FUSED_MOE_SC_GATHER"] = "0"
