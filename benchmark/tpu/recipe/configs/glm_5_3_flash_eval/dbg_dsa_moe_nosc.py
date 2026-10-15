from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash_eval.prof_dsa_mtp516_acc35 import CONFIG as _BASE

# Debug: MTP+DSA 8k1k with the fused MoE's SparseCore gathers off (XLA gathers instead),
# to test whether the MoE and sparse-attention SparseCore kernels collide.
CONFIG = deepcopy(_BASE)
CONFIG["runtime_env"]["SGLANG_TPU_FUSED_MOE_SC_GATHER"] = "0"
