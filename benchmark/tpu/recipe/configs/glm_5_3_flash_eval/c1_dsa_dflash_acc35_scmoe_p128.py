from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash_eval.c1_dsa_dflash_acc35 import CONFIG as _BASE

# DFlash + DSA 8k1k c1 at simulated accept 3.5, prefill chunked at 128 tokens (Linkun's GLM-5.3 DSA recipe size), with the fused
# MoE gathers back on the SparseCore (the indexer top-k stays on XLA). Tests whether smaller
# prefill graphs fit the SparseCore and recover the ~270 ms the XLA MoE gathers cost.
CONFIG = deepcopy(_BASE)
CONFIG["server_args"].update(
    {"chunked_prefill_size": 128, "max_prefill_tokens": 128, "tpu_compile_tokens": [128]}
)
CONFIG["runtime_env"]["SGLANG_TPU_FUSED_MOE_SC_GATHER"] = "1"
