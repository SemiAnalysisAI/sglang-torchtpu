from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash_eval.c1_dsa_dflash_acc35 import CONFIG as _BASE

# Baseline for the DSA TTFT work: the same 8k1k c1 DFlash run with DENSE attention
# (absorbed MLA on pallas_tpu_mla, FP8 latent), today's tree. Not the model's attention.
CONFIG = deepcopy(_BASE)
CONFIG["server_args"]["attention_backend"] = "pallas_tpu_mla"
CONFIG["runtime_env"]["SGLANG_TPU_GLM_FLASH_DENSE_MAX_CONTEXT"] = str(CONFIG["server_args"]["context_length"])
