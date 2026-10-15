from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash_eval.dbg_dense_radix import CONFIG as _BASE

# DIAGNOSTIC ONLY (not DSA): the KL consistency reference on dense MLA.
CONFIG = deepcopy(_BASE)
CONFIG["runtime_env"]["SGLANG_TPU_ENABLE_PROMPT_LOGPROB"] = "1"
