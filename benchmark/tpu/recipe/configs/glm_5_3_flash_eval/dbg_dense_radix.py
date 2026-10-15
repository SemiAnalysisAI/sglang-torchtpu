from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash_eval.dsa_nospec_radix import CONFIG as _BASE

# DIAGNOSTIC ONLY (not DSA): dense absorbed MLA + radix, to tell whether the prefill
# chunk-boundary logprob gap comes from DSA or from the KDA layers.
CONFIG = deepcopy(_BASE)
CONFIG["server_args"]["attention_backend"] = "pallas_tpu_mla"
CONFIG["runtime_env"]["SGLANG_TPU_GLM_FLASH_DENSE_MAX_CONTEXT"] = "10240"
