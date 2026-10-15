from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash_eval.dsa_nospec_radix import CONFIG as _BASE

# KL consistency (kl_test_utils): DSA no-spec with radix; prompt logprobs need every prefill row.
CONFIG = deepcopy(_BASE)
CONFIG["runtime_env"]["SGLANG_TPU_ENABLE_PROMPT_LOGPROB"] = "1"
