from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash_eval.dsa_nospec import CONFIG as _BASE

# Control for the prefix-cache logprob gap: no radix, prefill split into 1024-token chunks.
CONFIG = deepcopy(_BASE)
CONFIG["server_args"].update(
    chunked_prefill_size=1024, max_prefill_tokens=1024, tpu_compile_tokens=[1024]
)
CONFIG["runtime_env"]["TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE"] = None
