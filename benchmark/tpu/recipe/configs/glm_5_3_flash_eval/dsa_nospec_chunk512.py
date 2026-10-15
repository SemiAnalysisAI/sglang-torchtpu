from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash_eval.dsa_nospec import CONFIG as _BASE

# Control: no radix, prefill split into 512-token chunks.
CONFIG = deepcopy(_BASE)
CONFIG["server_args"].update(
    chunked_prefill_size=512, max_prefill_tokens=512, tpu_compile_tokens=[512]
)
CONFIG["runtime_env"]["TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE"] = None
