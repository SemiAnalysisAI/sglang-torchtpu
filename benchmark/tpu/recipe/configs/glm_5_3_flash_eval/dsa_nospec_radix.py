from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash_eval.dsa_nospec import CONFIG as _BASE

# Prefix cache (radix) on the recipe's DSA, no speculative decoding (KDA refuses spec + radix).
CONFIG = deepcopy(_BASE)
CONFIG["server_args"]["disable_radix_cache"] = False
# Radix keeps KDA state checkpoints: about mamba_ratio (5) state slots per running request.
CONFIG["server_args"].update(max_running_requests=2, torch_compile_max_bs=2, max_mamba_cache_size=16)
CONFIG["runtime_env"]["TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE"] = None
