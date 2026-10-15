from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash_eval.kl_dense_radix import CONFIG as _BASE

# KL cache-hit checks warm 32 prompts of ~3k tokens before generating: keep them all cached
# (KV for ~262k tokens, a KDA state snapshot slot per cached prefix).
CONFIG = deepcopy(_BASE)
CONFIG["server_args"].update(max_total_tokens=262144, max_mamba_cache_size=96)
