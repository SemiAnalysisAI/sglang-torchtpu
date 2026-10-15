from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash_eval.kl_dsa_radix import CONFIG as _BASE

# DSA prefix-cache KL check with a pool that boots (131072 tokens; 262144 halts the
# SparseCore): 24 prompts of ~3.5k tokens stay cached; a KDA snapshot slot per prefix.
CONFIG = deepcopy(_BASE)
CONFIG["server_args"].update(max_total_tokens=131072, max_mamba_cache_size=48)
