from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash_eval.kl_dsa_radix_big import CONFIG as _BASE

# Bisect: the halting KL config with the KDA state pool back at 16 slots (KV pool 262144 kept).
CONFIG = deepcopy(_BASE)
CONFIG["server_args"]["max_mamba_cache_size"] = 16
