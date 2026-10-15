from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash_eval.dsa_nospec import CONFIG as _BASE

# SparseCore-halt bisect: DSA no-spec, KV pool of 262144 tokens (20480, 65536, 131072 boot).
CONFIG = deepcopy(_BASE)
CONFIG["server_args"]["max_total_tokens"] = 262144
CONFIG["runtime_env"]["TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE"] = None
