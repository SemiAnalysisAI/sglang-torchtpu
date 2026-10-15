from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash_eval.prof_dsa_mtp516_acc35 import CONFIG as _BASE

# Debug: the 8k1k MTP+DSA config with 4096-token prefill chunks (one 4096 bucket).
CONFIG = deepcopy(_BASE)
CONFIG["server_args"].update(
    {"tpu_compile_tokens": [4096], "chunked_prefill_size": 4096, "max_prefill_tokens": 4096}
)
