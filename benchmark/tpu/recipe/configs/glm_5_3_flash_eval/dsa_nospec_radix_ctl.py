from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash_eval.dsa_nospec_radix import CONFIG as _BASE

# Control for the offload smoke: device-only radix cache, big enough that all of the
# test's prompts stay on device (no host offload).
CONFIG = deepcopy(_BASE)
CONFIG["server_args"].update(
    context_length=16384, max_total_tokens=131072, max_mamba_cache_size=48, enable_cache_report=True
)
