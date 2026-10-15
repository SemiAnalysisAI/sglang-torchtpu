from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash_eval.dsa_nospec import CONFIG as _BASE

# Debug: no-spec DSA without torch.compile, so a device fault surfaces at its op.
CONFIG = deepcopy(_BASE)
CONFIG["server_args"]["enable_torch_compile"] = False
