from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash_eval.prof_dsa_mtp516_acc35 import CONFIG as _BASE

# Debug: the 8k1k MTP+DSA config without torch.compile, so a device fault surfaces at
# the op that caused it instead of inside one fused graph.
CONFIG = deepcopy(_BASE)
CONFIG["server_args"]["enable_torch_compile"] = False
