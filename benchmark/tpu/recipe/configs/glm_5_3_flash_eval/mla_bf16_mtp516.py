from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash.mtp_s5 import CONFIG as _BASE
from benchmark.tpu.recipe.configs.glm_5_3_flash_eval.mla import apply

# bf16 latent: separates the fp8-KV effect on acceptance from the MLA path.
CONFIG = apply(deepcopy(_BASE), kv_dtype="bfloat16")
