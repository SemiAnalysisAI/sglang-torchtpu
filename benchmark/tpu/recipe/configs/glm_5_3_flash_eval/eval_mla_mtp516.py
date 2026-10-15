from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash.mtp_s5 import CONFIG as _BASE
from benchmark.tpu.recipe.configs.glm_5_3_flash_eval import long_ctx, mla

# Official MTP head (NEXTN 5/1/6) on the MLA bf16 latent; TPU verify does target-only
# sampling, so temperature / top_p are preserved.
CONFIG = long_ctx.apply(mla.apply(deepcopy(_BASE), kv_dtype="bfloat16"))
