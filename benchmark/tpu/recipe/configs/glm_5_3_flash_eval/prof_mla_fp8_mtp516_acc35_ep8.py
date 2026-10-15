from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash_eval.prof_mla_fp8_mtp516_acc35 import CONFIG as _BASE

# The accept-3.5 bench with expert-parallel MoE (36 experts per rank).
CONFIG = deepcopy(_BASE)
CONFIG["server_args"]["ep_size"] = 8
CONFIG["runtime_env"].pop("SGLANG_TPU_FUSED_MOE_STREAM_BS1", None)
