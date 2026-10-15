from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash_eval.cb_mla_mtp516 import CONFIG as _BASE

CONFIG = deepcopy(_BASE)
CONFIG["server_args"]["ep_size"] = 8
CONFIG["runtime_env"].pop("SGLANG_TPU_FUSED_MOE_STREAM_BS1", None)
