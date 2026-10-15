from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash_eval.cb_mla_dflash import CONFIG as _BASE

# c16 verify routes ~1k expert slots per layer: try expert parallel instead of
# TP-sharded experts. The streaming one-token MoE requires ep_size 1.
CONFIG = deepcopy(_BASE)
CONFIG["server_args"]["ep_size"] = 8
CONFIG["runtime_env"].pop("SGLANG_TPU_FUSED_MOE_STREAM_BS1", None)
