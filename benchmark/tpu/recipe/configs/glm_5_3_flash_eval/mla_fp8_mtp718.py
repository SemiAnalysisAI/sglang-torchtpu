from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash_eval.mla_fp8_mtp516 import CONFIG as _BASE

# More draft steps: each fused step is ~0.28 ms, verify grows by 2 tokens.
CONFIG = deepcopy(_BASE)
CONFIG["server_args"].update({"speculative_num_steps": 7, "speculative_num_draft_tokens": 8})
