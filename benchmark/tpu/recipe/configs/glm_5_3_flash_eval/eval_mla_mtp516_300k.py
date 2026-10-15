from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash_eval.eval_mla_mtp516 import CONFIG as _BASE

# 300k-token budget: the hardest AIME chains ran past 60k (no repetition, still verifying).
# MLA bf16 latent: 10 slots x 300k x 11 KiB ~ 33 GB per rank.
CONTEXT_LENGTH = 300_000
MAX_RUNNING = 10

CONFIG = deepcopy(_BASE)
CONFIG["server_args"].update(
    {"context_length": CONTEXT_LENGTH, "max_total_tokens": MAX_RUNNING * CONTEXT_LENGTH}
)
CONFIG["runtime_env"]["SGLANG_TPU_GLM_FLASH_DENSE_MAX_CONTEXT"] = str(CONTEXT_LENGTH)
