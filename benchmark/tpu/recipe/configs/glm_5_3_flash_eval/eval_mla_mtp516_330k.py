from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash_eval.eval_mla_mtp516 import CONFIG as _BASE

# Rerun of the AIME problems truncated at 290k: 320k-token budget plus the prompt.
CONTEXT_LENGTH = 330_000
MAX_RUNNING = 10

CONFIG = deepcopy(_BASE)
CONFIG["server_args"].update(
    {"context_length": CONTEXT_LENGTH, "max_total_tokens": MAX_RUNNING * CONTEXT_LENGTH}
)
CONFIG["runtime_env"]["SGLANG_TPU_GLM_FLASH_DENSE_MAX_CONTEXT"] = str(CONTEXT_LENGTH)
