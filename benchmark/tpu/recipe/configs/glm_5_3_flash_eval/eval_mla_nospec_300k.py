from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash_eval.eval_mla_nospec import CONFIG as _BASE

CONTEXT_LENGTH = 300_000
MAX_RUNNING = 10

CONFIG = deepcopy(_BASE)
CONFIG["server_args"].update(
    {"context_length": CONTEXT_LENGTH, "max_total_tokens": MAX_RUNNING * CONTEXT_LENGTH}
)
CONFIG["runtime_env"]["SGLANG_TPU_GLM_FLASH_DENSE_MAX_CONTEXT"] = str(CONTEXT_LENGTH)
