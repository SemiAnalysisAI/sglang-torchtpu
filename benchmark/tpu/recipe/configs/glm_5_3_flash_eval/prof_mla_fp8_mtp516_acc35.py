from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash_eval.mla_fp8_mtp516 import CONFIG as _BASE

# c1 at a simulated acceptance of 3.5 (3 or 4 per round, 50/50). The cookbook
# bench sends exactly 8192-token prompts, so only that prefill bucket compiles.
CONFIG = deepcopy(_BASE)
CONFIG["server_args"]["tpu_compile_tokens"] = [8192]
CONFIG["runtime_env"].update(
    {"SGLANG_SIMULATE_ACC_LEN": "3.5", "SGLANG_SIMULATE_ACC_METHOD": "match-expected"}
)
