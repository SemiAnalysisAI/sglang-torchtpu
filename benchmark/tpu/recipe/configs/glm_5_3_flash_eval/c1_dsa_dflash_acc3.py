from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash.dflash import CONFIG as _BASE

# 8k1k c1 DFlash on the recipe's DSA at simulated acceptance 3. Batch size 1 only: the
# cookbook's [1, 16] compile set pushes a cold boot's host memory past the slice's guard.
CONFIG = deepcopy(_BASE)
CONFIG["server_args"]["tpu_compile_tokens"] = [8192]
CONFIG["runtime_env"].update(
    {"SGLANG_SIMULATE_ACC_LEN": "3", "SGLANG_SIMULATE_ACC_METHOD": "match-expected"}
)
# Compile cache on shared storage only: the in-memory tier 2 sits in /dev/shm (pod RAM).
CONFIG["runtime_env"]["TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE"] = None
assert CONFIG["server_args"]["tpu_compile_bs"] == [1]
assert "SGLANG_TPU_GLM_FLASH_DENSE_MAX_CONTEXT" not in CONFIG["runtime_env"]
