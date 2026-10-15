from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash.mtp_s5 import CONFIG as _BASE

# 8k1k c1, MTP 5/1/6 at a simulated acceptance of 3.5, on the recipe's own DSA
# (pallas_tpu_mla_sparse, FP8 latent). Same knobs as prof_mla_fp8_mtp516_acc35.py
# otherwise: the cookbook bench sends exactly 8192-token prompts.
CONFIG = deepcopy(_BASE)
CONFIG["server_args"]["tpu_compile_tokens"] = [8192]
CONFIG["runtime_env"].update(
    {"SGLANG_SIMULATE_ACC_LEN": "3.5", "SGLANG_SIMULATE_ACC_METHOD": "match-expected"}
)
# Compile cache on shared storage only: the in-memory tier 2 sits in /dev/shm (pod RAM).
CONFIG["runtime_env"]["TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE"] = None
assert "SGLANG_TPU_GLM_FLASH_DENSE_MAX_CONTEXT" not in CONFIG["runtime_env"]
