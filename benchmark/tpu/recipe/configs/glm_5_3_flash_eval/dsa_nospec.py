from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash.server import CONFIG as _BASE

# Debug: the base recipe's DSA without speculative decoding, 8192-token prefill bucket only.
CONFIG = deepcopy(_BASE)
CONFIG["server_args"]["tpu_compile_tokens"] = [8192]
assert "SGLANG_TPU_GLM_FLASH_DENSE_MAX_CONTEXT" not in CONFIG["runtime_env"]
