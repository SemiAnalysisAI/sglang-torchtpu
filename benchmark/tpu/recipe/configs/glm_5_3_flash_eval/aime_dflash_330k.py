from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash_eval.eval_mla_dflash import CONFIG as _BASE

# AIME26 at a 320k-token budget (+ prompt) on a 3.3M-token pool (bf16 MLA latent,
# ~11 KiB/token/rank: ~36 GB). Batched DFlash/MTP needs 369d483ac1 (verify kv_lens
# one sequence short in a bucket-padded batch).
CONTEXT_LENGTH = 330_000
# Concurrency 8: the 32-way compile set pushed boot memory past the slice's guard.
CONCURRENCY = 8
MAX_RUNNING = CONCURRENCY + 2

CONFIG = deepcopy(_BASE)
CONFIG["server_args"].update(
    {
        "context_length": CONTEXT_LENGTH,
        "max_running_requests": MAX_RUNNING,
        "torch_compile_max_bs": MAX_RUNNING,
        "tpu_compile_bs": [1, 2, 4, 8],
        "max_total_tokens": 10 * CONTEXT_LENGTH,
        "max_mamba_cache_size": MAX_RUNNING + 2,
    }
)
CONFIG["runtime_env"]["SGLANG_TPU_GLM_FLASH_DENSE_MAX_CONTEXT"] = str(CONTEXT_LENGTH)
# Accuracy run: keep full Dynamo guards and recompile on an unforeseen batch
# shape instead of raising (the frozen compiler crashed a batched MTP run).
CONFIG["runtime_env"]["SGLANG_TPU_ALLOW_SPEC_RECOMPILE"] = "1"
