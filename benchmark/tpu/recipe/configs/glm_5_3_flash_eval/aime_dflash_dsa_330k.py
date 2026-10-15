from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash_eval.aime_dflash_330k import CONFIG as _BASE

# AIME26 at a 320k budget with the model's own sparse attention (pooled DSA, top-2048
# on pallas_tpu_mla_sparse) instead of the dense escape hatch. FP8 split latent pool,
# as in dsa_dflash.py; the DFlash draft (MHA) stays on RPA v3.
CONFIG = deepcopy(_BASE)
CONFIG["server_args"].update(
    {
        "attention_backend": "pallas_tpu_mla_sparse",
        "kv_cache_dtype": "fp8_e4m3",
        "speculative_draft_attention_backend": "pallas_tpu_rpa_v3",
    }
)
CONFIG["runtime_env"].pop("SGLANG_TPU_GLM_FLASH_DENSE_MAX_CONTEXT", None)
