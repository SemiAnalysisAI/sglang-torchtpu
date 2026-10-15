from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash.dflash import CONFIG as _BASE

# GLM-5.3-Flash with its own sparse attention (kpool DSA) on pallas_tpu_mla_sparse.
# The split MLA pool stores FP8 latents; the DFlash draft (MHA) stays on RPA v3.
CONFIG = deepcopy(_BASE)
CONFIG["server_args"].update(
    {
        "attention_backend": "pallas_tpu_mla_sparse",
        "kv_cache_dtype": "fp8_e4m3",
        "speculative_draft_attention_backend": "pallas_tpu_rpa_v3",
    }
)
CONFIG["runtime_env"].pop("SGLANG_TPU_GLM_FLASH_DENSE_MAX_CONTEXT", None)
