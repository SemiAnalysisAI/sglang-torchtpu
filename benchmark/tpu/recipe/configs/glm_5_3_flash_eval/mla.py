"""LEGACY, dense: absorbed-MLA overrides that move the target to pallas_tpu_mla (no
DSA). The recipes now run DSA on pallas_tpu_mla_sparse; use this only to compare
against the old dense numbers."""


def apply(config, kv_dtype="fp8_e4m3"):
    config["server_args"].update(
        {"attention_backend": "pallas_tpu_mla", "kv_cache_dtype": kv_dtype}
    )
    if config["server_args"].get("speculative_algorithm") == "DFLASH":
        # The DFlash draft model is plain MHA; keep it on the RPA kernel. The MTP
        # draft layer is MLA itself and shares pallas_tpu_mla.
        config["server_args"]["speculative_draft_attention_backend"] = "pallas_tpu_rpa_v3"
    config["runtime_env"]["SGLANG_TPU_GLM_FLASH_MLA"] = "1"
    return config
