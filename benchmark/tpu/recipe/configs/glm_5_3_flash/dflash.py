"""GLM-5.3-Flash with DFlash2 speculative decoding (block size 8), one tpu7x host.

Sample usage (from the repo root):
    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/glm_5_3_flash/dflash.py

Acceptance depends on the text, so compare configurations on the same prompt
set. (Earlier TPOT figures were measured with dense attention and no longer
apply.)
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash.server import CONFIG as BASE_CONFIG

CONFIG = deepcopy(BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "speculative_algorithm": "DFLASH",
        "speculative_draft_model_path": "incoai/GLM-5.3-Flash-DFlash2",
        "speculative_dflash_block_size": 8,
        # The DFlash draft is a plain MHA model with its own KV cache.
        "speculative_draft_attention_backend": "pallas_tpu_rpa_v3",
    }
)
CONFIG["runtime_env"].update(
    {
        # Keeps the draft's sampled tokens identical across TP ranks.
        "SGLANG_SPEC_TP_SYNC": "rng",
    }
)
