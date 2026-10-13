"""GLM-5.3-Flash with DFlash2 speculative decoding (block size 8), one tpu7x host.

Sample usage (from the repo root):
    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/glm_5_3_flash/dflash.py

8k/1k c=1 TPOT on v7x TP8: 3.34 ms on chat prompts (~3.3 accepted tokens per
round) vs 7.3-8.1 ms without speculation. Acceptance depends on the text, so
compare configurations on the same prompt set.
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash.server import CONFIG as BASE_CONFIG

CONFIG = deepcopy(BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "speculative_algorithm": "DFLASH",
        "speculative_draft_model_path": "incoai/GLM-5.3-Flash-DFlash2",
        "speculative_dflash_block_size": 8,
    }
)
CONFIG["runtime_env"].update(
    {
        # Keeps the draft's sampled tokens identical across TP ranks.
        "SGLANG_SPEC_TP_SYNC": "rng",
    }
)
