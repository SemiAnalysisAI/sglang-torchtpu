"""GLM-5.3-Flash MTP draft with 5 steps (verify 6), the GPU low-latency
cookbook's shape. Same as mtp.py otherwise.

Sample usage (from the repo root):
    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/glm_5_3_flash/mtp_s5.py
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash.mtp import CONFIG as MTP_CONFIG

NUM_STEPS = 5

CONFIG = deepcopy(MTP_CONFIG)
CONFIG["server_args"].update(
    {
        "speculative_num_steps": NUM_STEPS,
        "speculative_num_draft_tokens": NUM_STEPS + 1,
    }
)
