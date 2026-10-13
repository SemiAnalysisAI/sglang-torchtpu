"""GLM-5.3-Flash with its built-in MTP head (checkpoint layer 45) as a NEXTN
draft, one tpu7x host.

Sample usage (from the repo root):
    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/glm_5_3_flash/mtp.py

The draft is a single full-attention + MoE layer that shares the target's
embedding and lm_head; each draft step is one sequential forward, so
NUM_STEPS trades acceptance against per-round launches (mtp_s5.py: 5 steps).
TPU speculative decoding requires a top-1 chain.

To compare against the GPU cookbook at a fixed acceptance, prefix the launch
with SGLANG_SIMULATE_ACC_LEN=3.5 SGLANG_SIMULATE_ACC_METHOD=match-expected
(speed only: accepted tokens are then forced, so the text is meaningless).
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash.server import CONFIG as BASE_CONFIG

# Draft tokens per round; the target verifies NUM_STEPS + 1 tokens.
NUM_STEPS = 3

CONFIG = deepcopy(BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "speculative_algorithm": "NEXTN",
        # The MTP head lives in the target checkpoint.
        "speculative_draft_model_path": CONFIG["server_args"]["model_path"],
        "speculative_eagle_topk": 1,
        "speculative_num_steps": NUM_STEPS,
        "speculative_num_draft_tokens": NUM_STEPS + 1,
    }
)
CONFIG["runtime_env"].update({"SGLANG_SPEC_TP_SYNC": "rng"})
