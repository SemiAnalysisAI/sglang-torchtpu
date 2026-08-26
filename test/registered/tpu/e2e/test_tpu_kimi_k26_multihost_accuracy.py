"""Kimi-K2.6 (1T MoE, int4 W4A16, MoonViT VL) multi-host e2e accuracy.

The JobSet launches the server (recipe/kimi_k26_mh.sh, 2 hosts, tp=16); this
file attaches to it, under the K2.6 card protocol: temperature 1.0, top_p 1.0,
context 262144, and a 98304-token generation budget for the reasoning gates. Those are not the usual
house defaults -- the card states them explicitly, and at 4096 tokens GPQA
scores at chance because thinking is truncated mid-chain.

Pass counts are set by wall clock, not by taste. Measured on a 2-host v7x slice
(2026-08-26): under the card protocol Kimi clears 0.64 GPQA questions/minute at
32 concurrent, so a single 198-question pass is ~2.6h at the recipe's cap of 64.
The reasoning gates therefore run once; only MMLU, which is cheap, repeats. That
makes the floors wide on purpose -- the job's role is "serves the card protocol
end-to-end and lands in the right accuracy band", not a tight regression bound.

Baselines:
  gpqa    0.905  card GPQA-Diamond, pass@1
  aime25  0.950  the card reports AIME 2026 (0.964), not AIME'25; this is a
                 house estimate, and over one 30-question pass a single flipped
                 answer moves the score 3.3pp
  mmlu    0.900  the card does not report MMLU; house default, and the same
                 floor the R1 nightly uses

This is the text path only. MoonViT is untested and its precompile ladder is
disabled in the recipe; a vision gate (MMMU-Pro, card 0.794) is the follow-up.

The whole file runs under one `timeout` in jobset-tpu.yaml, so MULTIHOST_LAUNCH
declares the budget the six passes need (discovery takes the max across the
group; see discover_multihost.py). Kimi decodes slower than R1 and its
generation budget is 3x larger, hence the wider cap: ~2.6h for GPQA, ~0.5h for
AIME, ~1.5h for two MMLU passes, plus bring-up.
"""

import os

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST
from sglang.test.tpu import model_config_main

# est_time is the expected wall clock (bookkeeping); the enforced per-file
# cap is MULTIHOST_LAUNCH.timeout_per_file below.
register_tpu_ci(
    est_time=18000,
    suite="nightly-tpu-multihost",
    nightly=True,
)

MULTIHOST_LAUNCH = {
    "hosts": 2,
    "launch_script": "recipe/kimi_k26_mh.sh",
    "weights_pvc": "model-weights-rox",
    "timeout_per_file": 21600,
}

MODEL_PATH = os.environ.get("MODEL", "/weights/Kimi-K2.6")
BASE_URL = os.environ.get("BASE_URL", DEFAULT_URL_FOR_TEST)
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "3000"))

# Card protocol, section "Evaluation Details".
CARD_TEMPERATURE = 1.0
CARD_TOP_P = 1.0
REASONING_MAX_TOKENS = 98304

MODEL_CONFIG = {
    "model": MODEL_PATH,
    "test_name": "Kimi-K2.6-multihost",
    "variant": "2host-TP16+EP16-int4",
    "launch": {"external": True, "health_timeout": HEALTH_TIMEOUT, "tag": "k26-e2e"},
    "skip_unless": (
        os.environ.get("SGLANG_TPU_USE_LIVE_SERVER") == "1",
        "multi-host live-server not provided (the JobSet launches the server)",
    ),
    "gates": [
        {
            "dataset": "gpqa",
            "label": "gpqa_diamond",
            "baseline": 0.905,
            "atol": 0.15,
            "num_threads": 128,
            "temperature": CARD_TEMPERATURE,
            "top_p": CARD_TOP_P,
            "max_tokens": REASONING_MAX_TOKENS,
        },
        {
            "dataset": "aime25",
            "label": "aime25",
            "baseline": 0.950,
            "atol": 0.20,
            "num_threads": 128,
            "temperature": CARD_TEMPERATURE,
            "top_p": CARD_TOP_P,
            "max_tokens": REASONING_MAX_TOKENS,
        },
        {
            "dataset": "mmlu",
            "label": "mmlu",
            "baseline": 0.900,
            "atol": 0.10,
            "repeat": 2,
            "num_examples": 250,
            "num_threads": 128,
            "temperature": CARD_TEMPERATURE,
            "top_p": CARD_TOP_P,
            # Not a reasoning gate: MMLU is multiple-choice, so the card's
            # 98304-token budget only buys wall clock here.
            "max_tokens": 16384,
        },
    ],
}


if __name__ == "__main__":
    model_config_main(MODEL_CONFIG, base_url=BASE_URL)
