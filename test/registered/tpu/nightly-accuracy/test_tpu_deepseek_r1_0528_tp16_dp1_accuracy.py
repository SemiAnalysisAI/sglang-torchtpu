"""Original DeepSeek-R1-0528 TP16/EP16 FP8-KV evaluation, no DP attention.

One pass each of GPQA, AIME25 and 250 MMLU questions at concurrency 128.
"""

import os

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST
from sglang.test.tpu import model_config_main

register_tpu_ci(
    est_time=7200,
    suite="nightly-tpu-multihost-accuracy",
    nightly=True,
    labels=["multihost", "deepseek-r1", "accuracy"],
)

MULTIHOST_LAUNCH = {
    "hosts": 2,
    "launch_script": "recipe/configs/deepseek_r1_0528_fp8kv_mh.py",
    "weights_pvc": "model-weights-rox",
    "timeout_per_file": 10800,
}

BASE_URL = os.environ.get("BASE_URL", DEFAULT_URL_FOR_TEST)
MODEL_CONFIG = {
    "id": "accuracy-deepseek-r1-0528",
    "display_name": "DeepSeek R1-0528",
    "dashboard": "accuracy",
    "dashboard_model": "deepseek-r1-0528",
    "model": os.environ.get("MODEL", "/weights/DeepSeek-R1-0528"),
    "test_name": "DeepSeek-R1-0528-multihost-fp8kv",
    "variant": "2host-TP16+EP16+DP1-FP8KV",
    "launch": {
        "external": True,
        "health_timeout": float(os.environ.get("HEALTH_TIMEOUT", "3000")),
        "tag": "r1-fp8kv-e2e",
    },
    "skip_unless": (
        os.environ.get("SGLANG_TPU_USE_LIVE_SERVER") == "1",
        "multi-host live-server not provided (the JobSet launches the server)",
    ),
    "gates": [
        {
            "dataset": "gpqa",
            "label": "gpqa_diamond",
            "baseline": 0.715,
            "atol": 0.10,
            "repeat": 1,
            "num_threads": 128,
            "temperature": 0.6,
            "top_p": 0.95,
            "max_tokens": 32768,
        },
        {
            "dataset": "aime25",
            "label": "aime25",
            # Same broad serving-regression threshold as the non-DPA test.
            "baseline": 0.875,
            "atol": 0.20,
            "repeat": 1,
            "num_threads": 128,
            "temperature": 0.6,
            "top_p": 0.95,
            "max_tokens": 32768,
        },
        {
            "dataset": "mmlu",
            "label": "mmlu",
            "baseline": 0.90,
            "atol": 0.08,
            "repeat": 1,
            # Match the non-DPA test: 250 sampled questions per repeat.
            "num_examples": 250,
            "num_threads": 128,
            "temperature": 0.6,
            "top_p": 0.95,
            "max_tokens": 16384,
        },
    ],
}


if __name__ == "__main__":
    model_config_main(MODEL_CONFIG, base_url=BASE_URL)
