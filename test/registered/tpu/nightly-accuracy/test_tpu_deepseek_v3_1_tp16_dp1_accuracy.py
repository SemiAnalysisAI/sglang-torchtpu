"""DeepSeek-V3.1 TP16/EP16 FP8-KV evaluation, no DP attention.

One AIME25 pass with FP32 MoE scales, 32K output, and concurrency 30.
"""

import os

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST
from sglang.test.tpu import model_config_main

register_tpu_ci(
    est_time=7200,
    suite="nightly-tpu-multihost-accuracy",
    nightly=True,
    labels=["multihost", "deepseek-v3-1", "accuracy"],
)

MULTIHOST_LAUNCH = {
    "hosts": 2,
    "launch_script": "recipe/configs/deepseek_v3_1_fp8kv_mh.py",
    "weights_pvc": "deepseek-v3-1-rox",
    "timeout_per_file": 10800,
}

BASE_URL = os.environ.get("BASE_URL", DEFAULT_URL_FOR_TEST)
MODEL_CONFIG = {
    "id": "accuracy-deepseek-v3-1",
    "display_name": "DeepSeek V3.1",
    "dashboard": "accuracy",
    "dashboard_model": "deepseek-v3-1",
    "model": os.environ.get("MODEL", "/weights/DeepSeek-V3.1"),
    "test_name": "DeepSeek-V3.1-multihost-fp8kv",
    "variant": "2host-TP16+EP16+DP1-FP8KV",
    "launch": {
        "external": True,
        "health_timeout": float(os.environ.get("HEALTH_TIMEOUT", "3000")),
        "tag": "v3-1-fp8kv-e2e",
    },
    "skip_unless": (
        os.environ.get("SGLANG_TPU_USE_LIVE_SERVER") == "1",
        "multi-host live-server not provided (the JobSet launches the server)",
    ),
    "gates": [
        {
            "dataset": "aime25",
            "label": "aime25_32k",
            "report_only": True,
            "thinking": True,
            "repeat": 1,
            "num_threads": 30,
            "temperature": 0.6,
            "top_p": 0.95,
            "max_tokens": 32768,
        },
    ],
}


if __name__ == "__main__":
    model_config_main(MODEL_CONFIG, base_url=BASE_URL)
