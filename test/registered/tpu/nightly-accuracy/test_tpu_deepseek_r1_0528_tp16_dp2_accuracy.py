"""DeepSeek-R1-0528 DPA serving regression on one two-host TPU instance.

TP16/EP16/DP2 gives attention TP8 per group and a shared distributed MoE.
This is not two independent TP8 servers. Use concurrent GPQA, AIME25, and MMLU requests to
exercise DPA prefill, decode, and the shrinking-batch tail. The protocol and
broad accuracy floors match the existing non-DPA regression; this is
not the eight-repeat accuracy qualification or a DPA-vs-TP parity measurement.
"""

import os

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST
from sglang.test.tpu import model_config_main

register_tpu_ci(
    est_time=7200,
    suite="nightly-tpu-multihost-ondemand",
    nightly=True,
    labels=["multihost", "deepseek-r1", "dpa", "accuracy"],
)

MULTIHOST_LAUNCH = {
    "hosts": 2,
    "launch_script": "recipe/configs/deepseek_r1_0528_dpa_mh.py",
    "weights_pvc": "model-weights-rox",
    "timeout_per_file": 10800,
}

BASE_URL = os.environ.get("BASE_URL", DEFAULT_URL_FOR_TEST)
MODEL_CONFIG = {
    "id": "accuracy-deepseek-r1-0528-dpa",
    "display_name": "DeepSeek R1-0528 DPA",
    "dashboard": "accuracy",
    "dashboard_model": "deepseek-r1-0528-dpa",
    "model": os.environ.get("MODEL", "/weights/DeepSeek-R1-0528"),
    "test_name": "DeepSeek-R1-0528-multihost-dpa",
    "variant": "2host-TP16+EP16+DP2(attn-tp8)",
    "launch": {
        "external": True,
        "health_timeout": float(os.environ.get("HEALTH_TIMEOUT", "3000")),
        "tag": "r1-dpa-e2e",
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
