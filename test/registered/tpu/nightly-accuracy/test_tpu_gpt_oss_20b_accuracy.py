"""GPT-OSS-20B nightly GPQA, AIME25 and 250-question MMLU, one pass each."""

import os

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main
from sglang.test.tpu.server_test_suite import resolve_tpu_model_path

register_tpu_ci(
    est_time=7200,
    suite="nightly-tpu-accuracy-gpt-oss-20b",
    nightly=True,
    labels=["singlehost", "gpt-oss-20b", "accuracy"],
)

MODEL_CONFIG = {
    "model": "gpt-oss-20b",
    "test_name": "gpt-oss-20b-accuracy",
    "id": "accuracy-gpt-oss-20b",
    "display_name": "GPT-OSS 20B",
    "variant": "TP8+EP8",
    "dashboard": "accuracy",
    "dashboard_model": "gpt-oss-20b",
    "launch": {
        "recipe": "recipe/gpt_oss_120b/server.sh",
        "health_timeout": 3000,
        "tag": "gpt-oss-20b-accuracy",
        "env": {
            "MODEL": resolve_tpu_model_path("openai/gpt-oss-20b"),
            "EXTRA_ARGS": "--reasoning-parser gpt-oss --served-model-name gpt-oss-20b",
            "KV_CACHE_DTYPE": "",
            "PAGE_SIZE": "128",
            "DISABLE_RADIX_CACHE": "1",
            "MEM_FRACTION_STATIC": "0.60",
            "ISL": "4096",
            "OSL": "65536",
            "CONCUR": "16",
            "CONTEXT_LEN": "69632",
            "MAX_TOTAL_TOKENS": "1114112",
            "TPU_COMPILE_TOKENS": "4096",
            "TPU_COMPILE_BS": "1 2 4 8 16",
            "ENABLE_PROFILER_ENDPOINT": "0",
        },
    },
    "gates": [
        {
            "dataset": "gpqa",
            "label": "gpqa_diamond_high",
            "report_only": True,
            "repeat": 1,
            "num_threads": 16,
            "reasoning_effort": "high",
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": 65536,
        },
        {
            "dataset": "aime25",
            "label": "aime25_high",
            "report_only": True,
            "repeat": 1,
            "num_threads": 16,
            "reasoning_effort": "high",
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": 65536,
        },
        {
            "dataset": "mmlu",
            "label": "mmlu_high",
            "report_only": True,
            "repeat": 1,
            "num_examples": 250,
            "num_threads": 16,
            "reasoning_effort": "high",
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": 65536,
        },
    ],
}

if __name__ == "__main__":
    model_config_main(
        MODEL_CONFIG, base_url=os.environ.get("BASE_URL", "http://127.0.0.1:30000")
    )
