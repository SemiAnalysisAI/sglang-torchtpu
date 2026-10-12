# SPDX-License-Identifier: Apache-2.0
"""Single-host GPT-OSS-120B serving-time XLA recompilation gate (1x TPU v7-8)."""

import unittest

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.recompilation_gate import (
    assert_no_serving_recompilation,
    recompilation_gate_runner,
)
from sglang.test.tpu import (
    BASE_URL,
    GPT_OSS_120B,
    HEALTH_TIMEOUT,
    fixture_runtime_envs,
    run_model_config,
)
from sglang.test.tpu.utils.launcher_test_case import TPULauncherTestCase

register_tpu_ci(
    est_time=1800,
    suite="external-nightly-1host",
    nightly=True,
    labels=["singlehost", "gpt-oss-120b", "recompilation"],
)

GATES = [
    {
        "dataset": "recompilation_multi_shape",
        "label": "recompilation-multi-shape",
        "runner": recompilation_gate_runner(
            run_workload=True,
            prompt_word_counts=(8, 96, 320),
            batch_sizes=(1, 2, 4),
            max_new_tokens=16,
            use_baseline_diff=True,
        ),
        "fail_on_recompile": True,
    },
]

MODEL_CONFIG = {
    "model": GPT_OSS_120B["served_model_name"],
    "test_name": "gpt-oss-120b-singlehost-recompile",
    "variant": "TP4+EP4-singlehost-recompile",
    "gates": GATES,
}


class TestTpuGptOss120bSinglehostRecompile(TPULauncherTestCase):
    config_module = "recipe/configs/gpt_oss_120b/eval_singlehost.py"
    runtime_envs = {
        **fixture_runtime_envs(),
        "SGLANG_TPU_CHECK_RECOMPILATION": "1",
    }
    base_url = BASE_URL
    health_timeout = HEALTH_TIMEOUT
    health_endpoint = "/health_generate"
    tag = "gpt-oss-120b-singlehost-recompile"

    def test_no_serving_recompilation(self) -> None:
        run_model_config(MODEL_CONFIG, base_url=self.base_url)
        assert_no_serving_recompilation(self.base_url)


if __name__ == "__main__":
    unittest.main(verbosity=2)
