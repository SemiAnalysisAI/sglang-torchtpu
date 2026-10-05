"""Qwen3.5-397B-A17B-FP8 TPU GPQA Diamond score, four passes, on one host."""

import unittest

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import QWEN3_5_397B, make_model_config, make_runner
from sglang.test.tpu.utils.accuracy_base import AccuracyTestCase, gate_server_failures

register_tpu_ci(
    est_time=9000,
    suite="nightly-tpu-accuracy-qwen3-5-397b-gpqa",
    nightly=True,
    labels=["singlehost", "qwen3.5-397b", "accuracy"],
)

CONFIG_MODULE = QWEN3_5_397B["config_module"]

GATE = {
    "dataset": "gpqa",
    "label": "gpqa_diamond",
    "runner": make_runner(QWEN3_5_397B, "gpqa", n_repeats=4),
    "report_only": True,
}


def build_model_config():
    """Assemble this benchmark's gate against the model's shared identity."""
    return make_model_config(
        QWEN3_5_397B,
        test_name="qwen3-5-397b-gpqa",
        series_id="accuracy-qwen3-5-397b-gpqa",
        gate=GATE,
    )


class TestQwen35GPQA(AccuracyTestCase):
    """Score Qwen3.5-397B-FP8 gpqa_diamond on the server its recipe starts."""

    config_module = CONFIG_MODULE
    model_path = "/weights/Qwen3.5-397B-A17B-FP8"
    # One host at tp8 precompiles bigger per-core shapes than two at tp16:
    # the FP8 bring-up outlived the 1800 s default (run 37245212756).
    health_timeout = 3600
    tag = "qwen3-5-gpqa-e2e"

    @classmethod
    def build_model_config(cls):
        return build_model_config()

    def test_accuracy(self):
        self.run_evaluations()


class TestQwen35GPQAConfiguration(unittest.TestCase):
    """Host-side configuration tests, runnable without TPU hardware."""

    def test_gate_fits_the_server_its_recipe_starts(self):
        self.assertEqual(
            gate_server_failures(CONFIG_MODULE, QWEN3_5_397B["sampling"]), []
        )


if __name__ == "__main__":
    unittest.main()
