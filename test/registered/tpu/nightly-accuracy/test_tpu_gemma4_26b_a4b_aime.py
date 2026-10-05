"""Gemma-4-26B-A4B TPU AIME 2026 nightly score, mean of four passes."""

import unittest

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import GEMMA4_26B_A4B, make_model_config, make_runner
from sglang.test.tpu.utils.accuracy_base import AccuracyTestCase, gate_server_failures

register_tpu_ci(
    est_time=7200,
    suite="nightly-tpu-accuracy-gemma4-26b-a4b-aime",
    nightly=True,
    labels=["singlehost", "gemma4-26b-a4b", "accuracy"],
)

CONFIG_MODULE = "recipe/configs/gemma4_26b_a4b/aime.py"
MODEL_PATH = GEMMA4_26B_A4B["served_model_name"]
# 64k thinking budget: at 32k, 4% of the answers hit the cap and scored 0.
AIME_MODEL = {
    **GEMMA4_26B_A4B,
    "sampling": {**GEMMA4_26B_A4B["sampling"], "max_tokens": 65536, "num_threads": 32},
}

GATE = {
    "dataset": "aime26",
    "label": "aime26",
    "runner": make_runner(AIME_MODEL, "aime26", n_repeats=4),
    "report_only": True,
}


def build_model_config():
    """Assemble this benchmark's gate against the model's shared identity."""
    return make_model_config(
        AIME_MODEL,
        test_name="gemma4-26b-a4b-aime",
        series_id="accuracy-gemma4-26b-a4b-aime",
        gate=GATE,
    )


class TestGemma26BA4BAIME(AccuracyTestCase):
    """Score Gemma-4-26B-A4B aime26 on the server its recipe starts."""

    config_module = CONFIG_MODULE
    model_path = "/weights/gemma-4-26B-A4B-it"
    tag = "gemma4-26b-a4b-aime-e2e"

    @classmethod
    def build_model_config(cls):
        return build_model_config()

    def test_accuracy(self):
        self.run_evaluations()


class TestGemma26BA4BAIMEConfiguration(unittest.TestCase):
    """Host-side configuration tests, runnable without TPU hardware."""

    def test_gate_fits_the_server_its_recipe_starts(self):
        self.assertEqual(
            gate_server_failures(CONFIG_MODULE, AIME_MODEL["sampling"]), []
        )


if __name__ == "__main__":
    unittest.main()
