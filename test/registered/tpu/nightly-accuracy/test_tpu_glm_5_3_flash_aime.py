"""GLM-5.3-Flash TPU AIME 2026 nightly score, thinking on, mean of four passes."""

import unittest

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import GLM_5_3_FLASH, make_model_config, make_runner
from sglang.test.tpu.utils.accuracy_base import AccuracyTestCase, gate_server_failures

register_tpu_ci(
    est_time=10800,
    suite="nightly-tpu-accuracy-glm-5-3-flash-aime",
    nightly=True,
    labels=["singlehost", "glm-5.3-flash", "accuracy"],
)

CONFIG_MODULE = GLM_5_3_FLASH["config_module"]

GATE = {
    "dataset": "aime26",
    "label": "aime26",
    # Provisional: one dense-attention pass at 60k (15/30); replace with this
    # suite's first CI score.
    "baseline": 0.50,
    "atol": 0.12,
    "runner": make_runner(GLM_5_3_FLASH, "aime26", n_repeats=4),
}


def build_model_config():
    """Assemble this benchmark's gate against the model's shared identity."""
    return make_model_config(
        GLM_5_3_FLASH,
        test_name="glm-5-3-flash-aime",
        # The dashboard's series join key: never renamed, never shown.
        series_id="accuracy-glm-5-3-flash-aime",
        gate=GATE,
    )


class TestGlm53FlashAIME(AccuracyTestCase):
    """Score GLM-5.3-Flash aime26 on the server its recipe starts."""

    config_module = CONFIG_MODULE
    model_path = "/weights/GLM-5.3-Flash"
    tag = "glm-5-3-flash-aime-e2e"
    # Weight load alone measured 1113 s in CI, then 9 decode + 4 prefill graphs compile.
    health_timeout = 3600

    @classmethod
    def build_model_config(cls):
        return build_model_config()

    def test_accuracy(self):
        self.run_evaluations()


class TestGlm53FlashAIMEConfiguration(unittest.TestCase):
    """Host-side configuration tests, runnable without TPU hardware."""

    def test_gate_fits_the_server_its_recipe_starts(self):
        self.assertEqual(
            gate_server_failures(CONFIG_MODULE, GLM_5_3_FLASH["sampling"]), []
        )


if __name__ == "__main__":
    unittest.main()
