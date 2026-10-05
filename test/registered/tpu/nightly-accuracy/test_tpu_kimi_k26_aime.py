"""Kimi-K2.6 TPU AIME 2026 score, mean of four passes at the card's 98k budget,
on the MMMU recipe's two-host DPA4 server."""

import unittest

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import KIMI_K26, make_model_config, make_runner
from sglang.test.tpu.utils.accuracy_base import (
    AccuracyTestCase,
    live_server_only,
)

register_tpu_ci(
    est_time=14400,
    suite="nightly-tpu-multihost-accuracy",
    nightly=True,
    labels=["multihost", "kimi-k2.6", "accuracy"],
)

MULTIHOST_LAUNCH = {
    "hosts": 2,
    "launch_script": "recipe/configs/kimi_k26/mmmu.py",
    "weights_pvc": "model-weights",
    # Dispatch only (`model=kimi-k26` with `test_filter=aime`): no cron slot
    # has a second two-host pair free for it.
    "schedule": "kimi-text",
    "timeout_per_file": 14400,
}
CONFIG_MODULE = "recipe/configs/kimi_k26/mmmu.py"

GATE = {
    "dataset": "aime26",
    "label": "aime26",
    "runner": make_runner(KIMI_K26, "aime26", n_repeats=4),
    "report_only": True,
}


def build_model_config():
    """Assemble this benchmark's gate against the model's shared identity."""
    return make_model_config(
        KIMI_K26,
        test_name="kimi-k26-aime",
        series_id="accuracy-kimi-k26-aime",
        gate=GATE,
    )


@live_server_only
class TestKimiAIME(AccuracyTestCase):
    """Score this file's gate on the server the JobSet started."""

    config_module = CONFIG_MODULE
    tag = "kimi-aime-e2e"

    @classmethod
    def build_model_config(cls):
        return build_model_config()

    def test_accuracy(self):
        self.run_evaluations()


class TestKimiAIMEConfiguration(unittest.TestCase):
    """Host-side configuration tests, runnable without TPU hardware."""

    def test_the_jobset_launches_this_files_recipe(self):
        self.assertEqual(MULTIHOST_LAUNCH["launch_script"], CONFIG_MODULE)


if __name__ == "__main__":
    unittest.main()
