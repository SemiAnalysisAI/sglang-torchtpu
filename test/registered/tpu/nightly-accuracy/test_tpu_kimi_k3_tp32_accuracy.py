"""Kimi-K3 TP32/EP32 evaluation on one 2x2x4 slice, bf16 KV.

GSM8K (5-shot completion, greedy, full test set), then one pass of GPQA Diamond
and 250 MMLU questions in thinking mode with the model card's sampling. One file
for the three benchmarks: each file is one JobSet, and the 4-host bring-up
(~1 h: 40 min weight load, 13 min compile) would otherwise run three times on
the single slice.
"""

import unittest

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu.utils.accuracy_base import (
    AccuracyTestCase,
    gate_server_failures,
    live_server_only,
)

register_tpu_ci(
    est_time=10800,
    suite="nightly-tpu-multihost-accuracy",
    nightly=True,
    labels=["multihost", "kimi-k3", "accuracy"],
)

MULTIHOST_LAUNCH = {
    "hosts": 4,
    "launch_script": "recipe/configs/kimi_k3/accuracy.py",
    "weights_pvc": "model-weights",
    # Which accuracy-test-tpu.yml lane runs this file.
    "schedule": "nightly",
    # The three evals are estimated at 1-2 h of TP32 decode.
    "timeout_per_file": 10800,
    # A TP32 bring-up measured 57 min (40 min weight load, 13 min compile) with
    # the checkpoint on NFS: past the default 3000 s.
    "bringup_timeout": 5400,
}

CONFIG_MODULE = "recipe/configs/kimi_k3/accuracy.py"
# The recipe's served_model_name, which requests carry; the dashboard labels the
# series with dashboard_model below.
SERVED_MODEL_NAME = "moonshotai/Kimi-K3"
# The model card's sampling for single-step tasks. Kimi-K3 always thinks; the
# card's scores use reasoning_effort "max", "high" bounds the trace for CI.
CARD_SAMPLING = {
    "thinking": True,
    "reasoning_effort": "high",
    "temperature": 1.0,
    "top_p": 0.95,
}

GATES = [
    {
        "dataset": "gsm8k",
        "label": "gsm8k_full",
        "baseline": 0.959,  # measured on TPU
        "atol": 0.1,
        # The raw 5-shot prompt the GPU Kimi-K3 gate uses: no chat template, so
        # it isolates the backbone from the reasoning parser.
        "api": "completion",
        # Every question after the 5 few-shot examples (the eval caps 1319 at the
        # 1314 left, as for gpt-oss-120b).
        "num_examples": 1319,
        "num_threads": 64,
        "max_tokens": 512,
    },
    {
        "dataset": "gpqa",
        "label": "gpqa_diamond",
        # 198 questions x 1 pass.
        "baseline": 0.914,  # measured on TPU
        "atol": 0.1,
        "repeat": 1,
        # The KV pool holds ~550k tokens: 32 concurrent thinking traces fit while
        # they average under ~17k tokens; past that, and at 64, requests retract
        # and recompute.
        "num_threads": 32,
        "max_tokens": 32768,
        **CARD_SAMPLING,
    },
    {
        "dataset": "mmlu",
        "label": "mmlu",
        "baseline": 0.924,  # measured on TPU
        "atol": 0.1,
        "repeat": 1,
        # The same seeded 250 rows every model and every night.
        "num_examples": 250,
        "num_threads": 64,
        "max_tokens": 16384,
        **CARD_SAMPLING,
    },
]


def build_model_config():
    """Assemble this file's gates against the model's identity."""
    return {
        "id": "accuracy-kimi-k3",
        "display_name": "Kimi K3",
        "dashboard": "accuracy",
        "dashboard_model": "kimi-k3",
        "model": SERVED_MODEL_NAME,
        "test_name": "Kimi-K3-multihost",
        "variant": "4host-TP32+EP32",
        "gates": GATES,
    }


@live_server_only
class TestKimiK3Accuracy(AccuracyTestCase):
    """Score this file's gates on the server the JobSet started."""

    config_module = CONFIG_MODULE
    tag = "kimi-k3-e2e"

    @classmethod
    def build_model_config(cls):
        return build_model_config()

    def test_accuracy(self):
        self.run_evaluations()


class TestKimiK3AccuracyConfiguration(unittest.TestCase):
    """Host-side configuration tests, runnable without TPU hardware."""

    def test_the_jobset_launches_this_files_recipe(self):
        self.assertEqual(MULTIHOST_LAUNCH["launch_script"], CONFIG_MODULE)

    def test_every_gate_fits_the_server_its_recipe_starts(self):
        for gate in GATES:
            with self.subTest(gate=gate["label"]):
                self.assertEqual(gate_server_failures(CONFIG_MODULE, gate), [])


if __name__ == "__main__":
    unittest.main()
