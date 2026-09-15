"""Unit tests for Python recipe dictionaries and CLI flag conversion."""

import json
import sys
import unittest
from copy import deepcopy
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from benchmark.tpu.recipe.configs import config_eval
from benchmark.tpu.recipe.configs.base import CONFIG as BASE_CONFIG
from benchmark.tpu.recipe.configs.qwen3_0_6b import CONFIG


class TestTpuConfigEval(unittest.TestCase):
    def test_example_config(self):
        self.assertEqual(CONFIG["server_args"]["context_length"], 128)
        self.assertEqual(CONFIG["server_args"]["max_running_requests"], 1)
        self.assertEqual(CONFIG["runtime_env"]["TPU_VISIBLE_CHIPS"], "0")
        self.assertNotIn("model_path", BASE_CONFIG["server_args"])
        self.assertEqual(BASE_CONFIG["runtime_env"], {"SGLANG_PLATFORM": "tpu"})

    def test_config_overrides_do_not_modify_shared_defaults(self):
        original = deepcopy(CONFIG)
        original_base = deepcopy(BASE_CONFIG)
        config = deepcopy(CONFIG)
        config["server_args"].update(port=30001, context_length=256)
        config["server_args"]["tpu_compile_bs"].append(4)
        config["runtime_env"]["TPU_VISIBLE_CHIPS"] = "1"
        args = config_eval.server_args_to_cli(config["server_args"])
        self.assertEqual(args[args.index("--port") + 1], "30001")
        self.assertEqual(args[args.index("--context-length") + 1], "256")
        start = args.index("--tpu-compile-bs")
        self.assertEqual(args[start : start + 3], ["--tpu-compile-bs", "1", "4"])
        self.assertEqual(CONFIG, original)
        self.assertEqual(BASE_CONFIG, original_base)

    def test_cli_value_types(self):
        settings = {
            "model_path": "a b; $(not-a-command)",
            "port": 30000,
            "enable_torch_compile": True,
            "disable_radix_cache": False,
            "quantization": None,
            "cuda_graph_bs": [1, 4, 8],
            "json_model_override_args": {"rope_scaling": {"factor": 2}},
        }
        original = deepcopy(settings)
        args = config_eval.server_args_to_cli(settings)
        self.assertEqual(
            args[:-2],
            [
                "--model-path",
                "a b; $(not-a-command)",
                "--port",
                "30000",
                "--enable-torch-compile",
                "--cuda-graph-bs",
                "1",
                "4",
                "8",
            ],
        )
        flag, value = args[-2:]
        self.assertEqual(flag, "--json-model-override-args")
        self.assertEqual(json.loads(value), settings["json_model_override_args"])
        self.assertEqual(settings, original)

    def test_flags_are_visible_to_existing_ci_argument_checks(self):
        args = config_eval.server_args_to_cli(
            {"quantization": "modelopt_fp8", "lora_paths": {"adapter": "/tmp/adapter"}}
        )
        self.assertEqual(args[:2], ["--quantization", "modelopt_fp8"])
        self.assertEqual(args[2], "--lora-paths")
        self.assertEqual(json.loads(args[3]), {"adapter": "/tmp/adapter"})


if __name__ == "__main__":
    unittest.main()
