"""Unit tests for Python recipe loading, overrides, and CLI flag conversion."""

import json
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "benchmark" / "tpu"
sys.path.insert(0, str(ROOT))

from benchmark.tpu.recipe.configs import config_eval


class TestTpuConfigEval(unittest.TestCase):
    def test_only_python_recipes_are_accepted(self):
        for suffix in (".json", ".jsonnet", ".txt"):
            with (
                self.subTest(suffix=suffix),
                self.assertRaisesRegex(ValueError, "must end in .py"),
            ):
                config_eval.eval_config("config" + suffix)

    def test_example_config(self):
        config = config_eval.eval_config(BENCH / "recipe/configs/qwen3_0_6b.py")
        self.assertEqual(config["server_args"]["context_length"], 128)
        self.assertEqual(config["server_args"]["max_running_requests"], 1)
        self.assertEqual(config["runtime_env"]["TPU_VISIBLE_CHIPS"], "0")

    def test_python_overrides_do_not_modify_base_or_later_loads(self):
        recipe = BENCH / "recipe/configs/qwen3_0_6b.py"
        first = config_eval.eval_config(recipe)
        first["server_args"]["port"] = 9999
        second = config_eval.eval_config(recipe)
        base = sys.modules["benchmark.tpu.recipe.configs.base"].CONFIG
        self.assertEqual(second["server_args"]["port"], 30000)
        self.assertNotIn("model_path", base["server_args"])
        self.assertEqual(base["runtime_env"], {"SGLANG_PLATFORM": "tpu"})

    def test_python_recipe_requires_config_object(self):
        with tempfile.TemporaryDirectory() as tmp:
            recipe = Path(tmp) / "config.py"
            for source in (
                "value = 1",
                "CONFIG = []",
                'CONFIG = {"server_args": []}',
                'CONFIG = {"runtime_env": []}',
                'CONFIG = {"unknown": {}}',
            ):
                with self.subTest(source=source):
                    recipe.write_text(source)
                    with self.assertRaises(ValueError):
                        config_eval.eval_config(recipe)

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
            args[:-1],
            [
                "--model-path=a b; $(not-a-command)",
                "--port=30000",
                "--enable-torch-compile",
                "--cuda-graph-bs",
                "1",
                "4",
                "8",
            ],
        )
        flag, value = args[-1].split("=", 1)
        self.assertEqual(flag, "--json-model-override-args")
        self.assertEqual(json.loads(value), settings["json_model_override_args"])
        self.assertEqual(settings, original)

    def test_config_overrides_before_flag_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            recipe = Path(tmp) / "config.py"
            recipe.write_text(
                'CONFIG = {"server_args": {"port": 30000, "context_length": 128}, '
                '"runtime_env": {"EXAMPLE_SETTING": "original"}}\n'
                'CONFIG["server_args"].update(port=30001, context_length=256)\n'
                'CONFIG["runtime_env"]["EXAMPLE_SETTING"] = "overridden"\n'
            )
            config = config_eval.eval_config(recipe)
        self.assertEqual(
            config_eval.server_args_to_cli(config["server_args"]),
            ["--port=30001", "--context-length=256"],
        )
        self.assertEqual(config["runtime_env"], {"EXAMPLE_SETTING": "overridden"})


if __name__ == "__main__":
    unittest.main()
