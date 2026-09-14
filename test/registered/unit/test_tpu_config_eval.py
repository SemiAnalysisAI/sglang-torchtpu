"""Config evaluation and launcher process contract; no accelerator required."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "benchmark" / "tpu"
sys.path.insert(0, str(BENCH))

from launch_server_2 import build_launch, cli_args
from recipe.configs import config_eval


class TestTpuConfigEval(unittest.TestCase):
    def test_only_python_recipes_are_accepted(self):
        for suffix in (".json", ".jsonnet", ".txt"):
            with self.subTest(suffix=suffix), self.assertRaisesRegex(
                ValueError, "must end in .py"
            ):
                config_eval.eval_config("config" + suffix)

    def test_example_builds(self):
        config = config_eval.eval_config(BENCH / "recipe/configs/qwen3_0_6b.py")
        command, env = build_launch(config, inherited={})
        self.assertIn("--context-length=1300", command)
        self.assertIn("--max-running-requests=4", command)
        self.assertEqual(env["TPU_VISIBLE_CHIPS"], "0")

    def test_python_overrides_do_not_modify_base_or_later_loads(self):
        recipe = BENCH / "recipe/configs/qwen3_0_6b.py"
        first = config_eval.eval_config(recipe)
        first["server_args"]["port"] = 9999
        second = config_eval.eval_config(recipe)
        base = sys.modules["recipe.configs.base"].CONFIG
        self.assertEqual(second["server_args"]["port"], 30000)
        self.assertNotIn("model-path", base["server_args"])
        self.assertEqual(base["runtime_env"], {})

    def test_python_recipe_requires_config_object(self):
        with tempfile.TemporaryDirectory() as tmp:
            recipe = Path(tmp) / "config.py"
            for source in (
                "value = 1",
                "CONFIG = []",
                'CONFIG = {"server_args": []}',
                'CONFIG = {"unknown": {}}',
            ):
                with self.subTest(source=source):
                    recipe.write_text(source)
                    with self.assertRaises(ValueError):
                        config_eval.eval_config(recipe)

    def test_cli_types_and_overrides(self):
        settings = {
            "model-path": "a b; $(not-a-command)",
            "port": 30000,
            "enable-torch-compile": True,
            "disable-radix-cache": False,
            "quantization": None,
            "tpu-compile-bs": [1, 4, 8],
            "json-model-override-args": {"value": "a b"},
        }
        command, _ = build_launch({"server_args": settings}, ["--port", "30001"], {})
        self.assertIn("--model-path=a b; $(not-a-command)", command)
        self.assertNotIn("--port=30000", command)
        self.assertEqual(command[-2:], ["--port", "30001"])
        self.assertIn("--enable-torch-compile", command)
        self.assertNotIn("--disable-radix-cache", command)
        self.assertNotIn("--quantization", command)
        self.assertIn('--json-model-override-args={"value":"a b"}', command)
        start = command.index("--tpu-compile-bs")
        self.assertEqual(
            command[start : start + 4], ["--tpu-compile-bs", "1", "4", "8"]
        )
        for settings in (
            {"bad_key": 1},
            {"config": "x"},
            {"port": float("inf")},
            {"list": []},
            {"list": [True]},
            {"model-path": "x\0y"},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                cli_args(settings)

    def test_environment_sections_and_inherited_values(self):
        inherited = {"PATH": "/bin", "SGLANG_PLATFORM": "old", "TPU_VISIBLE_CHIPS": "3"}
        config = {
            "sglang_env": {"SGLANG_PLATFORM": "tpu"},
            "torch_tpu_env": {"TORCH_TPU_EXAMPLE": "1"},
            "runtime_env": {"TPU_VISIBLE_CHIPS": None},
        }
        _, env = build_launch(config, inherited=inherited)
        self.assertEqual(
            env, {"PATH": "/bin", "SGLANG_PLATFORM": "tpu", "TORCH_TPU_EXAMPLE": "1"}
        )
        self.assertEqual(inherited["SGLANG_PLATFORM"], "old")
        self.assertEqual(inherited["TPU_VISIBLE_CHIPS"], "3")
        for config in (
            {"runtime_env": {"SGLANG_PLATFORM": "tpu"}},
            {"sglang_env": {"SGLANG_PLATFORM": 1}},
            {"runtime_env": {"BAD=NAME": "1"}},
        ):
            with self.assertRaises(ValueError):
                build_launch(config, inherited={})

    def test_child_receives_environment_before_import_and_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "sglang"
            package.mkdir()
            (package / "__init__.py").write_text(
                'import os\nassert os.environ["SGLANG_PLATFORM"] == "tpu"\n'
                'assert os.environ["TORCH_TPU_EXAMPLE"] == "before-import"\n'
                'assert "TPU_VISIBLE_CHIPS" not in os.environ\n'
            )
            (package / "launch_server.py").write_text(
                "import json, sys\nprint(json.dumps(sys.argv[1:]))\nsys.exit(37)\n"
            )
            config = {
                "server_args": {
                    "model-path": "space ; $(literal)",
                    "api-key": "private",
                },
                "sglang_env": {"SGLANG_PLATFORM": "tpu"},
                "torch_tpu_env": {"TORCH_TPU_EXAMPLE": "before-import"},
                "runtime_env": {
                    "PYTHONPATH": str(root),
                    "TPU_VISIBLE_CHIPS": None,
                },
            }
            recipe = root / "config.py"
            recipe.write_text(
                f"CONFIG = {config!r}\n" 'CONFIG["server_args"]["port"] = 30001\n'
            )
            command = [
                sys.executable,
                str(BENCH / "launch_server_2.py"),
                "--config",
                str(recipe),
            ]
            env = {**os.environ, "TPU_VISIBLE_CHIPS": "7"}
            dry = subprocess.run(
                [*command, "--dry-run"], env=env, capture_output=True, text=True
            )
            self.assertEqual(dry.returncode, 0, dry.stderr)
            self.assertNotIn("private", dry.stdout)
            actual = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(actual.returncode, 37, actual.stderr)
            self.assertEqual(
                json.loads(actual.stdout),
                [
                    "--model-path=space ; $(literal)",
                    "--api-key=private",
                    "--port=30001",
                ],
            )


if __name__ == "__main__":
    unittest.main()
