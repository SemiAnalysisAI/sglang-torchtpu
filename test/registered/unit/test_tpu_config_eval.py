"""Unit tests for Python recipe dictionaries and CLI flag conversion."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from benchmark.tpu.launch_server import load_recipe, plan_launch
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


class TestNightlyAccuracyRecipes(unittest.TestCase):
    def test_deepseek_multihost_rendezvous_and_precision(self):
        for name, dp, parser in (
            ("deepseek_r1_0528_fp8kv_mh", 1, "deepseek-r1"),
            ("deepseek_r1_0528_dpa_mh", 2, "deepseek-r1"),
            ("deepseek_v3_1_fp8kv_mh", 1, "deepseek-v3"),
        ):
            config = load_recipe("recipe/configs/" + name + ".py")
            original = deepcopy(config)
            for rank in (0, 1):
                with self.subTest(recipe=name, rank=rank):
                    plan = plan_launch(
                        config,
                        {
                            "TPU_WORKER_ID": str(rank),
                            "TPU_WORKER_HOSTNAMES": "leader,worker",
                        },
                    )
                    args = plan.config.server_args
                    self.assertEqual(args["nnodes"], 2)
                    self.assertEqual(args["node_rank"], rank)
                    self.assertEqual(
                        args["dist_init_addr"],
                        "0.0.0.0:10011" if rank == 0 else "leader:10011",
                    )
                    self.assertEqual(
                        (args["tp_size"], args["ep_size"], args["dp_size"]),
                        (16, 16, dp),
                    )
                    self.assertEqual(plan.env["SGLANG_TPU_FP8_MOE_FP32_SCALES"], "1")
                    self.assertEqual(args["quantization"], "fp8")
                    self.assertEqual(args["kv_cache_dtype"], "fp8_e4m3")
                    self.assertEqual(args["reasoning_parser"], parser)
                    self.assertEqual(args["tpu_compile_bs"], [64, 128])
                    self.assertEqual(args["tpu_compile_tokens"], [8192 // dp])
                    self.assertEqual(args["chunked_prefill_size"], 8192)
                    self.assertEqual("--enable-dp-attention" in plan.command, dp > 1)
                    self.assertEqual("--enable-dp-lm-head" in plan.command, dp > 1)
            self.assertEqual(config, original)

    def test_jobsets_dispatch_python_and_legacy_shell_recipes(self):
        for template in ("jobset-tpu.yaml", "external/jobset-external-tpu.yaml"):
            text = (ROOT / ".github/tpu-ci" / template).read_text()
            start = text.index('case "${LAUNCH_SCRIPT}" in')
            dispatch = text[start : text.index("esac", start) + len("esac")]
            with tempfile.TemporaryDirectory() as directory:
                for executable in ("python3", "bash"):
                    path = Path(directory) / executable
                    path.write_text('#!/bin/sh\nprintf "%s\\n" "$0" "$@"\n')
                    path.chmod(0o755)
                for recipe, expected in (
                    (
                        "recipe/configs/deepseek_r1_0528_fp8kv_mh.py",
                        [
                            "python3",
                            "-m",
                            "benchmark.tpu.launch_server",
                            "--config",
                            "recipe/configs/deepseek_r1_0528_fp8kv_mh.py",
                        ],
                    ),
                    (
                        "recipe/legacy config.sh",
                        ["bash", "benchmark/tpu/recipe/legacy config.sh"],
                    ),
                ):
                    with self.subTest(template=template, recipe=recipe):
                        result = subprocess.run(
                            ["/bin/bash", "-c", dispatch],
                            check=True,
                            capture_output=True,
                            text=True,
                            env={
                                **os.environ,
                                "PATH": directory,
                                "LAUNCH_SCRIPT": recipe,
                            },
                        )
                        actual = result.stdout.splitlines()
                        actual[0] = Path(actual[0]).name
                        self.assertEqual(actual, expected)

    def test_kimi_full_mmmu_shards(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "mmmu_full", ROOT / "python/sglang/test/tpu/mmmu_full.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rows = list(range(1730))
        left = module.shard_rows(rows, 0, 2)
        right = module.shard_rows(rows, 1, 2)
        self.assertEqual(len(left), 865)
        self.assertEqual(len(right), 865)
        self.assertEqual(set(left) & set(right), set())
        self.assertEqual(sorted(left + right), list(enumerate(rows)))
        for invalid in (-1, 2):
            with self.assertRaises(ValueError):
                module.shard_rows(rows, invalid, 2)
        config = load_recipe("recipe/configs/kimi_k26_mmmu.py")
        plan = plan_launch(
            config, {"TPU_WORKER_ID": "1", "TPU_WORKER_HOSTNAMES": "a,b"}
        )
        self.assertEqual(plan.config.server_args["nnodes"], 2)
        self.assertEqual(plan.config.server_args["dp_size"], 4)
        self.assertEqual(plan.config.server_args["context_length"], 106496)
        self.assertNotIn("--quantization", plan.command)

    def test_full_mmmu_aggregation_rejects_missing_or_duplicate_ids(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "collect_full_evals", ROOT / ".github/tpu-ci/collect_full_evals.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        shards = [
            {
                "shard": i,
                "shards": 2,
                "questions": 865,
                "correct": 700,
                "ids": list(range(i, 1730, 2)),
            }
            for i in range(2)
        ]
        result = module.combine_shards(shards)
        self.assertEqual(result["correct"], 1400)
        with self.assertRaises(ValueError):
            module.combine_shards(shards[:1])
        shards[1]["ids"][0] = shards[0]["ids"][0]
        with self.assertRaises(ValueError):
            module.combine_shards(shards)

    def test_next_nightly_selects_first_run_after_cutoff(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "dispatch_after_nightly", ROOT / ".github/tpu-ci/dispatch_after_nightly.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        runs = [
            {"databaseId": 1, "createdAt": "2026-09-20T09:00:00Z"},
            {"databaseId": 3, "createdAt": "2026-09-22T09:00:00Z"},
            {"databaseId": 2, "createdAt": "2026-09-21T09:00:00Z"},
        ]
        cutoff = "2026-09-21T08:00:00Z"
        self.assertEqual(module.select_nightly(runs, cutoff)["databaseId"], 2)
        self.assertIsNone(module.select_nightly(runs[:1], cutoff))

    def test_gpt_oss_keeps_checkpoint_quantization_and_bf16_kv(self):
        config = load_recipe("recipe/configs/gpt_oss_20b_accuracy.py")
        plan = plan_launch(config, {})
        args = plan.config.server_args
        self.assertNotIn("--quantization", plan.command)
        self.assertEqual(args["dtype"], "bfloat16")
        self.assertEqual(args["kv_cache_dtype"], "auto")
        self.assertEqual(args["model_path"], "openai/gpt-oss-20b")
        self.assertEqual(args["max_running_requests"], 16)
        self.assertEqual(args["context_length"], 69632)
        self.assertEqual(args["max_total_tokens"], 1114112)
        self.assertEqual(args["tpu_compile_tokens"], [4096])
        self.assertEqual(args["tpu_compile_bs"], [1, 2, 4, 8, 16])
        self.assertEqual(plan.env["SGLANG_TPU_FUSED_MOE_SC_GATHER"], "0")


if __name__ == "__main__":
    unittest.main()
