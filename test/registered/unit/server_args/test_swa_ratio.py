import argparse
import dataclasses
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from sglang.srt.arg_groups.kv_cache_hook import handle_swa_ratio_default
from sglang.srt.arg_groups.model_overrides.deepseek_v4 import _deepseek_v4_overrides
from sglang.srt.arg_groups.overrides import declare_resolution
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.server_args_config_parser import ConfigArgumentMerger
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSWARatioIntent(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(cls.parser)

    def test_cli_default_and_explicit_default_are_distinct(self):
        for value in (None, "0.8", "0.1"):
            with self.subTest(value=value):
                argv = ["--model", "dummy"]
                if value is not None:
                    argv += ["--swa-full-tokens-ratio", value]
                args = ServerArgs.from_cli_args(self.parser.parse_args(argv))
                args.resolve_once()
                self.assertEqual(args.has_explicit_swa_ratio(), value is not None)
                self.assertEqual(
                    args.swa_full_tokens_ratio, None if value is None else float(value)
                )
                self.assertEqual(
                    args.resolved_dict()["swa_full_tokens_ratio"],
                    0.8 if value is None else float(value),
                )

    def test_yaml_value_is_explicit_and_cli_can_override(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("model-path: dummy\nswa-full-tokens-ratio: 0.1\n")
            merger = ConfigArgumentMerger(self.parser)
            for extra, expected in (
                ([], 0.1),
                (["--swa-full-tokens-ratio", "0.8"], 0.8),
            ):
                argv = merger.merge_config_with_args(["--config", str(path)] + extra)
                args = ServerArgs.from_cli_args(self.parser.parse_args(argv))
                self.assertEqual(args.swa_full_tokens_ratio, expected)

    def test_raw_intent_survives_worker_and_constructor_round_trips(self):
        for ratio in (None, 0.1, 0.8):
            with self.subTest(ratio=ratio):
                args = ServerArgs(model_path="dummy", swa_full_tokens_ratio=ratio)
                args.resolve_once()
                copies = (
                    pickle.loads(pickle.dumps(args)),
                    dataclasses.replace(args, port=31000),
                    ServerArgs(**dataclasses.asdict(args)),
                    args.replace_resolved("test", port=31000),
                )
                for copy in copies:
                    copy.resolve_once()
                    self.assertEqual(copy.has_explicit_swa_ratio(), ratio is not None)
                    self.assertEqual(copy.swa_full_tokens_ratio, ratio)
                    self.assertEqual(
                        copy.resolved_dict()["swa_full_tokens_ratio"],
                        0.8 if ratio is None else ratio,
                    )

    def test_default_fill_preserves_model_declaration(self):
        args = ServerArgs(model_path="dummy")
        declare_resolution(args, "model_default", swa_full_tokens_ratio=0.1)
        handle_swa_ratio_default(args)
        handle_swa_ratio_default(args)
        self.assertIsNone(args.swa_full_tokens_ratio)
        self.assertEqual(args.resolved_dict()["swa_full_tokens_ratio"], 0.1)

    def test_model_default_yields_to_explicit_point_eight(self):
        hf_config = SimpleNamespace(architectures=["DeepseekV4ForCausalLM"])
        for ratio in (None, 0.8):
            args = ServerArgs(
                model_path="dummy",
                device="cpu",
                moe_runner_backend="triton",
                swa_full_tokens_ratio=ratio,
            )
            overrides = _deepseek_v4_overrides(args, hf_config)
            if ratio is None:
                self.assertEqual(overrides["swa_full_tokens_ratio"], 0.1)
            else:
                self.assertNotIn("swa_full_tokens_ratio", overrides)

    def test_invalid_ratio_fails_even_for_dummy_model(self):
        for ratio in (0.0, -0.1, 1.1, float("nan"), float("inf")):
            with (
                self.subTest(ratio=ratio),
                self.assertRaisesRegex(ValueError, "swa-full-tokens-ratio"),
            ):
                ServerArgs(
                    model_path="dummy", swa_full_tokens_ratio=ratio
                ).resolve_once()


if __name__ == "__main__":
    unittest.main()
