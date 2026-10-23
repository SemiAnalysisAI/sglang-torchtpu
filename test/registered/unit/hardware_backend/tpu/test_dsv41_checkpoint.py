# SPDX-License-Identifier: Apache-2.0
"""Pure-stdlib contracts for EP selection before checkpoint payload reads.

Run directly with Python; no TPU, Torch, RunAI or CI registration import is
needed. Load the leaf module directly so package initialization stays outside
this test's dependency surface.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parents[5]
# CI registration is a stdlib-only, runtime no-op module. Load it directly so
# these host-only tests also run without importing SGLang's Torch dependencies.
_CI_SPEC = importlib.util.spec_from_file_location(
    "_dsv41_checkpoint_ci", _ROOT / "python/sglang/test/ci/ci_register.py"
)
_CI = importlib.util.module_from_spec(_CI_SPEC)
sys.modules[_CI_SPEC.name] = _CI
_CI_SPEC.loader.exec_module(_CI)
register_cpu_ci = _CI.register_cpu_ci
register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_SPEC = importlib.util.spec_from_file_location(
    "_dsv41_checkpoint_under_test", _ROOT / "sglang_tpu/dsv41/checkpoint.py"
)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
plan_expert_reads = _MODULE.plan_expert_reads


def expert_name(layer, expert, matrix="w1", kind="weight"):
    return f"layers.{layer}.ffn.experts.{expert}.{matrix}.{kind}"


def make_index(layers=2, experts=8):
    return {
        expert_name(layer, expert, matrix, kind): f"expert-{expert % 3}.safetensors"
        for layer in range(layers)
        for expert in range(experts)
        for matrix in ("w1", "w2", "w3")
        for kind in ("weight", "scale")
    }


class TestExpertReadPlan(unittest.TestCase):
    def plan(self, index, **overrides):
        options = dict(num_layers=2, num_experts=8, ep_size=4, ep_rank=0)
        options.update(overrides)
        return plan_expert_reads(index, **options)

    def test_full_40_layer_384_expert_eight_rank_partition(self):
        index = make_index(40, 384)
        nonexpert = {
            "tok_embeddings.weight": "replicated.safetensors",
            "norm.weight": "replicated.safetensors",
            "output.weight": "replicated.safetensors",
        }
        for layer in range(40):
            for suffix in (
                "attn.wq_a.weight",
                "ffn.gate.weight",
                "ffn.gate.e_score_correction_bias",
                "ffn.shared_experts.w1.weight",
                "engram.embeddings.weight",
                "engram.embeddings.scale",
                "future_parameter.weight",
            ):
                nonexpert[f"layers.{layer}.{suffix}"] = "replicated.safetensors"
        mtp = {"mtp.layers.0.ffn.experts.0.w1.weight": "mtp-only.safetensors"}
        index.update(nonexpert)
        index.update(mtp)
        all_experts = set(index) - set(nonexpert) - set(mtp)
        seen = set()
        for rank in range(8):
            with self.subTest(rank=rank):
                result = self.plan(
                    index, num_layers=40, num_experts=384, ep_size=8, ep_rank=rank
                )
                expected = {
                    expert_name(layer, expert, matrix, kind)
                    for layer in range(40)
                    for expert in range(rank * 48, (rank + 1) * 48)
                    for matrix in ("w1", "w2", "w3")
                    for kind in ("weight", "scale")
                }
                self.assertEqual(result.tensor_names, expected | set(nonexpert))
                self.assertEqual(result.local_expert_tensors, 40 * 48 * 6)
                self.assertEqual(result.total_tensors, len(index))
                self.assertFalse(seen & expected)
                seen.update(result.tensor_names - set(nonexpert))
                self.assertEqual(
                    result.shard_names, {index[name] for name in result.tensor_names}
                )
                self.assertNotIn("mtp-only.safetensors", result.shard_names)
        self.assertEqual(seen, all_experts)

    def test_single_rank_keeps_all_target_weights(self):
        index = make_index()
        result = self.plan(index, ep_size=1)
        self.assertEqual(result.tensor_names, set(index))
        self.assertEqual(result.local_expert_tensors, 2 * 8 * 6)

    def test_non_eight_way_ep_uses_configured_partition(self):
        index = make_index(3, 12)
        for rank in range(3):
            result = self.plan(
                index, num_layers=3, num_experts=12, ep_size=3, ep_rank=rank
            )
            expected = {
                expert_name(layer, expert, matrix, kind)
                for layer in range(3)
                for expert in range(rank * 4, (rank + 1) * 4)
                for matrix in ("w1", "w2", "w3")
                for kind in ("weight", "scale")
            }
            self.assertEqual(result.tensor_names, expected)

    def test_unknown_nonexpert_names_and_shared_engram_remain_on_every_rank(self):
        index = make_index()
        retained = {
            "layers.0.engram.table.weight": "host.safetensors",
            "layers.0.engram.table.scale": "host.safetensors",
            "layers.0.ffn.shared_experts.w1.weight": "shared.safetensors",
            "layers.0.ffn.experts_metadata.version": "future.safetensors",
            "mtp_like_but_not_mtp.weight": "future.safetensors",
        }
        index.update(retained)
        for rank in range(4):
            self.assertLessEqual(
                set(retained), self.plan(index, ep_rank=rank).tensor_names
            )

    def test_only_exact_mtp_prefix_is_excluded(self):
        index = make_index()
        index.update(
            {
                "mtp.output.weight": "mtp.safetensors",
                "mtp_state.weight": "state.safetensors",
            }
        )
        result = self.plan(index)
        self.assertNotIn("mtp.output.weight", result.tensor_names)
        self.assertIn("mtp_state.weight", result.tensor_names)

    def test_missing_each_matrix_weight_or_scale_fails(self):
        for matrix in ("w1", "w2", "w3"):
            for kind in ("weight", "scale"):
                with self.subTest(matrix=matrix, kind=kind):
                    index = make_index()
                    del index[expert_name(1, 7, matrix, kind)]
                    with self.assertRaises(ValueError):
                        self.plan(index)

    def test_partial_or_empty_index_fails(self):
        for index in ({}, {"norm.weight": "shared.safetensors"}, make_index(1, 8)):
            with self.subTest(tensor_count=len(index)):
                with self.assertRaises(ValueError):
                    self.plan(index)

    def test_incompatible_expert_naming_fails(self):
        for changed in (
            "model.layers.0.ffn.experts.0.w1.weight",
            "layers.0.ffn.experts.0.w4.weight",
            "layers.0.ffn.experts.0.w1.weight_scale_inv",
            "layers.0.ffn.experts.-1.w1.weight",
        ):
            with self.subTest(name=changed):
                index = make_index()
                shard = index.pop(expert_name(0, 0))
                index[changed] = shard
                with self.assertRaises(ValueError):
                    self.plan(index)

    def test_equal_cardinality_alias_cannot_hide_missing_expert_tensor(self):
        # Numeric aliases must not make a count-only completeness check accept
        # two names for one tensor while a different expert tensor is missing.
        for alias in (
            "layers.00.ffn.experts.0.w1.weight",
            "layers.0.ffn.experts.00.w1.weight",
        ):
            with self.subTest(alias=alias):
                index = make_index()
                shard = index.pop(expert_name(0, 1, "w1", "scale"))
                index[alias] = shard
                with self.assertRaises(ValueError):
                    self.plan(index)

    def test_out_of_range_layer_or_expert_fails_even_when_nonlocal(self):
        for name in (expert_name(2, 0), expert_name(0, 8)):
            index = make_index()
            index[name] = "extra.safetensors"
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    self.plan(index)

    def test_invalid_configuration_fails(self):
        for options in (
            {"num_layers": 0},
            {"num_layers": -1},
            {"num_experts": 0},
            {"num_experts": -1},
            {"ep_size": 0},
            {"ep_size": -1},
            {"ep_size": 3},
            {"ep_rank": -1},
            {"ep_rank": 4},
        ):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    self.plan(make_index(), **options)

    def test_input_is_not_mutated_and_rank_is_not_cached(self):
        index = make_index()
        original = dict(index)
        first = self.plan(index, ep_rank=0)
        second = self.plan(index, ep_rank=1)
        third = self.plan(index, ep_rank=0)
        self.assertEqual(index, original)
        self.assertNotEqual(first.tensor_names, second.tensor_names)
        self.assertEqual(first, third)
        self.assertIsInstance(first.tensor_names, frozenset)
        self.assertIsInstance(first.shard_names, frozenset)


class TestSelectiveLoader(unittest.TestCase):
    @contextmanager
    def modules(self, *, stream_error=False, missing_shard=False):
        """Load the real plugin adapters against tiny dependency fakes."""
        state = SimpleNamespace(
            calls=[],
            closed=0,
            entered=0,
            fallbacks=[],
            paths=[],
            parallel=SimpleNamespace(moe_ep_size=4, moe_ep_rank=0),
        )
        with TemporaryDirectory() as directory:
            index = make_index()
            index["layers.0.engram.table.weight"] = "host.safetensors"
            index["mtp.output.weight"] = "mtp.safetensors"
            Path(directory, "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": index})
            )
            state.directory = directory

            class BaseLoader:
                def __init__(self, config):
                    self.load_config = config

                def _get_weights_iterator(self, source):
                    state.fallbacks.append(source)
                    yield "parent-result", source

                def _prepare_weights(self, path, revision):
                    files = [f"{path}/{name}" for name in sorted(set(index.values()))]
                    if missing_shard:
                        files = [
                            name
                            for name in files
                            if not name.endswith("host.safetensors")
                        ]
                    return path, files

                def load_model(self, **kwargs):
                    return kwargs

            class Tensor:
                def numel(self):
                    return 8

                def element_size(self):
                    return 1

            class Streamer:
                def __enter__(self):
                    state.entered += 1
                    return self

                def __exit__(self, *args):
                    state.closed += 1

                def stream_files(self, files, *, device, is_distributed, tensor_names):
                    state.calls.append(
                        dict(
                            files=files,
                            device=device,
                            is_distributed=is_distributed,
                            tensor_names=tensor_names,
                        )
                    )
                    self.names = tensor_names

                def get_tensors(self):
                    self.assert_configured()
                    if stream_error:
                        raise RuntimeError("read failed")
                    for name in sorted(self.names):
                        yield name, Tensor()

                def assert_configured(self):
                    if not state.calls:
                        raise AssertionError("read preceded selection")

            class Storage:
                @staticmethod
                def get_path(path):
                    state.paths.append(path)
                    return directory

            def fake_module(name, **attrs):
                value = ModuleType(name)
                value.__dict__.update(attrs)
                return value

            fakes = {
                "sglang.srt.model_loader.loader": fake_module(
                    "loader", RunaiModelStreamerLoader=BaseLoader
                ),
                "sglang_tpu.dsv41.checkpoint": _MODULE,
                "runai_model_streamer": fake_module(
                    "runai", SafetensorsStreamer=Streamer
                ),
                "sglang.srt.model_loader.weight_utils": fake_module(
                    "weights", RUNAI_STREAMER_TENSOR_ATTR="_from_streamer"
                ),
                "sglang.srt.runtime_context": fake_module(
                    "runtime", get_parallel=lambda: state.parallel
                ),
                "sglang.srt.utils.runai_utils": fake_module(
                    "utils",
                    ObjectStorageModel=Storage,
                    is_runai_obj_uri=lambda p: p.startswith("gs://"),
                ),
            }
            with patch.dict(sys.modules, fakes):
                spec = importlib.util.spec_from_file_location(
                    "_dsv41_loader_under_test", _ROOT / "sglang_tpu/dsv41/loader.py"
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                with patch.dict(sys.modules, {"sglang_tpu.dsv41.loader": module}):
                    hook_spec = importlib.util.spec_from_file_location(
                        "_dsv41_hook_under_test",
                        _ROOT / "sglang_tpu/hooks/checkpoint.py",
                    )
                    hook = importlib.util.module_from_spec(hook_spec)
                    hook_spec.loader.exec_module(hook)
                    state.base, state.loader_class, state.hook = (
                        BaseLoader,
                        module.DSV41RunaiModelStreamerLoader,
                        hook,
                    )
                    state.config = SimpleNamespace(
                        model_path="gs://model/checkpoint",
                        revision=None,
                        is_draft_model=False,
                        hf_config=SimpleNamespace(
                            model_type="deepseek_v41",
                            num_hidden_layers=2,
                            n_routed_experts=8,
                        ),
                    )
                    state.load_config = SimpleNamespace(draft_model_idx=None)
                    state.source = SimpleNamespace(
                        model_or_path=state.config.model_path,
                        revision=None,
                        model_config=state.config,
                        prefix="",
                    )
                    state.loader = state.loader_class(state.load_config, state.config)
                    yield state

    def test_selection_precedes_read_and_retains_tensor_ownership_marker(self):
        with self.modules() as state:
            rows = list(state.loader._get_weights_iterator(state.source))
            request = state.calls[0]
            self.assertEqual(request["device"], "cpu")
            self.assertIs(request["is_distributed"], False)
            self.assertEqual(len(request["tensor_names"]), 2 * 2 * 6 + 1)
            self.assertTrue(
                all(getattr(tensor, "_from_streamer", False) for _, tensor in rows)
            )
            self.assertIn("layers.0.engram.table.weight", request["tensor_names"])
            self.assertNotIn("mtp.output.weight", request["tensor_names"])
            self.assertFalse(
                any(path.endswith("mtp.safetensors") for path in request["files"])
            )
            self.assertEqual(state.paths, [state.config.model_path])
            self.assertEqual(state.closed, 1)
            self.assertIs(state.loader_class.load_model, state.base.load_model)

    def test_context_is_closed_after_read_error(self):
        with self.modules(stream_error=True) as state:
            with self.assertRaisesRegex(RuntimeError, "read failed"):
                list(state.loader._get_weights_iterator(state.source))
            self.assertEqual(state.closed, 1)

    def test_missing_selected_shard_fails_before_stream_open(self):
        with self.modules(missing_shard=True) as state:
            with self.assertRaises(ValueError):
                list(state.loader._get_weights_iterator(state.source))
            self.assertEqual(state.entered, 0)

    def test_source_variants_delegate_to_parent(self):
        for change in (
            {"prefix": "secondary."},
            {"model_or_path": "gs://other"},
            {"revision": "other"},
            {"model_config": None},
        ):
            with self.subTest(change=change), self.modules() as state:
                source = SimpleNamespace(**(vars(state.source) | change))
                self.assertEqual(
                    list(state.loader._get_weights_iterator(source)),
                    [("parent-result", source)],
                )
                self.assertEqual(state.calls, [])
        with self.modules() as state:
            other = SimpleNamespace(**vars(state.config))
            state.source.model_config = other
            list(state.loader._get_weights_iterator(state.source))
            self.assertEqual(state.fallbacks, [state.source])

    def test_current_ep_rank_is_recomputed_for_each_iterator(self):
        with self.modules() as state:
            list(state.loader._get_weights_iterator(state.source))
            state.parallel.moe_ep_rank = 3
            list(state.loader._get_weights_iterator(state.source))
            self.assertIn(expert_name(0, 0), state.calls[0]["tensor_names"])
            self.assertNotIn(expert_name(0, 0), state.calls[1]["tensor_names"])
            self.assertIn(expert_name(0, 7), state.calls[1]["tensor_names"])

    def test_hook_only_selects_exact_base_loader_and_main_dsv41(self):
        with self.modules() as state:
            original = state.base(state.load_config)
            selected = state.hook.select_loader(
                original, state.load_config, state.config
            )
            self.assertIsInstance(selected, state.loader_class)
            for untouched in (object(), selected):
                self.assertIs(
                    state.hook.select_loader(
                        untouched, state.load_config, state.config
                    ),
                    untouched,
                )
            for attribute, value in (("is_draft_model", True),):
                setattr(state.config, attribute, value)
                self.assertIs(
                    state.hook.select_loader(original, state.load_config, state.config),
                    original,
                )
                setattr(state.config, attribute, False)
            state.config.hf_config.model_type = "other_model"
            self.assertIs(
                state.hook.select_loader(original, state.load_config, state.config),
                original,
            )
            state.config.hf_config.model_type = "deepseek_v41"
            state.load_config.draft_model_idx = 0
            self.assertIs(
                state.hook.select_loader(original, state.load_config, state.config),
                original,
            )


if __name__ == "__main__":
    unittest.main()
