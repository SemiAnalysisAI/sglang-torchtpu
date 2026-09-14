"""Execution-metadata residency and allocation-bound invariants."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

from test.registered.unit.hardware_backend.tpu.test_attention_pallas_rpa_v3 import (
    TestAttentionPallasRpaV3 as BackendFixture,
)

from sglang.srt.hardware_backend.tpu import eagle_device_execution as execution
from sglang.srt.managers.overlap_utils import FutureMap
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.runner.eager_runner import EagerRunner
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


class TestDeviceExecution(unittest.TestCase):
    def test_allocation_bound_survives_eager_batch_copy(self):
        batch = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=1,
            input_ids=torch.zeros(4, dtype=torch.int64),
            req_pool_indices=torch.tensor([0]),
            seq_lens=torch.tensor([128]),
            out_cache_loc=torch.arange(4),
            seq_lens_sum=None,
            _tpu_allocated_kv_upper_bound=256,
        )
        with patch("sglang.srt.model_executor.runner.eager_runner._is_tpu", True):
            copied = EagerRunner.load_batch(object.__new__(EagerRunner), batch)
        self.assertIsNot(copied, batch)
        self.assertEqual(copied._tpu_allocated_kv_upper_bound, 256)
        self.assertIsNone(copied.seq_lens_cpu)

    def test_configuration_limits_are_explicit(self):
        spec = SimpleNamespace(
            speculative_eagle_topk=1,
            speculative_num_steps=3,
            speculative_adaptive=False,
        )
        parallel = SimpleNamespace(dp_size=1, pp_size=1)
        runner = SimpleNamespace(
            spec_algorithm=SpeculativeAlgorithm.EAGLE3, page_size=64
        )
        with patch.dict(
            os.environ, {"SGLANG_TPU_EAGLE_DEVICE_EXECUTION": "1"}
        ), patch.object(execution, "get_spec", return_value=spec), patch.object(
            execution, "get_parallel", return_value=parallel
        ):
            self.assertTrue(execution.validate_configuration(runner))
            for obj, field, bad in (
                (parallel, "dp_size", 2),
                (parallel, "pp_size", 2),
                (spec, "speculative_eagle_topk", 2),
                (spec, "speculative_num_steps", 0),
                (spec, "speculative_adaptive", True),
                (runner, "page_size", 1),
                (runner, "spec_algorithm", SpeculativeAlgorithm.NONE),
            ):
                original = getattr(obj, field)
                with self.subTest(field=field):
                    setattr(obj, field, bad)
                    with self.assertRaises(ValueError):
                        execution.validate_configuration(runner)
                    setattr(obj, field, original)
            runner.model_config = SimpleNamespace(model_is_mrope=True)
            with self.assertRaises(ValueError):
                execution.validate_configuration(runner)

    def test_slot_snapshots_follow_reallocation_and_request_reordering(self):
        page_size = 64
        table = torch.cat(
            [
                torch.cat(
                    [
                        torch.arange(page * page_size, (page + 1) * page_size)
                        for page in row
                    ]
                )[None, :]
                for row in ([3, 8, 2, 11], [7, 4, 10, 1])
            ]
        )
        batch = SimpleNamespace(
            seq_lens_cpu=None,
            seq_lens=torch.tensor([63, 127]),
            req_pool_indices_cpu=torch.tensor([0, 1]),
            token_to_kv_pool_allocator=SimpleNamespace(page_size=page_size),
            reqs=[
                SimpleNamespace(kv=SimpleNamespace(kv_allocated_len=256))
                for _ in range(2)
            ],
        )
        with patch.object(
            execution, "capacity_for_batch", return_value=256
        ), patch.object(
            execution,
            "get_spec",
            return_value=SimpleNamespace(speculative_num_draft_tokens=4),
        ), patch.object(
            execution,
            "_functions",
            return_value=(execution._slots, execution._positions),
        ):
            for width in (3, 4):
                for dtype in (torch.int32, torch.int64):
                    batch.seq_lens = batch.seq_lens.to(dtype)

                    def expected():
                        return table[
                            batch.req_pool_indices_cpu[:, None],
                            batch.seq_lens[:, None] + torch.arange(width),
                        ].reshape(-1)

                    old_expected = expected().clone()
                    old_result = execution.slot_locations(batch, table, width)
                    torch.testing.assert_close(old_result, old_expected)
                    table[0, 64:128] = torch.arange(20 * page_size, 21 * page_size)
                    batch.req_pool_indices_cpu = batch.req_pool_indices_cpu.flip(0)
                    current = execution.slot_locations(batch, table, width)
                    torch.testing.assert_close(current, expected())
                    torch.testing.assert_close(old_result, old_expected)

    def test_uniform_positions_and_empty_padding(self):
        for dtype in (torch.int32, torch.int64):
            for lengths in ([], [63, 64, 127]):
                seq = torch.tensor(lengths, dtype=dtype)
                for width in (1, 4):
                    positions, starts = execution._positions(seq, width)
                    self.assertEqual(
                        positions.tolist(),
                        [n + i for n in lengths for i in range(width)],
                    )
                    self.assertEqual(
                        starts.tolist(), [i * width for i in range(len(lengths))]
                    )
                    self.assertEqual(positions.dtype, torch.int64)

    def test_gpu_does_not_use_tpu_capacity_fallback(self):
        future = object.__new__(FutureMap)
        future.publish_ready = None
        future.needs_cpu_seq_lens = False
        future.new_seq_lens_buf = torch.tensor([0, 127])
        batch = SimpleNamespace(
            spec_info=SimpleNamespace(future_indices=torch.tensor([1]))
        )
        with patch.dict(os.environ, {"SGLANG_TPU_EAGLE_DEVICE_EXECUTION": "1"}), patch(
            "sglang.srt.managers.overlap_utils._is_tpu", False
        ), patch.object(
            execution,
            "requires_cpu_capacity_mirror",
            side_effect=AssertionError("TPU path on GPU"),
        ):
            future.resolve_seq_lens_cpu(batch)
        self.assertIsNone(batch.seq_lens_cpu)
        self.assertEqual(batch.seq_lens.tolist(), [127])

    def test_active_rows_do_not_read_device_lengths(self):
        case = BackendFixture()
        try:
            backend = case._make_backend()
            backend.needs_cpu_seq_lens = False
            batch = SimpleNamespace(
                seq_lens_cpu=None, seq_lens=torch.tensor([128, 0, 0, 0]), _tpu_real_b=1
            )
            with patch(
                "torch.count_nonzero", side_effect=AssertionError("device read")
            ):
                self.assertEqual(backend._non_empty_batch_size(batch), 1)
        finally:
            case.doCleanups()

    def test_capacity_fallback_preserves_cpu_mirror(self):
        future = object.__new__(FutureMap)
        future.publish_ready = None
        future.needs_cpu_seq_lens = False
        future.fwd_prepare_d2h_stream = None
        future.new_seq_lens_buf = torch.tensor([0, 127], dtype=torch.int64)
        req = SimpleNamespace(kv=SimpleNamespace(kv_allocated_len=256))
        batch = SimpleNamespace(
            reqs=[req],
            spec_info=SimpleNamespace(future_indices=torch.tensor([1])),
            seq_lens_cpu=torch.tensor([126]),
            seq_lens_sum=126,
        )
        with patch.dict(os.environ, {"SGLANG_TPU_EAGLE_DEVICE_EXECUTION": "1"}), patch(
            "sglang.srt.managers.overlap_utils._is_tpu", True
        ), patch.object(execution, "capacity_for_batch", return_value=320):
            future.resolve_seq_lens_cpu(batch)
            self.assertIsNone(batch.seq_lens_cpu)
            self.assertIsNone(batch.seq_lens_sum)
            req.kv.kv_allocated_len = 384
            future.resolve_seq_lens_cpu(batch)
            self.assertEqual(batch.seq_lens_cpu.tolist(), [127])
            self.assertEqual(batch.seq_lens_sum, 127)


if __name__ == "__main__":
    unittest.main()
