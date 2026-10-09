"""Merging and filtering per-request tensors without a program per length pair."""

import itertools
import unittest
from math import prod
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt import runtime_context
from sglang.srt.hardware_backend.tpu import batch_rows
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CAPACITY = 16
LAYOUTS = [
    (torch.int64, ()),
    (torch.int32, ()),
    (torch.float32, ()),
    (torch.float32, (1,)),  # temperatures are [bs, 1]
    (torch.bool, ()),
]


def _rows(n, dtype, trailing, base=0):
    return (torch.arange(n * prod(trailing)) + base).to(dtype).reshape(n, *trailing)


class TestPaddedRows(unittest.TestCase):
    def setUp(self):
        state = patch.object(batch_rows, "_state", batch_rows._BatchRowsState())
        state.start()
        self.addCleanup(state.stop)

    def test_capacity_without_warmup_uses_per_dp_limit(self):
        cases = [
            (2048, 16, (), 128),
            (2048, 1, (), 2048),
            (129, 4, (), 32),
            (129, 4, (33,), 64),
            (None, 1, (), 1),
        ]
        for configured, attn_dp_size, lengths, expected in cases:
            with (
                self.subTest(configured=configured, attn_dp_size=attn_dp_size),
                patch.object(
                    runtime_context,
                    "get_schedule",
                    return_value=SimpleNamespace(max_running_requests=configured),
                ),
                patch.object(
                    runtime_context,
                    "get_parallel",
                    return_value=SimpleNamespace(attn_dp_size=attn_dp_size),
                ),
            ):
                self.assertEqual(batch_rows._capacity(*lengths), expected)

    def test_capacity_overflow_is_reported(self):
        batch_rows.configure_capacity(CAPACITY)
        with self.assertLogs(batch_rows.logger, level="WARNING") as logs:
            self.assertEqual(batch_rows._capacity(CAPACITY + 1), 2 * CAPACITY)
            self.assertEqual(batch_rows._capacity(CAPACITY + 1), 2 * CAPACITY)
        self.assertEqual(len(logs.output), 1)
        self.assertIn("exceed warmed capacity", logs.output[0])

    def test_padding_rejects_capacity_smaller_than_input(self):
        rows = torch.arange(3)
        with self.assertRaisesRegex(ValueError, "exceeds capacity"):
            batch_rows.padded_filter_rows(rows, torch.tensor([0]), 2)
        with self.assertRaisesRegex(ValueError, "exceeds capacity"):
            batch_rows.padded_merge_rows(rows, rows, 2)

    def test_merge_reuses_only_the_latest_index_across_tensor_layouts(self):
        uploads = []
        orig_to = torch.Tensor.to

        def to(tensor, *args, **kwargs):
            uploads.append(tensor)
            return orig_to(tensor, *args, **kwargs)

        with patch.object(torch.Tensor, "to", to):
            for na, nb in [(3, 2), (2, 4), (3, 2)]:
                for dtype, trailing in LAYOUTS:
                    a = torch.zeros((na, *trailing), dtype=dtype)
                    b = torch.ones((nb, *trailing), dtype=dtype)
                    result = batch_rows.padded_merge_rows(a, b, CAPACITY)
                    self.assertTrue(torch.equal(result, torch.cat([a, b])))
        self.assertEqual(len(uploads), 3)

    def test_filter_matches_plain_indexing(self):
        for (dtype, trailing), old in itertools.product(LAYOUTS, [1, 5, CAPACITY]):
            t = _rows(old, dtype, trailing)
            for keep in ([], [0], list(range(old)), list(range(0, old, 2)), [old - 1]):
                index = torch.tensor(keep, dtype=torch.int64)
                out = batch_rows.padded_filter_rows(t, index, CAPACITY)
                self.assertEqual(out.dtype, dtype)
                self.assertTrue(torch.equal(out, t[index]))

    def test_merge_matches_plain_cat(self):
        for (dtype, trailing), (na, nb) in itertools.product(
            LAYOUTS, [(0, 3), (3, 0), (5, 7), (CAPACITY, CAPACITY)]
        ):
            a = _rows(na, dtype, trailing)
            b = _rows(nb, dtype, trailing, base=100)
            out = batch_rows.padded_merge_rows(a, b, CAPACITY)
            self.assertEqual(out.dtype, dtype)
            self.assertTrue(torch.equal(out, torch.cat([a, b])))

    def test_merge_optional_rows_preserves_values_and_validity(self):
        layouts = [*LAYOUTS, (torch.int32, (3,)), (torch.float64, ())]
        for (dtype, trailing), (
            has_left,
            has_right,
        ), explicit_valid in itertools.product(
            layouts, itertools.product((False, True), repeat=2), (False, True)
        ):
            left = _rows(2, dtype, trailing) if has_left else None
            right = _rows(1, dtype, trailing, base=100) if has_right else None
            left_valid = (
                torch.tensor([True, False]) if has_left and explicit_valid else None
            )
            right_valid = (
                torch.tensor([False]) if has_right and explicit_valid else None
            )
            values, valid = batch_rows.merge_optional_rows(
                left, left_valid, 2, right, right_valid, 1, validity_device="cpu"
            )
            if not has_left and not has_right:
                self.assertIsNone(values)
                self.assertIsNone(valid)
                continue
            expected_left = (
                left if has_left else torch.zeros((2, *trailing), dtype=dtype)
            )
            expected_right = (
                right if has_right else torch.zeros((1, *trailing), dtype=dtype)
            )
            self.assertTrue(
                torch.equal(values, torch.cat([expected_left, expected_right]))
            )
            self.assertTrue(
                torch.equal(
                    valid,
                    torch.tensor(
                        [
                            has_left,
                            has_left and not explicit_valid,
                            has_right and not explicit_valid,
                        ]
                    ),
                )
            )

    def test_device_shapes_depend_on_one_length_at_a_time(self):
        """Every op sees at most one batch length; pairs never meet in one op."""
        seen = set()
        orig_pad, orig_select, orig_cat = (
            torch.nn.functional.pad,
            torch.Tensor.index_select,
            torch.cat,
        )

        def pad(t, spec):
            seen.add(("pad", tuple(t.shape)))
            return orig_pad(t, spec)

        def select(t, dim, index):
            seen.add(("select", tuple(t.shape), tuple(index.shape)))
            return orig_select(t, dim, index)

        def cat(ts, *args, **kwargs):
            if all(t.shape[0] == CAPACITY for t in ts):
                seen.add(("cat", tuple(tuple(t.shape) for t in ts)))
            return orig_cat(ts, *args, **kwargs)

        with (
            patch.object(batch_rows.F, "pad", pad),
            patch.object(torch.Tensor, "index_select", select),
            patch.object(batch_rows.torch, "cat", cat),
        ):
            for na, nb in [(3, 2), (3, 7), (9, 2)]:
                batch_rows.padded_merge_rows(torch.zeros(na), torch.zeros(nb), CAPACITY)
            for old, new in [(9, 8), (9, 3), (4, 3)]:
                keep = torch.arange(new, dtype=torch.int64)
                batch_rows.padded_filter_rows(torch.zeros(old), keep, CAPACITY)

        # Operand lengths each get one pad; results one select; the padded cat
        # is the same program for every pair.
        self.assertEqual(
            {s for s in seen if s[0] == "pad"},
            {("pad", (n,)) for n in (3, 2, 7, 9, 4)},
        )
        self.assertEqual(
            {s for s in seen if s[0] == "cat"},
            {("cat", ((CAPACITY,), (CAPACITY,)))},
        )
        self.assertEqual(
            {s[1] for s in seen if s[0] == "select"},
            {(2 * CAPACITY,), (CAPACITY,)},
        )

    def test_precompile_covers_every_length(self):
        selects = set()
        cats = set()
        orig_select = torch.Tensor.index_select
        orig_cat = torch.cat

        def select(t, dim, index):
            selects.add((t.dtype, tuple(t.shape), tuple(index.shape)))
            return orig_select(t, dim, index)

        def cat(tensors, *args, **kwargs):
            if all(t.shape[0] == CAPACITY for t in tensors):
                cats.add(tuple((t.dtype, tuple(t.shape)) for t in tensors))
            return orig_cat(tensors, *args, **kwargs)

        batch_rows.register_layout(torch.int32, (3,))
        batch_rows.register_layout(torch.int64, (3,))
        batch_rows.register_layout(torch.float64, ())
        batch_rows.register_layout(torch.bool, ())
        batch_rows.configure_capacity(CAPACITY - 1)
        with (
            patch.object(runtime_context, "get_schedule", side_effect=AssertionError),
            patch.object(torch.Tensor, "index_select", select),
            patch.object(torch, "cat", cat),
        ):
            batch_rows.precompile(torch.device("cpu"))
            self.assertEqual(batch_rows._capacity(), CAPACITY)
            self.assertEqual(batch_rows._capacity(CAPACITY - 1), CAPACITY)

        for dtype, trailing in [
            *LAYOUTS,
            (torch.int32, (3,)),
            (torch.int64, (3,)),
            (torch.float64, ()),
        ]:
            for n in range(CAPACITY + 1):
                self.assertIn((dtype, (CAPACITY, *trailing), (n,)), selects)
                self.assertIn((dtype, (2 * CAPACITY, *trailing), (n,)), selects)
        self.assertEqual(
            {shape[0] for _, _, shape in selects}, set(range(CAPACITY + 1))
        )
        for dtype_a, dtype_b, trailing in (
            (torch.int32, torch.int64, ()),
            (torch.int32, torch.int64, (3,)),
            (torch.float32, torch.float64, ()),
        ):
            a = (dtype_a, (CAPACITY, *trailing))
            b = (dtype_b, (CAPACITY, *trailing))
            self.assertIn((a, b), cats)
            self.assertIn((b, a), cats)

    def test_precompile_requires_explicit_capacity(self):
        with self.assertRaisesRegex(ValueError, "configure_capacity"):
            batch_rows.precompile(torch.device("cpu"))

    def test_precompile_freezes_layouts_and_capacity(self):
        batch_rows.configure_capacity(1)
        batch_rows.precompile(torch.device("cpu"))
        batch_rows.register_layout(torch.int64, ())
        with self.assertRaisesRegex(RuntimeError, "before precompile"):
            batch_rows.register_layout(torch.int64, (3,))
        for capacity in (1, 2):
            with self.assertRaisesRegex(RuntimeError, "before precompile"):
                batch_rows.configure_capacity(capacity)
        self.assertEqual(batch_rows._capacity(), 1)
        with self.assertRaisesRegex(RuntimeError, "before precompile"):
            batch_rows.register_layout(torch.int64, (3,))

    def test_non_tpu_tensors_keep_the_plain_ops(self):
        a, b = torch.arange(3), torch.arange(4)
        with patch.object(batch_rows, "_capacity", side_effect=AssertionError):
            self.assertTrue(torch.equal(batch_rows.merge_rows(a, b), torch.cat([a, b])))
            keep = torch.tensor([2, 0])
            self.assertTrue(torch.equal(batch_rows.filter_rows(a, keep), a[keep]))


if __name__ == "__main__":
    unittest.main()
