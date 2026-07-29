# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Byte-identical parity test for the vectorized PCP layout planner.

The naive triple-loop implementation of :func:`build_pcp_rank_major_token_order`
and :func:`pcp_local_token_counts` was replaced with vectorized numpy code
that computes each source token's assigned rank in one pass. This test
re-implements the naive versions inline and asserts byte-identical output
across many parametrized inputs — a regression here would silently
mis-route Q tokens across CP ranks, breaking attention correctness.

Pure Python; no TPU / JAX device required.
"""

from __future__ import annotations

import unittest

import numpy as np

from sglang.srt.layers.attention.pallas.pcp_streaming_rpa.pcp_layout import (
    build_pcp_rank_major_token_order,
    pcp_local_token_counts,
    pcp_query_chunk_ranges,
    pcp_query_start_offsets,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


# --- Naive reference implementations (byte-for-byte the pre-vectorization code) ---


def _naive_pcp_local_token_counts(
    num_scheduled_tokens_per_req,
    pcp_size,
    interleave_size,
    token_start_offsets_per_req=None,
):
    if pcp_size <= 1:
        return np.array([int(np.sum(num_scheduled_tokens_per_req))], dtype=np.int32)
    tokens = np.asarray(num_scheduled_tokens_per_req, dtype=np.int64)
    if tokens.size == 0:
        return np.zeros(pcp_size, dtype=np.int32)
    starts = pcp_query_start_offsets(tokens, token_start_offsets_per_req)
    counts = np.zeros(pcp_size, dtype=np.int64)
    for q_len, q_start in zip(tokens, starts):
        for rank in range(pcp_size):
            for chunk_start, chunk_end in pcp_query_chunk_ranges(
                int(q_len), int(q_start), rank, pcp_size, interleave_size
            ):
                counts[rank] += chunk_end - chunk_start
    return counts.astype(np.int32)


def _naive_build_pcp_rank_major_token_order(
    num_scheduled_tokens_per_req,
    pcp_size,
    interleave_size,
    padded_num_tokens,
    token_start_offsets_per_req=None,
):
    if pcp_size <= 1:
        total = int(np.sum(num_scheduled_tokens_per_req))
        token_order = np.full(padded_num_tokens, -1, dtype=np.int64)
        token_order[:total] = np.arange(total, dtype=np.int64)
        return token_order, np.arange(total, dtype=np.int64)
    local_padded_num_tokens = padded_num_tokens // pcp_size
    token_order = np.full(padded_num_tokens, -1, dtype=np.int64)
    total_num_tokens = int(np.sum(num_scheduled_tokens_per_req))
    inverse_order = np.full(total_num_tokens, -1, dtype=np.int64)
    starts = pcp_query_start_offsets(
        np.asarray(num_scheduled_tokens_per_req, dtype=np.int64),
        token_start_offsets_per_req,
    )
    req_start = 0
    rank_offsets = np.zeros(pcp_size, dtype=np.int64)
    for num_tokens, q_global_base in zip(num_scheduled_tokens_per_req, starts):
        num_tokens = int(num_tokens)
        for rank in range(pcp_size):
            for chunk_start, chunk_end in pcp_query_chunk_ranges(
                num_tokens, int(q_global_base), rank, pcp_size, interleave_size
            ):
                chunk_len = chunk_end - chunk_start
                dst_start = rank * local_padded_num_tokens + rank_offsets[rank]
                dst_end = dst_start + chunk_len
                if dst_end > (rank + 1) * local_padded_num_tokens:
                    raise ValueError(
                        "PCP local token count exceeds padded local capacity."
                    )
                local_chunk_start = chunk_start - q_global_base
                local_chunk_end = chunk_end - q_global_base
                src = np.arange(
                    req_start + local_chunk_start,
                    req_start + local_chunk_end,
                    dtype=np.int64,
                )
                token_order[dst_start:dst_end] = src
                inverse_order[src] = np.arange(dst_start, dst_end, dtype=np.int64)
                rank_offsets[rank] += chunk_len
        req_start += num_tokens
    if total_num_tokens and np.any(inverse_order < 0):
        raise ValueError("PCP token packing did not cover all tokens.")
    return token_order, inverse_order


# --- Parity test cases ---


CASES = [
    # (name, num_scheduled_per_req, pcp_size, interleave, padded, starts_or_None)
    ("single_req_pcp2_int1", [8], 2, 1, 8, None),
    ("single_req_pcp4_int1", [8], 4, 1, 8, None),
    ("single_req_pcp8_int1", [16], 8, 1, 16, None),
    ("single_req_pcp2_int2", [16], 2, 2, 16, None),
    ("single_req_pcp4_int2", [32], 4, 2, 32, None),
    ("single_req_pcp8_int128", [8192], 8, 128, 8192, None),
    ("multi_req_pcp2_int1", [4, 4], 2, 1, 8, None),
    ("multi_req_pcp4_int1", [8, 8, 8], 4, 1, 24, None),
    ("multi_req_pcp8_uneven", [16, 24, 8, 16], 8, 1, 64, None),
    ("multi_req_pcp4_int2", [16, 16], 4, 2, 32, None),
    # Offsets shift per-rank counts, so padded must fit the worst-case rank.
    # For [q_lens], pcp, interleave with offsets, the safe padded is
    # ~2 * total rounded to (pcp*interleave). Under-padding raises in both
    # naive and vectorized impls — see negative test below.
    ("with_offsets_pcp4", [8, 8], 4, 1, 32, [0, 100]),
    ("with_offsets_pcp8_int8", [64, 32], 8, 8, 192, [0, 512]),
    ("large_single_req_pcp8", [4096], 8, 1, 4096, None),
    ("large_multi_req_pcp8", [512, 512, 512, 512], 8, 1, 2048, None),
    ("padded_larger_than_total", [7], 2, 1, 8, None),
    ("padded_much_larger", [3, 5], 4, 1, 16, None),
]


class BuildTokenOrderParityTest(CustomTestCase):
    def test_all_cases_byte_identical(self) -> None:
        for name, scheduled, pcp_size, interleave, padded, offsets in CASES:
            with self.subTest(case=name):
                new_order, new_inv = build_pcp_rank_major_token_order(
                    scheduled,
                    pcp_size=pcp_size,
                    interleave_size=interleave,
                    padded_num_tokens=padded,
                    token_start_offsets_per_req=offsets,
                )
                ref_order, ref_inv = _naive_build_pcp_rank_major_token_order(
                    scheduled,
                    pcp_size=pcp_size,
                    interleave_size=interleave,
                    padded_num_tokens=padded,
                    token_start_offsets_per_req=offsets,
                )
                np.testing.assert_array_equal(new_order, ref_order)
                np.testing.assert_array_equal(new_inv, ref_inv)

    def test_random_stress(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(50):
            n_reqs = int(rng.integers(1, 6))
            pcp_size = int(rng.choice([2, 4, 8]))
            interleave = int(rng.choice([1, 2, 4, 8]))
            # Each req length is multiple of pcp*interleave to keep the layout well-defined.
            block = pcp_size * interleave
            scheduled = [int(rng.integers(1, 8)) * block for _ in range(n_reqs)]
            total = sum(scheduled)
            padded = ((total + block - 1) // block) * block
            offsets = [int(rng.integers(0, 4)) * block for _ in range(n_reqs)]
            new_order, new_inv = build_pcp_rank_major_token_order(
                scheduled, pcp_size, interleave, padded, offsets
            )
            ref_order, ref_inv = _naive_build_pcp_rank_major_token_order(
                scheduled, pcp_size, interleave, padded, offsets
            )
            np.testing.assert_array_equal(new_order, ref_order)
            np.testing.assert_array_equal(new_inv, ref_inv)

    def test_empty_batch(self) -> None:
        new_order, new_inv = build_pcp_rank_major_token_order(
            [], pcp_size=4, interleave_size=1, padded_num_tokens=8
        )
        # Empty batch: token_order all -1 padding, inverse_order empty.
        self.assertEqual(new_order.shape, (8,))
        self.assertTrue((new_order == -1).all())
        self.assertEqual(new_inv.shape, (0,))

    def test_under_padded_raises_matching_naive(self) -> None:
        # Under-padding forces per-rank count > local_padded_num_tokens.
        # Both naive and vectorized impls must raise ValueError.
        with self.assertRaises(ValueError):
            build_pcp_rank_major_token_order(
                [64, 32],
                pcp_size=8,
                interleave_size=8,
                padded_num_tokens=96,
                token_start_offsets_per_req=[0, 512],
            )
        with self.assertRaises(ValueError):
            _naive_build_pcp_rank_major_token_order(
                [64, 32],
                pcp_size=8,
                interleave_size=8,
                padded_num_tokens=96,
                token_start_offsets_per_req=[0, 512],
            )


class LocalTokenCountsParityTest(CustomTestCase):
    def test_all_cases_byte_identical(self) -> None:
        for name, scheduled, pcp_size, interleave, _, offsets in CASES:
            with self.subTest(case=name):
                new_counts = pcp_local_token_counts(
                    scheduled,
                    pcp_size=pcp_size,
                    interleave_size=interleave,
                    token_start_offsets_per_req=offsets,
                )
                ref_counts = _naive_pcp_local_token_counts(
                    scheduled,
                    pcp_size=pcp_size,
                    interleave_size=interleave,
                    token_start_offsets_per_req=offsets,
                )
                np.testing.assert_array_equal(new_counts, ref_counts)


if __name__ == "__main__":
    unittest.main()
