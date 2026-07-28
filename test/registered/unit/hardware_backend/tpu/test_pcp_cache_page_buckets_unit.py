# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for the pcp_cache_pages bucket helpers.

Covers :func:`pcp_cache_page_buckets` (static compile-bucket generator) and
:func:`round_up_pcp_cache_pages` (per-request quantizer). Pure Python; no
TPU / JAX device required.

The helpers are the compile-time bound used by the PCP gather-KV cache
phase to bound the ``lax.all_gather`` payload — a short request served
by a server with a large ``max_model_len`` must gather only its live KV
pages, not the whole ``max_num_blocks_per_req`` block-table width.
"""

from __future__ import annotations

import unittest

from sglang.srt.layers.cp.pcp_sequence_layout import (
    PCP_CACHE_PAGE_BUCKET_COUNT,
    pcp_cache_page_buckets,
    round_up_pcp_cache_pages,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class PcpCachePageBucketsTest(CustomTestCase):
    def test_zero_max_returns_only_zero(self) -> None:
        # No pages -> the single bucket is the empty-cache case.
        self.assertEqual(pcp_cache_page_buckets(0), [0])

    def test_one_max_returns_zero_and_one(self) -> None:
        self.assertEqual(pcp_cache_page_buckets(1), [0, 1])

    def test_negative_max_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be >= 0"):
            pcp_cache_page_buckets(-1)

    def test_max_two_gives_three_buckets(self) -> None:
        # {0, 2} + one log-spaced interior = at most 3 buckets.
        b = pcp_cache_page_buckets(2)
        self.assertEqual(b[0], 0)
        self.assertEqual(b[-1], 2)
        self.assertLessEqual(len(b), PCP_CACHE_PAGE_BUCKET_COUNT)

    def test_exact_bucket_tables(self) -> None:
        # Pin the exact bucket layout for values a reviewer can eyeball.
        # If the log-spacing / power-of-two rounding changes, these break loud
        # instead of silently drifting.
        cases = {
            0: [0],
            1: [0, 1],
            3: [0, 1, 2, 3],
            4: [0, 1, 2, 4],
            8: [0, 2, 4, 8],
            16: [0, 2, 4, 8, 16],
            1024: [0, 4, 32, 128, 1024],
            8192: [0, 8, 64, 512, 8192],
        }
        for maxn, expected in cases.items():
            with self.subTest(maxn=maxn):
                self.assertEqual(pcp_cache_page_buckets(maxn), expected)

    def test_max_1024_has_exactly_five_buckets(self) -> None:
        b = pcp_cache_page_buckets(1024)
        self.assertEqual(len(b), PCP_CACHE_PAGE_BUCKET_COUNT)
        self.assertEqual(len(b), len(set(b)))
        self.assertEqual(b, sorted(b))
        self.assertEqual(b[0], 0)
        self.assertEqual(b[-1], 1024)
        # Interior values are power-of-two.
        for v in b[1:-1]:
            self.assertEqual(v & (v - 1), 0, f"bucket {v} is not a power of two")

    def test_max_8192_full_five_bucket_spread(self) -> None:
        b = pcp_cache_page_buckets(8192)
        self.assertEqual(len(b), PCP_CACHE_PAGE_BUCKET_COUNT)
        self.assertEqual(b[0], 0)
        self.assertEqual(b[-1], 8192)
        # Interior monotonically increasing.
        self.assertEqual(b, sorted(b))

    def test_bucket_contains_zero_and_max_always(self) -> None:
        # Sweep several plausible max_num_blocks_per_req values.
        # Skip maxn=0 (already covered by test_zero_max_returns_only_zero,
        # would degenerately assert 0 is in {0} twice).
        for maxn in (1, 4, 16, 128, 512, 2048, 65536):
            b = pcp_cache_page_buckets(maxn)
            self.assertIn(0, b)
            self.assertIn(maxn, b)

    def test_result_is_cached(self) -> None:
        # Underlying _pcp_cache_page_buckets_cached uses lru_cache so repeated
        # calls with the same arg return the same tuple. Verify by identity on
        # the underlying tuple (list wrapper is a fresh copy each call).
        from sglang.srt.layers.cp.pcp_sequence_layout import (
            _pcp_cache_page_buckets_cached,
        )

        a = _pcp_cache_page_buckets_cached(1024)
        b = _pcp_cache_page_buckets_cached(1024)
        self.assertIs(a, b)


class RoundUpPcpCachePagesTest(CustomTestCase):
    def test_zero_tokens_returns_zero(self) -> None:
        # First-chunk case: no cached tokens, bucket is 0.
        self.assertEqual(
            round_up_pcp_cache_pages(
                num_computed_tokens=0, page_size=128, max_num_blocks_per_req=1024
            ),
            0,
        )

    def test_negative_tokens_returns_zero(self) -> None:
        self.assertEqual(
            round_up_pcp_cache_pages(
                num_computed_tokens=-5, page_size=128, max_num_blocks_per_req=1024
            ),
            0,
        )

    def test_zero_page_size_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "page_size must be > 0"):
            round_up_pcp_cache_pages(
                num_computed_tokens=128, page_size=0, max_num_blocks_per_req=1024
            )

    def test_picks_tightest_bucket(self) -> None:
        # max=1024, buckets look like [0, 4, 16, 128, 1024] (log-spaced po2).
        # 1 token -> 1 page live -> smallest non-zero bucket.
        page_size = 128
        max_n = 1024
        buckets = pcp_cache_page_buckets(max_n)
        picked = round_up_pcp_cache_pages(
            num_computed_tokens=1,
            page_size=page_size,
            max_num_blocks_per_req=max_n,
        )
        self.assertEqual(picked, buckets[1])  # smallest > 0

    def test_exact_page_boundary(self) -> None:
        # 128 tokens with page_size=128 -> exactly 1 live page. Picked bucket
        # should be the smallest bucket >= 1.
        max_n = 1024
        buckets = pcp_cache_page_buckets(max_n)
        picked = round_up_pcp_cache_pages(
            num_computed_tokens=128,
            page_size=128,
            max_num_blocks_per_req=max_n,
        )
        self.assertGreaterEqual(picked, 1)
        self.assertIn(picked, buckets)

    def test_full_context_returns_max(self) -> None:
        # A request that fills the block-table row round-trips to max_n.
        max_n = 1024
        picked = round_up_pcp_cache_pages(
            num_computed_tokens=max_n * 128,  # exactly max pages worth
            page_size=128,
            max_num_blocks_per_req=max_n,
        )
        self.assertEqual(picked, max_n)

    def test_overflow_returns_max(self) -> None:
        # If somehow more tokens than max_n * page_size, still returns max_n.
        max_n = 1024
        picked = round_up_pcp_cache_pages(
            num_computed_tokens=max_n * 128 + 999,
            page_size=128,
            max_num_blocks_per_req=max_n,
        )
        self.assertEqual(picked, max_n)

    def test_monotone_in_num_computed(self) -> None:
        # Increasing computed tokens must produce non-decreasing bucket.
        max_n = 1024
        prev = 0
        for tokens in (0, 1, 127, 128, 500, 1024, 4096, 16384, 65536):
            picked = round_up_pcp_cache_pages(
                num_computed_tokens=tokens,
                page_size=128,
                max_num_blocks_per_req=max_n,
            )
            self.assertGreaterEqual(picked, prev)
            prev = picked


if __name__ == "__main__":
    unittest.main()
