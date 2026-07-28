# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Pure-NumPy oracle parity test for the PCP streaming reference.

Compares :func:`execute_pcp_streaming_reference` (the NumPy oracle used by
the Pallas PCP streaming kernel's schedule) against a naive full-attention
reference on the shapes Qwen3-Coder presents (page_size=128, kv_packing=2
for bf16, head_dim=128).

Scope: this test only verifies the NumPy oracle + schedule + layout math.
The actual Pallas kernel is NOT exercised here (that requires a TPU host
and lives under ``test/manual/tpu/``). A kernel-vs-oracle parity test is
a natural follow-up.

No TPU / JAX device is required — runs on any host, so this lives under
``test/registered/unit/`` and runs in CI.

Usage::

    cd /path/to/sglang-torchtpu
    python -m unittest test.registered.unit.hardware_backend.tpu.test_pcp_streaming_reference_parity
"""

from __future__ import annotations

import unittest

import numpy as np

from sglang.srt.layers.attention.pallas.pcp_streaming_rpa.pcp_layout import (
    build_pcp_rank_major_token_order,
)
from sglang.srt.layers.attention.pallas.pcp_streaming_rpa.reference import (
    execute_pcp_streaming_reference,
)
from sglang.srt.layers.attention.pallas.pcp_streaming_rpa.schedule import (
    generate_pcp_streaming_schedule,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def _pack_q_rank_major(
    q_full: np.ndarray, token_order: np.ndarray, pcp_size: int
) -> np.ndarray:
    """Reorder (q_len, kv_heads, q_per_kv, head_dim) into rank-packed layout."""
    padded = np.zeros((token_order.size,) + q_full.shape[1:], dtype=np.float32)
    valid = token_order >= 0
    padded[valid] = q_full[token_order[valid]]
    local_padded = token_order.size // pcp_size
    return padded.reshape((pcp_size, local_padded) + q_full.shape[1:])


def _build_pcp_kv_cache(
    k_full: np.ndarray,
    v_full: np.ndarray,
    block_tables: np.ndarray,
    page_size: int,
    pcp_size: int,
) -> np.ndarray:
    """Split (kv_len, kv_heads, head_dim) K/V into per-rank paged layout."""
    num_pages = int(block_tables.max()) + 1
    _, kv_heads, head_dim = k_full.shape
    kv_cache = np.zeros(
        (pcp_size, num_pages, page_size, kv_heads, 2, head_dim),
        dtype=np.float32,
    )
    for pos in range(k_full.shape[0]):
        global_page = pos // page_size
        src_rank = global_page % pcp_size
        local_page_index = global_page // pcp_size
        page = int(block_tables[0, local_page_index])
        offset = pos % page_size
        kv_cache[src_rank, page, offset, :, 0, :] = k_full[pos]
        kv_cache[src_rank, page, offset, :, 1, :] = v_full[pos]
    return kv_cache


def _naive_full_attention(
    q_full: np.ndarray,
    k_full: np.ndarray,
    v_full: np.ndarray,
    q_global_base: int,
    sm_scale: float,
) -> np.ndarray:
    """(q_len, kv_heads, q_per_kv, head_dim) -> same, single-req causal attn."""
    q_len, kv_heads, q_per_kv, head_dim = q_full.shape
    out = np.zeros_like(q_full, dtype=np.float32)
    for q_idx in range(q_len):
        q_global = q_global_base + q_idx
        k_ctx = k_full[: q_global + 1]
        v_ctx = v_full[: q_global + 1]
        scores = np.einsum("hqd,shd->hqs", q_full[q_idx], k_ctx) * sm_scale
        scores = scores - np.max(scores, axis=-1, keepdims=True)
        probs = np.exp(scores)
        probs = probs / np.sum(probs, axis=-1, keepdims=True)
        out[q_idx] = np.einsum("hqs,shd->hqd", probs, v_ctx)
    return out


def _run_reference_case(
    *,
    q_len: int,
    q_global_base: int,
    kv_len: int,
    pcp_size: int,
    page_size: int,
    interleave_size: int,
    kv_heads: int,
    q_per_kv: int,
    head_dim: int,
    num_lanes: int,
    bq_sz: int,
    seed: int = 123,
) -> None:
    rng = np.random.default_rng(seed)
    q_full = rng.normal(size=(q_len, kv_heads, q_per_kv, head_dim)).astype(np.float32)
    k_full = rng.normal(size=(kv_len, kv_heads, head_dim)).astype(np.float32)
    v_full = rng.normal(size=(kv_len, kv_heads, head_dim)).astype(np.float32)
    block_tables = np.arange(
        (kv_len + page_size * pcp_size - 1) // (page_size * pcp_size),
        dtype=np.int32,
    ).reshape(1, -1)

    padded_num_tokens = ((q_len + pcp_size - 1) // pcp_size) * pcp_size
    token_order, inverse_order = build_pcp_rank_major_token_order(
        [q_len],
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        padded_num_tokens=padded_num_tokens,
        token_start_offsets_per_req=[q_global_base],
    )
    q_by_rank = _pack_q_rank_major(q_full, token_order, pcp_size)
    kv_cache = _build_pcp_kv_cache(k_full, v_full, block_tables, page_size, pcp_size)
    schedule = generate_pcp_streaming_schedule(
        kv_lens=[kv_len],
        cu_q_lens=[0, q_len],
        q_start_offsets=[q_global_base],
        block_tables=block_tables,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        num_lanes=num_lanes,
        bq_sz=bq_sz,
        pad_kv_pages_to_pcp_group=True,
    )
    sm_scale = 1.0 / float(np.sqrt(head_dim))

    packed_output = execute_pcp_streaming_reference(
        q_by_rank,
        kv_cache,
        schedule,
        sm_scale=sm_scale,
        interleave_size=interleave_size,
    )
    unpacked_output = packed_output.reshape((padded_num_tokens,) + q_full.shape[1:])[
        inverse_order
    ]
    expected = _naive_full_attention(q_full, k_full, v_full, q_global_base, sm_scale)
    np.testing.assert_allclose(unpacked_output, expected, rtol=1e-4, atol=1e-4)


class TestPcpStreamingReferenceParity(unittest.TestCase):
    """Reference vs naive full attention on Qwen3-Coder-shaped inputs."""

    def test_single_lane_pcp4(self) -> None:
        """Small full-prefill (q_len==kv_len==8, pcp_size=4)."""
        _run_reference_case(
            q_len=8,
            q_global_base=0,
            kv_len=8,
            pcp_size=4,
            page_size=2,
            interleave_size=2,
            kv_heads=2,
            q_per_kv=2,
            head_dim=4,
            num_lanes=1,
            bq_sz=2,
        )

    def test_full_prefill_pcp4(self) -> None:
        """Full-prefill (q_global_base=0, kv_len==q_len) at pcp_size=4."""
        _run_reference_case(
            q_len=16,
            q_global_base=0,
            kv_len=16,
            pcp_size=4,
            page_size=2,
            interleave_size=2,
            kv_heads=2,
            q_per_kv=2,
            head_dim=4,
            num_lanes=1,
            bq_sz=2,
        )

    def test_qwen3_coder_shape_scaled(self) -> None:
        """Qwen3-Coder-480B ratios except page_size (scaled 128 -> 8 for speed).

        head_dim=128 and GQA q_per_kv=8 are the real Qwen3-Coder-480B
        values; page_size / interleave / kv_heads / bq_sz are shrunk to
        keep the pure-NumPy oracle test cheap.
        """
        _run_reference_case(
            q_len=64,
            q_global_base=0,
            kv_len=64,
            pcp_size=4,
            page_size=8,
            interleave_size=8,
            kv_heads=2,
            q_per_kv=8,
            head_dim=128,
            num_lanes=1,
            bq_sz=8,
        )

    def test_decode_shape_padded_to_kernel_bucket(self) -> None:
        """Decode oracle at the smallest kernel-legal q_len.

        The PCP streaming schedule requires
        ``q_len % (pcp_size * interleave_size) == 0``. In real serve, decode
        (session-5's bbf24a893 route) pads its 1-real-token Q up to
        ``pcp_size * PCP_STREAMING_RPA_LOCAL_COMPILE_TOKEN_MULTIPLE``. The
        pure-NumPy oracle here isn't the padder — the padder lives in
        ``layers/cp/utils.py::_build_pcp_layout_plan`` — so this test just
        exercises the kernel-legal q_len that the padded batch presents
        after continuation from prior context.
        """
        _run_reference_case(
            q_len=16,
            q_global_base=16,  # 16 tokens already in KV
            kv_len=32,
            pcp_size=8,
            page_size=2,
            interleave_size=2,
            kv_heads=2,
            q_per_kv=2,
            head_dim=16,
            num_lanes=1,
            bq_sz=2,
        )

    def test_pcp_size_2(self) -> None:
        _run_reference_case(
            q_len=8,
            q_global_base=0,
            kv_len=8,
            pcp_size=2,
            page_size=2,
            interleave_size=2,
            kv_heads=2,
            q_per_kv=4,
            head_dim=16,
            num_lanes=1,
            bq_sz=2,
        )

    def test_pcp_size_8(self) -> None:
        _run_reference_case(
            q_len=32,
            q_global_base=0,
            kv_len=32,
            pcp_size=8,
            page_size=2,
            interleave_size=2,
            kv_heads=2,
            q_per_kv=2,
            head_dim=16,
            num_lanes=1,
            bq_sz=2,
        )

    def test_chunked_prefill_continuation(self) -> None:
        """Prior context already in KV cache; only new q_len rows are Q."""
        _run_reference_case(
            q_len=8,
            q_global_base=16,  # prior 16 tokens already in KV
            kv_len=24,
            pcp_size=4,
            page_size=2,
            interleave_size=2,
            kv_heads=2,
            q_per_kv=2,
            head_dim=8,
            num_lanes=1,
            bq_sz=2,
        )


if __name__ == "__main__":
    unittest.main()
