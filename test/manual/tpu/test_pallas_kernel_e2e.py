"""End-to-end tests for the Pallas ragged-paged-attention kernel.

These tests require a TPU device and torch_tpu installed. They:

* Allocate a fused KV cache, Q/K/V inputs, and the rectangular metadata
  produced by ``PallasAttnBackend._build_metadata``.
* Run the kernel both eagerly and through ``torch.compile(backend="tpu")``.
* Compare the result against a numpy/torch reference attention implementation
  for a small decode and a small extend case.
* Verify that ``torch.compile`` traces the kernel through the custom-op
  boundary without errors and produces numerically matching output.
"""

from __future__ import annotations

import os
import unittest

import numpy as np
import torch

# ``torch_tpu`` registers the ``tpu`` device + the ``tpu`` torch.compile
# backend as side-effects of import. ``compile`` registers the dynamo
# backend; importing it explicitly mirrors what ``apply_tpu_patches`` does
# at server startup so this file works standalone.
try:
    import torch_tpu  # noqa: F401
    from torch_tpu._internal import compile as _torch_tpu_compile  # noqa: F401

    _HAS_TPU = True
except Exception:  # pragma: no cover
    _HAS_TPU = False


def _has_tpu_device() -> bool:
    if not _HAS_TPU:
        return False
    try:
        torch.empty(1, device="tpu")
        return True
    except Exception:
        return False


_SKIP_REASON = "Requires torch_tpu + a TPU device"


def _ref_attention_torch(
    q: torch.Tensor,  # [num_tokens, num_q_heads, head_dim]
    k_full: torch.Tensor,  # [num_tokens_full, num_kv_heads, head_dim]
    v_full: torch.Tensor,  # [num_tokens_full, num_kv_heads, head_dim]
    cu_q_lens: torch.Tensor,  # [batch + 1]
    kv_lens: torch.Tensor,  # [batch]
    sm_scale: float,
) -> torch.Tensor:
    """Plain causal attention reference, one sequence at a time.

    Used as ground truth for the Pallas kernel output. Computes attention on
    CPU in float32 to avoid TPU-specific numerical paths bleeding into the
    reference.
    """
    out = torch.empty_like(q)
    num_q_heads = q.shape[1]
    num_kv_heads = k_full.shape[1]
    assert num_q_heads % num_kv_heads == 0
    repeats = num_q_heads // num_kv_heads
    kv_offset = 0
    for i in range(len(kv_lens)):
        q_start = int(cu_q_lens[i].item())
        q_end = int(cu_q_lens[i + 1].item())
        q_len = q_end - q_start
        kv_len = int(kv_lens[i].item())
        if q_len == 0:
            continue
        kv_q_gap = kv_len - q_len
        q_seq = q[q_start:q_end].float()
        k_seq = k_full[kv_offset : kv_offset + kv_len].float()
        v_seq = v_full[kv_offset : kv_offset + kv_len].float()
        kv_offset += kv_len
        # Repeat KV heads to match Q heads for GQA.
        k_seq = k_seq.repeat_interleave(repeats, dim=1)
        v_seq = v_seq.repeat_interleave(repeats, dim=1)
        # [q_len, num_heads, head_dim] -> [num_heads, q_len, head_dim]
        q_h = q_seq.movedim(0, 1)
        k_h = k_seq.movedim(0, 1)
        v_h = v_seq.movedim(0, 1)
        attn = torch.einsum("hqd,hkd->hqk", q_h, k_h) * sm_scale
        # Causal mask: q at absolute position kv_q_gap + i may attend to k positions
        # 0..kv_q_gap + i.
        q_positions = kv_q_gap + torch.arange(q_len)
        k_positions = torch.arange(kv_len)
        mask = q_positions[:, None] < k_positions[None, :]  # True = mask out
        attn = attn.masked_fill(mask[None, :, :], float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        out_h = torch.einsum("hqk,hkd->hqd", attn, v_h)
        out[q_start:q_end] = out_h.movedim(0, 1).to(out.dtype)
    return out


def _build_fused_kv_cache(
    num_blocks: int,
    page_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: str,
) -> torch.Tensor:
    """Return a zero-initialised fused KV cache shaped like the kernel wants.

    Layout: ``[num_blocks, page_size, num_kv_heads_x2 // kv_packing,
    kv_packing, head_dim]`` where ``kv_packing = 32 // bits_per_elem`` (2
    for bfloat16). K and V are interleaved along the kv-heads axis.
    """
    bits = torch.empty((), dtype=dtype).element_size() * 8
    kv_packing = 32 // bits
    num_kv_heads_x2 = num_kv_heads * 2  # already a multiple of kv_packing for bf16/f32
    assert num_kv_heads_x2 % kv_packing == 0
    return torch.zeros(
        (num_blocks, page_size, num_kv_heads_x2 // kv_packing, kv_packing, head_dim),
        dtype=dtype,
        device=device,
    )


def _flatten_fused_kv_to_loose(
    kv_cache: torch.Tensor, num_kv_heads: int
) -> torch.Tensor:
    """Materialise a logical ``[num_tokens, 2 * num_kv_heads, head_dim]`` view
    from the fused 5D cache so the reference can pull out K and V slices.
    """
    num_blocks, page_size, packed, packing, head_dim = kv_cache.shape
    return kv_cache.reshape(num_blocks * page_size, packed * packing, head_dim)[
        :, : 2 * num_kv_heads, :
    ]


def _build_rectangular_metadata(
    seq_lens: list[int],
    extend_seq_lens: list[int] | None,
    pages_per_seq: int,
    page_size: int,
    *,
    page_table: list[list[int]],
    device: str,
) -> dict:
    """Build the metadata tensors expected by the Pallas kernel via the
    same rectangular layout the backend uses.

    ``page_table[i][j]`` is the physical page index for page ``j`` of seq ``i``.
    """
    batch = len(seq_lens)
    if extend_seq_lens is None:
        extend_seq_lens = [1] * batch

    cu_q_lens = torch.tensor(
        [0] + list(np.cumsum(extend_seq_lens).tolist()), dtype=torch.int32, device=device
    )
    cu_kv_lens = torch.arange(batch + 1, dtype=torch.int32, device=device) * (
        pages_per_seq * page_size
    )

    page_indices = torch.tensor(
        [p for row in page_table for p in row] + [0] * (batch * pages_per_seq - sum(len(r) for r in page_table)),
        dtype=torch.int32,
        device=device,
    )

    kv_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    num_active = int((kv_lens_t > 0).sum().item())
    # decode-only distribution; we override at call site for extend mode.
    distribution_decode = torch.tensor(
        [num_active, num_active, num_active], dtype=torch.int32, device=device
    )
    distribution_extend = torch.tensor(
        [0, num_active, num_active], dtype=torch.int32, device=device
    )

    return {
        "kv_lens": kv_lens_t,
        "cu_q_lens": cu_q_lens,
        "cu_kv_lens": cu_kv_lens,
        "page_indices": page_indices,
        "distribution_decode": distribution_decode,
        "distribution_extend": distribution_extend,
    }


@unittest.skipUnless(_has_tpu_device(), _SKIP_REASON)
class TestPallasKernelEager(unittest.TestCase):
    """Run the Pallas kernel in eager mode and validate output."""

    @classmethod
    def setUpClass(cls):
        from sglang.srt.layers.attention.pallas_backend import (
            _build_pallas_kernel_op,
        )

        cls.op = _build_pallas_kernel_op()
        cls.device = "tpu"
        cls.dtype = torch.bfloat16
        cls.page_size = 16
        cls.num_q_heads = 4
        cls.num_kv_heads = 2  # GQA = 2x
        cls.head_dim = 128
        cls.sm_scale = cls.head_dim ** -0.5

    def _run_kernel(self, q, k, v, kv_cache, md, distribution):
        return self.op(
            q,
            k,
            v,
            kv_cache,
            md["kv_lens"],
            md["page_indices"],
            md["cu_q_lens"],
            md["cu_kv_lens"],
            distribution,
            float(self.sm_scale),
            None,  # sliding_window
            None,  # soft_cap
        )

    def _materialise_prefix_into_cache(
        self,
        kv_cache: torch.Tensor,
        page_table_row: list[int],
        k_prefix: torch.Tensor,  # [prefix_len, num_kv_heads, head_dim]
        v_prefix: torch.Tensor,  # [prefix_len, num_kv_heads, head_dim]
    ):
        """Write prefix K/V into the fused cache so the kernel can read them."""
        prefix_len = k_prefix.shape[0]
        # K and V interleaved along the kv-heads axis.
        # Build a token-flat 3D view: [num_blocks * page_size, 2*num_kv_heads, head_dim].
        num_kv_heads = k_prefix.shape[1]
        logical = _flatten_fused_kv_to_loose(kv_cache, num_kv_heads)
        # Write token positions 0..prefix_len of the first page(s) used by this seq.
        for tok in range(prefix_len):
            page_idx = page_table_row[tok // self.page_size]
            slot = page_idx * self.page_size + (tok % self.page_size)
            logical[slot, 0::2] = k_prefix[tok]
            logical[slot, 1::2] = v_prefix[tok]

    def test_decode_single_seq_matches_reference(self):
        """A single-sequence decode (1 new token, 31 prefix tokens) matches CPU SDPA."""
        seq_lens = [32]
        extend_lens = [1]
        prefix_len = 31
        num_blocks = 4
        pages_per_seq = num_blocks  # one seq using all blocks
        page_table = [[0, 1]]  # first 32 tokens spread over 2 pages

        kv_cache = _build_fused_kv_cache(
            num_blocks,
            self.page_size,
            self.num_kv_heads,
            self.head_dim,
            self.dtype,
            self.device,
        )

        # Generate prefix on CPU, copy to TPU.
        torch.manual_seed(0)
        k_prefix = torch.randn(prefix_len, self.num_kv_heads, self.head_dim, dtype=self.dtype)
        v_prefix = torch.randn(prefix_len, self.num_kv_heads, self.head_dim, dtype=self.dtype)
        self._materialise_prefix_into_cache(
            kv_cache, page_table[0], k_prefix.to(self.device), v_prefix.to(self.device)
        )

        q = torch.randn(1, self.num_q_heads, self.head_dim, dtype=self.dtype).to(self.device)
        k_new = torch.randn(1, self.num_kv_heads, self.head_dim, dtype=self.dtype).to(self.device)
        v_new = torch.randn(1, self.num_kv_heads, self.head_dim, dtype=self.dtype).to(self.device)

        md = _build_rectangular_metadata(
            seq_lens=seq_lens,
            extend_seq_lens=extend_lens,
            pages_per_seq=pages_per_seq,
            page_size=self.page_size,
            page_table=page_table,
            device=self.device,
        )

        out, updated_kv = self._run_kernel(q, k_new, v_new, kv_cache, md, md["distribution_decode"])
        kv_cache.copy_(updated_kv)
        torch.tpu.synchronize()

        # Build reference: full K/V is prefix + new.
        k_full = torch.cat([k_prefix, k_new.cpu()], dim=0)
        v_full = torch.cat([v_prefix, v_new.cpu()], dim=0)
        ref = _ref_attention_torch(
            q.cpu(),
            k_full,
            v_full,
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([32], dtype=torch.int32),
            self.sm_scale,
        )

        got = out.cpu().float()
        ref = ref.float()
        self.assertEqual(got.shape, ref.shape)
        max_diff = (got - ref).abs().max().item()
        self.assertLess(max_diff, 5e-2, f"Pallas vs reference max abs diff = {max_diff}")

    def test_extend_two_seqs_matches_reference(self):
        """Two-sequence extend (different q/kv lens) matches CPU SDPA."""
        # seq0: 4 new tokens on top of 12 prefix (kv_len=16, q_len=4)
        # seq1: 8 new tokens on top of 0 prefix  (kv_len=8,  q_len=8)
        seq_lens = [16, 8]
        extend_lens = [4, 8]
        prefix_lens = [12, 0]
        num_blocks = 6
        pages_per_seq = 3
        page_table = [[0, 1, 2], [3, 4, 5]]

        kv_cache = _build_fused_kv_cache(
            num_blocks,
            self.page_size,
            self.num_kv_heads,
            self.head_dim,
            self.dtype,
            self.device,
        )

        torch.manual_seed(1)
        k_prefixes = []
        v_prefixes = []
        for i, prefix_len in enumerate(prefix_lens):
            kp = torch.randn(prefix_len, self.num_kv_heads, self.head_dim, dtype=self.dtype)
            vp = torch.randn(prefix_len, self.num_kv_heads, self.head_dim, dtype=self.dtype)
            self._materialise_prefix_into_cache(
                kv_cache, page_table[i], kp.to(self.device), vp.to(self.device)
            )
            k_prefixes.append(kp)
            v_prefixes.append(vp)

        # Concatenated new (q, k, v) across both sequences, shape [sum(extend_lens), ...].
        total_q = sum(extend_lens)
        q = torch.randn(total_q, self.num_q_heads, self.head_dim, dtype=self.dtype).to(self.device)
        k_new = torch.randn(total_q, self.num_kv_heads, self.head_dim, dtype=self.dtype).to(
            self.device
        )
        v_new = torch.randn(total_q, self.num_kv_heads, self.head_dim, dtype=self.dtype).to(
            self.device
        )

        md = _build_rectangular_metadata(
            seq_lens=seq_lens,
            extend_seq_lens=extend_lens,
            pages_per_seq=pages_per_seq,
            page_size=self.page_size,
            page_table=page_table,
            device=self.device,
        )

        out, updated_kv = self._run_kernel(q, k_new, v_new, kv_cache, md, md["distribution_extend"])
        kv_cache.copy_(updated_kv)
        torch.tpu.synchronize()

        # Build reference K/V (prefix + new) per sequence; concatenate.
        k_full_list = []
        v_full_list = []
        offset = 0
        for i, ext in enumerate(extend_lens):
            k_full_list.append(k_prefixes[i])
            k_full_list.append(k_new[offset : offset + ext].cpu())
            v_full_list.append(v_prefixes[i])
            v_full_list.append(v_new[offset : offset + ext].cpu())
            offset += ext
        k_full = torch.cat(k_full_list, dim=0)
        v_full = torch.cat(v_full_list, dim=0)

        ref = _ref_attention_torch(
            q.cpu(),
            k_full,
            v_full,
            torch.tensor([0, 4, 12], dtype=torch.int32),
            torch.tensor(seq_lens, dtype=torch.int32),
            self.sm_scale,
        )

        got = out.cpu().float()
        ref = ref.float()
        self.assertEqual(got.shape, ref.shape)
        max_diff = (got - ref).abs().max().item()
        self.assertLess(max_diff, 5e-2, f"Pallas vs reference max abs diff = {max_diff}")


@unittest.skipUnless(_has_tpu_device(), _SKIP_REASON)
class TestPallasKernelTorchCompile(unittest.TestCase):
    """Run the Pallas kernel through torch.compile(backend='tpu') and validate output."""

    @classmethod
    def setUpClass(cls):
        from sglang.srt.layers.attention.pallas_backend import (
            _build_pallas_kernel_op,
        )

        cls.op = _build_pallas_kernel_op()
        cls.device = "tpu"
        cls.dtype = torch.bfloat16
        cls.page_size = 16
        cls.num_q_heads = 4
        cls.num_kv_heads = 2
        cls.head_dim = 128
        cls.sm_scale = cls.head_dim ** -0.5

    def _compiled_callable(self):
        op = self.op
        scale = float(self.sm_scale)

        def _run(q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, cu_kv_lens, distribution):
            out, updated_kv = op(
                q, k, v, kv_cache,
                kv_lens, page_indices, cu_q_lens, cu_kv_lens, distribution,
                scale, None, None,
            )
            kv_cache.copy_(updated_kv)
            return out

        return torch.compile(_run, backend="tpu", fullgraph=True, dynamic=False)

    def test_compile_decode_runs_and_matches_eager(self):
        """torch.compile path produces output bit-identical to the eager call."""
        seq_lens = [32]
        extend_lens = [1]
        prefix_len = 31
        num_blocks = 4
        pages_per_seq = num_blocks
        page_table = [[0, 1]]

        # Two identical caches: one for eager, one for compiled, so the
        # ``copy_`` mutations don't cross-contaminate.
        def _make_cache():
            return _build_fused_kv_cache(
                num_blocks,
                self.page_size,
                self.num_kv_heads,
                self.head_dim,
                self.dtype,
                self.device,
            )

        torch.manual_seed(2)
        k_prefix = torch.randn(prefix_len, self.num_kv_heads, self.head_dim, dtype=self.dtype)
        v_prefix = torch.randn(prefix_len, self.num_kv_heads, self.head_dim, dtype=self.dtype)
        q = torch.randn(1, self.num_q_heads, self.head_dim, dtype=self.dtype).to(self.device)
        k_new = torch.randn(1, self.num_kv_heads, self.head_dim, dtype=self.dtype).to(self.device)
        v_new = torch.randn(1, self.num_kv_heads, self.head_dim, dtype=self.dtype).to(self.device)

        md = _build_rectangular_metadata(
            seq_lens=seq_lens,
            extend_seq_lens=extend_lens,
            pages_per_seq=pages_per_seq,
            page_size=self.page_size,
            page_table=page_table,
            device=self.device,
        )

        # Helper to seed cache and run.
        def _seed_and_run(cache, run_fn):
            for tok in range(prefix_len):
                slot = page_table[0][tok // self.page_size] * self.page_size + (
                    tok % self.page_size
                )
                logical = _flatten_fused_kv_to_loose(cache, self.num_kv_heads)
                logical[slot, 0::2] = k_prefix[tok].to(self.device)
                logical[slot, 1::2] = v_prefix[tok].to(self.device)
            return run_fn(cache)

        # Eager path
        cache_eager = _make_cache()
        def _eager_run(cache):
            out, updated_kv = self.op(
                q, k_new, v_new, cache,
                md["kv_lens"], md["page_indices"], md["cu_q_lens"], md["cu_kv_lens"],
                md["distribution_decode"], float(self.sm_scale), None, None,
            )
            cache.copy_(updated_kv)
            return out

        out_eager = _seed_and_run(cache_eager, _eager_run)
        torch.tpu.synchronize()

        # Compiled path
        compiled = self._compiled_callable()
        cache_compiled = _make_cache()
        def _compiled_run(cache):
            return compiled(
                q, k_new, v_new, cache,
                md["kv_lens"], md["page_indices"], md["cu_q_lens"], md["cu_kv_lens"],
                md["distribution_decode"],
            )
        out_compiled = _seed_and_run(cache_compiled, _compiled_run)
        torch.tpu.synchronize()

        out_eager_cpu = out_eager.cpu().float()
        out_compiled_cpu = out_compiled.cpu().float()
        max_diff = (out_eager_cpu - out_compiled_cpu).abs().max().item()
        # bf16 fp accumulation may differ slightly between eager and compiled
        # if XLA fuses ops differently; allow a small tolerance.
        self.assertLess(max_diff, 1e-3, f"eager vs compiled max diff = {max_diff}")

        # And the compiled path itself should match CPU SDPA reference.
        k_full = torch.cat([k_prefix, k_new.cpu()], dim=0)
        v_full = torch.cat([v_prefix, v_new.cpu()], dim=0)
        ref = _ref_attention_torch(
            q.cpu(),
            k_full,
            v_full,
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([32], dtype=torch.int32),
            self.sm_scale,
        )
        max_ref_diff = (out_compiled_cpu - ref.float()).abs().max().item()
        self.assertLess(max_ref_diff, 5e-2, f"compiled vs reference max diff = {max_ref_diff}")


if __name__ == "__main__":
    unittest.main()
