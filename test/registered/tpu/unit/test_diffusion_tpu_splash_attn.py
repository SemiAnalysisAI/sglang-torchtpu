"""Numerics of the multimodal_gen TPU splash attention backend.

Every dispatch branch of ``tpu_attention`` is checked against a float32
reference: the custom Pallas kernel (long KV, including ragged KV tails and a
short sequence-parallel query shard), stock ``make_splash_mha`` for short
aligned cross-attention and medium or causal shapes (including zero-padded
KV, whose padded keys must not take softmax weight), and the SDPA fallback.

The numeric tests need TPU + torch_tpu and are skipped otherwise.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=600, suite="stage-a-test-tpu-unit")

import importlib.util
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.test.test_utils import CustomTestCase


def _skip_if_no_tpu() -> bool:
    try:
        return importlib.util.find_spec("torch_tpu") is None
    except Exception:
        return True


def _reference(q, k, v, scale, is_causal):
    """float32 attention on CPU, chunked over queries."""
    q, k, v = q.float(), k.float(), v.float()
    q_seq, kv_seq = q.shape[2], k.shape[2]
    out = torch.empty_like(q)
    for start in range(0, q_seq, 1024):
        stop = min(start + 1024, q_seq)
        scores = torch.matmul(q[:, :, start:stop], k.transpose(-2, -1)) * scale
        if is_causal:
            rows = torch.arange(start, stop)[:, None]
            cols = torch.arange(kv_seq)[None, :]
            scores = scores.masked_fill(cols > rows, float("-inf"))
        out[:, :, start:stop] = torch.matmul(torch.softmax(scores, dim=-1), v)
    return out


def _reference_lse(q, k, scale):
    """float32 natural-log log-sum-exp of the scaled scores, [B, H, S]."""
    return torch.logsumexp(
        torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale, dim=-1
    )


def _kv_info(row_offset, real_keys):
    return torch.tensor([row_offset, real_keys], dtype=torch.int32).to("tpu")


def _inputs(batch, heads, q_seq, kv_seq, *, q_gain=1.0, negative_scores=False):
    gen = torch.Generator().manual_seed(q_seq * 7919 + kv_seq)
    shape_q = (batch, heads, q_seq, 128)
    shape_kv = (batch, heads, kv_seq, 128)
    q = torch.randn(shape_q, generator=gen) * q_gain
    k = torch.randn(shape_kv, generator=gen)
    v = torch.randn(shape_kv, generator=gen)
    if negative_scores:
        # Every real logit is strongly negative, so a zero-padded key (logit 0)
        # would dominate the softmax if it were not masked.
        direction = torch.ones(128) / 128**0.5
        q = q * 0.1 + 4.0 * direction
        k = k * 0.1 - 4.0 * direction
    return q.bfloat16(), k.bfloat16(), v.bfloat16()


@unittest.skipIf(_skip_if_no_tpu(), "torch_tpu not available")
class TestTpuSplashAttentionNumerics(CustomTestCase):
    def _check(self, batch, heads, q_seq, kv_seq, *, is_causal=False, **kwargs):
        from sglang.multimodal_gen.runtime.layers.attention.backends.tpu_splash_attn import (
            tpu_attention,
        )

        q, k, v = _inputs(batch, heads, q_seq, kv_seq, **kwargs)
        scale = 128**-0.5
        want = _reference(q, k, v, scale, is_causal)
        q, k, v = q.to("tpu"), k.to("tpu"), v.to("tpu")
        got = tpu_attention(q, k, v, scale=scale, is_causal=is_causal).to("cpu")
        sdpa = F.scaled_dot_product_attention(
            q, k, v, scale=scale, is_causal=is_causal
        ).to("cpu")
        self.assertEqual(got.shape, want.shape)
        self.assertEqual(got.dtype, torch.bfloat16)
        self.assertFalse(torch.isnan(got).any().item())
        err = (got.float() - want).abs().max().item()
        sdpa_err = (sdpa.float() - want).abs().max().item()
        # bf16 inputs bound the achievable accuracy; peaked softmaxes amplify
        # it, so allow up to twice the error of bf16 SDPA on the same inputs.
        tol = max(2e-2, 2 * sdpa_err)
        print(f"{self._testMethodName}: max abs err {err:.3e} (sdpa {sdpa_err:.3e})")
        self.assertLess(err, tol, f"max abs error {err:.3e}, sdpa {sdpa_err:.3e}")

    # Custom kernel: non-causal, q_seq >= 256, kv_seq >= 512.
    def test_custom_aligned(self):
        self._check(1, 4, 4608, 4608)

    def test_custom_ragged_kv_tail(self):
        # 4128 = 2 * 2048 + 32: the last KV block is a 32-row partial block.
        self._check(1, 2, 4128, 4128)

    def test_custom_odd_kv_tail(self):
        # 8191 = 3 * 2048 + 2047: an odd-length partial last block.
        self._check(1, 2, 8191, 8191)

    def test_custom_short_query_shard(self):
        # A sequence-parallel query shard attending the gathered KV.
        self._check(1, 2, 880, 3520)

    def test_custom_batched(self):
        # B=2, as with batched CFG.
        self._check(2, 2, 8190, 8190)

    def test_custom_high_logit_gain(self):
        # Large logits make the softmax peaked and amplify rounding.
        self._check(1, 2, 8190, 8190, q_gain=8.0)

    def test_custom_negative_scores(self):
        self._check(1, 2, 4128, 4128, negative_scores=True)

    # Text cross-attention on the custom kernel: long query, short KV.
    def test_cross_attention_aligned_kv(self):
        self._check(1, 4, 4096, 512)

    def test_cross_attention_ragged_query(self):
        # No block size divides 4100 query rows; the edge block is masked.
        self._check(1, 4, 4100, 512)

    def test_cross_attention_unaligned_kv(self):
        # 600 keys: a ragged last KV block.
        self._check(1, 2, 18900, 600, negative_scores=True)

    def test_cross_attention_long_query(self):
        # Short KV leaves VMEM for the largest query block.
        self._check(1, 2, 37800, 512)

    # make_splash_mha, long query against fewer than 512 aligned keys.
    def test_splash_mha_cross_attention(self):
        for kv_seq in (256, 384):
            with self.subTest(kv_seq=kv_seq):
                self._check(1, 2, 4096, kv_seq)

    def test_cross_attention_kv_not_multiple_of_1024(self):
        self._check(1, 2, 4096, 1536, negative_scores=True)

    # Medium shapes (custom kernel) and causal (make_splash_mha).
    def test_medium_ragged_kv(self):
        # 1000 keys: a ragged last KV block.
        self._check(1, 4, 1000, 1000, negative_scores=True)

    def test_medium_aligned(self):
        self._check(1, 2, 1024, 1024)

    def test_short_query_long_padded_kv(self):
        # q_seq < 256 with kv >= 2048: make_splash_mha with segment-id masking.
        self._check(1, 2, 200, 4128, negative_scores=True)

    def test_long_query_unaligned_kv(self):
        # 1024 < kv < 2048, not 128-aligned: a ragged last KV block.
        self._check(1, 2, 4096, 1500, negative_scores=True)

    def test_causal_padded(self):
        self._check(1, 2, 1000, 1000, is_causal=True)

    # Short or unaligned-short KV stays on SDPA.
    def test_short_sequences(self):
        self._check(1, 4, 300, 77)

    def test_unaligned_short_kv(self):
        self._check(1, 2, 4096, 300)

    def test_gqa(self):
        from sglang.multimodal_gen.runtime.layers.attention.backends.tpu_splash_attn import (
            tpu_attention,
        )

        q, k, v = _inputs(1, 4, 4608, 4608)
        k, v = k[:, :2], v[:, :2]
        scale = 128**-0.5
        want = _reference(
            q, k.repeat_interleave(2, 1), v.repeat_interleave(2, 1), scale, False
        )
        got = tpu_attention(q.to("tpu"), k.to("tpu"), v.to("tpu"), scale=scale)
        self.assertLess((got.to("cpu").float() - want).abs().max().item(), 2e-2)

    def test_gqa_make_splash_mha(self):
        from sglang.multimodal_gen.runtime.layers.attention.backends.tpu_splash_attn import (
            tpu_attention,
        )

        q, k, v = _inputs(1, 4, 4096, 256)
        k, v = k[:, :2], v[:, :2]
        scale = 128**-0.5
        want = _reference(
            q, k.repeat_interleave(2, 1), v.repeat_interleave(2, 1), scale, False
        )
        got = tpu_attention(q.to("tpu"), k.to("tpu"), v.to("tpu"), scale=scale)
        self.assertLess((got.to("cpu").float() - want).abs().max().item(), 2e-2)

    def test_varlen_segments_use_sdpa(self):
        from sglang.multimodal_gen.runtime.layers.attention.backends import (
            tpu_splash_attn,
        )

        impl = tpu_splash_attn.TpuSplashAttentionImpl(
            num_heads=2, head_size=128, causal=False, softmax_scale=128**-0.5
        )
        lens = (3000, 2500)
        q, k, v = (
            torch.randn(sum(lens), 2, 128, dtype=torch.bfloat16) for _ in range(3)
        )
        cu = torch.tensor([0, lens[0], sum(lens)], dtype=torch.int32)
        with patch.object(
            tpu_splash_attn, "tpu_attention", side_effect=AssertionError("kernel")
        ):
            got = impl.forward_varlen(
                q.to("tpu"),
                k.to("tpu"),
                v.to("tpu"),
                cu_seqlens=cu.to("tpu"),
                max_seqlen=max(lens),
                cu_seqlens_host=(0, lens[0], sum(lens)),
            ).to("cpu")
        start = 0
        for n in lens:
            seg = slice(start, start + n)
            want = _reference(
                q[None, seg].transpose(1, 2),
                k[None, seg].transpose(1, 2),
                v[None, seg].transpose(1, 2),
                128**-0.5,
                False,
            ).transpose(1, 2)[0]
            self.assertLess((got[seg].float() - want).abs().max().item(), 2e-2)
            start += n

    def test_lse_matches_reference(self):
        from sglang.multimodal_gen.runtime.layers.attention.backends.tpu_splash_attn import (
            tpu_attention_with_lse,
        )

        scale = 128**-0.5
        for batch, heads, q_seq, kv_seq in ((1, 2, 4608, 4608), (2, 2, 880, 4128)):
            with self.subTest(q_seq=q_seq, kv_seq=kv_seq):
                q, k, v = _inputs(batch, heads, q_seq, kv_seq)
                want = _reference(q, k, v, scale, False)
                want_lse = _reference_lse(q, k, scale)
                out, lse = tpu_attention_with_lse(
                    q.to("tpu"), k.to("tpu"), v.to("tpu"), scale
                )
                out, lse = out.to("cpu"), lse.to("cpu")
                self.assertEqual(lse.shape, want_lse.shape)
                self.assertEqual(lse.dtype, torch.float32)
                self.assertLess((out.float() - want).abs().max().item(), 2e-2)
                self.assertLess((lse - want_lse).abs().max().item(), 2e-2)

    def test_impl_returns_softmax_lse(self):
        # The ring merge's contract: [b, s, h, d] output, [b, h, s] LSE.
        from sglang.multimodal_gen.runtime.layers.attention.backends.tpu_splash_attn import (
            TpuSplashAttentionImpl,
        )

        impl = TpuSplashAttentionImpl(
            num_heads=2, head_size=128, causal=False, softmax_scale=128**-0.5
        )
        q, k, v = _inputs(1, 2, 4608, 4608)
        out, lse = impl.forward(
            *(t.transpose(1, 2).to("tpu") for t in (q, k, v)),
            None,
            return_softmax_lse=True,
        )
        want = _reference(q, k, v, 128**-0.5, False).transpose(1, 2)
        self.assertEqual(out.shape, want.shape)
        self.assertLess((out.to("cpu").float() - want).abs().max().item(), 2e-2)
        want_lse = _reference_lse(q, k, 128**-0.5)
        self.assertLess((lse.to("cpu") - want_lse).abs().max().item(), 2e-2)

    def test_ring_step_skips_keys_past_the_real_count(self):
        # Keys past the real count are sequence-parallel padding; NaN there
        # proves they never contribute.
        from sglang.multimodal_gen.runtime.layers.attention.backends.tpu_splash_attn import (
            tpu_ring_attention_step,
        )

        scale = 128**-0.5
        for q_seq, kv_seq, real in ((4608, 4624, 4600), (2048, 8192, 6150)):
            with self.subTest(kv_seq=kv_seq, real=real):
                q, k, v = _inputs(1, 2, q_seq, kv_seq)
                k[:, :, real:], v[:, :, real:] = float("nan"), float("nan")
                want = _reference(q, k[:, :, :real], v[:, :, :real], scale, False)
                out, carry = tpu_ring_attention_step(
                    q.to("tpu"),
                    k.to("tpu"),
                    v.to("tpu"),
                    _kv_info(0, real),
                    None,
                    scale,
                )
                out = out.to("cpu").float()
                self.assertFalse(torch.isnan(out).any().item())
                self.assertFalse(torch.isnan(carry[1].to("cpu")).any().item())
                self.assertLess((out - want).abs().max().item(), 2e-2)

    def test_ring_step_resumes_an_earlier_partial(self):
        # Two key halves, the second step resuming the first, equal attention
        # over all keys.
        from sglang.multimodal_gen.runtime.layers.attention.backends.tpu_splash_attn import (
            tpu_ring_attention_step,
        )

        scale = 128**-0.5
        q, k, v = _inputs(1, 2, 4608, 2 * 4600)
        want = _reference(q, k, v, scale, False)
        qt = q.to("tpu")
        halves = [(t[:, :, :4600].to("tpu"), t[:, :, 4600:].to("tpu")) for t in (k, v)]
        _, carry = tpu_ring_attention_step(
            qt, halves[0][0], halves[1][0], _kv_info(0, 4600), None, scale
        )
        out, _ = tpu_ring_attention_step(
            qt, halves[0][1], halves[1][1], _kv_info(0, 4600), carry, scale
        )
        self.assertLess((out.to("cpu").float() - want).abs().max().item(), 2e-2)

    def test_ring_step_picks_shard_and_key_count_at_run_time(self):
        # Stacked shards with batch 2 and GQA: the row offset and real-key
        # count arrive as data, resuming the local shard's softmax, as the
        # padded ring does. NaN past the real keys proves they are unused.
        from sglang.multimodal_gen.runtime.layers.attention.backends.tpu_splash_attn import (
            tpu_ring_attention_step,
        )

        scale = 128**-0.5
        batch, heads, kv_heads, shard_len = 2, 4, 2, 4624
        gen = torch.Generator().manual_seed(0)

        def randn(*shape):
            return torch.randn(shape, generator=gen).bfloat16()

        q = randn(batch, heads, 4608, 128)
        k_local, v_local = (
            randn(batch, kv_heads, shard_len, 128),
            randn(batch, kv_heads, shard_len, 128),
        )
        k_stack, v_stack = (
            randn(2 * batch, kv_heads, shard_len, 128),
            randn(2 * batch, kv_heads, shard_len, 128),
        )
        for peer, real in ((0, shard_len), (1, 4600), (1, 2100)):
            with self.subTest(peer=peer, real=real):
                kk, vv = k_stack.clone(), v_stack.clone()
                rows = slice(peer * batch, (peer + 1) * batch)
                kk[rows, :, real:], vv[rows, :, real:] = float("nan"), float("nan")
                keys = torch.cat([k_local, kk[rows, :, :real]], dim=2)
                values = torch.cat([v_local, vv[rows, :, :real]], dim=2)
                want = _reference(
                    q,
                    keys.repeat_interleave(heads // kv_heads, 1),
                    values.repeat_interleave(heads // kv_heads, 1),
                    scale,
                    False,
                )
                qt = q.to("tpu")
                _, carry = tpu_ring_attention_step(
                    qt,
                    k_local.to("tpu"),
                    v_local.to("tpu"),
                    _kv_info(0, shard_len),
                    None,
                    scale,
                )
                out, _ = tpu_ring_attention_step(
                    qt,
                    kk.to("tpu"),
                    vv.to("tpu"),
                    _kv_info(peer * batch * kv_heads, real),
                    carry,
                    scale,
                )
                out = out.to("cpu").float()
                self.assertFalse(torch.isnan(out).any().item())
                self.assertLess((out - want).abs().max().item(), 2e-2)

    def test_ring_step_over_an_empty_shard_resumes_exactly(self):
        from sglang.multimodal_gen.runtime.layers.attention.backends.tpu_splash_attn import (
            tpu_ring_attention_step,
        )

        scale = 128**-0.5
        q, k, v = _inputs(1, 2, 4608, 4608)
        qt, kt, vt = q.to("tpu"), k.to("tpu"), v.to("tpu")
        empty_out, carry = tpu_ring_attention_step(
            qt, kt, vt, _kv_info(0, 0), None, scale
        )
        self.assertTrue(torch.all(empty_out.to("cpu") == 0))
        out, _ = tpu_ring_attention_step(qt, kt, vt, _kv_info(0, 4608), carry, scale)
        want = _reference(q, k, v, scale, False)
        self.assertLess((out.to("cpu").float() - want).abs().max().item(), 2e-2)

    def test_ring_step_under_torch_compile(self):
        import torch_tpu._internal.compile  # noqa: F401

        from sglang.multimodal_gen.runtime.layers.attention.backends.tpu_splash_attn import (
            _register_ops,
            tpu_ring_attention_step,
        )

        _register_ops()
        scale = 128**-0.5

        def two_steps(q, k, v, info_a, info_b):
            _, carry = tpu_ring_attention_step(q, k, v, info_a, None, scale)
            out, _ = tpu_ring_attention_step(q, k, v, info_b, carry, scale)
            return out

        q, k, v = _inputs(1, 2, 4608, 2 * 4608)
        stacked = [torch.cat(t.chunk(2, dim=2), dim=0) for t in (k, v)]
        want = _reference(q, k, v, scale, False)
        compiled = torch.compile(two_steps, backend="tpu", dynamic=False)
        got = compiled(
            q.to("tpu"),
            stacked[0].to("tpu"),
            stacked[1].to("tpu"),
            _kv_info(0, 4608),
            _kv_info(2, 4608),
        ).to("cpu")
        self.assertLess((got.float() - want).abs().max().item(), 2e-2)

    def test_backend_impl_layout(self):
        # The attention layer calls the impl with [B, S, H, D] tensors.
        from sglang.multimodal_gen.runtime.layers.attention.backends.tpu_splash_attn import (
            TpuSplashAttentionImpl,
        )

        impl = TpuSplashAttentionImpl(
            num_heads=4, head_size=128, causal=False, softmax_scale=128**-0.5
        )
        q, k, v = _inputs(1, 4, 4608, 4608)
        want = _reference(q, k, v, 128**-0.5, False).transpose(1, 2)
        got = impl.forward(
            q.transpose(1, 2).to("tpu"),
            k.transpose(1, 2).to("tpu"),
            v.transpose(1, 2).to("tpu"),
            None,
        ).to("cpu")
        self.assertEqual(got.shape, want.shape)
        self.assertLess((got.float() - want).abs().max().item(), 2e-2)

    def test_torch_compile_output_layout(self):
        # Dynamo traces the op through its fake impl, whose strides must match
        # the kernel's contiguous output for the layer's reshape to lower.
        import torch_tpu._internal.compile  # noqa: F401

        from sglang.multimodal_gen.runtime.layers.attention.backends.tpu_splash_attn import (
            TpuSplashAttentionImpl,
        )

        impl = TpuSplashAttentionImpl(
            num_heads=4, head_size=128, causal=False, softmax_scale=128**-0.5
        )

        def attn(q, k, v):
            out = impl.forward(q, k, v, None)
            return out.reshape(out.shape[0], out.shape[1], -1)

        # Custom kernel, and make_splash_mha with a padded query.
        for q_seq, kv_seq in ((4608, 4608), (200, 4128)):
            with self.subTest(q_seq=q_seq, kv_seq=kv_seq):
                q, k, v = _inputs(1, 4, q_seq, kv_seq)
                want = _reference(q, k, v, 128**-0.5, False).transpose(1, 2)
                compiled = torch.compile(attn, backend="tpu", dynamic=False)
                # Contiguous [B, S, H, D], as the layer's projections produce.
                got = compiled(
                    *(t.transpose(1, 2).contiguous().to("tpu") for t in (q, k, v))
                ).to("cpu")
                self.assertEqual(got.shape, (1, q_seq, 4 * 128))
                err = (got.float() - want.reshape(1, q_seq, -1)).abs().max().item()
                self.assertLess(err, 2e-2)

    def test_float32_falls_back_to_sdpa(self):
        from sglang.multimodal_gen.runtime.layers.attention.backends.tpu_splash_attn import (
            tpu_attention,
        )

        q, k, v = (t.float() for t in _inputs(1, 2, 2048, 2048))
        got = tpu_attention(q.to("tpu"), k.to("tpu"), v.to("tpu")).to("cpu")
        self.assertEqual(got.dtype, torch.float32)
        want = _reference(q, k, v, 128**-0.5, False)
        self.assertLess((got - want).abs().max().item(), 2e-2)


class TestTpuSplashDispatch(CustomTestCase):
    """Kernel selection logic; runs without a TPU."""

    def test_custom_kernel_selection(self):
        from sglang.multimodal_gen.runtime.layers.attention.backends.tpu_splash_attn import (
            _use_custom_kernel,
        )

        self.assertTrue(_use_custom_kernel(75600, 75600, False))
        self.assertTrue(_use_custom_kernel(880, 3520, False))
        self.assertTrue(_use_custom_kernel(18900, 512, False))
        self.assertFalse(_use_custom_kernel(75600, 256, False))
        self.assertFalse(_use_custom_kernel(200, 4128, False))
        self.assertFalse(_use_custom_kernel(4096, 4096, True))

    def test_cross_attn_block_kv_divides_kv(self):
        from sglang.multimodal_gen.runtime.layers.attention.backends.tpu_splash_attn import (
            _cross_attn_block_kv,
        )

        for kv_seq in range(128, 2048, 128):
            block_kv = _cross_attn_block_kv(kv_seq)
            self.assertEqual(kv_seq % block_kv, 0)
            self.assertLessEqual(block_kv, 1024)

    def test_short_kv_query_blocks_are_capped(self):
        from sglang.multimodal_gen.runtime.layers.attention.backends.tpu_splash_attn import (
            autotune_block_sizes,
        )

        for q_seq, kv_seq in ((75600, 512), (37800, 512), (55800, 1024)):
            self.assertLessEqual(autotune_block_sizes(q_seq, kv_seq)[0], 9472)
        # Long KV: the VMEM model binds first.
        self.assertEqual(autotune_block_sizes(75600, 75600), (6400, 2048, 2048))

    def test_splash_mha_blocks(self):
        from sglang.multimodal_gen.runtime.layers.attention.backends.tpu_splash_attn import (
            _splash_mha_block_sizes,
        )

        self.assertEqual(_splash_mha_block_sizes(4096, 384, False), (2048, 128))
        self.assertEqual(_splash_mha_block_sizes(4096, 256, False), (2048, 256))
        self.assertEqual(_splash_mha_block_sizes(4096, 4096, True), (1024, 1024))
        self.assertEqual(_splash_mha_block_sizes(1000, 1000, True), (512, 512))
        self.assertEqual(_splash_mha_block_sizes(200, 4128, False), (256, 256))

    def test_kernel_source_keys_the_compile_cache_once(self):
        import torch.compiler.config as compiler_config

        from sglang.multimodal_gen.runtime.layers.attention.backends import (
            tpu_splash_attn,
        )

        with patch.object(compiler_config, "cache_key_tag", "user-tag"):
            tpu_splash_attn._add_kernel_source_to_compile_cache_key()
            tpu_splash_attn._add_kernel_source_to_compile_cache_key()
            tags = compiler_config.cache_key_tag.split("|")
        self.assertEqual(tags[0], "user-tag")
        self.assertEqual(len(tags), 2)
        self.assertTrue(tags[1].startswith("diffusion-tpu-splash-attn:"))

    def test_sdpa_fallback_reasons(self):
        from sglang.multimodal_gen.runtime.layers.attention.backends import (
            tpu_splash_attn,
        )

        def reason(q_shape, kv_shape, *, is_causal=False, dtype=torch.bfloat16):
            q = torch.empty(q_shape, dtype=dtype, device="meta")
            k = torch.empty(kv_shape, dtype=dtype, device="meta")
            return tpu_splash_attn._sdpa_fallback_reason(q, k, k, is_causal)

        with patch.object(
            tpu_splash_attn, "_check_splash_available", return_value=True
        ):
            self.assertIsNone(reason((1, 4, 4608, 128), (1, 4, 4608, 128)))
            self.assertIsNone(reason((1, 4, 4096, 128), (1, 2, 512, 128)))
            self.assertEqual(reason((1, 4, 300, 128), (1, 4, 77, 128)), "short")
            self.assertEqual(reason((1, 4, 4096, 128), (1, 4, 300, 128)), "short")
            self.assertIsNone(reason((1, 4, 18900, 128), (1, 4, 600, 128)))
            self.assertEqual(
                reason((1, 4, 1000, 128), (1, 4, 2000, 128), is_causal=True),
                "ragged causal",
            )
            self.assertEqual(
                reason((1, 4, 4608, 64), (1, 4, 4608, 64)), "head_dim=64/64"
            )
            self.assertEqual(reason((1, 3, 4608, 128), (1, 2, 4608, 128)), "heads=3/2")
            self.assertEqual(
                reason((1, 4, 4608, 128), (1, 4, 4608, 128), dtype=torch.float32),
                "dtype=torch.float32/torch.float32/torch.float32",
            )
            self.assertEqual(reason((1, 4, 0, 128), (1, 4, 4608, 128)), "empty")
        with patch.object(
            tpu_splash_attn, "_check_splash_available", return_value=False
        ):
            self.assertEqual(
                reason((1, 4, 4608, 128), (1, 4, 4608, 128)), "pallas unavailable"
            )


if __name__ == "__main__":
    unittest.main()
