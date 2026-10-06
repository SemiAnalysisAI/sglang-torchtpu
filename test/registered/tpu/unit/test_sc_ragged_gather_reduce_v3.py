"""SparseCore gather-reduce v3 (SGLANG_TPU_USE_SC_GATHER_REDUCE_V3) vs v2 / reference."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=240, suite="stage-a-test-tpu-unit")

import unittest
from unittest import mock

import torch

from sglang.test.test_utils import CustomTestCase

try:
    import jax
    import jax.numpy as jnp
    import numpy as np

    _HAS_TPU = any(d.platform == "tpu" for d in jax.devices())
except Exception:  # pragma: no cover - non-TPU host
    _HAS_TPU = False

if _HAS_TPU:
    from test_fused_moe_situ_mxfp4 import _assert_close, _routing
    from test_pallas_fused_moe_ep import _make_inputs

_SKIP = "requires a TPU backend"

# Kimi-K3 per-rank shape scaled down: 28 local experts, top-16, SiTU.
E, HIDDEN, INTER, TOPK = 28, 1024, 768, 16
SITU = "situ:4.0:25.0"


def _force_sparse_core():
    """Run the v3 SparseCore kernel even below its TensorCore-fallback size."""
    from sglang.srt.layers.moe.pallas.sparse_core import ragged_gather_reduce_v3

    jax.clear_caches()
    return mock.patch.object(
        ragged_gather_reduce_v3, "use_tensorcore_fallback", return_value=False
    )


def _ref(x, indices, w, valid, topk):
    out = x[indices].astype(np.float32) * w[:, None]
    out = np.where(valid[:, None], out, 0)
    return out.reshape(-1, topk, x.shape[1]).sum(1)


@unittest.skipUnless(_HAS_TPU, _SKIP)
class TestRaggedGatherReduceV3Kernel(CustomTestCase):
    def _run(self, valid, force=True):
        from sglang.srt.layers.moe.pallas.sparse_core.ragged_gather_reduce_v3 import (
            ragged_gather_reduce,
        )

        n = valid.size
        rng = np.random.default_rng(n)
        indices = rng.permutation(n).astype(np.int32)
        # Padding / masked lanes gather row 0: make it a garbage row.
        p0, inv = int(np.argmax(indices == 0)), int(np.argmax(~valid))
        indices[[p0, inv]] = indices[[inv, p0]]
        w = rng.random(n).astype(np.float32)
        x = rng.standard_normal((n, HIDDEN)).astype(np.float32)
        x = np.array(jnp.asarray(x, jnp.bfloat16).astype(jnp.float32))
        ref = _ref(x, indices, w, valid, TOPK)
        # Rows only invalid routes point at were never written by the GMMs.
        x[indices[~valid]] = np.nan
        args = (jnp.asarray(x, jnp.bfloat16), jnp.asarray(indices), jnp.asarray(w))
        args += (jnp.asarray(valid),)
        if force:
            with _force_sparse_core():
                out = ragged_gather_reduce(*args, TOPK)
        else:
            out = ragged_gather_reduce(*args, TOPK)
        out = np.asarray(jax.block_until_ready(out).astype(jnp.float32))
        self.assertEqual(out.shape, ref.shape)
        _assert_close(self, out, ref, 1e-2)

    def test_random_masks(self):
        for tokens in (2, 4, 8, 64, 130):
            with self.subTest(tokens=tokens):
                rng = np.random.default_rng(tokens)
                self._run(rng.random(tokens * TOPK) < 0.4)

    def test_two_routes_per_token_fast_path(self):
        valid = np.zeros((64, TOPK), bool)
        valid[:, 3:5] = True
        self._run(valid.reshape(-1))

    def test_128_unpaired_routes_take_masked_path(self):
        # 128 valid routes in one block but not two per token: must not take
        # the unmasked pair path or read a padding row.
        valid = np.zeros((64, TOPK), bool)
        valid[::2, :3] = True
        valid[1::2, 0] = True
        self._run(valid.reshape(-1))

    def test_natural_size_uses_sparse_core(self):
        from sglang.srt.layers.moe.pallas.sparse_core import ragged_gather_reduce_v3

        tokens = 1024
        x = jnp.zeros((tokens * TOPK, HIDDEN), jnp.bfloat16)
        self.assertFalse(ragged_gather_reduce_v3.use_tensorcore_fallback(x))
        rng = np.random.default_rng(0)
        self._run(rng.random(tokens * TOPK) < 0.5, force=False)


@unittest.skipUnless(_HAS_TPU, _SKIP)
class TestFusedMoEGatherReduceV3(CustomTestCase):
    """EP _fused_moe_jax on the SparseCore combine, v3 flag on vs off."""

    def _fused(self, weights, m, flag, use_sc_gather=1, force=True):
        from sglang.srt.environ import envs
        from sglang.srt.layers.moe.pallas.fused_moe_op import _fused_moe_jax

        w1, w2, w1_s, w2_s = weights
        rng = np.random.default_rng(m)
        hidden = jnp.asarray(rng.standard_normal((m, HIDDEN)), jnp.bfloat16)
        # 28 local experts of 56 global; this rank owns the upper half.
        ids, w = _routing(m, 2 * E, TOPK, seed=m + 1)
        fn = jax.jit(
            lambda h: _fused_moe_jax(
                h,
                w1,
                w2,
                jnp.asarray(w, jnp.bfloat16),
                jnp.asarray(ids),
                w1_s,
                w2_s,
                None,
                None,
                TOPK,
                SITU,
                global_num_experts=2 * E,
                expert_group_offset=E,
                use_sc_gather=use_sc_gather,
            )
        )
        with envs.SGLANG_TPU_USE_SC_GATHER_REDUCE_V3.override(flag):
            if flag and force:
                with _force_sparse_core():
                    out = fn(hidden)
            else:
                out = fn(hidden)
        return np.asarray(jax.block_until_ready(out).astype(jnp.float32))

    def _check(self, weights):
        for m in (2, 4, 8, 64):
            with self.subTest(m=m):
                off = self._fused(weights, m, False)
                on = self._fused(weights, m, True)
                self.assertTrue(np.all(np.isfinite(off)))
                self.assertGreater(np.abs(off).max(), 0)
                _assert_close(self, on, off, 1e-2)

    def test_bf16(self):
        _, w1, w2, _, _ = _make_inputs(jax.random.key(0), 1, HIDDEN, INTER, E, TOPK)
        self._check((w1, w2, None, None))

    def test_natural_sparse_core_size_matches_tensorcore_gather(self):
        # 1024 tokens x top-16 is past the TensorCore-fallback cutoff, so v3
        # runs on SparseCore unpatched; reference is the TensorCore gather.
        _, w1, w2, _, _ = _make_inputs(jax.random.key(0), 1, HIDDEN, INTER, E, TOPK)
        weights = (w1, w2, None, None)
        ref = self._fused(weights, 1024, False, use_sc_gather=0)
        on = self._fused(weights, 1024, True, force=False)
        _assert_close(self, on, ref, 1e-2)

    def test_packed_mxfp4(self):
        from sglang.srt.layers.quantization.mxfp4_tpu_util import (
            process_mxfp4_moe_weights_native,
        )

        g = torch.Generator().manual_seed(0)
        u8 = lambda *s: torch.randint(0, 256, s, dtype=torch.uint8, generator=g)
        e8 = lambda *s: torch.randint(124, 131, s, dtype=torch.uint8, generator=g)
        N, H = INTER, HIDDEN
        w13, w13_s, _, w2, w2_s, _ = process_mxfp4_moe_weights_native(
            u8(E, 2 * N, H // 2),
            e8(E, 2 * N, H // 32),
            torch.zeros(E, 2 * N, dtype=torch.bfloat16),
            u8(E, H, N // 2),
            e8(E, H, N // 32),
            torch.zeros(E, H, dtype=torch.bfloat16),
            w13_interleaved=False,
        )
        self._check(tuple(jnp.asarray(t.numpy()) for t in (w13, w2, w13_s, w2_s)))


if __name__ == "__main__":
    unittest.main()
