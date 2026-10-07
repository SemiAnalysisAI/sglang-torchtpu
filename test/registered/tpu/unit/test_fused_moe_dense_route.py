"""Fused MoE multi-token dense route: local ids, sort-free rank permute."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=420, suite="stage-a-test-tpu-unit")

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
    from test_fused_moe_situ_mxfp4 import _assert_close, _moe_ref, _routing

_SKIP = "requires a TPU backend"

# Kimi-K3 per-rank shape scaled down: 28 local of 896 experts, top-16, SiTU.
E, E_GLOBAL, HIDDEN, INTER, TOPK = 28, 896, 1024, 768, 16
EP_OFFSET = 13 * E  # a middle rank: ids below and above the local window
SITU = "situ:4.0:25.0"
DENSE_THRESHOLD = 2048
MS = (2, 3, 4, 8, 16)


def _weights(kind):
    """(w1, w2, w1_scale, w2_scale) for the kernel, f32 (w1, w2) for the ref."""
    from sglang.srt.layers.quantization.mxfp4_tpu_util import (
        process_mxfp4_moe_weights_bf16,
        process_mxfp4_moe_weights_native,
    )

    if kind == "bf16":
        k1, k2 = jax.random.split(jax.random.key(0))
        w1 = jax.random.normal(k1, (E, HIDDEN, 2 * INTER)) * 0.05
        w2 = jax.random.normal(k2, (E, INTER, HIDDEN)) * 0.05
        w1, w2 = w1.astype(jnp.bfloat16), w2.astype(jnp.bfloat16)
        ref = tuple(np.asarray(t.astype(jnp.float32)) for t in (w1, w2))
        return (w1, w2, None, None), ref
    g = torch.Generator().manual_seed(0)
    u8 = lambda *s: torch.randint(0, 256, s, dtype=torch.uint8, generator=g)
    e8 = lambda *s: torch.randint(124, 131, s, dtype=torch.uint8, generator=g)
    N, H = INTER, HIDDEN
    hf = (
        u8(E, 2 * N, H // 2),
        e8(E, 2 * N, H // 32),
        torch.zeros(E, 2 * N, dtype=torch.bfloat16),
        u8(E, H, N // 2),
        e8(E, H, N // 32),
        torch.zeros(E, H, dtype=torch.bfloat16),
    )
    w13, w13_s, _, w2, w2_s, _ = process_mxfp4_moe_weights_native(
        *hf, w13_interleaved=False
    )
    w13_bf, _, w2_bf, _ = process_mxfp4_moe_weights_bf16(*hf, w13_interleaved=False)
    to_jax = lambda t: jnp.asarray(t.numpy())
    # bf16-rounded, as the existing MXFP4 reference test does.
    bf = lambda t: t.to(torch.bfloat16).float().numpy()
    return tuple(to_jax(t) for t in (w13, w2, w13_s, w2_s)), (bf(w13_bf), bf(w2_bf))


def _ep_routing(m, seed):
    """Global ids with 0-8 local experts per token, the rest on other ranks."""
    rng = np.random.default_rng(seed)
    local = np.arange(EP_OFFSET, EP_OFFSET + E)
    remote = np.setdiff1d(np.arange(E_GLOBAL), local)
    ids = []
    for _ in range(m):
        n_local = int(rng.integers(0, 9))
        row = np.concatenate(
            [
                rng.choice(local, n_local, replace=False),
                rng.choice(remote, TOPK - n_local, replace=False),
            ]
        )
        ids.append(rng.permutation(row))
    w = rng.random((m, TOPK)).astype(np.float32)
    return np.stack(ids).astype(np.int32), w / w.sum(-1, keepdims=True)


def _poison_unwritten(gmm_v2):
    """Fill the rows a no-zero-init grouped matmul leaves unwritten with NaN."""

    def wrapped(*args, **kwargs):
        out = gmm_v2(*args, **kwargs)
        if kwargs["zero_initialize"] or kwargs["group_offset"] is not None:
            return out
        written = jnp.arange(out.shape[0])[:, None] < jnp.sum(kwargs["group_sizes"])
        return jnp.where(written, out, jnp.nan).astype(out.dtype)

    return wrapped


@unittest.skipUnless(_HAS_TPU, _SKIP)
class TestFusedMoEDenseRoute(CustomTestCase):
    """_fused_moe_jax with the dense route vs the existing paths and a reference."""

    @classmethod
    def setUpClass(cls):
        cls.weights = {kind: _weights(kind) for kind in ("bf16", "mxfp4")}

    def _fused(self, kind, m, ep, *, route=True, threshold=DENSE_THRESHOLD):
        import importlib

        from sglang.srt.layers.moe.pallas import fused_moe_op

        gmm_mod = importlib.import_module("sglang.srt.layers.moe.pallas.gmm_v2")

        (w1, w2, w1_s, w2_s), _ = self.weights[kind]
        hidden = jnp.asarray(
            np.random.default_rng(m).standard_normal((m, HIDDEN)), jnp.bfloat16
        )
        ids, w = _ep_routing(m, m + 1) if ep else _routing(m, E, TOPK, m + 1)
        # Fresh closure per call: the patches apply at trace time.
        fn = jax.jit(
            lambda h: fused_moe_op._fused_moe_jax(
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
                global_num_experts=E_GLOBAL if ep else 0,
                expert_group_offset=EP_OFFSET if ep else 0,
                permute_dense_threshold=threshold,
            )
        )
        with (
            mock.patch.object(
                fused_moe_op, "_RANK_PERMUTE_MAX_ROWS", 256 if route else 0
            ),
            mock.patch.object(gmm_mod, "gmm_v2", _poison_unwritten(gmm_mod.gmm_v2)),
        ):
            out = jax.block_until_ready(fn(hidden))
        out = np.asarray(out.astype(jnp.float32))
        self.assertTrue(np.all(np.isfinite(out)))
        return out, hidden, ids, w

    def _reference(self, kind, hidden, ids, w, ep):
        _, (w1, w2) = self.weights[kind]
        local = ids - (EP_OFFSET if ep else 0)
        owned = (local >= 0) & (local < E)
        return _moe_ref(
            np.asarray(hidden.astype(jnp.float32)),
            w1,
            w2,
            np.where(owned, w, 0.0),
            np.where(owned, local, 0),
        )

    def _check(self, kind, ep):
        for m in MS:
            with self.subTest(kind=kind, ep=ep, m=m):
                new, hidden, ids, w = self._fused(kind, m, ep)
                old, *_ = self._fused(kind, m, ep, route=False)
                if ep:
                    # Owned rows move from the global window to a prefix, which
                    # can regroup the f32 combine sum: within one bf16 ulp.
                    np.testing.assert_allclose(new, old, rtol=2**-7, atol=1e-6)
                else:
                    # Same rows in the same stable order: bitwise.
                    np.testing.assert_array_equal(new, old)
                gather, *_ = self._fused(kind, m, ep, threshold=0)
                _assert_close(self, new, gather, 1e-2)
                ref = self._reference(kind, hidden, ids, w, ep)
                self.assertGreater(np.abs(ref).max(), 0)
                _assert_close(self, new, ref, 0.03)

    def test_bf16_all_local(self):
        self._check("bf16", ep=False)

    def test_bf16_ep(self):
        self._check("bf16", ep=True)

    def test_mxfp4_situ_all_local(self):
        self._check("mxfp4", ep=False)

    def test_mxfp4_situ_ep(self):
        self._check("mxfp4", ep=True)

    def test_out_of_range_ids_contribute_nothing(self):
        # -1 and >= global ids land on the sentinel, like non-local ids.
        from sglang.srt.layers.moe.pallas import fused_moe_op

        (kw1, kw2, _, _), _ = self.weights["bf16"]
        hidden = jnp.asarray(
            np.random.default_rng(0).standard_normal((4, HIDDEN)), jnp.bfloat16
        )
        bad_ids, w = _ep_routing(4, 1)
        bad_ids[:, :2] = np.array([-1, E_GLOBAL + 5])
        out = fused_moe_op._fused_moe_jax(
            hidden,
            kw1,
            kw2,
            jnp.asarray(w, jnp.bfloat16),
            jnp.asarray(bad_ids),
            None,
            None,
            None,
            None,
            TOPK,
            SITU,
            global_num_experts=E_GLOBAL,
            expert_group_offset=EP_OFFSET,
            permute_dense_threshold=DENSE_THRESHOLD,
        )
        out = np.asarray(jax.block_until_ready(out).astype(jnp.float32))
        ref = self._reference("bf16", hidden, bad_ids, w, ep=True)
        _assert_close(self, out, ref, 0.03)


if __name__ == "__main__":
    unittest.main()
