"""Pallas fused MoE on layer-stacked experts must match that layer's 3D weights."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=90, suite="stage-a-test-tpu-unit", chips=1)

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

_SKIP = "requires a TPU backend"

H, N = 512, 256
# EP4 rank 2: experts [16, 24) of 32 are local.
LOCAL, GLOBAL, EP_OFFSET, TOPK = 8, 32, 16, 4
# The stacked calls compute the middle layer, which neither a dropped nor a
# doubled layer offset lands on.
LAYERS, LAYER = 3, 1
SITU = "situ:4.0:25.0"
DENSE_THRESHOLD = 2048


def _mxfp4_layer(seed):
    """One layer's packed MXFP4 (w1, w2, w1_scale, w2_scale), loader layout."""
    from sglang.srt.layers.quantization.mxfp4_tpu_util import (
        process_mxfp4_moe_weights_native,
    )

    g = torch.Generator().manual_seed(seed)
    u8 = lambda *s: torch.randint(0, 256, s, dtype=torch.uint8, generator=g)
    e8 = lambda *s: torch.randint(122, 127, s, dtype=torch.uint8, generator=g)
    zeros = lambda *s: torch.zeros(*s, dtype=torch.bfloat16)
    w13, w13_s, _, w2, w2_s, _ = process_mxfp4_moe_weights_native(
        u8(LOCAL, 2 * N, H // 2),
        e8(LOCAL, 2 * N, H // 32),
        zeros(LOCAL, 2 * N),
        u8(LOCAL, H, N // 2),
        e8(LOCAL, H, N // 32),
        zeros(LOCAL, H),
        w13_interleaved=False,
    )
    return tuple(jnp.asarray(t.numpy()) for t in (w13, w2, w13_s, w2_s))


def _bf16_layer(seed):
    k1, k2 = jax.random.split(jax.random.key(seed))
    w1 = (jax.random.normal(k1, (LOCAL, H, 2 * N)) * 0.05).astype(jnp.bfloat16)
    w2 = (jax.random.normal(k2, (LOCAL, N, H)) * 0.05).astype(jnp.bfloat16)
    return w1, w2, None, None


def _routing(m, ep, seed):
    """Top-k ids (global under EP, two local slots per token), with a -1 slot
    and, for m > 1, a fully padded last token."""
    rng = np.random.default_rng(seed)
    remote = np.setdiff1d(np.arange(GLOBAL), np.arange(EP_OFFSET, EP_OFFSET + LOCAL))
    rows = []
    for _ in range(m):
        if ep:
            local = EP_OFFSET + rng.choice(LOCAL, 2, replace=False)
            row = np.r_[local, rng.choice(remote, TOPK - 2, replace=False)]
        else:
            row = rng.choice(LOCAL, TOPK, replace=False)
        rows.append(rng.permutation(row))
    ids = np.stack(rows).astype(np.int32)
    ids[0, -1] = -1
    if m > 1:
        ids[-1] = -1
    w = rng.random((m, TOPK)).astype(np.float32)
    return ids, w / w.sum(-1, keepdims=True)


def _fused(weights, ids, w, *, ep, act, layer_index=None, threshold=0):
    from sglang_tpu.moe.pallas.fused_moe_op import _fused_moe_jax

    w1, w2, w1_scale, w2_scale = weights
    m = ids.shape[0]
    hidden = jnp.asarray(np.random.default_rng(m).standard_normal((m, H)), jnp.bfloat16)
    out = _fused_moe_jax(
        hidden,
        w1,
        w2,
        jnp.asarray(w, jnp.bfloat16),
        jnp.asarray(ids),
        w1_scale,
        w2_scale,
        None,
        None,
        TOPK,
        act,
        global_num_experts=GLOBAL if ep else 0,
        expert_group_offset=EP_OFFSET if ep else 0,
        permute_dense_threshold=threshold,
        w_layer_index=(
            None if layer_index is None else jnp.asarray([layer_index], jnp.int32)
        ),
    )
    return np.asarray(jax.block_until_ready(out).astype(jnp.float32))


@unittest.skipUnless(_HAS_TPU, _SKIP)
class TestFusedMoEStackedExperts(CustomTestCase):
    """[1, L, E, k, n] weights (the K3 megakernel's shared layout) with
    per-layer scales, at layer LAYER, vs that layer's own 3D weights."""

    @classmethod
    def setUpClass(cls):
        cls.layers = {
            "mxfp4": [_mxfp4_layer(seed) for seed in range(LAYERS)],
            "bf16": [_bf16_layer(seed) for seed in range(LAYERS)],
        }

    def _stacked(self, kind):
        layers = self.layers[kind]
        w1 = jnp.stack([layer[0] for layer in layers])[None]
        w2 = jnp.stack([layer[1] for layer in layers])[None]
        _, _, w1_scale, w2_scale = layers[LAYER]
        return w1, w2, w1_scale, w2_scale

    def _check(self, kind, act, ids, w, *, ep, threshold=0):
        got = _fused(
            self._stacked(kind),
            ids,
            w,
            ep=ep,
            act=act,
            layer_index=LAYER,
            threshold=threshold,
        )
        want = _fused(
            self.layers[kind][LAYER], ids, w, ep=ep, act=act, threshold=threshold
        )
        self.assertGreater(np.abs(want).max(), 0)
        # Same tiles and rows; only the weight block index differs: bitwise.
        np.testing.assert_array_equal(got, want)

    def test_sort_route_mxfp4_situ_and_bf16(self):
        # Five tokens pad to eight; under EP gmm_v2 also gets a group_offset.
        for kind, act in (("mxfp4", SITU), ("bf16", "silu")):
            for ep in (False, True):
                with self.subTest(kind=kind, ep=ep):
                    ids, w = _routing(5, ep, seed=1)
                    self._check(kind, act, ids, w, ep=ep)

    def test_dense_rank_route(self):
        from sglang_tpu.moe.pallas import fused_moe_op

        ids, w = _routing(4, True, seed=2)
        with mock.patch.object(
            fused_moe_op, "_dense_rank_route", wraps=fused_moe_op._dense_rank_route
        ) as route_spy:
            self._check("mxfp4", SITU, ids, w, ep=True, threshold=DENSE_THRESHOLD)
        self.assertEqual(route_spy.call_count, 2)

    def test_single_token_kernel_keeps_gmm_v2_for_stacked(self):
        """single_token_moe has no layer offset: stacked weights take gmm_v2
        (sort or single-token route) even with the kernel's own shape gates
        forced open, as the 3D weights take the kernel."""
        from sglang_tpu.envs import envs as tpu_envs
        from sglang_tpu.moe.pallas import single_token_moe

        ids, w = _routing(1, True, seed=3)
        for threshold in (0, DENSE_THRESHOLD):
            with (
                self.subTest(threshold=threshold),
                mock.patch.object(single_token_moe, "validated", lambda *a: True),
                mock.patch.object(single_token_moe, "eligible", lambda *a: True),
                mock.patch.object(
                    single_token_moe,
                    "single_token_moe",
                    wraps=single_token_moe.single_token_moe,
                ) as kernel_spy,
            ):
                with tpu_envs.SGLANG_TPU_ENABLE_SINGLE_TOKEN_MOE_KERNEL.override(True):
                    _fused(self.layers["mxfp4"][LAYER], ids, w, ep=True, act=SITU)
                    self.assertEqual(kernel_spy.call_count, 1)
                    got = _fused(
                        self._stacked("mxfp4"),
                        ids,
                        w,
                        ep=True,
                        act=SITU,
                        layer_index=LAYER,
                        threshold=threshold,
                    )
                    self.assertEqual(kernel_spy.call_count, 1)
                with tpu_envs.SGLANG_TPU_ENABLE_SINGLE_TOKEN_MOE_KERNEL.override(False):
                    want = _fused(
                        self.layers["mxfp4"][LAYER],
                        ids,
                        w,
                        ep=True,
                        act=SITU,
                        threshold=threshold,
                    )
                self.assertGreater(np.abs(want).max(), 0)
                np.testing.assert_array_equal(got, want)

    def test_streaming_kernel_keeps_gmm_v2_for_stacked(self):
        """maybe_moe_stream unpacks 3D weights, so stacked weights skip it."""
        from sglang_tpu.envs import envs as tpu_envs
        from sglang_tpu.moe.pallas import moe_stream

        ids, w = _routing(2, False, seed=4)
        act = "silu_and_mul_clamp:7.0"
        with (
            tpu_envs.SGLANG_TPU_ENABLE_STREAMING_MOE_KERNEL.override(True),
            mock.patch.object(
                moe_stream, "maybe_moe_stream", wraps=moe_stream.maybe_moe_stream
            ) as stream_spy,
        ):
            got = _fused(
                self._stacked("bf16"), ids, w, ep=False, act=act, layer_index=LAYER
            )
        self.assertFalse(stream_spy.called)
        want = _fused(self.layers["bf16"][LAYER], ids, w, ep=False, act=act)
        self.assertGreater(np.abs(want).max(), 0)
        np.testing.assert_array_equal(got, want)


@unittest.skipUnless(_HAS_TPU, _SKIP)
class TestStackedExpertValidation(CustomTestCase):
    """A layer index and stacked weights come together or not at all, and
    stacking does not relax the per-layer group-count checks."""

    def _gmm(self, rhs, rhs_scale=None, rhs_layer_index=None):
        """Trace (not run) gmm_v2; the checks under test fire at trace time."""
        from sglang_tpu.moe.pallas.gmm_v2 import gmm_v2

        return jax.eval_shape(
            lambda: gmm_v2(
                jnp.zeros((16, H), jnp.bfloat16),
                rhs,
                jnp.full((LOCAL,), 2, jnp.int32),
                rhs_scale,
                packed_fp4=rhs.dtype == jnp.uint8,
                rhs_layer_index=rhs_layer_index,
            )
        )

    def test_stacked_weights_need_layer_index(self):
        w1, w2, _, _ = _bf16_layer(0)
        ids, w = _routing(2, False, seed=5)
        with self.assertRaisesRegex(ValueError, "need w_layer_index"):
            _fused(
                (w1[None, None], w2[None, None], None, None),
                ids,
                w,
                ep=False,
                act="silu",
            )
        with self.assertRaisesRegex(AssertionError, "needs rhs_layer_index"):
            self._gmm(w1[None, None])

    def test_layer_index_needs_stacked_weights(self):
        w1, w2, _, _ = _bf16_layer(0)
        ids, w = _routing(2, False, seed=5)
        with self.assertRaisesRegex(ValueError, "w_layer_index needs"):
            _fused((w1, w2, None, None), ids, w, ep=False, act="silu", layer_index=0)
        with self.assertRaisesRegex(AssertionError, "needs a layer-stacked rhs"):
            self._gmm(w1, rhs_layer_index=jnp.zeros((1,), jnp.int32))

    def test_scale_group_count_is_still_exact(self):
        w1, _, w1_scale, _ = _mxfp4_layer(0)
        stacked_w1 = jnp.stack([w1] * LAYERS)[None]
        layer_index = jnp.asarray([LAYER], jnp.int32)
        # The matching per-layer scale traces, so the failures below are the
        # group count alone.
        self._gmm(w1, w1_scale)
        self._gmm(stacked_w1, w1_scale, rhs_layer_index=layer_index)
        # 3D: a scale for half the weight's groups.
        with self.assertRaises(AssertionError):
            self._gmm(w1, w1_scale[: LOCAL // 2])
        # Stacked: scales stacked over layers instead of the layer's own.
        with self.assertRaises(AssertionError):
            self._gmm(
                stacked_w1,
                jnp.concatenate([w1_scale] * LAYERS),
                rhs_layer_index=layer_index,
            )


if __name__ == "__main__":
    unittest.main()
