# SPDX-License-Identifier: Apache-2.0
"""The streaming one-token MoE must equal the route-meta + gmm_v2 fused-MoE path."""

import unittest

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

try:
    import jax
    import jax.numpy as jnp
    import numpy as np

    _HAS_TPU = any(d.platform == "tpu" for d in jax.devices())
except Exception:  # pragma: no cover - non-TPU host
    _HAS_TPU = False

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit")

# GLM-5.3-Flash at TP8 with tensor-parallel experts: 288 experts of width 256.
HIDDEN, INTER, EXPERTS, TOPK, LIMIT = 4096, 256, 288, 8, 10.0


@unittest.skipUnless(_HAS_TPU, "requires a JAX TPU backend")
class TestMoeStreamBs1(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        keys = iter(jax.random.split(jax.random.PRNGKey(0), 8))
        f8 = jnp.float8_e4m3fn
        cls.w1 = (
            jax.random.normal(next(keys), (EXPERTS, HIDDEN, 2 * INTER)) * 0.5
        ).astype(f8)
        cls.w2 = (jax.random.normal(next(keys), (EXPERTS, INTER, HIDDEN)) * 0.5).astype(
            f8
        )
        cls.s1 = jax.random.uniform(
            next(keys), (EXPERTS, 1, 1, 2 * INTER), minval=1e-3, maxval=1e-2
        )
        cls.s2 = jax.random.uniform(
            next(keys), (EXPERTS, 1, 1, HIDDEN), minval=1e-3, maxval=1e-2
        )
        cls.keys = keys

    def inputs(self, seed):
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(seed), 3)
        x = (jax.random.normal(k1, (1, HIDDEN)) * 3).astype(jnp.bfloat16)
        ids = jax.random.permutation(k2, EXPERTS)[:TOPK].astype(jnp.int32)
        wts = jax.nn.softmax(jax.random.normal(k3, (1, TOPK))).astype(jnp.bfloat16)
        return x, ids.reshape(1, TOPK), wts

    def test_bit_identical_to_route_meta_path(self):
        from sglang_tpu.kernels.moe_stream_bs1 import (
            moe_stream_bs1,
        )
        from sglang_tpu.moe.pallas.fused_moe_op import _fused_moe_jax

        for seed in range(5):
            with self.subTest(seed=seed):
                x, ids, wts = self.inputs(seed)
                expected = _fused_moe_jax(
                    x,
                    self.w1,
                    self.w2,
                    wts,
                    ids,
                    self.s1,
                    self.s2,
                    None,
                    None,
                    topk=TOPK,
                    activation_name=f"silu_and_mul_clamp:{LIMIT}",
                    permute_dense_threshold=512,
                )
                actual = moe_stream_bs1(
                    x, ids, wts, self.w1, self.s1, self.w2, self.s2, LIMIT
                )
                np.testing.assert_array_equal(
                    np.asarray(actual, np.float32), np.asarray(expected, np.float32)
                )

    def test_declines_outside_one_token_ep1(self):
        from sglang_tpu.moe.pallas.fused_moe_op import _maybe_moe_stream_bs1

        x, ids, wts = self.inputs(0)
        common = dict(activation_name=f"silu_and_mul_clamp:{LIMIT}", w_quant_bits=0)
        args = (self.w1, self.w2, wts, ids, self.s1, self.s2, None, None)
        self.assertIsNotNone(
            _maybe_moe_stream_bs1(
                x, *args, no_combine=False, use_group_offset=False, **common
            )
        )
        # Expert parallelism, no_combine and multi-token batches keep the gmm path.
        self.assertIsNone(
            _maybe_moe_stream_bs1(
                x, *args, no_combine=False, use_group_offset=True, **common
            )
        )
        self.assertIsNone(
            _maybe_moe_stream_bs1(
                x, *args, no_combine=True, use_group_offset=False, **common
            )
        )
        # Plain SiLU fuses the activation into gmm_v2 and keeps that path.
        self.assertIsNone(
            _maybe_moe_stream_bs1(
                x,
                *args,
                no_combine=False,
                use_group_offset=False,
                activation_name="silu",
                w_quant_bits=0,
            )
        )
        x2 = jnp.concatenate([x, x])
        self.assertIsNone(
            _maybe_moe_stream_bs1(
                x2, *args, no_combine=False, use_group_offset=False, **common
            )
        )


if __name__ == "__main__":
    unittest.main()
