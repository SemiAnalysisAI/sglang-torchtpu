"""Numerical coverage for the opt-in FP8 EP padding sentinel."""

import functools
import unittest

import numpy as np
import torch  # noqa: F401

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_tpu_ci(est_time=180, suite="stage-a-test-tpu-unit")

try:
    import jax
    import jax.numpy as jnp

    _HAS_TPU = any(d.platform == "tpu" for d in jax.devices())
except Exception:
    _HAS_TPU = False


@unittest.skipUnless(_HAS_TPU, "requires a JAX TPU backend")
class TestFp8MoePaddingSentinel(CustomTestCase):
    def test_alignment_and_bucket_padding_match_with_empty_and_active_ranks(self):
        from sglang.srt.layers.moe.pallas.fused_moe_op import _fused_moe_jax

        rng = np.random.default_rng(20261002)
        # GLM expert topology/top-k; reduced matrix widths keep this a unit test.
        w1 = jnp.asarray(rng.normal(0, 24, (16, 256, 256)), jnp.float8_e4m3fn)
        w2 = jnp.asarray(rng.normal(0, 24, (16, 128, 256)), jnp.float8_e4m3fn)
        s1 = jnp.asarray(rng.uniform(0.0004, 0.0008, (16, 1, 1, 256)), jnp.bfloat16)
        s2 = jnp.asarray(rng.uniform(0.0004, 0.0008, (16, 1, 1, 256)), jnp.bfloat16)

        for offset in (0, 112):
            for use_sc, dense_threshold in ((1, 0), (1, 3), (1, 32), (0, 0)):
                for no_combine in (False, True):
                    base = functools.partial(
                        _fused_moe_jax,
                        w1_scale=s1,
                        w2_scale=s2,
                        w1_bias=None,
                        w2_bias=None,
                        topk=8,
                        activation_name="silu",
                        global_num_experts=256,
                        expert_group_offset=offset,
                        use_sc_gather=use_sc,
                        permute_dense_threshold=dense_threshold,
                        no_combine=no_combine,
                    )
                    old = jax.jit(functools.partial(base, skip_pad_experts=False))
                    new = jax.jit(functools.partial(base, skip_pad_experts=True))
                    for local_routes in (0, 1, 8):
                        ids = np.concatenate(
                            (
                                np.arange(offset + 1, offset + 1 + local_routes),
                                np.arange(200, 200 + 8 - local_routes),
                            )
                        ).astype(np.int32)[None]
                        weights = rng.uniform(0.1, 1, (1, 8)).astype(np.float32)
                        weights /= weights.sum()
                        x = rng.normal(0, 1, (1, 256)).astype(np.float32)
                        # m=1 exercises alignment padding within the bridge;
                        # m=2 exercises an already-padded compile bucket.
                        for m in (1, 2):
                            ids_old, weights_m, x_m = ids, weights, x
                            if m == 2:
                                ids_old = np.pad(ids, ((0, 1), (0, 0)))
                                weights_m = np.pad(weights, ((0, 1), (0, 0)))
                                # Bucket padding need not have zero hidden
                                # states; zero route weights/invalid IDs must
                                # be sufficient to suppress its contribution.
                                x_m = np.concatenate(
                                    (x, rng.normal(0, 1, (1, 256))), axis=0
                                )
                            ids_new = ids_old.copy()
                            if m == 2:
                                ids_new[1] = 256
                            args = (
                                jnp.asarray(x_m, jnp.bfloat16),
                                w1,
                                w2,
                                jnp.asarray(weights_m, jnp.bfloat16),
                            )
                            reference = np.asarray(old(*args, jnp.asarray(ids_old)))
                            actual = np.asarray(new(*args, jnp.asarray(ids_new)))
                            with self.subTest(
                                offset=offset,
                                sc=use_sc,
                                dense=dense_threshold,
                                combine=not no_combine,
                                local_routes=local_routes,
                                tokens=m,
                            ):
                                self.assertTrue(np.isfinite(actual).all())
                                if no_combine and m == 2:
                                    # The old expert-zero placeholder can
                                    # produce unweighted pad outputs; downstream
                                    # routing weights suppressed them. The new
                                    # sentinel directly returns zeros instead.
                                    np.testing.assert_array_equal(
                                        actual[:1], reference[:1]
                                    )
                                else:
                                    np.testing.assert_array_equal(actual, reference)
                                if m == 2:
                                    np.testing.assert_array_equal(actual[1], 0)


if __name__ == "__main__":
    unittest.main()
