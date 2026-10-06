"""TPU dispatch coverage and full-MoE parity with the unchanged native combine."""

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

    _HAS_TPU = any(device.platform == "tpu" for device in jax.devices())
except Exception:
    _HAS_TPU = False


def reference_dispatch(x, ids):
    padded_ids = jnp.pad(ids.reshape(8), (0, 8), constant_values=256)
    permutation = jnp.argsort(padded_ids, stable=True)
    tokens = jnp.repeat(jnp.arange(2), 8)[permutation]
    xs = jnp.pad(x, ((0, 1), (0, 0)))[tokens]
    groups = jax.nn.one_hot(padded_ids, 256, dtype=jnp.int32).sum(0)
    return xs, groups, jnp.argsort(permutation, stable=True)[:8]


@unittest.skipUnless(_HAS_TPU, "requires a JAX TPU backend")
class TestFp8MoeBs1Routing(CustomTestCase):
    def test_dispatch_stable_ranks_and_padding(self):
        from sglang.srt.hardware_backend.tpu.fp8_moe_bs1_routing import dispatch_bs1

        rng = np.random.default_rng(9791)
        dispatch = jax.jit(dispatch_bs1)
        reference = jax.jit(reference_dispatch)
        for case in range(120):
            offset = (0, 112, 240)[case % 3]
            route_ids = rng.integers(0, 256, size=(1, 8), dtype=np.int32)
            route_ids[0, : case % 9] = offset + np.arange(case % 9) % 16
            if case % 5 == 0:
                route_ids[0, -2:] = 256
            if case % 7 == 0:
                route_ids[0, :2] = offset
            if case == 119:
                route_ids[:] = 256
            ids = jnp.asarray(route_ids)
            x = jnp.asarray(rng.standard_normal((1, 6144)), dtype=jnp.bfloat16)
            actual = dispatch(x, ids)
            expected = reference(x, ids)
            with self.subTest(case=case, offset=offset):
                for actual_part, expected_part in zip(actual, expected):
                    np.testing.assert_array_equal(actual_part, expected_part)

    def test_full_glm_shape_matches_native_moe(self):
        import ml_dtypes

        from sglang.srt.layers.moe.pallas.fused_moe_op import _fused_moe_jax

        rng = np.random.default_rng(20261002)
        weights = []
        scales = []
        # The guard deliberately accepts the actual GLM shape only. Generate
        # one expert at a time to avoid multi-GB FP32 host temporaries.
        for k, n in ((6144, 4096), (2048, 6144)):
            host = np.empty((16, k, n), dtype=ml_dtypes.float8_e4m3fn)
            for expert in range(16):
                host[expert] = np.clip(
                    rng.standard_normal((k, n), dtype=np.float32) * 96, -448, 448
                )
            weights.append(jax.block_until_ready(jax.device_put(host)))
            del host
            scales.append(
                jnp.asarray(
                    rng.uniform(0.0004, 0.0008, (16, 1, 1, n)), jnp.bfloat16
                )
            )
        w1, w2 = weights
        s1, s2 = scales
        for offset in (0, 112, 240):
            common = functools.partial(
                _fused_moe_jax,
                w1_scale=s1,
                w2_scale=s2,
                w1_bias=None,
                w2_bias=None,
                topk=8,
                activation_name="silu",
                global_num_experts=256,
                expert_group_offset=offset,
                use_sc_gather=1,
                permute_dense_threshold=3,
                skip_pad_experts=True,
            )
            native = jax.jit(functools.partial(common, bs1_fused_routing=False))
            fused = jax.jit(functools.partial(common, bs1_fused_routing=True))
            local = np.arange(offset, offset + 16, dtype=np.int32)
            other = np.setdiff1d(np.arange(256, dtype=np.int32), local)
            for local_count in (0, 1, 4, 8):
                ids = np.concatenate(
                    (
                        rng.choice(local, local_count, replace=False),
                        rng.choice(other, 8 - local_count, replace=False),
                    )
                )
                rng.shuffle(ids)
                scores = rng.uniform(0.1, 1, (1, 8)).astype(np.float32)
                scores /= scores.sum()
                x = jnp.asarray(rng.standard_normal((1, 6144)), jnp.bfloat16)
                args = (
                    x,
                    w1,
                    w2,
                    jnp.asarray(scores, jnp.bfloat16),
                    jnp.asarray(ids[None], jnp.int32),
                )
                expected = native(*args)
                actual = fused(*args)
                with self.subTest(offset=offset, local_count=local_count):
                    np.testing.assert_array_equal(actual, expected)
                    self.assertTrue(np.isfinite(np.asarray(actual)).all())


if __name__ == "__main__":
    unittest.main()
