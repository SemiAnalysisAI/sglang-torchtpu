"""Qualified TPU7x expert-pair integration against the native GMM/combine.

Fresh Python/JAX functions are traced under each process-static setting.
This deliberately avoids changing flags underneath a cached public Torch op.
The dependent chain reuses weights to test arithmetic, not model performance.
"""

import importlib
import unittest
from unittest import mock

import numpy as np
import torch  # noqa: F401

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=300, suite="stage-a-test-tpu-unit")

try:
    import jax
    import jax.numpy as jnp

    _HAS_TPU = any(device.platform == "tpu" for device in jax.devices())
except Exception:
    _HAS_TPU = False


@unittest.skipUnless(_HAS_TPU, "requires a JAX TPU backend")
class TestGlmFp8FusedExperts(unittest.TestCase):
    def test_native_combine_empty_shards_offsets_and_dependent_chain(self):
        import ml_dtypes

        from sglang.srt.hardware_backend.tpu.glm_fp8_fused_experts import (
            supports_glm_fused_experts,
        )

        if not supports_glm_fused_experts(True, fp32_scales=False):
            self.skipTest("requires the qualified TPU7x hardware geometry")

        bridge = importlib.import_module(
            "sglang.srt.layers.moe.pallas.fused_moe_op"
        )
        gmm = importlib.import_module("sglang.srt.layers.moe.pallas.gmm_v2")
        rng = np.random.default_rng(20261003)
        weights, scales = [], []
        # Bound host temporaries to one expert. Per-channel normalization
        # exercises the full FP8 range, including values at +/-448.
        for k, n in ((6144, 4096), (2048, 6144)):
            host = np.empty((16, k, n), dtype=ml_dtypes.float8_e4m3fn)
            scale = np.empty((16, 1, 1, n), dtype=np.float32)
            for expert in range(16):
                raw = rng.standard_normal((k, n), dtype=np.float32) * 0.015
                channel_scale = np.abs(raw).max(axis=0) / np.float32(448)
                host[expert] = raw / channel_scale
                scale[expert, 0, 0] = channel_scale
            weights.append(jax.block_until_ready(jax.device_put(host)))
            scales.append(jnp.asarray(scale, jnp.bfloat16))
            del host, raw
        w1, w2 = weights
        s1, s2 = scales

        def make_case(offset, count, zero):
            # Include first/last active experts and gaps in the shard.
            local = offset + np.array([15, 0, 7, 3, 11, 5, 13, 1], np.int32)
            outside = np.setdiff1d(np.arange(256), np.arange(offset, offset + 16))
            ids = np.concatenate(
                (local[:count], rng.choice(outside, 8 - count, replace=False))
            ).astype(np.int32)
            rng.shuffle(ids)
            scores = rng.uniform(0.1, 1, (1, 8)).astype(np.float32)
            scores /= scores.sum(axis=-1, keepdims=True)
            x = rng.standard_normal((1, 6144), dtype=np.float32)
            if zero:
                x.fill(0)
            return (
                jnp.asarray(x, jnp.bfloat16), w1, w2,
                jnp.asarray(scores, jnp.bfloat16),
                jnp.asarray(ids[None], jnp.int32), s1, s2,
            )

        def compile_variant(offset, enabled, iterations, example):
            # Each invocation creates a distinct Python function object, so
            # earlier traces cannot capture the other static flag's branch.
            def run(x, a, b, scores, ids, sa, sb):
                original_x = x
                local = (ids >= offset) & (ids < offset + 16)
                for _ in range(iterations):
                    current_ids = ids
                    if iterations > 1:
                        bits = jax.lax.bitcast_convert_type(
                            x[0, 0].astype(jnp.float32), jnp.int32
                        )
                        shift = (bits >> 16) & 15
                        shifted = ((ids - offset + shift) & 15) + offset
                        current_ids = jnp.where(local, shifted, ids)
                    x = bridge._fused_moe_jax(
                        x, a, b, scores, current_ids, sa, sb, None, None,
                        8, "silu", global_num_experts=256,
                        expert_group_offset=offset, use_sc_gather=1,
                        permute_dense_threshold=3, skip_pad_experts=True,
                        bs1_fused_routing=True,
                    )
                    if iterations > 1:
                        x = x + original_x * jnp.bfloat16(0.125)
                return x

            with (
                mock.patch.object(bridge, "_GLM_FUSED_EXPERTS", enabled),
                mock.patch.object(gmm, "_FP32_MOE_SCALES", False),
                mock.patch.object(gmm, "_GLM_GMM_TWO_RHS_BUFFERS", True),
            ):
                compiled = jax.jit(run).lower(*example).compile()
            hlo = compiled.as_text()
            self.assertEqual("glm_bs1_fused_experts_lookahead" in hlo, enabled)
            if not enabled:
                self.assertIn("gmm_v2", hlo)
            return compiled

        for offset in (0, 112, 240):
            example = make_case(offset, 3, False)
            compiled = {
                (enabled, iterations): compile_variant(
                    offset, enabled, iterations, example
                )
                for enabled in (False, True)
                for iterations in (1, 4)
            }
            for count in (0, 1, 3, 8):
                for zero in (False, True):
                    values = make_case(offset, count, zero)
                    for iterations in (1, 4):
                        expected = jax.block_until_ready(
                            compiled[False, iterations](*values)
                        )
                        actual = jax.block_until_ready(
                            compiled[True, iterations](*values)
                        )
                        with self.subTest(
                            offset=offset, count=count,
                            zero=zero, iterations=iterations,
                        ):
                            self.assertEqual(actual.dtype, jnp.bfloat16)
                            self.assertTrue(
                                np.isfinite(np.asarray(actual, np.float32)).all()
                            )
                            np.testing.assert_array_equal(
                                np.asarray(actual).view(np.uint16),
                                np.asarray(expected).view(np.uint16),
                            )


if __name__ == "__main__":
    unittest.main()
