"""Invariant tests for TPU NVFP4 W4A8 numerical boundaries."""

import unittest
from unittest import mock

import numpy as np

from sglang.test.ci.ci_register import register_cpu_ci, register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_tpu_ci(est_time=5, suite="stage-a-test-tpu-unit")

try:
    import jax
    import jax.numpy as jnp

    _HAS_JAX = True
except ImportError:  # pragma: no cover - JAX is absent on some CPU CI agents
    _HAS_JAX = False

if _HAS_JAX:
    from sglang.srt.layers.moe.pallas.fused_rs import gmm_fused_rs_nodedup
    from sglang.srt.layers.moe.pallas.fused_rs.fused_moe import (
        _post_expert_rms_norm,
        _weighted_topk_reduce,
    )
    from sglang.srt.layers.moe.pallas.fused_rs.gmm_fused_rs_nodedup import (
        _quantize_fp8_blockwise,
    )


@unittest.skipUnless(_HAS_JAX, "requires JAX + Pallas")
class TestNvfp4W4A8Numerics(CustomTestCase):
    class _TpuInfo:
        num_lanes = 128
        mxu_column_size = 128
        vmem_capacity_bytes = 128 * 1024 * 1024

        @staticmethod
        def get_sublane_tiling(dtype):
            return 4 if jnp.dtype(dtype) == jnp.dtype(jnp.float32) else 8

    def test_direct_fp8_quantization_uses_fp32_block_amax(self):
        values = jnp.stack(
            (
                jnp.concatenate(
                    (
                        jnp.linspace(-3.25, 7.5, 512, dtype=jnp.float32),
                        jnp.zeros(512, dtype=jnp.float32),
                    )
                ),
                jnp.linspace(-19.0, 11.0, 1024, dtype=jnp.float32),
            )
        )

        quantized, scales = _quantize_fp8_blockwise(values, 512)

        expected_scales = jnp.max(
            jnp.abs(values.reshape(2, 2, 512)), axis=-1
        ) / jnp.float32(448.0)
        blocks = values.reshape(2, 2, 512)
        amax = jnp.max(jnp.abs(blocks), axis=-1, keepdims=True)
        amax_safe = jnp.where(amax == 0, jnp.ones_like(amax), amax)
        expected_quantized = (blocks / amax_safe * 448.0).astype(jnp.float8_e4m3fn)
        np.testing.assert_array_equal(np.asarray(scales), np.asarray(expected_scales))
        np.testing.assert_array_equal(
            np.asarray(quantized), np.asarray(expected_quantized.reshape(2, 1024))
        )
        self.assertEqual(quantized.dtype, jnp.float8_e4m3fn)
        self.assertEqual(float(scales[0, 1]), 0.0)
        np.testing.assert_array_equal(
            np.asarray(quantized[0, 512:]), np.zeros(512, dtype=np.float32)
        )
        self.assertEqual(float(jnp.max(jnp.abs(quantized[0, :512]))), 448.0)

    def test_direct_fp8_quantization_rejects_split_blocks(self):
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            _quantize_fp8_blockwise(jnp.ones((2, 513), jnp.float32), 512)

    def test_direct_fp8_preserves_baseline_tiles(self):
        shape = dict(
            size_m=2048,
            size_k1=4096,
            size_n1=28672,
            size_k2=14336,
            size_n2=4096,
            size_group=16,
            size_lhs_group=256,
            ep_size=16,
            out_dtype=jnp.bfloat16,
            w1_dtype=jnp.uint8,
            w2_dtype=jnp.uint8,
            is_quantized=True,
            quant_block_size=512,
            act_fn="silu",
            vmem_limit_bytes=120 * 1024 * 1024,
        )
        with (
            mock.patch.object(
                gmm_fused_rs_nodedup.pltpu,
                "get_tpu_info",
                return_value=self._TpuInfo(),
            ),
            mock.patch.object(
                gmm_fused_rs_nodedup,
                "get_fused_rs_tuned_block_sizes",
                side_effect=lambda *args, **kwargs: args[9],
            ),
        ):
            baseline = gmm_fused_rs_nodedup._select_fused_rs_block_sizes(**shape)
            direct = gmm_fused_rs_nodedup._select_fused_rs_block_sizes(
                **shape, direct_fp8_intermediate=True
            )

        self.assertEqual(direct, baseline)
        self.assertEqual(direct.tile_k2 % 512, 0)

    def test_fp32_gmm2_output_uses_dtype_aware_tiles(self):
        shape = dict(
            size_m=2048,
            size_k1=4096,
            size_n1=28672,
            size_k2=14336,
            size_n2=4096,
            size_group=16,
            size_lhs_group=256,
            ep_size=16,
            out_dtype=jnp.bfloat16,
            w1_dtype=jnp.uint8,
            w2_dtype=jnp.uint8,
            is_quantized=True,
            quant_block_size=512,
            act_fn="silu",
            vmem_limit_bytes=120 * 1024 * 1024,
        )
        with (
            mock.patch.object(
                gmm_fused_rs_nodedup.pltpu,
                "get_tpu_info",
                return_value=self._TpuInfo(),
            ),
            mock.patch.object(
                gmm_fused_rs_nodedup,
                "get_fused_rs_tuned_block_sizes",
                side_effect=AssertionError("bf16 tuned entry was reused"),
            ),
        ):
            fp32 = gmm_fused_rs_nodedup._select_fused_rs_block_sizes(
                **shape, fp32_gmm2_output=True
            )

        self.assertEqual(fp32.aligned_n2 % (128 * 4), 0)

    def test_fp32_topk_combine_matches_single_rounding_reference(self):
        experts = jnp.asarray(
            [
                [[4096.0, 1.0], [-4096.0, 3.0], [2.0, -5.0]],
                [[17.0, -13.0], [11.0, 7.0], [-23.0, 19.0]],
            ],
            dtype=jnp.bfloat16,
        )
        weights = jnp.asarray(
            [[0.50001, 0.49999, 0.00003], [0.2, 0.3, 0.5]], dtype=jnp.float32
        )

        actual = _weighted_topk_reduce(
            experts,
            weights,
            output_dtype=jnp.bfloat16,
            fp32_combine=True,
        )
        expected = jnp.sum(
            experts.astype(jnp.float32) * weights[:, :, None], axis=1
        ).astype(jnp.bfloat16)

        self.assertEqual(actual.dtype, jnp.bfloat16)
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    def test_post_expert_norm_stays_fp32_and_ignores_padding(self):
        experts = jnp.arange(2 * 2 * 256, dtype=jnp.float32).reshape(2, 2, 256)
        experts = (experts / 31.0).astype(jnp.bfloat16)
        experts = experts.at[:, :, 192:].set(jnp.bfloat16(1000.0))
        weight = jnp.linspace(-0.25, 0.25, 192, dtype=jnp.float32).astype(jnp.bfloat16)

        actual = _post_expert_rms_norm(experts, weight)

        valid = (
            experts[:, :, :192].astype(jnp.float32)
            * (weight.astype(jnp.float32) + 1.0)[None, None, :]
        )
        variance = jnp.sum(valid**2, axis=-1, keepdims=True) / 192
        expected_valid = valid * jax.lax.rsqrt(variance + 1e-8)
        expected = jnp.pad(expected_valid, ((0, 0), (0, 0), (0, 64)))
        self.assertEqual(actual.dtype, jnp.float32)
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


if __name__ == "__main__":
    unittest.main()
