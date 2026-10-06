"""Static pipeline eligibility; imports allocate no device arrays."""

import dataclasses
import importlib
import unittest
from unittest import mock

import jax.numpy as jnp

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

gmm = importlib.import_module("sglang.srt.layers.moe.pallas.gmm_v2")


def fixture(fuse_act):
    size_k, size_n, tile_n = (
        (6144, 2048, 1024) if fuse_act else (2048, 6144, 6144)
    )
    return gmm.GmmConfigs(
        dims=gmm.Dimensions(16, size_k, size_n, 16, 256, 16),
        tiles=gmm.TileSizes(16, size_k, tile_n),
        lhs_cfgs=gmm.InputConfigs(
            dtype=jnp.bfloat16,
            quant_dtype=jnp.float8_e4m3fn,
            quant_block_size=512,
        ),
        rhs_cfgs=gmm.InputConfigs(
            dtype=jnp.float8_e4m3fn,
            quant_dtype=jnp.float8_e4m3fn,
            quant_block_size=size_k,
            has_scale=True,
            packing=4,
            num_quant_blocks=1,
            scale_size_n=size_n * (2 if fuse_act else 1),
            scale_block_n=1,
            scale_dtype=jnp.bfloat16,
        ),
        out_dtype=jnp.bfloat16,
        acc_dtype=jnp.bfloat16,
        zero_init=False,
        fuse_act=fuse_act,
    )


class TestGlmGmmRhsBuffers(unittest.TestCase):
    def test_both_qualified_projections_and_default_off(self):
        for fuse_act in ("silu", None):
            cfgs = fixture(fuse_act)
            for enabled, expected in ((False, 3), (True, 2)):
                with self.subTest(fuse_act=fuse_act, enabled=enabled):
                    with mock.patch.object(gmm, "_GLM_GMM_TWO_RHS_BUFFERS", enabled):
                        with mock.patch.object(gmm, "_FP32_MOE_SCALES", False):
                            self.assertEqual(gmm._rhs_buffer_count(cfgs), expected)
            with mock.patch.object(gmm, "_GLM_GMM_TWO_RHS_BUFFERS", True):
                with mock.patch.object(gmm, "_FP32_MOE_SCALES", True):
                    self.assertEqual(gmm._rhs_buffer_count(cfgs), 3)

    def test_other_shapes_dtypes_scales_and_tiles_keep_three(self):
        cases = [
            ("dims", "size_m", 32),
            ("dims", "size_group", 8),
            ("dims", "size_lhs_group", 16),
            ("dims", "size_lhs_sublane", 8),
            ("dims", "size_k", 4096),
            ("dims", "size_n", 4096),
            ("tiles", "tile_m", 32),
            ("tiles", "tile_k", 512),
            ("tiles", "tile_n", 512),
            ("lhs_cfgs", "dtype", jnp.float32),
            ("lhs_cfgs", "quant_dtype", None),
            ("lhs_cfgs", "quant_block_size", 128),
            ("rhs_cfgs", "dtype", jnp.float8_e5m2),
            ("rhs_cfgs", "quant_dtype", jnp.int4),
            ("rhs_cfgs", "packing", 8),
            ("rhs_cfgs", "has_scale", False),
            ("rhs_cfgs", "has_bias", True),
            ("rhs_cfgs", "num_quant_blocks", 16),
            ("rhs_cfgs", "quant_block_size", 128),
            ("rhs_cfgs", "scale_block_n", 128),
            ("rhs_cfgs", "scale_size_n", 128),
            ("rhs_cfgs", "scale_dtype", jnp.float32),
            (None, "out_dtype", jnp.float32),
            (None, "acc_dtype", jnp.float32),
            (None, "zero_init", True),
            (None, "fuse_act", "gelu"),
        ]
        with mock.patch.object(gmm, "_GLM_GMM_TWO_RHS_BUFFERS", True):
            with mock.patch.object(gmm, "_FP32_MOE_SCALES", False):
                for fuse_act in ("silu", None):
                    for parent, name, value in cases:
                        with self.subTest(fuse_act=fuse_act, parent=parent, name=name):
                            cfgs = fixture(fuse_act)
                            if parent is None:
                                cfgs = dataclasses.replace(cfgs, **{name: value})
                            else:
                                child = dataclasses.replace(
                                    getattr(cfgs, parent), **{name: value}
                                )
                                cfgs = dataclasses.replace(cfgs, **{parent: child})
                            self.assertEqual(gmm._rhs_buffer_count(cfgs), 3)


if __name__ == "__main__":
    unittest.main()
