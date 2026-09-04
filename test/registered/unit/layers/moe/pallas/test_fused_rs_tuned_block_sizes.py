"""Regression tests for fused-RS tuned-table composition."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.layers.moe.pallas.fused_rs.v2_fused_rs_tuned_block_sizes import (
    FUSED_RS_TUNED_BLOCK_SIZES,
)
from sglang.test.test_utils import CustomTestCase


class TestFusedRsTunedBlockSizes(CustomTestCase):
    def test_large_routed_bucket_uses_established_schedule(self):
        dtype_table = FUSED_RS_TUNED_BLOCK_SIZES["TPU7x"][
            "lhs_bfloat16_rhs_float4_e2m1fn"
        ]
        shape_table = dtype_table[
            "k-4096_n-8192_g-16_lhs_quant_rhs_bs-256"
            "+k-4096_n-4096_g-16_lhs_quant_rhs_bs-256"
        ]
        self.assertEqual(shape_table["65536_silu"], shape_table["8192_silu"])


if __name__ == "__main__":
    import unittest

    unittest.main()
