"""Metadata guard coverage without creating devices or expert tensors."""

import unittest

import jax
import jax.numpy as jnp

from sglang.srt.hardware_backend.tpu.fp8_moe_bs1_routing import supports_bs1_routing
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def fixture():
    tensor = jax.ShapeDtypeStruct
    return {
        "hidden_states": tensor((1, 6144), jnp.bfloat16),
        "w1": tensor((16, 6144, 4096), jnp.float8_e4m3fn),
        "w2": tensor((16, 2048, 6144), jnp.float8_e4m3fn),
        "topk_weights": tensor((1, 8), jnp.bfloat16),
        "topk_ids": tensor((1, 8), jnp.int32),
        "w1_scale": tensor((16, 1, 1, 4096), jnp.bfloat16),
        "w2_scale": tensor((16, 1, 1, 6144), jnp.bfloat16),
        "w1_bias": None,
        "w2_bias": None,
        "topk": 8,
        "activation_name": "silu",
        "no_combine": False,
        "global_num_experts": 256,
        "expert_group_offset": 0,
        "use_sc_gather": 1,
        "w_quant_bits": 0,
        "permute_dense_threshold": 3,
        "skip_pad_experts": True,
    }


class TestFp8MoeBs1RoutingGuard(unittest.TestCase):
    def test_all_qualified_expert_parallel_ranks(self):
        args = fixture()
        for offset in range(0, 256, 16):
            with self.subTest(offset=offset):
                args["expert_group_offset"] = offset
                self.assertTrue(supports_bs1_routing(**args))

    def test_unsupported_contracts_fall_back(self):
        tensor = jax.ShapeDtypeStruct
        cases = [
            ("hidden_states", tensor((2, 6144), jnp.bfloat16)),
            ("hidden_states", tensor((1, 6144), jnp.float32)),
            ("hidden_states", tensor((1, 4096), jnp.bfloat16)),
            ("w1", tensor((8, 6144, 4096), jnp.float8_e4m3fn)),
            ("w2", tensor((16, 2048, 6144), jnp.float8_e5m2)),
            ("w2", tensor((16, 1024, 6144), jnp.uint8)),
            ("topk_ids", tensor((1, 8), jnp.int64)),
            ("topk_ids", tensor((1, 4), jnp.int32)),
            ("topk_weights", tensor((1, 8), jnp.float32)),
            ("w1_scale", None),
            ("w2_scale", None),
            ("w1_scale", tensor((16, 48, 1, 4096), jnp.bfloat16)),
            ("w2_scale", tensor((16, 1, 1, 6144), jnp.float32)),
            ("w1_bias", tensor((16, 1, 4096), jnp.bfloat16)),
            ("w2_bias", tensor((16, 1, 6144), jnp.bfloat16)),
            ("topk", 4),
            ("activation_name", "silu_clamped:7.0"),
            ("activation_name", "gelu"),
            ("no_combine", True),
            ("global_num_experts", 16),
            ("expert_group_offset", -16),
            ("expert_group_offset", 15),
            ("expert_group_offset", 256),
            ("use_sc_gather", 0),
            ("w_quant_bits", 4),
            ("permute_dense_threshold", 0),
            ("permute_dense_threshold", 32),
            ("skip_pad_experts", False),
        ]
        for name, value in cases:
            with self.subTest(name=name, value=value):
                args = fixture()
                args[name] = value
                self.assertFalse(supports_bs1_routing(**args))


if __name__ == "__main__":
    unittest.main()
