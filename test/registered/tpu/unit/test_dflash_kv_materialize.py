"""Correctness coverage for vectorized DFlash KV materialization on TPU."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

import sglang.srt.hardware_backend.tpu.dflash_kv_materialize as kv_materialize
from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit")

from sglang.srt.hardware_backend.tpu.env_override import apply_tpu_env_overrides

# Serving applies this before the first TPU operation. This standalone test
# imports the TorchTPU/JAX bridge directly, so it must do the same.
apply_tpu_env_overrides()

from sglang.srt.hardware_backend.tpu.dflash_kv_materialize import (
    _multi_layer_scatter_into,
    _pack_fused_kv_rows,
    _pad_prefill_materialize_inputs,
    _scatter_available,
    _select_prefill_token_bucket,
)
from sglang.test.test_utils import CustomTestCase


class TestPackFusedKvRows(CustomTestCase):
    def test_interleaves_kv_heads_and_zero_pads(self):
        layers, tokens, heads, head_dim = 2, 3, 2, 4
        k = torch.arange(layers * tokens * heads * head_dim).reshape(
            layers, tokens, heads, head_dim
        )
        v = k + 1000

        packed = _pack_fused_kv_rows(
            k_all=k,
            v_all=v,
            head_num=heads,
            head_dim=head_dim,
            padded_head_dim=8,
            num_kv_heads_x2=8,
            physical_tail=(2, 4, 8),
            store_dtype=torch.bfloat16,
        )
        logical = packed.view(layers, tokens, 8, 8).float()

        expected_k = k.to(torch.bfloat16).float()
        expected_v = v.to(torch.bfloat16).float()
        torch.testing.assert_close(logical[:, :, 0, :head_dim], expected_k[:, :, 0])
        torch.testing.assert_close(logical[:, :, 1, :head_dim], expected_v[:, :, 0])
        torch.testing.assert_close(logical[:, :, 2, :head_dim], expected_k[:, :, 1])
        torch.testing.assert_close(logical[:, :, 3, :head_dim], expected_v[:, :, 1])
        self.assertTrue(torch.count_nonzero(logical[:, :, :4, head_dim:]) == 0)
        self.assertTrue(torch.count_nonzero(logical[:, :, 4:, :]) == 0)

    def test_prefill_inputs_pad_to_static_bucket(self):
        hidden = torch.arange(15, dtype=torch.float32).reshape(3, 5)
        positions = torch.tensor([7, 8, 9], dtype=torch.int64)
        cache_loc = torch.tensor([11, 12, 13], dtype=torch.int64)

        padded_hidden, padded_positions, padded_cache_loc, valid_mask = (
            _pad_prefill_materialize_inputs(
                target_hidden=hidden,
                positions=positions,
                cache_loc=cache_loc,
                bucket_tokens=8,
            )
        )

        self.assertEqual(tuple(padded_hidden.shape), (8, 5))
        self.assertTrue(torch.equal(padded_hidden[:3], hidden))
        self.assertTrue(torch.count_nonzero(padded_hidden[3:]) == 0)
        self.assertTrue(
            torch.equal(padded_positions, torch.tensor([7, 8, 9, 0, 0, 0, 0, 0]))
        )
        self.assertTrue(
            torch.equal(padded_cache_loc, torch.tensor([11, 12, 13, 0, 0, 0, 0, 0]))
        )
        self.assertTrue(
            torch.equal(
                valid_mask,
                torch.tensor([True, True, True, False, False, False, False, False]),
            )
        )

    def test_selects_smallest_covering_prefill_bucket(self):
        buckets = (64, 128, 256)
        self.assertEqual(
            _select_prefill_token_bucket(num_tokens=1, token_buckets=buckets), 64
        )
        self.assertEqual(
            _select_prefill_token_bucket(num_tokens=64, token_buckets=buckets), 64
        )
        self.assertEqual(
            _select_prefill_token_bucket(num_tokens=65, token_buckets=buckets), 128
        )
        self.assertEqual(
            _select_prefill_token_bucket(num_tokens=256, token_buckets=buckets), 256
        )
        self.assertIsNone(
            _select_prefill_token_bucket(num_tokens=257, token_buckets=buckets)
        )


class TestMultiLayerScatterRouting(CustomTestCase):
    def test_dispatches_one_fixed_schema_op_per_layer(self):
        """Any model-static layer count maps to one donated scatter per cache."""
        srcs = [torch.full((2, 3), layer) for layer in range(3)]
        dsts = [torch.zeros((8, 3)) for _ in srcs]
        locations = torch.tensor([2, 5], dtype=torch.int32)

        with (
            patch.object(kv_materialize, "_jax_import_error", None),
            patch.object(
                kv_materialize,
                "_make_scatter_one_layer_kernel",
                return_value=lambda *args: args[-1],
            ) as make_kernel,
        ):
            outputs = _multi_layer_scatter_into(
                srcs=srcs,
                dsts=dsts,
                dest_offsets=locations,
            )

        make_kernel.assert_called_once_with()
        self.assertEqual(len(outputs), 3)
        for output, dst in zip(outputs, dsts):
            self.assertIs(output, dst)


class TestMultiLayerScatter(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch_tpu  # noqa: F401

            cls.device = torch.device("tpu")
            torch.zeros(1, device=cls.device)
            if not _scatter_available():
                raise RuntimeError("TorchTPU Pallas bridge is unavailable")
        except Exception as error:  # noqa: BLE001 - any TPU runtime failure skips
            raise unittest.SkipTest(f"TPU not available: {error}")

    def test_masked_token_rows_update_all_layers(self):
        # A non-eight layer count guards the generic fixed-schema registration.
        num_layers, num_tokens, num_slots = 3, 4, 16
        trailing = (1, 4, 128)
        srcs_cpu = []
        for layer in range(num_layers):
            rows = torch.arange(num_tokens * 4 * 128, dtype=torch.float32).reshape(
                num_tokens, *trailing
            )
            srcs_cpu.append((rows + layer * 10000).to(torch.bfloat16))

        srcs = [source.to(self.device) for source in srcs_cpu]
        dsts = [
            torch.zeros(
                (num_slots, *trailing), dtype=torch.bfloat16, device=self.device
            )
            for _ in range(num_layers)
        ]
        locations = torch.tensor([3, 7, 11, 15], dtype=torch.int32, device=self.device)
        valid_mask = torch.tensor([1, 0, 1, 0], dtype=torch.int32, device=self.device)

        outputs = _multi_layer_scatter_into(
            srcs=srcs,
            dsts=dsts,
            dest_offsets=locations,
            valid_mask=valid_mask,
        )
        for layer, output in enumerate(outputs):
            expected = torch.zeros((num_slots, *trailing), dtype=torch.bfloat16)
            expected[3].copy_(srcs_cpu[layer][0])
            expected[11].copy_(srcs_cpu[layer][2])
            self.assertTrue(torch.equal(output.cpu(), expected))


if __name__ == "__main__":
    unittest.main(verbosity=2)
