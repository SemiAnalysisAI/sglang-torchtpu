"""Gating and layout contract of the TPU KN linear weight (CPU, TPU flag patched)."""

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

import sglang.srt.layers.quantization.unquant as unquant
from sglang.srt.environ import envs
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _layer(bias=False, allow=True):
    layer = ReplicatedLinear(64, 96, bias=bias, params_dtype=torch.float32)
    with torch.no_grad():
        layer.weight.copy_(torch.randn(96, 64))
        if bias:
            layer.bias.copy_(torch.randn(96))
    if allow is not None:
        layer.allow_kn_weight = allow
    return layer


def _process(layer, *, tpu=True, flag=True):
    with (
        patch.object(unquant, "_is_tpu", tpu),
        patch.object(unquant, "_is_cpu", False),
        envs.SGLANG_TPU_ENABLE_LINEAR_KN_WEIGHT.override(flag),
    ):
        layer.quant_method.process_weights_after_loading(layer)


def _apply(layer, x, tpu=True):
    with patch.object(unquant, "_is_tpu", tpu):
        out, _ = layer(x)
    return out


class TestUnquantTpuKnWeight(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)

    def _assert_unchanged(self, layer, before):
        self.assertIsNone(getattr(layer, "weight_kn", None))
        self.assertTrue(layer.weight.is_contiguous())
        self.assertEqual(layer.weight.data_ptr(), before)

    def test_gated_off(self):
        for kwargs, allow in [
            (dict(flag=False), True),
            (dict(tpu=False), True),
            (dict(), None),  # layer not opted in
            (dict(), False),
        ]:
            with self.subTest(kwargs=kwargs, allow=allow):
                layer = _layer(allow=allow)
                before = layer.weight.data_ptr()
                _process(layer, **kwargs)
                self._assert_unchanged(layer, before)

    @torch.no_grad()
    def test_kn_layout_keeps_nk_view_and_output(self):
        for bias in (False, True):
            with self.subTest(bias=bias):
                layer = _layer(bias=bias)
                x = torch.randn(5, 64)
                ref = F.linear(x, layer.weight.clone(), layer.bias)
                _process(layer)
                # Consumers keep seeing [out, in]; storage is [in, out].
                self.assertEqual(tuple(layer.weight.shape), (96, 64))
                self.assertTrue(layer.weight_kn.is_contiguous())
                self.assertEqual(tuple(layer.weight_kn.shape), (64, 96))
                self.assertEqual(layer.weight_kn.data_ptr(), layer.weight.data_ptr())
                torch.testing.assert_close(_apply(layer, x), ref)

    @torch.no_grad()
    def test_reload_writes_through_and_process_is_idempotent(self):
        layer = _layer()
        _process(layer)
        storage = layer.weight_kn.data_ptr()
        new_w = torch.randn(96, 64)
        layer.weight_loader(layer.weight, new_w)
        _process(layer)
        self.assertEqual(layer.weight_kn.data_ptr(), storage)
        x = torch.randn(3, 64)
        torch.testing.assert_close(_apply(layer, x), F.linear(x, new_w))


if __name__ == "__main__":
    unittest.main()
