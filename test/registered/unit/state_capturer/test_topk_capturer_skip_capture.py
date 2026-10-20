"""BaseTopkCapturer.skip_capture: a skipped forward reports -1 rows and resets."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.state_capturer.base import BaseDeviceCache, BaseTopkCapturer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_NUM_TOKENS, _NUM_LAYERS, _TOPK = 16, 2, 3


def _capturer() -> BaseTopkCapturer:
    # The real host cache pins an mmap arena; a plain tensor has the same indexing.
    cap = object.__new__(BaseTopkCapturer)
    cap.num_layers, cap.topk_size = _NUM_LAYERS, _TOPK
    cap.host_cache = SimpleNamespace(
        buffer=torch.zeros(_NUM_TOKENS, _NUM_LAYERS, _TOPK, dtype=torch.int32)
    )
    cap.device_cache = BaseDeviceCache(
        max_batch_size=8,
        num_layers=_NUM_LAYERS,
        topk_size=_TOPK,
        device="cpu",
        name="t",
    )
    cap.skip_capture = False
    return cap


def _forward(cap: BaseTopkCapturer, out_cache_loc: list[int], *, no_copy_to_cpu: bool):
    batch = len(out_cache_loc)
    for layer_id in range(_NUM_LAYERS):
        ids = torch.arange(batch * _TOPK, dtype=torch.int32).reshape(batch, _TOPK)
        cap.capture(layer_id=layer_id, topk_indices=ids + 100 * layer_id)
    fb = SimpleNamespace(
        out_cache_loc=torch.tensor(out_cache_loc), out_cache_loc_virtual=None
    )
    out = cap.on_forward_end(
        forward_batch=fb,
        can_run_graph=False,
        cuda_graph_batch=None,
        no_copy_to_cpu=no_copy_to_cpu,
    )
    if out is not None:
        out.finalize()


class TestSkipCapture(CustomTestCase):
    def test_skipped_forward_reports_unknown_and_next_forward_captures(self):
        for no_copy_to_cpu in (False, True):
            cap = _capturer()
            _forward(cap, [5, 6], no_copy_to_cpu=no_copy_to_cpu)
            captured = cap.host_cache.buffer[[5, 6]].clone()

            cap.skip_capture = True
            _forward(cap, [0, 1, 2], no_copy_to_cpu=no_copy_to_cpu)
            host = cap.host_cache.buffer
            self.assertTrue(torch.all(host[[0, 1, 2]] == -1))
            # Nothing was written to the device rows the skipped forward owned.
            self.assertTrue(torch.all(cap.device_cache.buffer[2] == 0))
            self.assertFalse(cap.skip_capture)

            # A forward nobody marked (e.g. an eager one) captures again.
            _forward(cap, [7, 8], no_copy_to_cpu=no_copy_to_cpu)
            self.assertTrue(torch.equal(host[[7, 8]], captured))
            self.assertTrue(torch.all(host[[0, 1, 2]] == -1))


if __name__ == "__main__":
    unittest.main()
