"""PP dynamic chunking sizes a new request's first chunk by its matched prefix."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _scheduler(*, enable_dynamic_chunking: bool, predicted: int = 1024):
    s = Scheduler.__new__(Scheduler)
    s.enable_dynamic_chunking = enable_dynamic_chunking
    s.predict_next_chunk_size = MagicMock(return_value=predicted)
    return s


def _req(*, prefix_len: int, host_hit_length: int = 0):
    return SimpleNamespace(
        prefix_indices=[0] * prefix_len, host_hit_length=host_hit_length
    )


class TestDynamicFirstChunkCap(CustomTestCase):
    def test_no_cap_without_dynamic_chunking(self):
        s = _scheduler(enable_dynamic_chunking=False)
        self.assertIsNone(s._dynamic_first_chunk_cap(_req(prefix_len=4096)))
        s.predict_next_chunk_size.assert_not_called()

    def test_no_cap_without_matched_prefix(self):
        # A cold request keeps the static chunk; only a prefix adds attention cost.
        s = _scheduler(enable_dynamic_chunking=True)
        self.assertIsNone(s._dynamic_first_chunk_cap(_req(prefix_len=0)))
        s.predict_next_chunk_size.assert_not_called()

    def test_cap_sized_by_prefix_and_host_hit(self):
        # A host-tier hit is loaded back, not recomputed, but it is still KV
        # history the first chunk attends over.
        s = _scheduler(enable_dynamic_chunking=True, predicted=1024)
        cap = s._dynamic_first_chunk_cap(_req(prefix_len=4096, host_hit_length=512))
        self.assertEqual(cap, 1024)
        s.predict_next_chunk_size.assert_called_once_with(4608)


if __name__ == "__main__":
    unittest.main()
