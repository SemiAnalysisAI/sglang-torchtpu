"""Unit tests for Scheduler.init_overlap stream gating.

Overlap scheduling on a backend with real multi-stream execution (CUDA/ROCm)
uses CUDA-style forward/copy/schedule streams. A single-queue backend such as
torch_tpu serialises all device work on one PJRT queue, so cross-stream ordering
is a no-op and a non-default stream only relabels async-copy fences — there
overlap comes from async dispatch plus a scoped completion event instead.

``Platform.supports_overlap_streams()`` is the seam that selects between the two,
and ``Scheduler.init_overlap`` consumes it via ``self.use_overlap_streams``. These
tests pin that gating in isolation (no real worker / device), since the rest of
the overlap path is now shared between TPU and GPU.
"""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

_SCHED_MOD = "sglang.srt.managers.scheduler"


def _make_scheduler(*, enable_overlap: bool) -> Scheduler:
    """A bare Scheduler with only the attributes init_overlap touches."""
    s = Scheduler.__new__(Scheduler)
    s.device = "cpu"
    s.forward_stream = object()  # sentinel; only used on the stream path
    s.enable_overlap = enable_overlap
    s.spec_algorithm = MagicMock()
    s.max_running_requests = 8
    s.chunked_prefill_size = 2048
    s.model_config = SimpleNamespace(context_len=4096)
    return s


def _run_init_overlap(s: Scheduler, *, use_mlx: bool, supports_streams: bool):
    """Invoke init_overlap with use_mlx() and the platform capability stubbed.

    Returns the MagicMock standing in for ``torch.get_device_module(device)`` so
    callers can assert whether stream factories were touched.
    """
    mock_platform = MagicMock()
    mock_platform.supports_overlap_streams.return_value = supports_streams
    mock_device_module = MagicMock()
    with patch(f"{_SCHED_MOD}.use_mlx", return_value=use_mlx), patch(
        f"{_SCHED_MOD}.current_platform", mock_platform
    ), patch("torch.get_device_module", return_value=mock_device_module):
        Scheduler.init_overlap(s)
    return mock_device_module


class TestInitOverlapStreamGating(CustomTestCase):
    def test_single_queue_skips_streams(self):
        """supports_overlap_streams=False -> nullcontext stand-ins, no real
        forward/copy/schedule streams, but the FutureMap is still built."""
        s = _make_scheduler(enable_overlap=True)
        dm = _run_init_overlap(s, use_mlx=False, supports_streams=False)

        self.assertFalse(s.use_overlap_streams)
        self.assertIsInstance(s.forward_stream_ctx, nullcontext)
        self.assertIsInstance(s.copy_stream_ctx, nullcontext)
        # No non-default streams are created on a single-queue backend.
        self.assertFalse(hasattr(s, "copy_stream"))
        self.assertFalse(hasattr(s, "schedule_stream"))
        dm.stream.assert_not_called()
        dm.Stream.assert_not_called()
        # Overlap bookkeeping is shared with the GPU path.
        self.assertIs(s.future_map, s.spec_algorithm.create_future_map.return_value)
        s.spec_algorithm.create_future_map.assert_called_once_with(
            8, 2048, 4096, "cpu"
        )
        self.assertEqual(s.batch_record_buf, [None, None])
        self.assertEqual(s.batch_record_ct, 0)

    def test_multi_stream_creates_streams(self):
        """supports_overlap_streams=True -> CUDA-style forward/copy streams from
        the device module (unchanged GPU behavior)."""
        s = _make_scheduler(enable_overlap=True)
        dm = _run_init_overlap(s, use_mlx=False, supports_streams=True)

        self.assertTrue(s.use_overlap_streams)
        dm.stream.assert_any_call(s.forward_stream)
        dm.Stream.assert_called_once_with()
        self.assertIs(s.copy_stream, dm.Stream.return_value)
        self.assertIs(s.forward_stream_ctx, dm.stream(s.forward_stream))
        self.assertIs(s.future_map, s.spec_algorithm.create_future_map.return_value)
        self.assertEqual(s.batch_record_buf, [None, None])

    def test_single_queue_overlap_disabled_has_no_future_map(self):
        """With overlap disabled, the single-queue path still avoids streams and
        leaves future_map None (mirrors the GPU disabled path)."""
        s = _make_scheduler(enable_overlap=False)
        dm = _run_init_overlap(s, use_mlx=False, supports_streams=False)

        self.assertFalse(s.use_overlap_streams)
        self.assertIsInstance(s.forward_stream_ctx, nullcontext)
        self.assertIsNone(s.future_map)
        self.assertFalse(hasattr(s, "copy_stream"))
        dm.Stream.assert_not_called()

    def test_mlx_is_untouched(self):
        """MLX keeps its own early-return path: no streams, no FutureMap, and an
        empty result_queue. use_mlx() wins regardless of the stream capability."""
        s = _make_scheduler(enable_overlap=True)
        dm = _run_init_overlap(s, use_mlx=True, supports_streams=True)

        self.assertFalse(s.use_overlap_streams)
        self.assertIsNone(s.future_map)
        self.assertEqual(len(s.result_queue), 0)
        self.assertFalse(hasattr(s, "forward_stream_ctx"))
        dm.stream.assert_not_called()
        dm.Stream.assert_not_called()


if __name__ == "__main__":
    unittest.main()
