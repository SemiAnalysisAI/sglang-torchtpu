"""The core batch-rows seam: plain defaults, backend registration, scheduler warmup."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers import batch_rows  # noqa: E402
from sglang.srt.managers import scheduler as scheduler_module  # noqa: E402

register_cpu_ci(est_time=25, suite="base-a-test-cpu")


class _RecordingBackend(batch_rows.BatchRowsBackend):
    def __init__(self):
        self.calls = []

    def merge_rows(self, a, b):
        self.calls.append("merge")
        return super().merge_rows(a, b)

    def filter_rows(self, t, keep_device):
        self.calls.append("filter")
        return super().filter_rows(t, keep_device)

    def warmup(self, max_running_requests, device):
        self.calls.append(("warmup", max_running_requests, device))


class TestBatchRowsBackend(CustomTestCase):
    def setUp(self):
        restore = patch.object(batch_rows, "_backend", batch_rows.BatchRowsBackend())
        restore.start()
        self.addCleanup(restore.stop)

    def test_default_backend_runs_the_plain_ops(self):
        a, b, keep = torch.arange(3), torch.arange(4), torch.tensor([2, 0])
        self.assertTrue(torch.equal(batch_rows.merge_rows(a, b), torch.cat([a, b])))
        self.assertTrue(torch.equal(batch_rows.filter_rows(a, keep), a[keep]))
        batch_rows.warmup_batch_rows(8, torch.device("cpu"))

    def test_registered_backend_serves_merge_filter_and_scheduler_warmup(self):
        with self.assertRaises(TypeError):
            batch_rows.register_batch_rows_backend(object())
        backend = _RecordingBackend()
        batch_rows.register_batch_rows_backend(backend)
        a = torch.arange(3)
        batch_rows.merge_rows(a, a)
        batch_rows.filter_rows(a, torch.tensor([1]))

        # The scheduler warms whichever backend is registered, on every platform.
        scheduler = SimpleNamespace(
            server_args=SimpleNamespace(
                unsafe_save_torch_compile_bundle=None,
                unsafe_load_torch_compile_bundle=None,
            ),
            tp_worker=SimpleNamespace(
                init_cuda_graphs=lambda: None,
                model_runner=SimpleNamespace(max_running_requests=48, device="cpu"),
            ),
            draft_worker=None,
        )
        with patch.object(
            scheduler_module, "current_platform", SimpleNamespace(is_tpu=lambda: False)
        ):
            scheduler_module.Scheduler.init_all_cuda_graphs(scheduler)
        self.assertEqual(
            backend.calls, ["merge", "filter", ("warmup", 48, torch.device("cpu"))]
        )


if __name__ == "__main__":
    unittest.main()
