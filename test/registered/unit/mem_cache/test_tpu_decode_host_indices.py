"""TPU decode allocation must not read request indices back from the device."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.mem_cache import allocation
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestTpuDecodeHostIndices(unittest.TestCase):
    def test_host_pool_uses_reordered_cpu_indices_without_device_copy(self):
        table = torch.arange(32, dtype=torch.int32).reshape(4, 8)
        device_indices = Mock()
        device_indices.cpu.side_effect = AssertionError(
            "unexpected device synchronization"
        )
        pool = SimpleNamespace(
            req_to_token=table, write=Mock(), alloc_aux_to_lengths=Mock()
        )
        batch = SimpleNamespace(
            maybe_evict_swa=Mock(),
            seq_lens_cpu=torch.tensor([3, 5]),
            req_pool_indices=device_indices,
            req_pool_indices_cpu=torch.tensor([2, 0]),
            req_to_token_pool=pool,
            tree_cache=Mock(),
            model_config=SimpleNamespace(is_encoder_decoder=False),
            reqs=[
                SimpleNamespace(
                    kv=SimpleNamespace(kv_allocated_len=n, kv_committed_len=n)
                )
                for n in (3, 5)
            ],
        )
        slots = torch.tensor([19, 5], dtype=torch.int32)
        with (
            patch.object(allocation, "_is_tpu", True),
            patch.object(allocation, "_is_npu", False),
            patch.object(allocation, "_alloc_page_size", return_value=128),
            patch.object(
                allocation, "alloc_paged_token_slots_decode", return_value=slots
            ) as allocate,
        ):
            result = allocation.alloc_for_decode(batch, token_per_req=1)
        self.assertIs(result, slots)
        torch.testing.assert_close(
            allocate.call_args.kwargs["last_loc"],
            torch.tensor([18, 4], dtype=torch.int32),
        )
        torch.testing.assert_close(
            allocate.call_args.kwargs["seq_lens_cpu"], torch.tensor([4, 6])
        )
        self.assertEqual([r.kv.kv_committed_len for r in batch.reqs], [4, 6])
        device_indices.cpu.assert_not_called()
        torch.testing.assert_close(
            pool.write.call_args.args[0][0], batch.req_pool_indices_cpu
        )


if __name__ == "__main__":
    unittest.main()
