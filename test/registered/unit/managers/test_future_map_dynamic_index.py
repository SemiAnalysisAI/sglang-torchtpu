"""Unit tests for the FutureMap dynamic-index store path (overlap_utils).

On backends that bake a buffer WRITE offset as a compile-time constant
(torch_tpu / XLA), the overlap FutureMap routes future-buffer writes through
``_store`` with a tensor-indexed ``index_copy_`` instead of a ``buf[indices] =
values`` setitem, so the compiled store graph shape stays static across overlap
steps (no per-step XLA recompile). These tests pin two things on CPU (no TPU
needed):

* ``_device_use_dynamic_index`` selects that path only for the listed backends.
* The dynamic (``index_copy_``) store is **bit-identical** to the plain setitem
  store — same buffer contents — for the unique-row writes the relay issues.
  That equivalence is the whole correctness claim; it is device-independent, so
  we force the flag on a CPU FutureMap and diff the two paths.

Upstream reworked FutureMap from a circular future-index buffer into a
pool-indexed relay (writes keyed by req_pool_idx via publish/stash); the
dynamic-index ``_store`` seam (fork PR #78) is preserved and is what we test.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.hardware_backend.tpu.overlap_utils_tpu import (
    store_future_map_tpu,
)
from sglang.srt.managers.overlap_utils import (
    _DYNAMIC_INDEX_DEVICE_TYPES,
    FutureMap,
    _device_use_dynamic_index,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# A non-speculative SpeculativeAlgorithm stub. FutureMap.__init__ only stores it
# (spec methods are hit lazily on the first stash, which these tests do not do).
_SPEC_NONE = SimpleNamespace(is_none=lambda: True)


def _make_future_map(*, req_pool_size=16, device="cpu") -> FutureMap:
    # FutureMap sizes its per-req_pool_idx buffers from
    # req_to_token_pool.req_to_token.shape[0].
    req_to_token_pool = SimpleNamespace(
        req_to_token=torch.zeros((req_pool_size, 1), dtype=torch.int64)
    )
    return FutureMap(
        device=device,
        spec_algo=_SPEC_NONE,
        req_to_token_pool=req_to_token_pool,
        needs_cpu_seq_lens=True,
    )


class TestDeviceUseDynamicIndex(CustomTestCase):
    def test_cpu_and_cuda_are_static(self):
        self.assertFalse(_device_use_dynamic_index(torch.device("cpu")))
        # torch.device("cuda") parses without a GPU present.
        self.assertFalse(_device_use_dynamic_index(torch.device("cuda")))
        self.assertFalse(_device_use_dynamic_index("cpu"))

    def test_tpu_is_dynamic_via_type_attr(self):
        # Object with a ``.type`` attribute (the torch.device shape) -> hasattr branch.
        self.assertTrue(_device_use_dynamic_index(SimpleNamespace(type="tpu")))

    def test_tpu_is_dynamic_via_string(self):
        # Plain strings -> the ``str(device).split(":")`` fallback branch.
        self.assertTrue(_device_use_dynamic_index("tpu"))
        self.assertTrue(_device_use_dynamic_index("tpu:0"))

    def test_device_list_is_the_single_source_of_truth(self):
        # Every listed type maps to dynamic; an unlisted one does not.
        for dev_type in _DYNAMIC_INDEX_DEVICE_TYPES:
            self.assertTrue(_device_use_dynamic_index(dev_type))
        self.assertNotIn("cpu", _DYNAMIC_INDEX_DEVICE_TYPES)


class TestFutureMapInit(CustomTestCase):
    def test_cpu_uses_static_path(self):
        fm = _make_future_map(req_pool_size=8)
        # CPU is not a dynamic-index backend, so writes use the plain setitem.
        self.assertFalse(fm._use_dynamic_index)
        # Buffers are sized by the req->token pool row count.
        self.assertEqual(fm.output_tokens_buf.shape[0], 8)
        self.assertEqual(fm.new_seq_lens_buf.shape[0], 8)

    def test_tpu_reserves_scratch_row_without_growing_request_pool(self):
        with patch("sglang.srt.managers.overlap_utils._is_tpu", True):
            fm = _make_future_map(req_pool_size=8)
        fm.spec_algo = SimpleNamespace(is_some=lambda: True, need_topk=lambda: True)
        payload = SimpleNamespace(
            topk_p=torch.ones((2, 1)),
            topk_index=torch.zeros((2, 1), dtype=torch.int64),
            hidden_states=torch.zeros((2, 3)),
            draft_probs=None,
        )
        with patch(
            "sglang.srt.speculative.spec_utils.spec_need_hidden_states",
            return_value=True,
        ):
            fm._maybe_init_forward_bufs(payload)
        self.assertEqual(fm.req_pool_size, 8)
        self.assertEqual(fm.new_seq_lens_buf.shape, (8,))
        self.assertEqual(fm.output_tokens_buf.shape, (9,))
        self.assertEqual(fm.topk_p_buf.shape, (9, 1))
        self.assertEqual(fm.topk_index_buf.shape, (9, 1))
        self.assertEqual(fm.hidden_states_buf.shape, (9, 3))


class TestDynamicIndexEquivalence(CustomTestCase):
    """``_store``'s dynamic (index_copy_) path must equal the static setitem path."""

    def test_store_dynamic_matches_static_across_writes(self):
        static_fm = _make_future_map(
            req_pool_size=16
        )  # _use_dynamic_index=False on CPU
        dyn_fm = _make_future_map(req_pool_size=16)
        dyn_fm._use_dynamic_index = True
        self.assertFalse(static_fm._use_dynamic_index)

        # Unique-row writes (the only kind the relay issues): a mix of contiguous
        # intervals, a singleton, and a set that revisits row 0 in a later step.
        index_sets = [
            [0, 1, 2, 3],
            [4, 5, 6, 7, 8],
            [9, 10, 11],
            [12],
            [13, 14, 15, 0],
        ]
        sbuf = torch.zeros(16, dtype=torch.int64)
        dbuf = torch.zeros(16, dtype=torch.int64)
        for step, idxs in enumerate(index_sets):
            indices = torch.tensor(idxs, dtype=torch.int64)
            values = torch.arange(len(idxs), dtype=torch.int64) + 1000 * (step + 1)
            static_fm._store(sbuf, indices, values)
            dyn_fm._store(dbuf, indices, values)
            self.assertTrue(
                torch.equal(sbuf, dbuf),
                msg=f"step {step} idxs {idxs}: buffers diverged\n{sbuf}\n{dbuf}",
            )

    def test_store_casts_values_to_buffer_dtype(self):
        # _store casts values to the buffer dtype before writing (main #134: the
        # sampler returns int64 ids that must land in the buffer's dtype).
        fm = _make_future_map(req_pool_size=8)
        fm._use_dynamic_index = True
        buf = torch.zeros(8, dtype=torch.int64)
        fm._store(buf, torch.tensor([1, 2]), torch.tensor([5, 6], dtype=torch.int32))
        self.assertEqual(buf[1].item(), 5)
        self.assertEqual(buf[2].item(), 6)


class TestFusedFutureMapStore(CustomTestCase):
    def test_store_updates_all_relay_buffers(self):
        indices = torch.tensor([1, 3], dtype=torch.int64)
        output, topk_p, topk_index, hidden = store_future_map_tpu(
            indices=indices,
            future_map=SimpleNamespace(
                output_tokens_buf=torch.zeros(5, dtype=torch.int64),
                topk_p_buf=torch.zeros((5, 1), dtype=torch.float32),
                topk_index_buf=torch.zeros((5, 1), dtype=torch.int64),
                need_hidden_states=False,
                req_pool_size=4,
                _tpu_precompiled_batch_sizes=(1, 2, 4, 8),
                hidden_states_buf=None,
            ),
            payload=SimpleNamespace(
                bonus_tokens=torch.tensor([7, 9], dtype=torch.int32),
                topk_p=torch.tensor([[0.25], [0.75]], dtype=torch.float32),
                topk_index=torch.tensor([[4], [6]], dtype=torch.int32),
            ),
        )
        self.assertIsNone(hidden)
        self.assertEqual(output.tolist(), [0, 7, 0, 9, 0])
        self.assertTrue(
            torch.equal(topk_p[:, 0], torch.tensor([0.0, 0.25, 0.0, 0.75, 0.0]))
        )
        self.assertEqual(topk_index[:, 0].tolist(), [0, 4, 0, 6, 0])

    def test_hidden_store_updates_hidden_buffer(self):
        indices = torch.tensor([0, 2], dtype=torch.int64)
        output, topk_p, topk_index, hidden = store_future_map_tpu(
            indices=indices,
            future_map=SimpleNamespace(
                output_tokens_buf=torch.zeros(4, dtype=torch.int64),
                topk_p_buf=torch.zeros((4, 1), dtype=torch.float32),
                topk_index_buf=torch.zeros((4, 1), dtype=torch.int64),
                hidden_states_buf=torch.zeros((4, 3), dtype=torch.bfloat16),
                need_hidden_states=True,
                req_pool_size=3,
                _tpu_precompiled_batch_sizes=(1, 2, 4, 8),
            ),
            payload=SimpleNamespace(
                bonus_tokens=torch.tensor([5, 8], dtype=torch.int64),
                topk_p=torch.tensor([[0.4], [0.6]], dtype=torch.float32),
                topk_index=torch.tensor([[2], [3]], dtype=torch.int64),
                hidden_states=torch.tensor(
                    [[1, 2, 3], [4, 5, 6]], dtype=torch.bfloat16
                ),
            ),
        )
        self.assertEqual(output.tolist(), [5, 0, 8, 0])
        self.assertTrue(torch.equal(topk_p[:, 0], torch.tensor([0.4, 0.0, 0.6, 0.0])))
        self.assertEqual(topk_index[:, 0].tolist(), [2, 0, 3, 0])
        self.assertTrue(
            torch.equal(
                hidden,
                torch.tensor(
                    [[1, 2, 3], [0, 0, 0], [4, 5, 6], [0, 0, 0]],
                    dtype=torch.bfloat16,
                ),
            )
        )


if __name__ == "__main__":
    unittest.main()
