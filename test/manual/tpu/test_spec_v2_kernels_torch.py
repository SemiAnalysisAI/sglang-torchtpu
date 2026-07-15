"""CPU differential tests for the spec-v2 TPU torch-fallback kernels.

torch_tpu has no Triton, so ``spec_utils_v2`` provides pure-torch fallbacks
for its helper kernels. These tests force the TPU path (``_is_tpu = True``) on a CPU
tensor and check each fallback against an independent Python reference of the
original Triton kernel's semantics, across randomized shapes. No TPU needed.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import sglang.srt.speculative.spec_utils_v2 as ev2
from sglang.test.test_utils import CustomTestCase


class _ForceTpu:
    """Context manager: make the kernel fallbacks take the TPU branch."""

    def __enter__(self):
        self._prev = ev2._is_tpu
        ev2._is_tpu = True
        return self

    def __exit__(self, *a):
        ev2._is_tpu = self._prev


# --------------------------------------------------------------------------
# Independent Python references for the Triton kernels.
# --------------------------------------------------------------------------


def _ref_draft_cache_locs_p1(req_pool_indices, req_to_token, seq_lens, topk, steps):
    bs = seq_lens.shape[0]
    L = topk * steps
    out = torch.empty(bs * L, dtype=req_to_token.dtype)
    for i in range(bs):
        ks = int(seq_lens[i])
        r = int(req_pool_indices[i])
        out[i * L : (i + 1) * L] = req_to_token[r, ks : ks + L]
    return out


def _ref_fill_bonus(accept_tokens, accept_lens, num_draft):
    bs = accept_lens.shape[0]
    flat = accept_tokens.reshape(-1)  # kernel reads accept_tokens flat
    out = torch.empty(bs, dtype=torch.int32)
    for i in range(bs):
        idx = num_draft * i + int(accept_lens[i]) - 1
        out[i] = flat[idx]
    return out


def _ref_fill_accepted(accept_index, out_cache_loc):
    ai = accept_index.reshape(-1)  # kernel reads accept_index flat
    ocl = out_cache_loc.reshape(-1)
    size = ai.shape[0]
    out = torch.zeros(size, dtype=torch.int64)
    k = 0
    for i in range(size):
        src = int(ai[i])
        if src > -1:
            out[k] = ocl[src]
            k += 1
    return out


def _ref_extend_cache_locs(req_pool_indices, req_to_token, start, end, total):
    out = torch.zeros(total, dtype=torch.int64)
    off = 0
    for i in range(req_pool_indices.shape[0]):
        s, e, r = int(start[i]), int(end[i]), int(req_pool_indices[i])
        n = e - s
        out[off : off + n] = req_to_token[r, s:e]
        off += n
    return out


class TestSpecV2KernelTorchFallbacks(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_assign_draft_cache_locs_page_size_1(self):
        for bs, topk, steps in [(1, 1, 3), (4, 1, 4), (8, 1, 3), (3, 1, 1)]:
            pool_w = 256
            req_to_token = torch.randint(0, 1_000_000, (16, pool_w), dtype=torch.int64)
            req_pool_indices = torch.randperm(16)[:bs]
            seq_lens = torch.randint(0, pool_w - topk * steps, (bs,))
            out = torch.zeros(bs * topk * steps, dtype=torch.int64)
            with _ForceTpu():
                ev2.assign_draft_cache_locs_page_size_1_func(
                    req_pool_indices, req_to_token, seq_lens, out, topk, steps
                )
            ref = _ref_draft_cache_locs_p1(
                req_pool_indices, req_to_token, seq_lens, topk, steps
            )
            self.assertTrue(torch.equal(out, ref), f"bs={bs} topk={topk} steps={steps}")

    def test_draft_cache_router_prefers_host_shadows(self):
        req_to_token = torch.arange(4 * 32, dtype=torch.int64).reshape(4, 32)
        device_rows = torch.tensor([0, 0], dtype=torch.int64)
        device_lens = torch.tensor([0, 0], dtype=torch.int64)
        host_rows = torch.tensor([2, 3], dtype=torch.int64)
        host_lens = torch.tensor([5, 7], dtype=torch.int64)
        out = torch.zeros(4, dtype=torch.int64)

        with _ForceTpu():
            ev2.assign_draft_cache_locs_page_size_1_func(
                device_rows,
                req_to_token,
                device_lens,
                out,
                topk=1,
                speculative_num_steps=2,
                host_req_pool_indices=lambda: host_rows,
                host_seq_lens=host_lens,
            )

        ref = _ref_draft_cache_locs_p1(host_rows, req_to_token, host_lens, 1, 2)
        self.assertTrue(torch.equal(out, ref))

    def test_decode_cache_router_prefers_host_shadows(self):
        req_to_token = torch.zeros((4, 32), dtype=torch.int32)
        host_out = torch.tensor([101, 102, 103, 104], dtype=torch.int32)
        device_out = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
        allocator = SimpleNamespace(
            page_size=4,
            last_alloc_extend_cpu=host_out,
        )
        tree_cache = SimpleNamespace(token_to_kv_pool_allocator=allocator)
        req_to_token_pool = SimpleNamespace(
            req_to_token=req_to_token,
            host_resident=True,
        )
        device_rows = torch.tensor([0, 0], dtype=torch.int64)
        host_rows = torch.tensor([2, 3], dtype=torch.int64)
        host_rows_factory = Mock(return_value=host_rows)
        cur = torch.tensor([5, 7], dtype=torch.int32)
        nxt = torch.tensor([7, 9], dtype=torch.int32)
        last_loc = torch.tensor([10, 20], dtype=torch.int32)

        with (
            patch.object(ev2, "get_last_loc", return_value=last_loc) as get_last,
            patch.object(
                ev2,
                "alloc_paged_token_slots_extend",
                return_value=device_out,
            ),
            patch.object(ev2, "assign_req_to_token_pool_func") as assign,
        ):
            out = ev2.allocate_and_assign_decode_cache(
                tree_cache=tree_cache,
                req_to_token_pool=req_to_token_pool,
                req_pool_indices=device_rows,
                cur_kv_lens_cpu=cur,
                nxt_kv_lens_cpu=nxt,
                num_needed_tokens=4,
                batch_size=2,
                device=torch.device("cpu"),
                host_req_pool_indices=host_rows_factory,
            )

        self.assertIs(out, device_out)
        host_rows_factory.assert_called_once_with()
        self.assertIs(get_last.call_args.args[1], host_rows)
        assign_args = assign.call_args.args
        self.assertIs(assign_args[0], host_rows)
        self.assertIs(assign_args[2], cur)
        self.assertIs(assign_args[3], nxt)
        self.assertIs(assign_args[4], host_out)

    def test_decode_cache_router_moves_lengths_for_device_pool(self):
        req_to_token = torch.empty((4, 32), dtype=torch.int32, device="meta")
        req_to_token_pool = SimpleNamespace(
            req_to_token=req_to_token,
            host_resident=False,
        )
        allocator = SimpleNamespace(page_size=4)
        tree_cache = SimpleNamespace(token_to_kv_pool_allocator=allocator)
        req_pool_indices = torch.empty(2, dtype=torch.int64, device="meta")
        cur = torch.tensor([5, 7], dtype=torch.int32)
        nxt = torch.tensor([7, 9], dtype=torch.int32)
        last_loc = torch.empty(2, dtype=torch.int32, device="meta")
        device_out = torch.empty(4, dtype=torch.int32, device="meta")
        host_rows_factory = Mock()

        with (
            patch.object(ev2, "get_last_loc", return_value=last_loc) as get_last,
            patch.object(
                ev2,
                "alloc_paged_token_slots_extend",
                return_value=device_out,
            ),
            patch.object(ev2, "assign_req_to_token_pool_func") as assign,
        ):
            ev2.allocate_and_assign_decode_cache(
                tree_cache=tree_cache,
                req_to_token_pool=req_to_token_pool,
                req_pool_indices=req_pool_indices,
                cur_kv_lens_cpu=cur,
                nxt_kv_lens_cpu=nxt,
                num_needed_tokens=4,
                batch_size=2,
                device=torch.device("meta"),
                host_req_pool_indices=host_rows_factory,
            )

        host_rows_factory.assert_not_called()
        self.assertEqual(get_last.call_args.args[2].device.type, "meta")
        assign_args = assign.call_args.args
        self.assertIs(assign_args[0], req_pool_indices)
        self.assertEqual(assign_args[2].device.type, "meta")
        self.assertEqual(assign_args[3].device.type, "meta")
        self.assertIs(assign_args[4], device_out)

    def test_fill_bonus_tokens(self):
        for bs, num_draft in [(1, 5), (4, 5), (8, 4), (7, 8), (2, 4)]:
            # Real caller passes a [bs, num_draft] tensor (predict[accept_index]),
            # which the kernel reads flat. Use the 2D shape here.
            accept_tokens = torch.randint(0, 32000, (bs, num_draft), dtype=torch.int32)
            # accept_lens in [1, num_draft] (includes the bonus token).
            accept_lens = torch.randint(1, num_draft + 1, (bs,), dtype=torch.int32)
            bonus = torch.empty(bs, dtype=torch.int32)
            with _ForceTpu():
                ev2.fill_bonus_tokens_func(accept_tokens, accept_lens, bonus, num_draft)
            ref = _ref_fill_bonus(accept_tokens, accept_lens, num_draft)
            self.assertTrue(torch.equal(bonus, ref), f"bs={bs} num_draft={num_draft}")

    def test_fill_accepted_out_cache_loc(self):
        # (bs, num_draft) -> the real caller passes a 2D accept_index the kernel
        # reads flat; size == bs*num_draft.
        for bs, num_draft in ((1, 4), (2, 4), (4, 4), (8, 4)):
            size = bs * num_draft
            out_cache_loc = torch.randint(0, 100000, (size,), dtype=torch.int64)
            flat_ai = torch.full((size,), -1, dtype=torch.int64)
            perm = torch.randperm(size)
            naccept = int(torch.randint(0, size + 1, (1,)))
            flat_ai[perm[:naccept]] = perm[:naccept]  # accepted -> a valid src
            accept_index = flat_ai.reshape(bs, num_draft)  # 2D like the caller
            accepted = torch.zeros(size, dtype=torch.int64)
            with _ForceTpu():
                ev2.fill_accepted_out_cache_loc_func(
                    accept_index, out_cache_loc, accepted
                )
            ref = _ref_fill_accepted(accept_index, out_cache_loc)
            self.assertTrue(
                torch.equal(accepted, ref), f"bs={bs} num_draft={num_draft}"
            )

    def test_assign_extend_cache_locs(self):
        for bs in (1, 4, 8):
            pool_w = 256
            req_to_token = torch.randint(0, 1_000_000, (16, pool_w), dtype=torch.int64)
            req_pool_indices = torch.randperm(16)[:bs]
            start = torch.randint(0, 100, (bs,))
            lengths = torch.randint(1, 6, (bs,))  # variable per-request length
            end = start + lengths
            total = int(lengths.sum())
            out = torch.zeros(total, dtype=torch.int64)
            with _ForceTpu():
                ev2.assign_extend_cache_locs_kernel_func(
                    req_pool_indices, req_to_token, start, end, out
                )
            ref = _ref_extend_cache_locs(
                req_pool_indices, req_to_token, start, end, total
            )
            self.assertTrue(torch.equal(out, ref), f"bs={bs}")

    def test_extend_cache_router_prefers_host_shadows(self):
        req_to_token = torch.arange(4 * 32, dtype=torch.int64).reshape(4, 32)
        device_rows = torch.tensor([0, 0], dtype=torch.int64)
        device_lens = torch.tensor([0, 0], dtype=torch.int64)
        host_rows = torch.tensor([2, 3], dtype=torch.int64)
        host_lens = torch.tensor([5, 7], dtype=torch.int64)

        with _ForceTpu():
            out = ev2.assign_extend_cache_locs_func(
                device_rows,
                req_to_token,
                device_lens,
                batch_size=2,
                draft_token_num=2,
                device=torch.device("cpu"),
                host_req_pool_indices=lambda: host_rows,
                host_seq_lens=host_lens,
            )

        ref = _ref_extend_cache_locs(
            host_rows, req_to_token, host_lens, host_lens + 2, total=4
        )
        self.assertTrue(torch.equal(out, ref))

    def test_empty_batch_is_noop(self):
        with _ForceTpu():
            ev2.assign_draft_cache_locs_page_size_1_func(
                torch.empty(0, dtype=torch.int64),
                torch.zeros(4, 16, dtype=torch.int64),
                torch.empty(0, dtype=torch.int64),
                torch.empty(0, dtype=torch.int64),
                1,
                3,
            )  # must not raise


if __name__ == "__main__":
    unittest.main()
