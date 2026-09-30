"""Unit tests for the TPU prefill context-parallel layout, K/V row writes and
forward glue (CPU tensors)."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from sglang.srt.hardware_backend.tpu.attention_pallas_rpa_v3 import (
    TpuPallasRpaV3AttnBackend,
)
from sglang.srt.hardware_backend.tpu.compilation.bucketer import Bucketer
from sglang.srt.hardware_backend.tpu.cp_forward import (
    TpuCpBatchLayout,
    all_gather_and_write_kv,
    cp_last_token_rows,
    cp_local_embeds,
    cp_token_inputs,
    fused_kv_rows,
    get_tpu_cp_layout,
    install_cp_logits_forward,
    prepare_tpu_cp_forward_batch,
    wrap_cp_embedding_forward,
)
from sglang.srt.hardware_backend.tpu.cp_layout import (
    build_tpu_cp_layout,
    local_row_count,
    owned_blocks,
    warmup_request_lengths,
    zigzag_block_sizes,
)
from sglang.srt.hardware_backend.tpu.kv_row_write import (
    kv_write_segments,
    write_rows_torch,
    write_run_bound,
)
from sglang.srt.layers import dp_attention
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_CP_FORWARD = "sglang.srt.hardware_backend.tpu.cp_forward"


def _expand_write_segments(segments, num_segments):
    """``{gathered row: KV slot}`` covered by the write runs."""
    rows = {}
    for src, dst, n in segments[:, : int(num_segments)].T.tolist():
        for i in range(n):
            rows[src + i] = dst + i
    return rows


class _FakeGatherGroup:
    """Context-parallel group of ``world_size`` ranks whose all-gather places
    this rank's rows first and ``other_rows`` (rank-major) after them; with
    ``per_call`` the other ranks' tensors are consumed one list per call."""

    def __init__(self, world_size, other_rows, per_call=False):
        self.world_size = world_size
        self.other_rows = list(other_rows)
        self.per_call = per_call

    def all_gather_into_tensor(self, output, input):
        others = self.other_rows.pop(0) if self.per_call else self.other_rows
        output.copy_(torch.cat([input] + others, dim=0))


class _FakeReduceGroup:
    """Sum all-reduce where the other ranks contribute ``other``."""

    def __init__(self, other):
        self.other = other

    def all_reduce(self, tensor):
        return tensor + self.other


def _layout(cp_size=4, cp_rank=1, extend=(40, 24), prefix=(0, 100), pad_b=4, bucket=32):
    host = build_tpu_cp_layout(
        extend_lens=list(extend),
        prefix_lens=list(prefix),
        cp_size=cp_size,
        cp_rank=cp_rank,
        pad_b=pad_b,
        local_rows_padded=bucket,
    )
    rows = torch.from_numpy(np.clip(host.local_rows, 0, None))
    return TpuCpBatchLayout(host=host, local_rows=rows)


class TestZigzagBlocks(unittest.TestCase):
    def test_block_sizes_sum_to_length_and_front_load_the_remainder(self):
        self.assertEqual(zigzag_block_sizes(35, 4), [5, 5, 5, 4, 4, 4, 4, 4])
        self.assertEqual(zigzag_block_sizes(64, 8), [4] * 16)
        for length in (16, 17, 100, 4095):
            self.assertEqual(sum(zigzag_block_sizes(length, 8)), length)

    def test_local_rows_partition_the_tokens_across_ranks(self):
        extend_lens = [40, 24, 17]
        total = sum(local_row_count(extend_lens, 4, rank) for rank in range(4))
        self.assertEqual(total, sum(extend_lens))

    def test_local_rows_are_equal_when_lengths_divide_evenly(self):
        # 64 -> 4 tokens per block, 128 -> 8: every rank holds 2 blocks of each.
        extend_lens = [64, 128]
        counts = {local_row_count(extend_lens, 8, rank) for rank in range(8)}
        self.assertEqual(counts, {24})


class TestBuildLayout(unittest.TestCase):
    CP = 4
    EXTEND = [40, 24]
    PREFIX = [0, 100]
    PAD_B = 4
    BUCKET = 32

    def _layout(self, rank):
        return build_tpu_cp_layout(
            extend_lens=self.EXTEND,
            prefix_lens=self.PREFIX,
            cp_size=self.CP,
            cp_rank=rank,
            pad_b=self.PAD_B,
            local_rows_padded=self.BUCKET,
        )

    def test_every_token_is_owned_by_exactly_one_rank(self):
        seen = np.zeros(sum(self.EXTEND), dtype=np.int64)
        for rank in range(self.CP):
            layout = self._layout(rank)
            real = layout.local_rows[: layout.local_len]
            self.assertTrue((layout.local_rows[layout.local_len :] == -1).all())
            np.add.at(seen, real, 1)
        self.assertTrue((seen == 1).all())

    def test_local_order_is_request_major_with_adjacent_blocks(self):
        layout = self._layout(1)
        blocks0 = zigzag_block_sizes(self.EXTEND[0], self.CP)
        starts0 = np.concatenate([[0], np.cumsum(blocks0)])
        early, late = owned_blocks(self.CP, 1)
        expected = np.concatenate(
            [
                np.arange(starts0[early], starts0[early + 1]),
                np.arange(starts0[late], starts0[late + 1]),
            ]
        )
        self.assertEqual(
            layout.local_rows[: expected.shape[0]].tolist(), expected.tolist()
        )

    def test_gathered_rows_are_identical_on_every_rank(self):
        reference = self._layout(0)
        for rank in range(1, self.CP):
            layout = self._layout(rank)
            self.assertTrue((layout.gathered_req == reference.gathered_req).all())
            self.assertTrue((layout.gathered_pos == reference.gathered_pos).all())
        self.assertEqual(reference.gathered_req.shape[0], self.CP * self.BUCKET)
        # Every (request, offset) pair appears once; padding rows are -1 twice.
        valid = reference.gathered_req >= 0
        pairs = set(
            zip(
                reference.gathered_req[valid].tolist(),
                reference.gathered_pos[valid].tolist(),
            )
        )
        self.assertEqual(len(pairs), sum(self.EXTEND))
        self.assertTrue((reference.gathered_pos[~valid] == -1).all())

    def test_pseudo_rows_follow_the_owned_blocks(self):
        for rank in range(self.CP):
            layout = self._layout(rank)
            early, late = owned_blocks(self.CP, rank)
            for i, length in enumerate(self.EXTEND):
                blocks = zigzag_block_sizes(length, self.CP)
                prefix = self.PREFIX[i]
                self.assertEqual(
                    layout.pseudo_kv_lens[2 * i], prefix + sum(blocks[: early + 1])
                )
                self.assertEqual(layout.pseudo_q_lens[2 * i], blocks[early])
                self.assertEqual(
                    layout.pseudo_kv_lens[2 * i + 1], prefix + sum(blocks[: late + 1])
                )
                self.assertEqual(layout.pseudo_q_lens[2 * i + 1], blocks[late])
            # Padded requests are zero rows at the tail.
            self.assertTrue((layout.pseudo_kv_lens[2 * len(self.EXTEND) :] == 0).all())
            self.assertTrue((layout.pseudo_q_lens[2 * len(self.EXTEND) :] == 0).all())
            self.assertEqual(
                int(layout.pseudo_q_lens.sum()),
                local_row_count(self.EXTEND, self.CP, rank),
            )

    def test_rank_zero_owns_every_last_token(self):
        layout = self._layout(0)
        self.assertEqual(layout.last_owner_mask.tolist(), [1, 1, 0, 0])
        for i, length in enumerate(self.EXTEND):
            row = layout.last_rows[i]
            token = layout.local_rows[row]
            self.assertEqual(token, sum(self.EXTEND[: i + 1]) - 1)
        for rank in range(1, self.CP):
            other = self._layout(rank)
            self.assertEqual(other.last_owner_mask.tolist(), [0] * self.PAD_B)
            self.assertEqual(other.last_rows.tolist(), [0] * self.PAD_B)

    def test_rejects_short_requests_and_overfull_buckets(self):
        with self.assertRaisesRegex(ValueError, "below the zigzag minimum"):
            build_tpu_cp_layout(
                extend_lens=[7],
                prefix_lens=[0],
                cp_size=4,
                cp_rank=0,
                pad_b=1,
                local_rows_padded=8,
            )
        with self.assertRaisesRegex(ValueError, "exceed the request bucket"):
            build_tpu_cp_layout(
                extend_lens=[16, 16],
                prefix_lens=[0, 0],
                cp_size=4,
                cp_rank=0,
                pad_b=1,
                local_rows_padded=8,
            )
        with self.assertRaisesRegex(ValueError, "exceed the local token bucket"):
            build_tpu_cp_layout(
                extend_lens=[64],
                prefix_lens=[0],
                cp_size=4,
                cp_rank=0,
                pad_b=1,
                local_rows_padded=8,
            )


class TestWarmupRequestLengths(unittest.TestCase):
    def test_lengths_fill_the_local_bucket_on_every_rank(self):
        for local_rows, cp_size, max_reqs in ((64, 4, 8), (2048, 8, 64), (16, 8, 1)):
            lengths = warmup_request_lengths(
                local_rows=local_rows, cp_size=cp_size, max_reqs=max_reqs
            )
            self.assertLessEqual(len(lengths), max_reqs)
            for rank in range(cp_size):
                self.assertEqual(local_row_count(lengths, cp_size, rank), local_rows)
            for length in lengths:
                self.assertEqual(length % (2 * cp_size), 0)

    def test_rejects_odd_or_empty_buckets(self):
        with self.assertRaises(ValueError):
            warmup_request_lengths(local_rows=3, cp_size=4, max_reqs=1)
        with self.assertRaises(ValueError):
            warmup_request_lengths(local_rows=8, cp_size=4, max_reqs=0)


class TestKvWriteSegments(unittest.TestCase):
    def test_runs_break_at_slot_gaps_and_padding(self):
        # Slots 5-8 are one run; 9 skipped; 10-11; padding rows; 30 alone.
        slots = np.array([5, 6, 7, 8, 10, 11, 0, 0, 30])
        valid = np.array([1, 1, 1, 1, 1, 1, 0, 0, 1], dtype=bool)
        segments, count = kv_write_segments(slots, valid, max_segments=8)
        self.assertEqual(count, 3)
        self.assertEqual(segments.shape, (3, 8))
        self.assertEqual(
            segments[:, :3].T.tolist(), [[0, 5, 4], [4, 10, 2], [8, 30, 1]]
        )
        self.assertEqual(int(segments[:, 3:].abs().sum()), 0)

    def test_padding_between_consecutive_slots_splits_the_run(self):
        slots = np.array([5, 6, 7, 8])
        valid = np.array([1, 1, 0, 1], dtype=bool)
        segments, count = kv_write_segments(slots, valid)
        self.assertEqual(count, 2)
        self.assertEqual(segments[:, :2].T.tolist(), [[0, 5, 2], [3, 8, 1]])

    def test_long_runs_split_into_staging_pieces(self):
        slots = np.arange(100, 100 + 3000)
        segments, count = kv_write_segments(slots, np.ones(3000, dtype=bool))
        self.assertEqual(count, 3)
        self.assertEqual(
            segments[:, :3].T.tolist(),
            [[0, 100, 1024], [1024, 1124, 1024], [2048, 2148, 952]],
        )

    def test_no_valid_rows_gives_no_runs(self):
        segments, count = kv_write_segments(
            np.zeros(4, dtype=np.int64), np.zeros(4, dtype=bool)
        )
        self.assertEqual(count, 0)
        self.assertEqual(int(segments.abs().sum()), 0)

    def test_too_many_runs_is_rejected(self):
        slots = np.arange(0, 20, 2)
        with self.assertRaises(ValueError):
            kv_write_segments(slots, np.ones(10, dtype=bool), max_segments=4)


class TestWriteRowsFallback(unittest.TestCase):
    def test_cpu_write_matches_scatter(self):
        torch.manual_seed(0)
        kv = torch.zeros(6, 4, 2, 2, 8, dtype=torch.bfloat16)
        rows = torch.randn(10, 2, 2, 8).to(torch.bfloat16)
        slots = np.array([9, 10, 11, 20, 21, 0, 0, 3, 4, 5])
        valid = np.array([1, 1, 1, 1, 1, 0, 0, 1, 1, 1], dtype=bool)
        segments, count = kv_write_segments(slots, valid)
        write_rows_torch(torch.tensor([count], dtype=torch.int32), segments, rows, kv)
        expected = torch.zeros_like(kv).flatten(0, 1)
        expected[torch.from_numpy(slots[valid])] = rows[torch.from_numpy(valid)]
        self.assertTrue(torch.equal(kv.flatten(0, 1), expected))


class TestFusedKvRows(unittest.TestCase):
    def test_interleaves_k_and_v_heads_and_pads_to_the_pool_layout(self):
        k = torch.randn(3, 2, 128, dtype=torch.bfloat16)
        v = torch.randn(3, 2, 128, dtype=torch.bfloat16)
        kv_cache = torch.zeros(4, 8, 2, 2, 128, dtype=torch.bfloat16)
        rows = fused_kv_rows(k, v, kv_cache)
        self.assertEqual(tuple(rows.shape), (3, 2, 2, 128))
        self.assertTrue(torch.equal(rows[:, 0, 0], k[:, 0]))
        self.assertTrue(torch.equal(rows[:, 0, 1], v[:, 0]))
        self.assertTrue(torch.equal(rows[:, 1, 0], k[:, 1]))
        self.assertTrue(torch.equal(rows[:, 1, 1], v[:, 1]))

        # One head packed four wide with a narrower head dim: zero padding.
        k = torch.ones(2, 1, 64, dtype=torch.bfloat16)
        v = 2 * torch.ones(2, 1, 64, dtype=torch.bfloat16)
        kv_cache = torch.zeros(4, 8, 1, 4, 128, dtype=torch.bfloat16)
        rows = fused_kv_rows(k, v, kv_cache)
        self.assertEqual(tuple(rows.shape), (2, 1, 4, 128))
        self.assertTrue(torch.equal(rows[:, 0, 0, :64], k[:, 0]))
        self.assertTrue(torch.equal(rows[:, 0, 1, :64], v[:, 0]))
        self.assertEqual(float(rows[:, 0, 0, 64:].abs().sum()), 0.0)
        self.assertEqual(float(rows[:, 0, 2:].abs().sum()), 0.0)

    def test_fp8_inputs_are_handled_as_bytes(self):
        k = torch.randn(2, 1, 128).to(torch.float8_e4m3fn)
        v = torch.randn(2, 1, 128).to(torch.float8_e4m3fn)
        kv_cache = torch.zeros(4, 8, 1, 2, 128, dtype=torch.float8_e4m3fn)
        rows = fused_kv_rows(k, v, kv_cache)
        self.assertEqual(rows.dtype, torch.uint8)
        self.assertTrue(torch.equal(rows[:, 0, 0], k[:, 0].view(torch.uint8)))
        self.assertTrue(torch.equal(rows[:, 0, 1], v[:, 0].view(torch.uint8)))


class TestAllGatherAndWriteKv(unittest.TestCase):
    def _run(self, dtype):
        torch.manual_seed(0)
        k0 = torch.randn(4, 1, 128, dtype=torch.bfloat16)
        v0 = torch.randn(4, 1, 128, dtype=torch.bfloat16)
        k1 = torch.randn(4, 1, 128, dtype=torch.bfloat16)
        v1 = torch.randn(4, 1, 128, dtype=torch.bfloat16)
        kv_cache = torch.zeros(8, 4, 1, 2, 128, dtype=dtype)
        # Rank 0 owns slots 5, 6, 9, 10; rank 1 owns 17, 18 and two padding rows.
        dest_slots = torch.tensor([5, 6, 9, 10, 17, 18, 0, 0])
        segments, count = kv_write_segments(
            dest_slots.numpy(), np.array([1, 1, 1, 1, 1, 1, 0, 0], dtype=bool)
        )
        if dtype == torch.float8_e4m3fn:
            # An fp8 pool gathers the bf16 K/V (one all-gather each) and
            # quantizes the gathered rows.
            quantize = lambda k, v: (k.to(dtype), v.to(dtype))  # noqa: E731
            group = _FakeGatherGroup(2, [[k1], [v1]], per_call=True)
            k1q, v1q = quantize(k1, v1)
            k0q, v0q = quantize(k0, v0)
        else:
            quantize = None
            group = _FakeGatherGroup(2, [fused_kv_rows(k1, v1, kv_cache)])
            k0q, v0q, k1q, v1q = k0, v0, k1, v1
        all_gather_and_write_kv(
            kv_cache=kv_cache,
            k=k0,
            v=v0,
            write_segments=segments,
            write_num_segments=torch.tensor([count], dtype=torch.int32),
            group=group,
            quantize=quantize,
        )
        flat = (
            kv_cache.view(torch.uint8).flatten(0, 1)
            if dtype == torch.float8_e4m3fn
            else kv_cache.flatten(0, 1)
        )
        expected = torch.cat(
            [fused_kv_rows(k0q, v0q, kv_cache), fused_kv_rows(k1q, v1q, kv_cache)],
            dim=0,
        )
        for row, slot in enumerate(dest_slots.tolist()[:6]):
            self.assertTrue(torch.equal(flat[slot], expected[row]))
        # Untouched slots stay zero.
        self.assertEqual(float(flat[1:5].to(torch.float32).abs().sum()), 0.0)

    def test_bf16_rows_land_at_their_slots(self):
        self._run(torch.bfloat16)

    def test_fp8_rows_land_at_their_slots(self):
        self._run(torch.float8_e4m3fn)

    def test_fp8_pool_without_quantize_is_rejected(self):
        """Compute-dtype rows must not reach an fp8 pool unquantized."""
        k = torch.randn(4, 1, 128, dtype=torch.bfloat16)
        kv_cache = torch.zeros(8, 4, 1, 2, 128, dtype=torch.float8_e4m3fn)
        with self.assertRaisesRegex(ValueError, "needs `quantize`"):
            all_gather_and_write_kv(
                kv_cache=kv_cache,
                k=k,
                v=k,
                write_segments=torch.zeros(3, 1, dtype=torch.int32),
                write_num_segments=torch.tensor([0], dtype=torch.int32),
                group=_FakeGatherGroup(2, []),
            )


class TestRankTensorSelection(unittest.TestCase):
    def test_cp_slot_rows_use_the_rank_tensor_offset(self):
        device = torch.device("cpu")
        saved = dict(dp_attention._RANK_TENSORS)
        self.addCleanup(dp_attention._RANK_TENSORS.update, saved)
        self.addCleanup(dp_attention._RANK_TENSORS.clear)
        dp_attention._RANK_TENSORS[("attn_cp", device)] = torch.tensor(
            1, dtype=torch.int64
        )
        rows = dp_attention._cp_rank_slot_rows(slot_rows=6, num_rows=5, device=device)
        self.assertEqual(rows.tolist(), [6, 7, 8, 9, 10])

    def test_cp_token_inputs_follow_the_layout(self):
        layout = _layout()
        input_ids = torch.arange(64) + 1
        positions = torch.arange(64) * 10
        global_ids, ids, pos = cp_token_inputs(input_ids, positions, layout)
        host = layout.host
        self.assertEqual(global_ids.shape[0], host.cp_size * host.local_rows_padded)
        self.assertEqual(global_ids[:64].tolist(), input_ids.tolist())
        self.assertEqual(int(global_ids[64:].abs().sum()), 0)
        self.assertEqual(ids.shape[0], host.local_rows_padded)
        real = host.local_len
        self.assertEqual(ids[:real].tolist(), (host.local_rows[:real] + 1).tolist())
        self.assertEqual(pos[:real].tolist(), (host.local_rows[:real] * 10).tolist())

    def test_embedding_wrapper_embeds_global_ids_and_runs_local_rows(self):
        """The vocab-parallel embedding all-reduces across the group, so it must
        see the same (global) ids on every rank; the model then runs on the
        rank's rows only."""
        table = torch.arange(20, dtype=torch.float32).reshape(10, 2)
        calls = []

        def fn(input_ids, positions, forward_batch, input_embeds=None, extra=None):
            calls.append((input_ids, positions, input_embeds, extra))
            return "out"

        forward = wrap_cp_embedding_forward(fn, lambda ids: table[ids])
        global_ids = torch.tensor([3, 1, 4, 1, 5, 9, 0, 0])
        local_rows = torch.tensor([0, 5, 2])
        out = forward(
            global_ids,
            torch.tensor([0, 5, 2]),
            None,
            cp_local_rows=local_rows,
            extra="x",
        )
        self.assertEqual(out, "out")
        ids, pos, embeds, extra = calls[0]
        self.assertEqual(ids.tolist(), [3, 9, 4])
        self.assertTrue(torch.equal(embeds, table[[3, 9, 4]]))
        self.assertEqual(extra, "x")
        # Without local rows the call passes straight through.
        forward(global_ids, None, None, extra="y")
        self.assertIsNone(calls[1][2])
        self.assertIs(calls[1][0], global_ids)

    def test_embedding_wrapper_fills_a_positional_input_embeds_slot(self):
        """A caller that passes input_embeds positionally must not also get it as
        a keyword: binding it twice is a TypeError, and the arity must not change
        because dynamo keys its warmup entries on len(args)."""
        table = torch.arange(20, dtype=torch.float32).reshape(10, 2)
        calls = []

        def fn(input_ids, positions, forward_batch, input_embeds=None, proxy=None):
            calls.append((input_ids, input_embeds, proxy))
            return "out"

        forward = wrap_cp_embedding_forward(fn, lambda ids: table[ids])
        global_ids = torch.tensor([3, 1, 4, 1, 5, 9, 0, 0])
        sentinel = object()
        out = forward(
            global_ids,
            torch.tensor([0, 5, 2]),
            None,
            None,
            sentinel,
            cp_local_rows=torch.tensor([0, 5, 2]),
        )
        self.assertEqual(out, "out")
        ids, embeds, proxy = calls[0]
        self.assertEqual(ids.tolist(), [3, 9, 4])
        self.assertTrue(torch.equal(embeds, table[[3, 9, 4]]))
        # The trailing positional (pp_proxy_tensors in the real model) survives.
        self.assertIs(proxy, sentinel)

    def test_cp_local_embeds_match_the_rows_cp_token_inputs_selects(self):
        """Caller-supplied embeddings arrive at the batch's global token count
        while every other token-indexed graph input is local, so a forward that
        takes input_embeds must narrow them to the same rows and the same width
        as the ids -- otherwise the compiled ladder, built at the local width,
        never matches at serve time."""
        layout = _layout()
        host = layout.host
        embeds = torch.arange(64, dtype=torch.float32).reshape(64, 1).expand(64, 3)
        local = cp_local_embeds(embeds, layout)
        _, ids, _ = cp_token_inputs(torch.arange(64), torch.arange(64), layout)
        self.assertEqual(local.shape[0], ids.shape[0])
        self.assertEqual(local.shape[0], host.local_rows_padded)
        real = host.local_len
        self.assertEqual(local[:real, 0].tolist(), host.local_rows[:real].tolist())

    def test_cp_local_embeds_zero_pad_reaches_the_global_row_space(self):
        """Local rows index cp_size * local_rows_padded rows, which is wider
        than the batch's own token count, so a short embedding must be padded
        rather than indexed out of bounds."""
        layout = _layout()
        host = layout.host
        short = host.cp_size * host.local_rows_padded - 8
        local = cp_local_embeds(torch.ones(short, 3), layout)
        self.assertEqual(local.shape, (host.local_rows_padded, 3))


class TestLastTokenRows(unittest.TestCase):
    def test_owned_rows_are_kept_and_the_rest_come_from_the_reduce(self):
        hidden = torch.arange(8, dtype=torch.float32).reshape(8, 1).expand(8, 4)
        other = torch.full((4, 4), 100.0)
        rows = cp_last_token_rows(
            hidden,
            last_rows=torch.tensor([3, 7, 0, 0]),
            last_owner_mask=torch.tensor([1, 1, 0, 0], dtype=torch.int32),
            group=_FakeReduceGroup(other),
        )
        self.assertEqual(rows[:, 0].tolist(), [103.0, 107.0, 100.0, 100.0])


class TestLogitsForwardWrapper(unittest.TestCase):
    def _wrapper(self, cp_last_rows, cp_last_owner_mask=None, other=None):
        original = MagicMock(return_value="out")
        processor = SimpleNamespace(forward=original, compute_input_logprobs=False)
        wrapper = install_cp_logits_forward(processor)
        self.assertIs(install_cp_logits_forward(processor), wrapper)
        processor.original_forward = original
        metadata = SimpleNamespace(
            cp_last_rows=cp_last_rows, cp_last_owner_mask=cp_last_owner_mask
        )
        self._patch(
            "get_forward_context",
            lambda: SimpleNamespace(
                attn_backend=SimpleNamespace(forward_metadata=metadata)
            ),
        )
        self._patch(
            "get_parallel",
            lambda: SimpleNamespace(
                attn_cp_group=_FakeReduceGroup(other if other is not None else 0.0)
            ),
        )
        self._patch("ForwardBatch", SimpleNamespace)
        return wrapper, processor

    def _patch(self, name, value):
        patcher = patch(f"{_CP_FORWARD}.{name}", value)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_batches_without_cp_rows_go_straight_through(self):
        wrapper, processor = self._wrapper(cp_last_rows=None)
        batch = SimpleNamespace()
        out = wrapper("ids", "hidden", "head", batch, None, None)
        self.assertEqual(out, "out")
        processor.original_forward.assert_called_once_with(
            "ids", "hidden", "head", batch, None, None
        )

    def test_cp_batches_run_on_the_reduced_last_rows(self):
        hidden = torch.arange(8, dtype=torch.float32).reshape(8, 1).expand(8, 2)
        wrapper, processor = self._wrapper(
            cp_last_rows=torch.tensor([3, 7, 0, 0]),
            cp_last_owner_mask=torch.tensor([1, 1, 0, 0], dtype=torch.int32),
        )
        built = []

        class _Meta:
            @classmethod
            def from_forward_batch(cls, batch, *, compute_input_logprobs):
                meta = SimpleNamespace(
                    extend_seq_lens=None, compute_input_logprobs=compute_input_logprobs
                )
                built.append(meta)
                return meta

        self._patch("LogitsMetadata", _Meta)
        batch = SimpleNamespace()
        wrapper("ids", hidden, "head", batch)
        args = processor.original_forward.call_args.args
        self.assertEqual(args[1][:, 0].tolist(), [3.0, 7.0, 0.0, 0.0])
        self.assertIs(args[3], built[0])
        self.assertEqual(built[0].extend_seq_lens.tolist(), [1, 1, 1, 1])
        self.assertFalse(built[0].compute_input_logprobs)


class TestPrepareForwardBatch(unittest.TestCase):
    def _batch(self, extend=(40, 24), prefix=(0, 100)):
        total = sum(extend)
        bs = len(extend)
        lens = torch.tensor(extend, dtype=torch.int32)
        return SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            input_ids=torch.arange(total),
            positions=torch.arange(total),
            out_cache_loc=torch.arange(total),
            input_embeds=None,
            mrope_positions=None,
            seq_lens=lens + torch.tensor(prefix, dtype=torch.int32),
            orig_seq_lens=lens.clone(),
            req_pool_indices=torch.arange(bs),
            req_pool_indices_cpu=torch.arange(bs),
            extend_seq_lens=lens.clone(),
            extend_prefix_lens=torch.tensor(prefix, dtype=torch.int32),
            extend_start_loc=torch.zeros(bs, dtype=torch.int32),
            encoder_lens=None,
            extend_seq_lens_cpu=list(extend),
            extend_prefix_lens_cpu=list(prefix),
            extend_logprob_start_lens_cpu=None,
            mm_inputs=None,
            spec_info=None,
            batch_size=bs,
            attn_cp_metadata=SimpleNamespace(
                per_rank_logical_token=None,
                per_rank_actual_token=None,
                max_rank_len=None,
            ),
        )

    def test_buckets_local_rows_and_pads_request_fields(self):
        batch = self._batch()
        real_q, real_b = prepare_tpu_cp_forward_batch(
            batch,
            t_bucketer=Bucketer(name="Token", buckets=[16, 32, 64]),
            b_bucketer=Bucketer(name="Bs", buckets=[1, 2, 4]),
            extend_bs=None,
            cp_size=4,
            cp_rank=1,
            device=torch.device("cpu"),
        )
        self.assertEqual((real_q, real_b), (64, 2))
        layout = get_tpu_cp_layout(batch)
        # 40 -> blocks of 5, 24 -> blocks of 3: 16 local rows on every rank.
        self.assertEqual(layout.host.local_rows_padded, 16)
        self.assertEqual(layout.host.local_len, 16)
        self.assertEqual(layout.local_rows.shape[0], 16)
        self.assertEqual(batch.attn_cp_metadata.per_rank_actual_token, [16] * 4)
        self.assertEqual(batch.attn_cp_metadata.max_rank_len, [16] * 4)
        # Request fields are padded to the request bucket; token fields keep
        # their global row count.
        self.assertEqual(batch.seq_lens.shape[0], 4)
        self.assertEqual(batch.req_pool_indices_cpu.shape[0], 4)
        self.assertEqual(batch.extend_seq_lens_cpu, [40, 24, 0, 0])
        self.assertEqual(batch.batch_size, 4)
        self.assertEqual(batch.input_ids.shape[0], 64)

    def test_rejects_mixed_batches(self):
        batch = self._batch()
        batch.input_ids = torch.arange(70)
        with self.assertRaisesRegex(ValueError, "pure extend batch"):
            prepare_tpu_cp_forward_batch(
                batch,
                t_bucketer=Bucketer(name="Token", buckets=[16, 32, 64]),
                b_bucketer=Bucketer(name="Bs", buckets=[1, 2, 4]),
                extend_bs=None,
                cp_size=4,
                cp_rank=0,
                device=torch.device("cpu"),
            )


class TestCpMetadata(unittest.TestCase):
    CONTEXT_LEN = 1024
    POOL_WIDTH = 2048
    PAGE_SIZE = 64

    def _make_backend(self, req_to_token, full_to_swa=None):
        override = get_context().override_server_args(page_size=self.PAGE_SIZE)
        override.install()
        self.addCleanup(override.restore)
        model_runner = SimpleNamespace(
            device=torch.device("cpu"),
            page_size=self.PAGE_SIZE,
            model_config=SimpleNamespace(context_len=self.CONTEXT_LEN),
            req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
            token_to_kv_pool=SimpleNamespace(full_to_swa_index_mapping=full_to_swa),
            is_hybrid_swa=full_to_swa is not None,
            kv_cache_dtype=torch.bfloat16,
            spec_algorithm=SimpleNamespace(is_none=lambda: True),
        )
        with patch(
            "sglang.srt.hardware_backend.tpu.attention_pallas_rpa_base."
            "_build_pallas_kernel_op",
            return_value=None,
        ):
            return TpuPallasRpaV3AttnBackend(model_runner)

    def test_two_rows_per_request_over_duplicated_page_rows(self):
        req_to_token = torch.zeros(2, self.POOL_WIDTH, dtype=torch.int32)
        req_to_token[0] = 1000 + torch.arange(self.POOL_WIDTH)
        req_to_token[1] = 5000 + torch.arange(self.POOL_WIDTH)
        backend = self._make_backend(req_to_token)
        layout = _layout(
            cp_size=4, cp_rank=1, extend=(40, 24), prefix=(0, 100), pad_b=4
        )
        host = layout.host
        batch = SimpleNamespace(
            seq_lens=torch.tensor([40, 124, 0, 0], dtype=torch.int32),
            req_pool_indices_cpu=torch.tensor([0, 1, 0, 0]),
        )
        metadata = backend._build_cp_metadata(batch, layout)

        pages_per_seq = (backend.max_context_len + self.PAGE_SIZE - 1) // self.PAGE_SIZE
        self.assertEqual(metadata.kv_lens.tolist(), host.pseudo_kv_lens.tolist())
        self.assertEqual(
            metadata.cu_q_lens.tolist(),
            [0] + np.cumsum(host.pseudo_q_lens).tolist(),
        )
        self.assertEqual(
            metadata.cu_kv_lens.tolist(),
            [i * pages_per_seq * self.PAGE_SIZE for i in range(9)],
        )
        pages = metadata.page_indices.reshape(8, pages_per_seq)
        self.assertTrue(torch.equal(pages[0], pages[1]))
        self.assertTrue(torch.equal(pages[2], pages[3]))
        self.assertEqual(
            pages[0].tolist(), [(1000 + 64 * j) // 64 for j in range(pages_per_seq)]
        )
        self.assertEqual(
            pages[2].tolist(), [(5000 + 64 * j) // 64 for j in range(pages_per_seq)]
        )
        self.assertEqual(metadata.distribution.tolist(), [0, 4, 4])
        self.assertEqual(int(metadata.active_num_tokens), host.local_len)

        expected_slots = {}
        for row, (req, pos) in enumerate(
            zip(host.gathered_req.tolist(), host.gathered_pos.tolist())
        ):
            if req >= 0:
                expected_slots[row] = int(
                    req_to_token[req, host.prefix_lens[req] + pos]
                )
        self.assertEqual(
            _expand_write_segments(
                metadata.cp_write_segments, metadata.cp_write_num_segments
            ),
            expected_slots,
        )
        self.assertIsNone(metadata.swa_page_indices)
        self.assertIsNone(metadata.cp_swa_write_segments)
        self.assertEqual(metadata.cp_last_rows.tolist(), host.last_rows.tolist())
        self.assertEqual(
            metadata.cp_last_owner_mask.tolist(), host.last_owner_mask.tolist()
        )
        # The same builder is reached through init_forward_metadata.
        batch.attn_cp_metadata = None
        setattr(batch, "_tpu_cp_layout", layout)
        backend.token_to_kv_pool.tpu_commit_pending_loads = lambda: None
        backend.init_forward_metadata(batch)
        self.assertEqual(
            _expand_write_segments(
                backend.forward_metadata.cp_write_segments,
                backend.forward_metadata.cp_write_num_segments,
            ),
            expected_slots,
        )

    def test_hybrid_swa_rows_and_slots_are_translated(self):
        req_to_token = torch.zeros(2, self.POOL_WIDTH, dtype=torch.int32)
        req_to_token[0] = 1000 + torch.arange(self.POOL_WIDTH)
        req_to_token[1] = 5000 + torch.arange(self.POOL_WIDTH)
        # Full-pool slot s lives at SWA slot s // 2 (page 64 -> SWA page 32).
        full_to_swa = torch.arange(8192, dtype=torch.int64) // 2
        backend = self._make_backend(req_to_token, full_to_swa=full_to_swa)
        layout = _layout(
            cp_size=4, cp_rank=1, extend=(40, 24), prefix=(0, 100), pad_b=4
        )
        batch = SimpleNamespace(
            seq_lens=torch.tensor([40, 124, 0, 0], dtype=torch.int32),
            req_pool_indices_cpu=torch.tensor([0, 1, 0, 0]),
        )
        metadata = backend._build_cp_metadata(batch, layout)

        pages_per_seq = (backend.max_context_len + self.PAGE_SIZE - 1) // self.PAGE_SIZE
        full_pages = metadata.page_indices.reshape(8, pages_per_seq)
        swa_pages = metadata.swa_page_indices.reshape(8, pages_per_seq)
        self.assertTrue(
            torch.equal(
                swa_pages,
                (
                    full_to_swa[full_pages.to(torch.int64) * self.PAGE_SIZE]
                    // self.PAGE_SIZE
                ).to(torch.int32),
            )
        )
        full_rows = _expand_write_segments(
            metadata.cp_write_segments, metadata.cp_write_num_segments
        )
        self.assertEqual(
            _expand_write_segments(
                metadata.cp_swa_write_segments, metadata.cp_swa_write_num_segments
            ),
            {row: int(full_to_swa[slot]) for row, slot in full_rows.items()},
        )

    def test_hybrid_swa_rejects_unmapped_extend_rows(self):
        req_to_token = torch.zeros(2, self.POOL_WIDTH, dtype=torch.int32)
        req_to_token[0] = 1000 + torch.arange(self.POOL_WIDTH)
        req_to_token[1] = 5000 + torch.arange(self.POOL_WIDTH)
        full_to_swa = torch.zeros(8192, dtype=torch.int64)  # nothing mapped
        backend = self._make_backend(req_to_token, full_to_swa=full_to_swa)
        layout = _layout(
            cp_size=4, cp_rank=1, extend=(40, 24), prefix=(0, 100), pad_b=4
        )
        batch = SimpleNamespace(
            seq_lens=torch.tensor([40, 124, 0, 0], dtype=torch.int32),
            req_pool_indices_cpu=torch.tensor([0, 1, 0, 0]),
        )
        with self.assertRaisesRegex(RuntimeError, "without an SWA slot"):
            backend._build_cp_metadata(batch, layout)

    def test_hybrid_swa_accepts_warmup_rows_on_the_padding_slot(self):
        req_to_token = torch.zeros(2, self.POOL_WIDTH, dtype=torch.int32)
        full_to_swa = torch.zeros(8192, dtype=torch.int64)  # nothing mapped
        backend = self._make_backend(req_to_token, full_to_swa=full_to_swa)
        layout = _layout(
            cp_size=4, cp_rank=1, extend=(40, 24), prefix=(0, 100), pad_b=4
        )
        batch = SimpleNamespace(
            seq_lens=torch.tensor([40, 124, 0, 0], dtype=torch.int32),
            req_pool_indices_cpu=torch.tensor([0, 1, 0, 0]),
        )
        metadata = backend._build_cp_metadata(batch, layout)
        swa_rows = _expand_write_segments(
            metadata.cp_swa_write_segments, metadata.cp_swa_write_num_segments
        )
        self.assertEqual(swa_rows, {})

    def test_padding_rows_are_not_written_and_the_table_is_bounded(self):
        """Warmup rows all point at slot 0, the reserved page, and produce no
        runs; the run table is prefetched into SMEM, so it is sized for a served
        batch (one run per row would not fit at cp 8)."""
        req_to_token = torch.zeros(2, self.POOL_WIDTH, dtype=torch.int32)
        layout = _layout(
            cp_size=4, cp_rank=1, extend=(40, 24), prefix=(0, 100), pad_b=4
        )
        batch = SimpleNamespace(
            seq_lens=torch.tensor([40, 124, 0, 0], dtype=torch.int32),
            req_pool_indices_cpu=torch.tensor([0, 1, 0, 0]),
        )
        metadata = self._make_backend(req_to_token)._build_cp_metadata(batch, layout)
        self.assertEqual(int(metadata.cp_write_num_segments), 0)
        rows = layout.host.cp_size * layout.host.local_rows_padded
        bound = write_run_bound(rows, page_size=self.PAGE_SIZE, blocks=2 * 4 * 4)
        self.assertEqual(tuple(metadata.cp_write_segments.shape), (3, bound))


if __name__ == "__main__":
    unittest.main()
