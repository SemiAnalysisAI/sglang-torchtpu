"""CPU tests for serving a hybrid linear-attention model over Raiden P/D.

Kimi-K3 keeps MLA latent KV in the paged ``TpuFusedMLATokenToKVPool`` and KDA
conv / recurrent state in the slot-indexed ``TpuMambaPool``. These pin what the
TPU P/D path relies on beyond the symmetric state transfer, without a chip:

* the Mamba pool describes how its state tensors are TP-sharded, which a
  decode at a narrower attention TP reassembles its rows from;
* both pools back up and restore a request for decode retraction;
* state slots freed while a forward may still write them wait on the KV free
  fence, like KV pages;
* the fan-in pieces: state row assembly, the RESHARD_STATE message, and whole
  MLA pages pushed from every source.

The protocol around them is in ``test_disagg_raiden_transfer.py``.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from sglang.srt.configs.mamba_utils import (
    KimiLinearCacheParams,
    KimiLinearStateShape,
    Mamba2StateDType,
)
from sglang.srt.disaggregation.base.conn import StateType
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.kv_free_fence import KVFreeFence
from sglang_tpu.kv_pool import TpuFusedMLATokenToKVPool
from sglang_tpu.mamba_pool import AT_SLOT, TpuMambaPool, TpuMambaSlotAllocator

LAYERS = [1, 2, 4]
NUM_HEADS, HEAD_DIM, K = 4, 32, 4
SIZE = 8


def _mamba_pool(width: int = 1) -> TpuMambaPool:
    shape = KimiLinearStateShape.create(
        tp_world_size=1,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        conv_kernel_size=K,
    )
    params = KimiLinearCacheParams(
        dtype=Mamba2StateDType(conv=torch.float32, temporal=torch.float32),
        layers=LAYERS,
        shape=shape,
    )
    return TpuMambaPool(
        size=SIZE,
        spec_state_size=SIZE,
        cache_params=params,
        mamba_layer_ids=LAYERS,
        device="cpu",
        speculative_num_draft_tokens=None if width == 1 else width,
    )


def _mla_pool(page_size: int = 16) -> TpuFusedMLATokenToKVPool:
    return TpuFusedMLATokenToKVPool(
        size=64,
        page_size=page_size,
        dtype=torch.bfloat16,
        kv_lora_rank=128,
        qk_rope_head_dim=64,
        layer_num=2,
        device="cpu",
        enable_memory_saver=False,
    )


def _fill(tensors) -> None:
    generator = torch.Generator().manual_seed(0)
    for tensor in tensors:
        tensor.copy_(torch.randn(tensor.shape, generator=generator).to(tensor.dtype))


class TestMambaStateDescription(CustomTestCase):
    def test_entries_follow_conv_then_temporal_layers(self):
        pool = _mamba_pool()
        proj = NUM_HEADS * HEAD_DIM
        ptrs, lens, items = pool.get_contiguous_buf_infos()
        states = pool.get_state()
        self.assertEqual(ptrs, [t.data_ptr() for t in states])
        self.assertEqual(lens, [t.nbytes for t in states])
        self.assertEqual(items, [t[0].nbytes for t in states])
        self.assertEqual(pool.get_state_layer_ids(), LAYERS + LAYERS)
        # conv [K-1, 1, q|k|v] shards its channel axis, after K-1 rows;
        # temporal [n_v, d_k, d_v] shards its head axis.
        self.assertEqual(
            pool.get_state_dim_per_tensor(), [3 * proj] * 3 + [NUM_HEADS] * 3
        )
        self.assertEqual(pool.get_state_slice_axes(), [2] * 3 + [0] * 3)
        self.assertEqual(
            pool.get_state_conv_shard_groups(), [[proj, proj, proj]] * 3 + [None] * 3
        )

    def test_hybrid_pool_registers_one_mamba_component(self):
        from sglang.srt.disaggregation.utils import setup_state_kv_args
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

        mamba = _mamba_pool()
        pool = HybridLinearKVPool.__new__(HybridLinearKVPool)
        pool.mamba_pool = mamba
        pool.full_kv_pool = _mla_pool()
        pool.use_dsa = False
        kv_args = SimpleNamespace()
        setup_state_kv_args(kv_args, pool)
        self.assertEqual(kv_args.state_types, [StateType.MAMBA])
        self.assertEqual(kv_args.state_layer_ids, [LAYERS + LAYERS])
        self.assertEqual(
            kv_args.state_item_lens, [[t[0].nbytes for t in mamba.get_state()]]
        )


class TestRetractionBackup(CustomTestCase):
    def _assert_only_row(self, pool, row, want):
        for got, expected in zip(pool.get_state(), want):
            torch.testing.assert_close(got[row], expected)
            others = torch.ones(got.shape[0], dtype=torch.bool)
            others[row] = False
            self.assertEqual(float(got[others].abs().sum()), 0.0)

    def test_mamba_slot_round_trip(self):
        pool = _mamba_pool()
        _fill(pool.get_state())
        before = [t.clone() for t in pool.get_state()]
        saved = pool.get_cpu_copy(torch.tensor(3))
        for tensor in pool.get_state():
            tensor.zero_()
        pool.load_cpu_copy(saved, torch.tensor([5]))
        self._assert_only_row(pool, 5, [t[3] for t in before])

    def test_backup_reads_the_committed_window_row(self):
        """After a verify the committed state is a row of the request's window.
        The hybrid pool hands the request's pool row down to find it, and the
        restore leaves it in the resumed request's slot row."""
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

        mamba = _mamba_pool(width=2)
        _fill(mamba.get_state())
        before = [t.clone() for t in mamba.get_state()]
        # Slot 3's request holds pool row 2; its last verify kept window row 1.
        mamba.select_checkpoints(torch.tensor([3]), torch.tensor([1]))
        committed = mamba.window_start + 2 * 2 + 1
        pool = HybridLinearKVPool.__new__(HybridLinearKVPool)
        pool.full_kv_pool = _mla_pool()
        pool.mamba_pool = mamba
        pool._mamba_translate = lambda ids: ids
        saved = pool.get_cpu_copy(
            torch.arange(16, 21), mamba_indices=torch.tensor(3), req_pool_index=2
        )
        for tensor in mamba.get_state():
            tensor.zero_()
        # The resumed request holds slot 5 and pool row 6; a stale verify
        # offset on slot 5 must not survive the restore.
        mamba.checkpoint_read_offsets[5] = 0
        pool.load_cpu_copy(
            saved,
            torch.arange(32, 37),
            mamba_indices=torch.tensor(5),
            req_pool_index=6,
        )
        self._assert_only_row(mamba, 5, [t[committed] for t in before])
        self.assertEqual(int(mamba.checkpoint_read_offsets[5]), AT_SLOT)

    def test_backup_without_the_request_row_refuses_a_window_state(self):
        pool = _mamba_pool(width=2)
        # At its slot row (after extend), the state needs no request row.
        self.assertEqual(len(pool.get_cpu_copy(torch.tensor(3))), len(pool.get_state()))
        pool.select_checkpoints(torch.tensor([3]), torch.tensor([0]))
        with self.assertRaisesRegex(ValueError, "req_pool_indices"):
            pool.get_cpu_copy(torch.tensor(3))

    def test_mla_tokens_round_trip_full_and_partial_pages(self):
        pool = _mla_pool(page_size=16)
        _fill(pool.kv_buffer)
        before = [buf.clone() for buf in pool.kv_buffer]
        # Page 1 whole, then 5 tokens of page 2.
        src = torch.cat([torch.arange(16, 32), torch.arange(32, 37)])
        saved = pool.get_cpu_copy(src)
        dst = torch.cat([torch.arange(48, 64), torch.arange(64, 69)])
        pool.load_cpu_copy(saved, dst)
        for layer in range(2):
            rows = before[layer].view(-1, before[layer].shape[-1])
            got = pool.kv_buffer[layer].view(-1, rows.shape[-1])
            torch.testing.assert_close(got[dst], rows[src])
            # The rest of the partly written page keeps its rows.
            torch.testing.assert_close(got[69:80], rows[69:80])

    def test_mla_filter_keeps_selected_rows(self):
        pool = _mla_pool(page_size=16)
        _fill(pool.kv_buffer)
        saved = pool.get_cpu_copy(torch.arange(16, 32))
        mask = torch.zeros(16, dtype=torch.bool)
        mask[::2] = True
        filtered = pool.filter_cpu_copy(saved, mask)
        torch.testing.assert_close(torch.cat(filtered[0]), torch.cat(saved[0])[mask])


class _Result(SimpleNamespace):
    copy_done = None


class TestFencedStateSlots(CustomTestCase):
    def setUp(self):
        self.syncs = 0

        def sync():
            self.syncs += 1

        self.fence = KVFreeFence(sync)
        self.slots = TpuMambaSlotAllocator(size=4, device="cpu")
        self.fence.attach(self.slots)

    def _in_flight(self):
        self.fence.on_batch_formed()
        result = _Result()
        self.fence.on_forward_launched(result)
        return result

    def test_idle_free_returns_the_slot_at_once(self):
        slot = self.slots.alloc(1)
        self.slots.free(slot)
        self.assertEqual(len(self.slots.free_slots), 4)

    def test_free_during_a_forward_waits_for_it(self):
        slot = self.slots.alloc(1)
        result = self._in_flight()
        self.slots.free(slot)
        self.assertEqual(len(self.slots.free_slots), 3)
        self.assertEqual(self.slots.available_size(), 4)
        self.fence.on_forward_done(result)
        self.assertEqual(len(self.slots.free_slots), 4)
        self.assertEqual(self.slots.held_pages.num_pages, 0)

    def test_dma_allocation_syncs_before_reusing_a_held_slot(self):
        taken = self.slots.alloc(4)
        self._in_flight()
        self.slots.free(taken[:1])
        got = self.slots.alloc(1)
        self.assertEqual(self.syncs, 1)
        self.assertEqual(got.tolist(), taken[:1].tolist())


class TestStateFanIn(CustomTestCase):
    """Decode TP < prefill TP: each decode rank assembles its state slot from
    ``ratio`` sources, and replicated MLA pages move whole."""

    def _shards(self, full, axis, groups, tp):
        """Rank-local rows of ``full`` sharded on ``axis`` (per sub-block)."""
        if groups is None:
            return np.split(full, tp, axis=axis)
        parts = np.split(full, np.cumsum(groups)[:-1], axis=axis)
        return [
            np.concatenate([np.split(p, tp, axis=axis)[r] for p in parts], axis=axis)
            for r in range(tp)
        ]

    def test_assembly_matches_sharding_at_the_decode_tp(self):
        from sglang_tpu.disaggregation.raiden.reshard_schedule import (
            assemble_state_fan_in,
        )

        rng = np.random.default_rng(0)
        groups = [64, 32, 32]
        conv = rng.standard_normal((K - 1, 1, sum(groups))).astype(np.float32)
        temporal = rng.standard_normal((8, 4, 4)).astype(np.float32)
        src_tp, dst_tp = 4, 2
        src_conv = self._shards(conv, 2, groups, src_tp)
        src_temp = self._shards(temporal, 0, None, src_tp)
        want_conv = self._shards(conv, 2, groups, dst_tp)
        want_temp = self._shards(temporal, 0, None, dst_tp)
        ratio = src_tp // dst_tp
        for d in range(dst_tp):
            sources = [
                [src_conv[s], src_temp[s]] for s in range(d * ratio, (d + 1) * ratio)
            ]
            got = assemble_state_fan_in(sources, [2, 0], [groups, None], src_tp)
            np.testing.assert_array_equal(got[0], want_conv[d])
            np.testing.assert_array_equal(got[1], want_temp[d])

    def test_assembly_rejects_groups_that_do_not_cover_the_row(self):
        from sglang_tpu.disaggregation.raiden.reshard_schedule import (
            assemble_state_fan_in,
        )

        row = np.zeros((3, 1, 10), np.float32)
        with self.assertRaisesRegex(ValueError, "cover"):
            assemble_state_fan_in([[row], [row]], [2], [[16, 8, 8]], 4)

    def test_state_message_round_trips_rows(self):
        from sglang_tpu.disaggregation.raiden.serde import ReshardState

        rows = [
            np.arange(12, dtype=np.float32).reshape(3, 1, 4),
            np.arange(8, dtype=np.float32).reshape(2, 2, 2),
        ]
        frames = ReshardState(room=7, src_rank=3, rows=rows).to_frames()
        got = ReshardState.from_frames(frames)
        self.assertEqual((got.room, got.src_rank), (7, 3))
        for a, b in zip(got.rows, rows):
            np.testing.assert_array_equal(a, b)
        with self.assertRaisesRegex(ValueError, "rows declared"):
            ReshardState.from_frames(frames[:-1])

    def test_mla_pages_move_whole_from_every_source(self):
        from sglang_tpu.disaggregation.raiden.reshard_schedule import (
            WholeBlockGeometry,
            build_receiver_request,
            build_sender_request,
        )

        pb2 = SimpleNamespace(
            StartTransferRequest=_FakeRequest,
            MEMORY_TYPE_DRAM=1,
            MEMORY_TYPE_HBM=2,
        )
        geom = WholeBlockGeometry(4, 2, block_bytes=4096)
        send = build_sender_request(
            pb2,
            uuid=1,
            req_id="1",
            peer="p",
            prefill_rank=3,
            decode_rank=1,
            geom=geom,
            src_block_ids=[5, 6],
            dst_block_ids=[8, 9],
            live_tokens=[128, 40],
        )
        entries = send.shard_push_schedules[0].entries
        self.assertEqual(
            [
                (
                    e.src_block_id,
                    e.dst_block_id,
                    e.src_offset_bytes,
                    e.size_bytes,
                    e.count,
                )
                for e in entries
            ],
            [(5, 8, 0, 4096, 1), (6, 9, 0, 4096, 1)],
        )
        recv = build_receiver_request(
            pb2,
            uuid=1,
            req_id="1",
            peer="p",
            decode_rank=1,
            geom=geom,
            sources=[(2, [5, 6]), (3, [5, 6])],
            dst_block_ids=[8, 9],
            live_tokens=[128, 40],
        )
        self.assertEqual(recv.expected_block_count, 4)


class _FakeEntries(list):
    def add(self):
        entry = SimpleNamespace()
        self.append(entry)
        return entry


class _FakeSchedule:
    def __init__(self):
        self.entries = _FakeEntries()


class _FakeSchedules(dict):
    def __missing__(self, key):
        self[key] = _FakeSchedule()
        return self[key]


class _FakeRequest:
    def __init__(self):
        self.shard_push_schedules = _FakeSchedules()


if __name__ == "__main__":
    unittest.main()
