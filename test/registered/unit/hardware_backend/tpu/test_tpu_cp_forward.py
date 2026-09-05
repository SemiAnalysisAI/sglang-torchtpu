"""Unit tests for the TPU prefill context-parallel forward glue (CPU tensors)."""

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
    cp_token_inputs,
    ensure_cp_rank_tensor,
    fused_kv_rows,
    get_tpu_cp_layout,
    install_cp_logits_forward,
    pinned_extend_seq_lens_cpu,
    prepare_tpu_cp_forward_batch,
    select_cp_local_rows,
    wrap_cp_embedding_forward,
)
from sglang.srt.hardware_backend.tpu.cp_layout import build_tpu_cp_layout
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class _FakeGatherGroup:
    """Context-parallel group of ``world_size`` ranks whose all-gather places
    this rank's rows first and ``other_rows`` (rank-major) after them."""

    def __init__(self, world_size, other_rows):
        self.world_size = world_size
        self.other_rows = other_rows

    def all_gather_into_tensor(self, output, input):
        output.copy_(torch.cat([input] + self.other_rows, dim=0))


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
        k0 = torch.randn(4, 1, 128).to(dtype)
        v0 = torch.randn(4, 1, 128).to(dtype)
        k1 = torch.randn(4, 1, 128).to(dtype)
        v1 = torch.randn(4, 1, 128).to(dtype)
        kv_cache = torch.zeros(8, 4, 1, 2, 128, dtype=dtype)
        # Rank 0 owns slots 5, 6, 9, 10; rank 1 owns 17, 18 and two padding rows.
        dest_slots = torch.tensor([5, 6, 9, 10, 17, 18, 0, 0])
        group = _FakeGatherGroup(2, [fused_kv_rows(k1, v1, kv_cache)])
        all_gather_and_write_kv(
            kv_cache=kv_cache, k=k0, v=v0, dest_slots=dest_slots, group=group
        )
        flat = (
            kv_cache.view(torch.uint8).flatten(0, 1)
            if dtype == torch.float8_e4m3fn
            else kv_cache.flatten(0, 1)
        )
        expected = torch.cat(
            [fused_kv_rows(k0, v0, kv_cache), fused_kv_rows(k1, v1, kv_cache)], dim=0
        )
        for row, slot in enumerate(dest_slots.tolist()[:6]):
            self.assertTrue(torch.equal(flat[slot], expected[row]))
        # Untouched slots stay zero.
        self.assertEqual(float(flat[1:5].to(torch.float32).abs().sum()), 0.0)

    def test_bf16_rows_land_at_their_slots(self):
        self._run(torch.bfloat16)

    def test_fp8_rows_land_at_their_slots(self):
        self._run(torch.float8_e4m3fn)


class TestRankTensorSelection(unittest.TestCase):
    def test_select_uses_the_rank_tensor_offset(self):
        ensure_cp_rank_tensor(torch.device("cpu"), 1)
        hidden = torch.arange(4 * 6, dtype=torch.float32).reshape(24, 1).expand(24, 3)
        rows = select_cp_local_rows(hidden, slot_rows=6, num_rows=5)
        self.assertEqual(rows[:, 0].tolist(), [6.0, 7.0, 8.0, 9.0, 10.0])

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

    def test_pinned_extend_seq_lens_are_batch_independent(self):
        self.assertEqual(
            pinned_extend_seq_lens_cpu(3, cp_size=8, cp_active=True), [16, 16, 16]
        )
        self.assertEqual(
            pinned_extend_seq_lens_cpu(2, cp_size=8, cp_active=False), [0, 0]
        )


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
        wrapper._get_forward_context = lambda: SimpleNamespace(
            attn_backend=SimpleNamespace(forward_metadata=metadata)
        )
        wrapper._get_parallel = lambda: SimpleNamespace(
            attn_cp_group=_FakeReduceGroup(other if other is not None else 0.0)
        )
        return wrapper, processor

    def test_batches_without_cp_rows_go_straight_through(self):
        wrapper, processor = self._wrapper(cp_last_rows=None)
        batch = SimpleNamespace()
        wrapper._forward_batch_cls = SimpleNamespace
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
        wrapper._forward_batch_cls = SimpleNamespace
        built = []

        class _Meta:
            @classmethod
            def from_forward_batch(cls, batch, *, compute_input_logprobs):
                meta = SimpleNamespace(
                    extend_seq_lens=None, compute_input_logprobs=compute_input_logprobs
                )
                built.append(meta)
                return meta

        wrapper._logits_metadata_cls = _Meta
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
    DRAFT_TOKENS = 8

    def _make_backend(self, req_to_token):
        override = get_context().override_server_args(
            speculative_num_draft_tokens=self.DRAFT_TOKENS,
            speculative_algorithm="EAGLE3",
            speculative_num_steps=None,
            speculative_eagle_topk=1,
            page_size=self.PAGE_SIZE,
        )
        override.install()
        self.addCleanup(override.restore)
        server_args = SimpleNamespace(
            speculative_num_draft_tokens=self.DRAFT_TOKENS,
            speculative_algorithm="EAGLE3",
            speculative_num_steps=None,
            speculative_eagle_topk=1,
            page_size=self.PAGE_SIZE,
        )
        model_runner = SimpleNamespace(
            device=torch.device("cpu"),
            page_size=self.PAGE_SIZE,
            model_config=SimpleNamespace(context_len=self.CONTEXT_LEN),
            server_args=server_args,
            req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
            token_to_kv_pool=SimpleNamespace(),
            is_hybrid_swa=False,
            kv_cache_dtype=torch.bfloat16,
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

        expected_slots = []
        for req, pos in zip(host.gathered_req.tolist(), host.gathered_pos.tolist()):
            if req < 0:
                expected_slots.append(0)
            else:
                expected_slots.append(
                    int(req_to_token[req, host.prefix_lens[req] + pos])
                )
        self.assertEqual(metadata.cp_dest_slots.tolist(), expected_slots)
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
            backend.forward_metadata.cp_dest_slots.tolist(), expected_slots
        )


if __name__ == "__main__":
    unittest.main()
