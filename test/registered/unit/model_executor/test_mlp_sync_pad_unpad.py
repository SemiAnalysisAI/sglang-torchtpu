"""Unit tests for the DP-attention MLP-sync pad/unpad round-trip.

``prepare_mlp_sync_batch`` pads per-request tensors (positions / seq_lens /
req_pool_indices) by appending dummy rows after the real ones so all DP ranks
agree on tensor shapes. ``post_forward_mlp_sync_batch`` must slice them back so
post-forward consumers — seeded sampling (which asserts positions rows ==
sampling rows), ngram token-table updates — never see the padding.

Pure dataclass logic — CPU only.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.scheduler_components import dp_attn
from sglang.srt.managers.scheduler_components.dp_attn import (
    MLPSyncBatchInfo,
    _select_active_tp_replica_info,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _mock_model_runner(seq_len_fill_value: int = 1) -> MagicMock:
    runner = MagicMock()
    runner.attn_backend.get_cuda_graph_seq_len_fill_value.return_value = (
        seq_len_fill_value
    )
    return runner


def _logits_output(num_rows: int) -> LogitsProcessorOutput:
    return LogitsProcessorOutput(
        next_token_logits=torch.randn(num_rows, 16), hidden_states=None
    )


def _global_top_k_output(num_rows: int) -> LogitsProcessorOutput:
    return LogitsProcessorOutput(
        next_token_logits=None,
        next_token_global_top_k_logits=torch.randn(num_rows, 8),
        next_token_global_top_k_token_ids=torch.arange(num_rows * 8).reshape(
            num_rows, 8
        ),
        hidden_states=None,
    )


def _mlp_sync_info(*, tp_size: int = 4, cp_size: int = 1) -> MLPSyncBatchInfo:
    return MLPSyncBatchInfo(
        dp_size=2,
        tp_size=tp_size,
        cp_size=cp_size,
        num_tokens=0,
        num_tokens_for_logprob=0,
        can_run_decode_cuda_graph=True,
        can_run_prefill_cuda_graph=False,
        is_extend_in_batch=False,
        local_can_run_tbo=True,
        local_forward_mode=ForwardMode.IDLE.value,
    )


def _sync_row(
    mode: ForwardMode,
    num_tokens: int,
    *,
    num_tokens_for_logprob: int | None = None,
    can_run_decode_graph: bool = False,
    can_run_prefill_graph: bool | None = None,
) -> list[int]:
    """Build a metadata row that prepare_mlp_sync_batch_raw can emit."""
    if num_tokens_for_logprob is None:
        num_tokens_for_logprob = num_tokens
    if can_run_prefill_graph is None:
        can_run_prefill_graph = mode.is_idle()
    return [
        num_tokens,
        num_tokens_for_logprob,
        int(can_run_decode_graph),
        int(mode.is_extend()),
        0,  # TBO disabled
        mode.value,
        int(can_run_prefill_graph),
    ]


def _run_mocked_mlp_sync_all_gather(
    gathered: torch.Tensor,
    *,
    is_tpu: bool,
    active_ranks: torch.Tensor | None = None,
    use_all_reduce: bool = False,
    tp_size: int = 4,
    cp_size: int = 1,
) -> MLPSyncBatchInfo:
    info = _mlp_sync_info(tp_size=tp_size, cp_size=cp_size)
    if active_ranks is None:
        active_ranks = torch.ones(
            gathered.shape[0] * gathered.shape[1], dtype=torch.int64
        )
    tp_group = SimpleNamespace(active_ranks_cpu=active_ranks)

    def mock_collective(output, *_args, **_kwargs):
        output.copy_(gathered.flatten() if output.ndim == 1 else gathered)

    with (
        patch.object(dp_attn, "_is_tpu", is_tpu),
        patch.object(dp_attn, "get_tp_group", return_value=tp_group),
        patch.object(torch.distributed, "get_rank", return_value=0),
        patch.object(
            torch.distributed,
            "all_reduce" if use_all_reduce else "all_gather_into_tensor",
            side_effect=mock_collective,
        ),
    ):
        info.all_gather(
            device="cpu",
            group=MagicMock(),
            use_all_reduce=use_all_reduce,
        )

    return info


class TestMlpSyncPadUnpad(CustomTestCase):
    def test_tpu_dp_sync_selects_busiest_replica_per_dp_rank(self):
        idle = _sync_row(ForwardMode.IDLE, 0)
        decode = _sync_row(ForwardMode.DECODE, 4)
        extend_tie = _sync_row(ForwardMode.EXTEND, 4)
        extend = _sync_row(ForwardMode.EXTEND, 2)
        gathered = torch.tensor(
            [
                [idle, extend, decode, extend_tie],
                [extend, idle, idle, idle],
            ],
            dtype=torch.int64,
        )

        selected = _select_active_tp_replica_info(gathered)

        self.assertEqual(selected[0].tolist(), decode)
        self.assertEqual(selected[1].tolist(), extend)

    def test_tpu_all_gather_uses_active_replica_metadata(self):
        idle = _sync_row(ForwardMode.IDLE, 0)
        decode = _sync_row(ForwardMode.DECODE, 1, num_tokens_for_logprob=7)
        gathered = torch.tensor(
            [
                [idle, decode, decode, decode],
                [idle, idle, idle, idle],
            ],
            dtype=torch.int64,
        )

        info = _run_mocked_mlp_sync_all_gather(gathered, is_tpu=True)

        self.assertEqual(info.global_num_tokens, [1, 0])
        self.assertEqual(info.global_num_tokens_for_logprob, [7, 0])
        self.assertEqual(info.selected_info_cpu[0].tolist(), decode)
        self.assertEqual(info.selected_info_cpu[1].tolist(), idle)
        self.assertFalse(info.can_run_decode_cuda_graph)
        self.assertFalse(info.can_run_prefill_cuda_graph)

    def test_tpu_all_gather_keeps_extend_vote_hidden_by_token_count_tie(self):
        idle = _sync_row(ForwardMode.IDLE, 0)
        decode = _sync_row(ForwardMode.DECODE, 1)
        extend = _sync_row(ForwardMode.EXTEND, 1)
        gathered = torch.tensor(
            [
                [decode, extend, decode, decode],
                [idle, idle, idle, idle],
            ],
            dtype=torch.int64,
        )

        info = _run_mocked_mlp_sync_all_gather(gathered, is_tpu=True)

        self.assertEqual(info.selected_info_cpu[0].tolist(), decode)
        self.assertTrue(info.is_extend_in_batch)

    def test_non_tpu_all_gather_keeps_tp0_canonical(self):
        idle = _sync_row(
            ForwardMode.IDLE,
            0,
            can_run_decode_graph=True,
            can_run_prefill_graph=True,
        )
        decode = _sync_row(
            ForwardMode.DECODE,
            1,
            num_tokens_for_logprob=7,
            can_run_decode_graph=True,
        )
        gathered = torch.tensor(
            [
                [idle, decode, decode, decode],
                [idle, idle, idle, idle],
            ],
            dtype=torch.int64,
        )

        info = _run_mocked_mlp_sync_all_gather(gathered, is_tpu=False)

        self.assertEqual(info.global_num_tokens, [0, 0])
        self.assertEqual(info.global_num_tokens_for_logprob, [0, 0])
        self.assertEqual(info.selected_info_cpu.tolist(), [idle, idle])
        self.assertTrue(info.can_run_decode_cuda_graph)
        self.assertTrue(info.can_run_prefill_cuda_graph)

    def test_tpu_all_gather_masks_inactive_replica_before_selection(self):
        idle = _sync_row(ForwardMode.IDLE, 0)
        stale_decode = _sync_row(ForwardMode.DECODE, 9)
        gathered = torch.tensor(
            [
                [idle, stale_decode, idle, idle],
                [idle, idle, idle, idle],
            ],
            dtype=torch.int64,
        )
        active_ranks = torch.tensor([1, 0, 1, 1, 1, 1, 1, 1])

        info = _run_mocked_mlp_sync_all_gather(
            gathered,
            is_tpu=True,
            active_ranks=active_ranks,
        )

        self.assertEqual(info.global_num_tokens, [0, 0])
        self.assertEqual(info.selected_info_cpu.tolist(), [idle, idle])

    def test_tpu_all_reduce_selects_across_tp_and_cp_replicas(self):
        idle = _sync_row(ForwardMode.IDLE, 0)
        decode = _sync_row(ForwardMode.DECODE, 3, num_tokens_for_logprob=2)
        gathered = torch.tensor(
            [
                [idle, idle, idle, decode],
                [idle, idle, idle, idle],
            ],
            dtype=torch.int64,
        )

        info = _run_mocked_mlp_sync_all_gather(
            gathered,
            is_tpu=True,
            use_all_reduce=True,
            tp_size=2,
            cp_size=2,
        )

        self.assertEqual(info.global_num_tokens, [3, 0])
        self.assertEqual(info.selected_info_cpu[0].tolist(), decode)

    def test_tpu_non_gather_batch_uses_selected_local_dp_count(self):
        info = _mlp_sync_info()
        info.global_num_tokens = [3, 5]
        info.global_num_tokens_for_logprob = [7, 11]
        batch = SimpleNamespace()

        with (
            patch.object(dp_attn, "_is_tpu", True),
            patch.object(
                dp_attn,
                "get_parallel",
                return_value=SimpleNamespace(attn_dp_rank=1),
            ),
        ):
            dp_attn._update_gather_batch(
                batch,
                info,
                require_mlp_tp_gather=False,
                skip_global_metadata=False,
            )

        self.assertEqual(batch.global_num_tokens, [5])
        self.assertEqual(batch.global_num_tokens_for_logprob, [11])

    def test_init_mlp_sync_metadata_scales_speculative_request_width(self):
        spec_info = SimpleNamespace(
            num_tokens_per_req=4,
            num_tokens_for_logprob_per_req=2,
        )
        fb = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=2,
            input_ids=torch.arange(8),
            req_pool_indices=torch.tensor([0, 1]),
            seq_lens=torch.tensor([5, 6]),
            out_cache_loc=torch.arange(8),
            seq_lens_sum=11,
            positions=torch.arange(8),
            spec_info=spec_info,
        )
        batch = SimpleNamespace(
            global_num_tokens=[2, 0, 3],
            global_num_tokens_for_logprob=[2, 0, 3],
            can_run_decode_cuda_graph=True,
        )

        fb.init_mlp_sync_metadata(batch, torch.device("cpu"))

        self.assertEqual(fb.original_global_num_tokens_cpu, [2, 0, 3])
        self.assertEqual(fb.global_num_tokens_cpu, [8, 0, 12])
        self.assertEqual(fb.global_num_tokens_for_logprob_cpu, [4, 0, 6])
        torch.testing.assert_close(fb.global_num_tokens_gpu, torch.tensor([8, 0, 12]))
        torch.testing.assert_close(
            fb.global_num_tokens_for_logprob_gpu, torch.tensor([4, 0, 6])
        )
        self.assertTrue(fb.can_run_decode_cuda_graph)

    def test_draft_input_without_hidden_states_can_be_padded(self):
        spec_info = SimpleNamespace(
            is_draft_input=lambda: True,
            hidden_states=None,
        )
        fb = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=1,
            input_ids=torch.tensor([11]),
            req_pool_indices=torch.tensor([5]),
            seq_lens=torch.tensor([7]),
            out_cache_loc=torch.tensor([0]),
            seq_lens_sum=7,
            positions=torch.tensor([6]),
            seq_lens_cpu=torch.tensor([7]),
            lora_ids=[None],
            spec_info=spec_info,
        )

        fb._pad_inputs_to_size(_mock_model_runner(), num_tokens=2, bs=1)

        self.assertIsNone(spec_info.hidden_states)

    def test_dp_cuda_graph_batch_size_uses_raw_request_counts(self):
        fb = SimpleNamespace(original_global_num_tokens_cpu=[3, 11, 7])
        self.assertEqual(DecodeCudaGraphRunner._max_dp_batch_size(fb), 11)

        fb.original_global_num_tokens_cpu = None
        with self.assertRaisesRegex(RuntimeError, "raw per-rank request counts"):
            DecodeCudaGraphRunner._max_dp_batch_size(fb)

    def test_decode_post_forward_unpads_per_request_tensors(self):
        fb = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=3,
            input_ids=torch.tensor([11, 12, 13]),
            req_pool_indices=torch.tensor([5, 6, 7]),
            seq_lens=torch.tensor([7, 8, 9]),
            out_cache_loc=torch.tensor([0, 1, 2]),
            seq_lens_sum=24,
            positions=torch.tensor([6, 7, 8]),
            seq_lens_cpu=torch.tensor([7, 8, 9]),
            lora_ids=[None, None, None],
        )
        # Mirror the decode arm of prepare_mlp_sync_batch: record the original
        # batch size, adopt the synced (padded) one, then pad the inputs.
        padded = 5
        fb._original_batch_size = fb.batch_size
        fb.batch_size = padded
        fb._pad_inputs_to_size(_mock_model_runner(), num_tokens=padded, bs=padded)

        # Padding appends dummy rows after the real ones.
        self.assertEqual(fb.positions.shape[0], padded)
        self.assertEqual(fb.seq_lens.shape[0], padded)
        self.assertEqual(fb.req_pool_indices.shape[0], padded)
        torch.testing.assert_close(fb.positions[:3], torch.tensor([6, 7, 8]))

        logits_output = _logits_output(padded)
        fb.post_forward_mlp_sync_batch(logits_output)

        self.assertEqual(fb.batch_size, 3)
        torch.testing.assert_close(fb.positions, torch.tensor([6, 7, 8]))
        torch.testing.assert_close(fb.seq_lens, torch.tensor([7, 8, 9]))
        torch.testing.assert_close(fb.req_pool_indices, torch.tensor([5, 6, 7]))
        torch.testing.assert_close(fb.seq_lens_cpu, torch.tensor([7, 8, 9]))
        self.assertEqual(logits_output.next_token_logits.shape[0], 3)
        # Seeded sampling asserts positions rows == sampled (real) rows.
        self.assertEqual(fb.positions.shape[0], fb.batch_size)

        global_top_k_output = _global_top_k_output(padded)
        fb.post_forward_mlp_sync_batch(global_top_k_output)
        self.assertIsNone(global_top_k_output.next_token_logits)
        self.assertEqual(global_top_k_output.next_token_global_top_k_logits.shape[0], 3)
        self.assertEqual(
            global_top_k_output.next_token_global_top_k_token_ids.shape[0], 3
        )

    def test_extend_post_forward_unpads_positions(self):
        fb = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=2,
            input_ids=torch.arange(7),
            req_pool_indices=torch.tensor([1, 2]),
            seq_lens=torch.tensor([3, 4]),
            out_cache_loc=torch.arange(7),
            seq_lens_sum=7,
            positions=torch.tensor([0, 1, 2, 0, 1, 2, 3]),
            seq_lens_cpu=torch.tensor([3, 4]),
            lora_ids=[None, None],
        )
        # Extend keeps batch_size; only token-level tensors get padded.
        fb._original_batch_size = fb.batch_size
        fb._pad_inputs_to_size(_mock_model_runner(), num_tokens=10, bs=2)

        self.assertEqual(fb.positions.shape[0], 10)

        logits_output = _logits_output(10)
        fb.post_forward_mlp_sync_batch(logits_output)

        torch.testing.assert_close(fb.positions, torch.tensor([0, 1, 2, 0, 1, 2, 3]))
        torch.testing.assert_close(fb.seq_lens, torch.tensor([3, 4]))
        # sample() derives prefill sampling positions from seq_lens - 1, so the
        # row count must match the real request count.
        self.assertEqual((fb.seq_lens - 1).shape[0], fb.batch_size)
        self.assertEqual(logits_output.next_token_logits.shape[0], 7)

        global_top_k_output = _global_top_k_output(10)
        fb.post_forward_mlp_sync_batch(global_top_k_output)
        self.assertIsNone(global_top_k_output.next_token_logits)
        self.assertEqual(global_top_k_output.next_token_global_top_k_logits.shape[0], 2)
        self.assertEqual(
            global_top_k_output.next_token_global_top_k_token_ids.shape[0], 2
        )

    def test_draft_extend_dummy_request_pads_cpu_and_gpu_lens(self):
        spec_info = MagicMock()
        spec_info.num_tokens_per_req = 4
        spec_info.is_draft_input.return_value = False
        fb = ForwardBatch(
            forward_mode=ForwardMode.DRAFT_EXTEND_V2,
            batch_size=1,
            input_ids=torch.empty(0, dtype=torch.int64),
            req_pool_indices=torch.empty(0, dtype=torch.int64),
            seq_lens=torch.empty(0, dtype=torch.int64),
            seq_lens_sum=0,
            out_cache_loc=torch.empty(0, dtype=torch.int64),
            positions=torch.empty(0, dtype=torch.int64),
            seq_lens_cpu=torch.empty(0, dtype=torch.int64),
            extend_seq_lens=torch.empty(0, dtype=torch.int32),
            extend_prefix_lens=torch.empty(0, dtype=torch.int64),
            extend_seq_lens_cpu=[],
            extend_prefix_lens_cpu=[],
            extend_logprob_start_lens_cpu=[],
            spec_info=spec_info,
        )

        fb._pad_inputs_to_size(_mock_model_runner(), num_tokens=4, bs=1)

        torch.testing.assert_close(
            fb.extend_seq_lens, torch.tensor([4], dtype=torch.int32)
        )
        torch.testing.assert_close(fb.extend_prefix_lens, torch.tensor([0]))
        self.assertEqual(fb.extend_seq_lens_cpu, [4])
        self.assertEqual(fb.extend_prefix_lens_cpu, [0])
        self.assertEqual(fb.extend_logprob_start_lens_cpu, [0])


if __name__ == "__main__":
    unittest.main()
