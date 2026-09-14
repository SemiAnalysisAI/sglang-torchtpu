from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.tpu.attention_pallas_rpa_base import (
    BaseTpuPallasRpaAttnBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize("lengths", [[4, 4, 0, 0], [5, 9, 0, 0]])
@pytest.mark.parametrize("mirror_kind", ["list", "tensor", "missing"])
def test_host_query_lengths_match_padded_device_lengths_without_readback(
    lengths, mirror_kind
):
    backend = object.__new__(BaseTpuPallasRpaAttnBackend)
    backend.page_size = 64
    backend.max_context_len = 128
    backend._is_hybrid_swa = False
    backend._is_extend_including_draft = lambda mode: True
    backend._get_cached_rpa_distribution = lambda *args: torch.zeros(
        3, dtype=torch.int32
    )
    backend.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.arange(8 * 128, dtype=torch.int32).view(8, 128)
    )
    mirror = (
        lengths
        if mirror_kind == "list"
        else torch.tensor(lengths) if mirror_kind == "tensor" else None
    )
    # The meta tensor cannot be copied to CPU. A present host mirror must be
    # sufficient; the missing-mirror case exercises the existing fallback.
    device_lengths = (
        torch.empty(4, device="meta") if mirror is not None else torch.tensor(lengths)
    )
    batch = SimpleNamespace(
        seq_lens=torch.tensor([68, 73, 0, 0]),
        seq_lens_cpu=torch.tensor([68, 73, 0, 0]),
        req_pool_indices_cpu=torch.tensor([4, 1, 0, 0]),
        extend_seq_lens=device_lengths,
        extend_seq_lens_cpu=mirror,
        forward_mode=SimpleNamespace(
            is_decode=lambda: False, is_target_verify=lambda: False
        ),
    )
    with patch.object(
        envs.SGLANG_TPU_SPEC_DECODE_HOST_QUERY_LENS, "get", return_value=True
    ):
        metadata = backend._build_metadata(batch, non_empty_bs=2)
    expected = torch.cat(
        (torch.zeros(1, dtype=torch.int32), torch.tensor(lengths).cumsum(0).int())
    )
    torch.testing.assert_close(metadata.cu_q_lens, expected)
    assert metadata.active_num_tokens.item() == sum(lengths)


@pytest.mark.parametrize("padding", ["general", "eagle"])
@pytest.mark.parametrize("physical_bs", [2, 4])
@pytest.mark.parametrize("mirror_kind", ["missing", "partial", "list", "tensor"])
@pytest.mark.parametrize("enabled", [False, True])
def test_query_mirror_survives_padding_before_metadata(
    padding, physical_bs, mirror_kind, enabled
):
    from sglang.srt.hardware_backend.tpu.compilation.bucketer import Bucketer
    from sglang.srt.hardware_backend.tpu.compilation.compiled_forward_shim import (
        _pad_forward_batch_inputs,
        _restore_forward_batch_inputs,
    )
    from sglang.srt.hardware_backend.tpu.compilation.spec_decode_compile_policy_eagle3 import (
        Eagle3SpecDecodeCompilePolicyMixin,
    )
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

    mirror = {
        "missing": None,
        "partial": [4],
        "list": [4, 4],
        "tensor": torch.tensor([4, 4]),
    }[mirror_kind]
    batch = ForwardBatch(
        forward_mode=(
            ForwardMode.EXTEND if padding == "general" else ForwardMode.DRAFT_EXTEND_V2
        ),
        batch_size=2,
        input_ids=torch.arange(8),
        positions=torch.arange(8),
        req_pool_indices=torch.tensor([0, 1]),
        req_pool_indices_cpu=torch.tensor([0, 1]),
        seq_lens=torch.tensor([68, 73]),
        seq_lens_cpu=torch.tensor([68, 73]),
        seq_lens_sum=141,
        out_cache_loc=torch.arange(8),
        extend_seq_lens=torch.tensor([4, 4], dtype=torch.int32),
        extend_seq_lens_cpu=mirror,
        extend_prefix_lens=torch.tensor([64, 69]),
        extend_prefix_lens_cpu=[64, 69],
        extend_start_loc=torch.tensor([0, 4]),
        extend_num_tokens=8,
        spec_info=SimpleNamespace(num_tokens_per_req=4, is_draft_input=lambda: False),
    )
    backend = object.__new__(BaseTpuPallasRpaAttnBackend)
    backend.kernel_spec = SimpleNamespace(uses_draft_extend_v2=True)
    backend._is_hybrid_swa = False
    backend._host_metadata_cache = None
    with patch.object(
        envs.SGLANG_TPU_SPEC_DECODE_HOST_QUERY_LENS, "get", return_value=enabled
    ), patch(
        "sglang.srt.layers.dp_attention.is_dp_attention_enabled", return_value=False
    ):
        if padding == "general":
            _pad_forward_batch_inputs(
                batch,
                Bucketer(name="Token", buckets=[16]),
                Bucketer(name="Batch", buckets=[physical_bs]),
                extend_bs=physical_bs,
            )
        else:
            policy = object.__new__(Eagle3SpecDecodeCompilePolicyMixin)
            policy.b_bucketer = Bucketer(name="Batch", buckets=[physical_bs])
            policy.draft_token_num = 4
            policy._dp_peer_draft_extend_rows = lambda _: 0
            policy.model_runner = SimpleNamespace(
                attn_backend=SimpleNamespace(
                    get_cpu_graph_seq_len_fill_value=lambda: 0,
                    get_cuda_graph_seq_len_fill_value=lambda: 0,
                )
            )
            policy._pad_draft_extend(batch)
        metadata = backend._build_metadata_host(
            batch,
            64,
            torch.device("cpu"),
            physical_bs,
            torch.arange(4 * 128, dtype=torch.int32).reshape(4, 128),
            2,
            torch.zeros(3, dtype=torch.int32),
            non_empty_bs=2,
        )
    expected = torch.cat(
        (torch.zeros(1, dtype=torch.int32), batch.extend_seq_lens.cumsum(0).int())
    )
    torch.testing.assert_close(metadata.cu_q_lens, expected)
    assert metadata.active_num_tokens.item() == 8
    assert batch.extend_seq_lens_cpu == batch.extend_seq_lens.tolist()
    _restore_forward_batch_inputs(batch, 8, 2)
    assert batch.extend_seq_lens_cpu == [4, 4]
    assert batch.extend_seq_lens.tolist() == [4, 4]


def test_complete_query_mirror_does_not_read_device_during_padding():
    from sglang.srt.hardware_backend.tpu.compilation.compiled_forward_shim import (
        _ensure_host_query_lens,
    )

    batch = SimpleNamespace(
        extend_seq_lens=torch.empty(2, device="meta"), extend_seq_lens_cpu=[4, 4]
    )
    with patch.object(
        envs.SGLANG_TPU_SPEC_DECODE_HOST_QUERY_LENS, "get", return_value=True
    ):
        _ensure_host_query_lens(batch)
    assert batch.extend_seq_lens_cpu == [4, 4]


@pytest.mark.parametrize("idle", [False, True])
def test_eagle_query_producer_precedes_idle_normalization(idle):
    from unittest.mock import Mock

    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.speculative import eagle_worker_common as common

    bs = 0 if idle else 2
    batch = SimpleNamespace(
        seq_lens=torch.full((bs,), 64),
        seq_lens_cpu=None,
        forward_mode=ForwardMode.IDLE if idle else ForwardMode.DECODE,
        model_config=SimpleNamespace(vocab_size=100),
    )
    forward = SimpleNamespace(
        seq_lens=batch.seq_lens,
        seq_lens_cpu=None,
        extend_seq_lens_cpu=None,
        forward_mode=ForwardMode.IDLE if idle else ForwardMode.DRAFT_EXTEND_V2,
        mark_forward_metadata_ready=lambda: None,
    )

    def normalize(fb):
        if idle:
            assert fb.extend_seq_lens_cpu is None
            # An idle DP rank acquires the peer's physical request layout.
            fb.seq_lens = torch.full((4,), 64)
            fb.extend_seq_lens = torch.full((4,), 4)
            fb.extend_seq_lens_cpu = [4] * 4
            fb.forward_mode = ForwardMode.DRAFT_EXTEND_V2
        else:
            assert fb.extend_seq_lens_cpu == [4, 4]
            fb.extend_seq_lens = torch.full((2,), 4)

    runner = SimpleNamespace(
        spec_algorithm=SimpleNamespace(is_standalone=lambda: False),
        compile_policy=SimpleNamespace(
            normalize_spec_forward_batch=normalize,
            prepare_spec_forward_batch=lambda fb: None,
        ),
        attn_backend=SimpleNamespace(init_forward_metadata=Mock()),
    )
    with patch.object(
        common.ForwardBatch, "init_new", return_value=forward
    ), patch.object(common, "maybe_detect_oob"):
        result = common.prepare_for_draft_extend(
            SimpleNamespace(num_front_tokens=0),
            batch,
            torch.zeros(bs * 4, dtype=torch.int64),
            4,
            runner,
            None,
            return_hidden_states_before_norm=False,
        )
    assert result.extend_seq_lens_cpu == result.extend_seq_lens.tolist()


def test_padding_does_not_mutate_a_borrowed_cpu_query_list():
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

    original = [4, 4]
    batch = ForwardBatch(
        forward_mode=ForwardMode.DRAFT_EXTEND_V2,
        batch_size=2,
        input_ids=torch.arange(8),
        positions=torch.arange(8),
        req_pool_indices=torch.tensor([0, 1]),
        seq_lens=torch.tensor([68, 73]),
        seq_lens_sum=141,
        out_cache_loc=torch.arange(8),
        extend_seq_lens=torch.tensor(original),
        extend_seq_lens_cpu=original,
        spec_info=SimpleNamespace(num_tokens_per_req=4, is_draft_input=lambda: False),
    )
    runner = SimpleNamespace(
        attn_backend=SimpleNamespace(
            get_cpu_graph_seq_len_fill_value=lambda: 0,
            get_cuda_graph_seq_len_fill_value=lambda: 0,
        )
    )
    batch._pad_inputs_to_size(runner, 16, 4)
    assert original == [4, 4]
    assert batch.extend_seq_lens_cpu == batch.extend_seq_lens.tolist() == [4, 4, 4, 4]
