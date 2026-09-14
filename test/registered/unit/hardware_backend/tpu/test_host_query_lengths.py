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
    with patch.object(envs.SGLANG_TPU_EAGLE_HOST_QUERY_LENS, "get", return_value=True):
        metadata = backend._build_metadata(batch, non_empty_bs=2)
    expected = torch.cat(
        (torch.zeros(1, dtype=torch.int32), torch.tensor(lengths).cumsum(0).int())
    )
    torch.testing.assert_close(metadata.cu_q_lens, expected)
    assert metadata.active_num_tokens.item() == sum(lengths)
