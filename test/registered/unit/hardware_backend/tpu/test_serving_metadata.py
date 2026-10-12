"""Host metadata ownership and coverage of eager serving shapes."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from sglang.srt.runtime_context import get_context
from sglang.srt.sampling import sampling_batch_info as sampling
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.compilation.bucketer import Bucketer
from sglang_tpu.compilation.decoder_compile_policy import (
    DecoderTpuCompilePolicy,
)
from sglang_tpu.serving_op_precompile import precompile_serving_ops

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def config():
    with get_context().override_server_args(enable_torch_compile=True):
        yield


def make_info(temps, *, device="meta", staged=True):
    reqs = []
    for i, temp in enumerate(temps):
        params = SamplingParams(temperature=temp, top_k=i + 1, sampling_seed=i + 23)
        reqs.append(
            SimpleNamespace(
                sampling_params=params,
                return_sampling_mask=False,
                sampling_logprobs_mode=None,
                custom_logit_processor=None,
            )
        )
    batch = SimpleNamespace(
        reqs=reqs, device=device, batch_input_device="cpu" if staged else None
    )
    with patch.object(
        sampling.penaltylib,
        "BatchedPenalizerOrchestrator",
        return_value=MagicMock(is_required=False),
    ):
        return sampling.SamplingBatchInfo.from_schedule_batch(batch, 32)


@pytest.mark.parametrize("seeded", [False, True])
def test_public_parameters_stay_on_device_and_filter_merge_preserve_order(seeded):
    with get_context().override_server_args(enable_deterministic_inference=seeded):
        info = make_info([0.2, 0.6, 0.8])
        before = info.copy_for_forward()
        before_host = info.host_sampling_params.host
        info.filter_batch([2, 0], torch.empty(2, dtype=torch.int64, device="meta"))
        other = make_info([0.4])
        info.merge_batch(other)
        host = info.host_sampling_params.host
        torch.testing.assert_close(
            host["temperatures"].flatten(), torch.tensor([0.8, 0.2, 0.4])
        )
        assert host["top_ks"].tolist() == [3, 1, 1]
        if seeded:
            assert host["sampling_seed"].tolist() == [25, 23, 23]
        else:
            assert host["sampling_seed"] is None
        for name, value in host.items():
            if value is not None:
                public = getattr(info, name)
                assert public.device.type == "meta" and public.shape == value.shape
                assert public.dtype == value.dtype
                assert getattr(before, name) is not public
        torch.testing.assert_close(
            before_host["temperatures"].flatten(), torch.tensor([0.2, 0.6, 0.8])
        )
        assert before.host_sampling_params is None


def test_unstaged_batch_keeps_base_path():
    info = make_info([0.2, 0.8], staged=False)
    assert info.host_sampling_params is None
    assert info.temperatures.device.type == "meta"


@pytest.mark.parametrize("inplace", [False, True])
def test_mutated_parameters_cannot_use_stale_host_mirror(inplace):
    info = make_info([0.2, 0.8], device="cpu")
    if inplace:
        info.temperatures.add_(1)
    else:
        info.temperatures = torch.tensor([[3.0], [4.0]])
    assert info._current_host_sampling_params() is None
    expected = info.temperatures[1:].clone()
    info.filter_batch([1], torch.tensor([1]))
    torch.testing.assert_close(info.temperatures, expected)


@pytest.mark.parametrize("mode", ["prefill", "decode", "null"])
def test_serving_warmup_covers_partial_buckets_without_touching_live_relay(mode):
    policy = object.__new__(DecoderTpuCompilePolicy)
    policy.b_bucketer = Bucketer("batch", [1, 2, 4])
    policy.t_bucketer = Bucketer("tokens", [8, 16])
    policy.extend_bs = 4
    live = torch.full((9,), 99, dtype=torch.int64)
    stores = []

    def store(buf, indices, values):
        assert buf is not live
        stores.append(indices.numel())
        buf.index_copy_(0, indices, values)

    scheduler = SimpleNamespace(
        tp_worker=SimpleNamespace(model_runner=SimpleNamespace(compile_policy=policy)),
        spec_algorithm=SpeculativeAlgorithm.NONE,
        max_running_requests=4,
        model_config=SimpleNamespace(vocab_size=16),
        future_map=SimpleNamespace(output_tokens_buf=live, _store=store),
    )
    shapes = []
    argmax = torch.argmax

    def record_argmax(logits, **kwargs):
        shapes.append((logits.shape[0], logits._base.shape[0]))
        return argmax(logits, **kwargs)

    with (
        get_context().override_server_args(
            enable_torch_compile=True, disaggregation_mode=mode
        ),
        patch("torch.argmax", side_effect=record_argmax),
    ):
        precompile_serving_ops(
            compile_policy=policy,
            future_map=scheduler.future_map,
            spec_algorithm=scheduler.spec_algorithm,
            max_running_requests=scheduler.max_running_requests,
            vocab_size=scheduler.model_config.vocab_size,
        )
    assert stores == [1, 1, 2, 2, 3, 3, 4, 4]
    assert live.tolist() == [99] * 9
    expected = set()
    for logical in range(1, 5):
        if mode != "decode":
            expected.add((logical, 4))
        if mode != "prefill":
            expected.add((logical, policy.b_bucketer.next_bucket(logical)))
    assert set(shapes) == expected


def test_mixed_tail_warmup_covers_token_buckets_and_tail_widths():
    policy = object.__new__(DecoderTpuCompilePolicy)
    policy.b_bucketer = Bucketer("batch", [1, 2, 4])
    policy.t_bucketer = Bucketer("tokens", [8, 16])
    policy.extend_bs = 4
    live = torch.full((9,), 99, dtype=torch.int64)
    future = SimpleNamespace(
        output_tokens_buf=live,
        _store=lambda buf, idx, val: buf.index_copy_(0, idx, val),
    )
    seen = []
    original = torch.Tensor.index_copy_

    def record(tensor, dim, indices, values):
        if tensor.numel() in (8, 16):
            seen.append((tensor.numel(), indices.numel()))
        return original(tensor, dim, indices, values)

    with (
        get_context().override_server_args(
            enable_torch_compile=True, disaggregation_mode="null"
        ),
        patch.object(torch.Tensor, "index_copy_", record),
    ):
        precompile_serving_ops(
            compile_policy=policy,
            future_map=future,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            max_running_requests=4,
            vocab_size=7,
            enable_mixed_chunk=True,
        )
    assert set(seen) == {(tokens, bs) for tokens in [8, 16] for bs in [1, 2, 3, 4]}
    assert live.tolist() == [99] * 9


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
