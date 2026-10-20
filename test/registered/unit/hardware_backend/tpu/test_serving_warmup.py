"""State isolation for eager TPU serving warmups."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.managers import scheduler as scheduler_module
from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu import output_logprobs, serving_warmup
from sglang_tpu.compilation.pending_call import PendingCall
from sglang_tpu.compilation.warmup_queue import current_warmup_queue

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Sampler:
    enable_deterministic = False

    def __init__(self, fail):
        self.fail = fail
        self.greedy_modes = set()

    def __call__(self, output, info, return_logprob, top_nums, token_ids, positions):
        self.greedy_modes.add(info.is_all_greedy)
        torch.rand_like(output.next_token_logits)
        if self.fail:
            raise RuntimeError("sampler failed")
        assert return_logprob
        assert positions.dtype == torch.int64
        output.next_token_logprobs = output.next_token_logits[:, 0]
        return torch.zeros(positions.shape, dtype=torch.int64)


@pytest.mark.parametrize("fail", (False, True))
def test_sampling_warmup_restores_rng_on_success_and_failure(fail):
    sampler = _Sampler(fail)
    runner = SimpleNamespace(
        is_draft_worker=False,
        pp_group=SimpleNamespace(is_last_rank=True),
        model_config=SimpleNamespace(is_generation=True, vocab_size=8),
        sampler=sampler,
        device="cpu",
        max_running_requests=2,
    )
    before = torch.random.get_rng_state().clone()
    with (
        patch.object(
            serving_warmup,
            "get_spec",
            return_value=SimpleNamespace(speculative_algorithm=None),
        ),
        patch.object(
            serving_warmup,
            "get_exec",
            return_value=SimpleNamespace(graph=SimpleNamespace(torch_compile_max_bs=2)),
        ),
        patch.object(serving_warmup, "warmup_output_top_logprobs"),
    ):
        if fail:
            with pytest.raises(RuntimeError, match="sampler failed"):
                serving_warmup.warmup_sampling(runner)
        else:
            serving_warmup.warmup_sampling(runner)
            assert sampler.greedy_modes == {False, True}
    assert torch.equal(torch.random.get_rng_state(), before)


def test_request_relay_warmup_preserves_live_storage():
    tokens = torch.full((9,), -1, dtype=torch.int64)
    lengths = torch.full((9,), -2, dtype=torch.int64)
    stores = []

    def store(buf, indices, values):
        assert buf.data_ptr() not in (tokens.data_ptr(), lengths.data_ptr())
        buf.index_copy_(0, indices, values)
        stores.append((indices.clone(), values.clone()))

    relay = SimpleNamespace(
        spec_algo=SimpleNamespace(is_none=lambda: True),
        device="cpu",
        output_tokens_buf=tokens,
        new_seq_lens_buf=lengths,
        req_pool_size=9,
        _store=store,
    )
    serving_warmup.warmup_request_relay(relay, 3)
    assert torch.equal(tokens, torch.full_like(tokens, -1))
    assert torch.equal(lengths, torch.full_like(lengths, -2))
    assert {indices.numel() for indices, _ in stores} == {1, 2, 3}


def test_output_sort_compilation_overlaps_dynamo_model_warmup():
    events = []
    runner = SimpleNamespace(
        is_draft_worker=False,
        pp_group=SimpleNamespace(is_last_rank=True),
        model_config=SimpleNamespace(is_generation=True, vocab_size=8),
        device="cpu",
        max_running_requests=2,
    )

    def compile_sort(logprobs, *, async_compile):
        assert async_compile
        bs = logprobs.shape[0]
        events.append(("submit_sort", bs))
        return PendingCall(lambda: events.append(("finish_sort", bs)))

    def init_graphs():
        assert len(current_warmup_queue().calls) == 2
        events.append("submit_model")
        current_warmup_queue().calls.append(
            PendingCall(lambda: events.append("finish_model"))
        )

    scheduler = SimpleNamespace(
        server_args=SimpleNamespace(
            unsafe_save_torch_compile_bundle=None,
            unsafe_load_torch_compile_bundle=None,
        ),
        tp_worker=SimpleNamespace(model_runner=runner, init_cuda_graphs=init_graphs),
        draft_worker=None,
    )
    platform = Mock()
    platform.is_tpu.return_value = True
    platform.precompile_sampling.side_effect = serving_warmup.precompile_sampling
    with (
        envs.SGLANG_DIRECT_COMPILE_ASYNC.override(True),
        patch.object(scheduler_module, "current_platform", platform),
        patch("torch.distributed.is_initialized", return_value=False),
        patch.object(
            serving_warmup,
            "get_spec",
            return_value=SimpleNamespace(speculative_algorithm=None),
        ),
        patch.object(
            serving_warmup,
            "get_exec",
            return_value=SimpleNamespace(
                graph=SimpleNamespace(
                    enable_torch_compile=True,
                    torch_compile_max_bs=2,
                    tpu_compile_tracer="dynamo",
                )
            ),
        ),
        patch.object(
            output_logprobs,
            "_compiled_sort",
            return_value=SimpleNamespace(precompile=compile_sort),
        ),
    ):
        Scheduler.init_all_cuda_graphs(scheduler)
    assert events == [
        ("submit_sort", 1),
        ("submit_sort", 2),
        "submit_model",
        ("finish_sort", 1),
        ("finish_sort", 2),
        "finish_model",
    ]


@pytest.mark.parametrize(
    "speculative_algorithm,enabled", [("EAGLE3", True), (None, False)]
)
def test_sort_precompile_does_not_expand_speculative_or_uncompiled_warmup(
    speculative_algorithm, enabled
):
    runner = SimpleNamespace(
        is_draft_worker=False,
        pp_group=SimpleNamespace(is_last_rank=True),
        model_config=SimpleNamespace(is_generation=True),
    )
    with (
        patch.object(
            serving_warmup,
            "get_spec",
            return_value=SimpleNamespace(speculative_algorithm=speculative_algorithm),
        ),
        patch.object(
            serving_warmup,
            "get_exec",
            return_value=SimpleNamespace(
                graph=SimpleNamespace(enable_torch_compile=enabled)
            ),
        ),
        patch.object(serving_warmup, "precompile_output_top_logprobs") as precompile,
    ):
        serving_warmup.precompile_sampling(runner)
    precompile.assert_not_called()
