"""Finite TPU logits shapes and preservation of logical request outputs."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.hardware_backend.tpu.compilation import decoder_compile_policy as m
from sglang.srt.hardware_backend.tpu.compilation.bucketer import Bucketer
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.runtime_context import get_context
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def runtime():
    with get_context().override_server_args(enable_torch_compile=True):
        yield


def make_policy(processor=None):
    owner = SimpleNamespace(get_input_embeddings=lambda: None)
    model = SimpleNamespace(
        get_image_feature=lambda: None,
        model=owner,
        logits_processor=processor,
        pp_group=SimpleNamespace(is_last_rank=True),
        lm_head=None,
    )
    runner = SimpleNamespace(
        model=model,
        model_config=SimpleNamespace(is_generation=True, hidden_size=4),
        spec_algorithm=SpeculativeAlgorithm.NONE,
        dtype=torch.float32,
        device="cpu",
    )
    policy = m.Qwen3VLDecoderTpuCompilePolicy(model_runner=runner, owner=owner)
    policy.b_bucketer = Bucketer("batch", [1, 2, 4])
    policy.t_bucketer = Bucketer("token", [8, 16])
    policy.extend_bs = 4
    return policy, model


def batch(mode, real_q=5, real_b=2, physical_q=8):
    fb = ForwardBatch(
        forward_mode=mode,
        batch_size=4,
        input_ids=torch.arange(physical_q),
        req_pool_indices=torch.arange(4),
        seq_lens=torch.ones(4, dtype=torch.int32),
        out_cache_loc=torch.arange(physical_q),
        seq_lens_sum=real_q,
        capture_hidden_mode=CaptureHiddenMode.NULL,
        extend_seq_lens=torch.tensor([3, 2, 0, 0], dtype=torch.int32),
        extend_seq_lens_cpu=[3, 2, 0, 0],
        mm_input_embeds=torch.zeros(physical_q, 4),
    )
    fb._tpu_real_q, fb._tpu_real_b = real_q, real_b
    return fb


@pytest.mark.parametrize(
    "mode", [ForwardMode.EXTEND, ForwardMode.MIXED, ForwardMode.DECODE]
)
def test_bucketed_hidden_states_preserve_real_logits_and_metadata(mode):
    import copy

    from sglang.srt.layers.logits_processor import LogitsMetadata

    processor = LogitsProcessor(SimpleNamespace(vocab_size=7))
    policy, model = make_policy(processor)
    head = torch.nn.Linear(4, 7, bias=False)
    fb = batch(mode) if mode != ForwardMode.DECODE else batch(mode, 3, 3, 4)
    original = copy.deepcopy(fb)
    hidden = (
        torch.arange(fb.input_ids.numel() * 4, dtype=torch.float32).reshape(-1, 4) / 11
    )
    expected_hidden = hidden[: fb._tpu_real_q]
    expected = processor(
        None,
        expected_hidden,
        head,
        LogitsMetadata(forward_mode=mode, extend_seq_lens=torch.tensor([3, 2])),
    ).next_token_logits
    actual_hidden = policy.finalize_forward_batch(hidden, fb)
    if mode != ForwardMode.DECODE:
        assert actual_hidden is hidden  # no length-specialized slice before gather
    else:
        assert actual_hidden.shape == (3, 4)
    actual = processor(fb.input_ids, actual_hidden, head, fb).next_token_logits
    torch.testing.assert_close(actual, expected)
    assert fb.batch_size == original._tpu_real_b
    assert fb.seq_lens.numel() == original._tpu_real_b
    assert fb.input_ids.numel() == original._tpu_real_q
    assert fb.mm_input_embeds.shape[0] == original._tpu_real_q


@pytest.mark.parametrize(
    "excluded",
    [
        "other_owner",
        "embedding",
        "nonfinal_pp",
        "speculative",
        "capture",
        "text_only",
        "logprob",
    ],
)
def test_excluded_paths_keep_existing_hidden_state_contract(excluded):
    policy, model = make_policy()
    fb = batch(ForwardMode.EXTEND)
    if excluded == "other_owner":
        policy.owner = object()
    elif excluded == "embedding":
        policy.model_runner.model_config.is_generation = False
    elif excluded == "nonfinal_pp":
        model.pp_group.is_last_rank = False
    elif excluded == "speculative":
        policy.model_runner.spec_algorithm = SpeculativeAlgorithm.EAGLE
    elif excluded == "logprob":
        fb.return_logprob = True
    elif excluded == "text_only":
        del model.get_image_feature
    else:
        fb.capture_hidden_mode = CaptureHiddenMode.FULL
    result = policy.finalize_forward_batch(torch.zeros(8, 4), fb)
    assert result.shape == (5, 4)
    assert fb.batch_size == 2


def test_warmup_covers_logical_batches_and_their_physical_views():
    seen = []

    def processor(ids, h, head, metadata):
        if metadata.forward_mode == ForwardMode.DECODE:
            bs = h.shape[0]
            physical = h._base.shape[0] if h._base is not None else bs
            seen.append(("decode", physical, bs, None))
        else:
            bs = metadata.extend_seq_lens.numel()
            base = metadata.extend_seq_lens._base
            seen.append(
                ("extend", h.shape[0], bs, base.numel() if base is not None else None)
            )
        return LogitsProcessorOutput(next_token_logits=torch.zeros(bs, 7))

    policy, _ = make_policy(processor)
    policy._precompile_eager_logits(ForwardMode.EXTEND)
    policy._precompile_eager_logits(ForwardMode.DECODE)
    assert seen == [
        ("extend", q, bs, 4 if bs < 4 else None) for q in [8, 16] for bs in range(1, 5)
    ] + [
        ("decode", 1, 1, None),
        ("decode", 2, 2, None),
        ("decode", 4, 3, None),
        ("decode", 4, 4, None),
    ]


def test_warmup_runs_the_real_logits_processor():
    processor = LogitsProcessor(SimpleNamespace(vocab_size=7))
    policy, model = make_policy(processor)
    model.lm_head = torch.nn.Linear(4, 7, bias=False)
    with patch("torch.argmax", wraps=torch.argmax) as argmax:
        policy._precompile_eager_logits(ForwardMode.EXTEND)
        policy._precompile_eager_logits(ForwardMode.DECODE)
    assert [call.args[0].shape for call in argmax.call_args_list] == [
        torch.Size([bs, 7]) for _ in range(3) for bs in range(1, 5)
    ]


@pytest.mark.parametrize("lengths,starts", [([3, 2], [0, 0]), ([2, 3], [2, 1])])
def test_prompt_logprob_pruning_uses_only_real_hidden_rows(lengths, starts):
    from sglang.srt.layers.logits_processor import LogitsMetadata

    processor = LogitsProcessor(SimpleNamespace(vocab_size=7))
    policy, _ = make_policy(processor)
    fb = batch(ForwardMode.EXTEND)
    fb.return_logprob = True
    hidden = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    actual_hidden = policy.finalize_forward_batch(hidden, fb)
    assert actual_hidden.shape == (5, 4)
    metadata = LogitsMetadata(
        forward_mode=ForwardMode.EXTEND,
        extend_return_logprob=True,
        extend_seq_lens_cpu=lengths,
        extend_logprob_start_lens_cpu=starts,
    )
    actual = processor._get_pruned_states(actual_hidden, None, None, metadata)
    expected = processor._get_pruned_states(hidden[:5], None, None, metadata)
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[3], expected[3])
