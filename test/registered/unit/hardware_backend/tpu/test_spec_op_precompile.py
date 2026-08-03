import torch

import sglang.srt.hardware_backend.tpu.spec_op_precompile as spec_op_precompile
from sglang.srt.hardware_backend.tpu.spec_op_precompile import (
    _build_op_calls,
    _find_fused_ops,
    precompile_spec_v2_ops,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def test_fused_spec_ops_are_discovered_from_decorator_registry():
    names = (
        "_finalize_topk1_tree",
        "greedy_verify_fused_func",
        "verify_post_fused_func",
        "draft_extend_select_argmax_fused_func",
        "store_future_map_spec_hidden",
        "_gather_spec_extras_hidden_and_cache_locs_tpu",
    )

    assert set(_find_fused_ops(names)) == set(names)


def test_op_calls_cover_both_serving_bonus_token_dtypes():
    calls = _build_op_calls(
        device=torch.device("cpu"),
        physical_bs=4,
        num_verify_tokens=4,
        num_spec_steps=3,
        verify_vocab_size=11,
        draft_vocab_size=7,
        verify_logits_dtype=torch.float32,
        draft_logits_dtype=torch.float32,
        hidden_size=4,
        hidden_dtype=torch.bfloat16,
        req_pool_size=32,
        warm_device_accept=True,
    )
    store_calls = [
        args for name, args in calls if name == "store_future_map_spec_hidden"
    ]
    calls_by_name = {name: args for name, args in calls}

    assert [args[5].dtype for args in store_calls] == [torch.int64, torch.int32]
    assert calls_by_name["greedy_verify_fused_func"][0].shape == (16, 11)
    assert calls_by_name["draft_extend_select_argmax_fused_func"][0].shape == (
        16,
        7,
    )


def test_precompile_spec_v2_ops_covers_physical_and_logical_shapes(monkeypatch):
    seen = {}

    def fake_find_ops(names):
        def make_op(name):
            def op(*args):
                seen.setdefault(name, []).append(args[0].shape[0])

            return op

        return {name: make_op(name) for name in names}

    monkeypatch.setattr(spec_op_precompile, "_find_fused_ops", fake_find_ops)
    shapes = precompile_spec_v2_ops(
        device=torch.device("cpu"),
        physical_batch_buckets=(1, 4, 16),
        num_verify_tokens=4,
        num_spec_steps=3,
        verify_vocab_size=8,
        draft_vocab_size=6,
        verify_logits_dtype=torch.float32,
        draft_logits_dtype=torch.float32,
        hidden_size=4,
        hidden_dtype=torch.bfloat16,
        req_pool_size=32,
        warm_device_accept=True,
    )

    assert shapes == ((1, 4), (4, 16), (16, 64))
    assert seen["_finalize_topk1_tree"] == list(range(1, 17))
    assert seen["draft_extend_select_argmax_fused_func"] == [
        4,
        16,
        16,
        16,
        *([64] * 12),
    ]
    assert seen["store_future_map_spec_hidden"] == [
        batch_size
        for batch_size in range(1, 17)
        for _bonus_dtype in range(2)
    ]
    assert seen["_gather_spec_extras_hidden_and_cache_locs_tpu"] == list(
        range(1, 17)
    )
