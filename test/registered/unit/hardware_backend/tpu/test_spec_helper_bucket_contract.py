"""Bucket padding must reuse graphs without changing live-request results."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.compilation.compiled_op import get_compiled_ops
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.tpu import eagle_utils_tpu as eagle
from sglang.srt.hardware_backend.tpu import overlap_utils_tpu as overlap
from sglang.srt.hardware_backend.tpu import spec_op_precompile
from sglang.srt.hardware_backend.tpu.spec_batch_padding import physical_batch_size
from sglang.srt.hardware_backend.tpu.spec_op_warmup import (
    SpecOpWarmupContext,
    make_draft_extend_prepare_args,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

BUCKETS = (4, 8)


class TestSpecHelperBucketContract(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.enterContext(
            patch.object(envs.SGLANG_ENABLE_TORCH_COMPILE, "get", return_value=True)
        )

    def test_compile_disabled_preserves_logical_batch_size(self):
        with patch.object(envs.SGLANG_ENABLE_TORCH_COMPILE, "get", return_value=False):
            for buckets in ((), BUCKETS):
                for bs in (0, 3, 4, 9):
                    with self.subTest(buckets=buckets, bs=bs):
                        self.assertEqual(physical_batch_size(bs, buckets), bs)

    def test_prepare_reuses_warmup_after_compilation_is_frozen(self):
        """Production warmup must cover padded runtime shapes and dtypes."""
        graphs = []

        def backend(graph, inputs):
            graphs.append(graph)
            return graph.forward

        compiled = torch.compile(
            eagle.fused_draft_extend_prepare_func.__wrapped__,
            backend=backend,
            fullgraph=True,
            dynamic=False,
        )
        for bs in BUCKETS:
            context = SpecOpWarmupContext(
                device=torch.device("cpu"),
                physical_bs=bs,
                num_verify_tokens=4,
                vocab_size=16,
                draft_vocab_size=16,
                hidden_size=2,
                hidden_dtype=torch.bfloat16,
                req_pool_size=10,
            )
            compiled(*make_draft_extend_prepare_args(context))
        self.assertEqual(len(graphs), len(BUCKETS))
        with (
            patch.object(eagle, "fused_draft_extend_prepare_func", compiled),
            torch.compiler.set_stance("fail_on_recompile"),
        ):
            for bs in (0, 3, 4, 5, 7, 8, 3, 0):
                with self.subTest(bs=bs):
                    seq = torch.arange(bs, dtype=torch.int64) + 10
                    accept = torch.arange(bs, dtype=torch.int32) % 4 + 1
                    tokens = torch.arange(bs * 4, dtype=torch.int32)
                    result = eagle.draft_extend_prepare_func(
                        seq, accept, tokens, 4, physical_batch_buckets=BUCKETS
                    )
                    self.assertEqual(result[0].tolist(), (accept - 1).tolist())
                    self.assertEqual(result[1].tolist(), tokens.tolist())
                    self.assertEqual(
                        result[2].tolist(),
                        [10 + i + j for i in range(bs) for j in range(4)],
                    )
                    self.assertEqual(result[3].tolist(), [4 * i for i in range(bs)])
                    self.assertEqual(
                        result[4].tolist(), [4 * i + i % 4 for i in range(bs)]
                    )
        self.assertEqual(len(graphs), len(BUCKETS))

    def test_store_padding_preserves_live_slots_and_reuses_graphs(self):
        graphs = []

        def backend(graph, inputs):
            graphs.append(graph)
            return graph.forward

        compiled = torch.compile(
            overlap.fused_future_map_store_tpu.__wrapped__,
            backend=backend,
            fullgraph=True,
            dynamic=False,
        )
        registered = [
            op
            for op in get_compiled_ops()
            if op.op is overlap.fused_future_map_store_tpu
        ]
        self.assertEqual(len(registered), 2)
        warmup_ops = tuple(
            SimpleNamespace(op=compiled, make_warmup_args=op.make_warmup_args)
            for op in registered
        )
        with patch.object(
            spec_op_precompile, "get_compiled_ops", return_value=warmup_ops
        ):
            spec_op_precompile.precompile_spec_v2_ops(
                device=torch.device("cpu"),
                physical_batch_buckets=BUCKETS,
                num_verify_tokens=4,
                vocab_size=16,
                draft_vocab_size=16,
                hidden_size=2,
                hidden_dtype=torch.bfloat16,
                req_pool_size=10,
            )
        self.assertEqual(len(graphs), 2 * len(BUCKETS))
        for with_hidden in (False, True):
            with self.subTest(with_hidden=with_hidden):
                fm = SimpleNamespace(
                    req_pool_size=10,
                    _tpu_precompiled_batch_sizes=BUCKETS,
                    need_hidden_states=with_hidden,
                    output_tokens_buf=torch.arange(11, dtype=torch.int64),
                    topk_p_buf=torch.arange(11, dtype=torch.float32).view(11, 1),
                    topk_index_buf=torch.arange(11, dtype=torch.int64).view(11, 1),
                    hidden_states_buf=(
                        torch.arange(22, dtype=torch.bfloat16).view(11, 2)
                        if with_hidden
                        else None
                    ),
                )
                with (
                    patch.object(overlap, "fused_future_map_store_tpu", compiled),
                    torch.no_grad(),
                    torch.compiler.set_stance("fail_on_recompile"),
                ):
                    for bs in (0, 3, 4, 5, 7, 8, 3, 0):
                        # Include slot zero and the last slot to catch unsafe dummy indices.
                        indices = torch.tensor(
                            [9, 0, 4, 2, 7, 1, 6, 8][:bs], dtype=torch.int64
                        )
                        payload = SimpleNamespace(
                            bonus_tokens=torch.arange(bs, dtype=torch.int32) + 100,
                            topk_p=torch.arange(bs, dtype=torch.float32).view(bs, 1)
                            + 20,
                            topk_index=torch.arange(bs, dtype=torch.int32).view(bs, 1)
                            + 30,
                            hidden_states=(
                                torch.arange(bs * 2, dtype=torch.bfloat16).view(bs, 2)
                                + 40
                                if with_hidden
                                else None
                            ),
                        )
                        old = (
                            fm.output_tokens_buf,
                            fm.topk_p_buf,
                            fm.topk_index_buf,
                            fm.hidden_states_buf,
                        )
                        values = (
                            payload.bonus_tokens,
                            payload.topk_p,
                            payload.topk_index,
                            payload.hidden_states,
                        )
                        result = overlap.store_future_map_tpu(
                            future_map=fm, indices=indices, payload=payload
                        )
                        for original, value, actual in zip(old, values, result):
                            if original is None:
                                self.assertIsNone(actual)
                                continue
                            expected = original.clone()
                            expected[indices] = value.to(expected.dtype)
                            self.assertEqual(actual.shape, original.shape)
                            torch.testing.assert_close(
                                actual[: fm.req_pool_size],
                                expected[: fm.req_pool_size],
                                rtol=0,
                                atol=0,
                            )
                        (
                            fm.output_tokens_buf,
                            fm.topk_p_buf,
                            fm.topk_index_buf,
                            fm.hidden_states_buf,
                        ) = result
                self.assertEqual(len(graphs), 2 * len(BUCKETS))

    def test_select_reuses_buckets_including_a_padded_idle_peer(self):
        graphs = []

        def backend(graph, inputs):
            graphs.append(graph)
            return graph.forward

        compiled = torch.compile(
            eagle.fused_draft_extend_select_argmax_func.__wrapped__,
            backend=backend,
            fullgraph=True,
            dynamic=False,
        )
        with patch.object(eagle, "fused_draft_extend_select_argmax_func", compiled):
            for logical_bs, input_bs in (
                (3, 3),
                (4, 4),
                (0, 8),
                (5, 8),
                (7, 7),
                (8, 8),
                (3, 4),
            ):
                with self.subTest(logical_bs=logical_bs, input_bs=input_bs):
                    logits = torch.arange(input_bs * 4 * 16, dtype=torch.float32).view(
                        input_bs * 4, 16
                    )
                    hidden = torch.arange(input_bs * 4 * 2, dtype=torch.bfloat16).view(
                        input_bs * 4, 2
                    )
                    select = torch.arange(logical_bs, dtype=torch.int64) * 4 + 3
                    result = eagle.draft_extend_select_argmax_func(
                        logits,
                        hidden,
                        select,
                        physical_batch_buckets=BUCKETS,
                        num_draft_tokens=4,
                    )
                    torch.testing.assert_close(result[0], logits[select])
                    torch.testing.assert_close(result[1], hidden[select])
                    self.assertEqual(result[2].tolist(), [[1.0]] * logical_bs)
                    self.assertEqual(result[3].tolist(), [[15]] * logical_bs)
        self.assertEqual(len(graphs), len(BUCKETS))

    def test_overflow_and_missing_compile_buckets_fail_before_dispatch(self):
        with self.assertRaisesRegex(AssertionError, "exceeds"):
            physical_batch_size(9, BUCKETS)
        with self.assertRaisesRegex(AssertionError, "Missing"):
            physical_batch_size(3, ())


if __name__ == "__main__":
    unittest.main()
