"""Unit tests for decoder TPU compile policy warmup."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.hardware_backend.tpu.compilation.bucketer import Bucketer
from sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy import (
    DecoderTpuCompilePolicy,
    _materialize_out,
)
from sglang.srt.hardware_backend.tpu.compilation.spec_decode_compile_policy import (
    SpecDecoderTpuCompilePolicy,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from sglang.srt.speculative.eagle_info import (
    EagleDraftExtendInput,
    EagleVerifyInput,
)
from sglang.srt.speculative.eagle_utils import verify_tree_greedy_from_logits
from sglang.srt.speculative.spec_info import AcceptIndices
from sglang.test.test_utils import CustomTestCase


class _FakeAttnBackend:
    """Records ``init_forward_metadata`` calls so tests can assert shape."""

    def __init__(self):
        self.calls: list[tuple[int, int]] = []  # (num_tokens, bs)

    def init_forward_metadata(self, forward_batch):
        self.calls.append(
            (forward_batch.input_ids.shape[0], forward_batch.seq_lens.shape[0])
        )


def _model_runner(device="cpu"):
    """Minimal ModelRunner stand-in for unit tests.

    ``prepare_forward_batch`` is included because ``_run_one`` calls it
    before ``init_forward_metadata`` (mirrors the runtime chain installed
    by ``DecoderTpuCompilePolicy.install()``)."""
    return SimpleNamespace(
        device=device,
        model=SimpleNamespace(
            forward=MagicMock(),
            prepare_forward_batch=MagicMock(),
        ),
        attn_backend=_FakeAttnBackend(),
        req_to_token_pool=SimpleNamespace(),
        token_to_kv_pool=SimpleNamespace(),
        server_args=SimpleNamespace(
            speculative_algorithm=None,
            page_size=1,
            dp_size=1,
        ),
    )


def _make_policy(model_runner=None, token_buckets=(64, 256), bs_buckets=(1, 4)):
    if model_runner is None:
        model_runner = _model_runner()
    t_bucketer = Bucketer(name="Token", buckets=list(token_buckets))
    b_bucketer = Bucketer(name="Bs", buckets=list(bs_buckets))
    return DecoderTpuCompilePolicy(
        model_runner=model_runner,
        t_bucketer=t_bucketer,
        b_bucketer=b_bucketer,
        owner=model_runner.model,
    )


def _target_and_padding(policy):
    return policy.owner, policy


def _run_cases(policy, mode):
    with patch(
        "sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy."
        "get_tensor_model_parallel_rank",
        return_value=1,
    ):
        if mode == ForwardMode.EXTEND:
            policy._precompile_extend()
        else:
            policy._precompile_decode()


class TestMakeDummyForwardBatch(CustomTestCase):
    """``_make_dummy_forward_batch`` shapes valid + ghost slots correctly."""

    def test_extend_shapes_and_ghost_tokens(self):
        mr = _model_runner()
        policy = _make_policy(mr)
        _, padding = _target_and_padding(policy)
        fb = padding._make_dummy_forward_batch(
            num_tokens=64, bs=4, mode=ForwardMode.EXTEND
        )
        # Token-axis tensors padded to num_tokens; bs=4 valid + 60 ghost.
        self.assertEqual(fb.input_ids.shape[0], 64)
        self.assertEqual(fb.positions.shape[0], 64)
        self.assertEqual(fb.out_cache_loc.shape[0], 64)
        self.assertEqual(fb.input_ids[:4].tolist(), [1, 1, 1, 1])
        self.assertEqual(fb.input_ids[4:].sum().item(), 0)
        self.assertEqual(fb.input_ids.dtype, torch.int64)
        self.assertEqual(fb.positions.dtype, torch.int64)
        # Per-request tensors at bs.
        self.assertEqual(fb.seq_lens.shape[0], 4)
        self.assertEqual(fb.seq_lens.dtype, torch.int64)
        self.assertEqual(fb.extend_seq_lens.shape[0], 4)
        self.assertEqual(fb.extend_seq_lens.tolist(), [1, 1, 1, 1])
        self.assertEqual(fb.extend_logprob_start_lens_cpu, [0, 0, 0, 0])
        # KV writes go to dummy slot 0 across the whole token range.
        self.assertEqual(fb.out_cache_loc.sum().item(), 0)

    def test_decode_has_no_extend_fields(self):
        mr = _model_runner()
        policy = _make_policy(mr)
        _, padding = _target_and_padding(policy)
        fb = padding._make_dummy_forward_batch(
            num_tokens=8, bs=8, mode=ForwardMode.DECODE
        )
        # Decode: one query token per running seq.
        self.assertEqual(fb.input_ids.shape[0], 8)
        self.assertEqual(fb.seq_lens.shape[0], 8)
        # Extend fields are None in decode (the path skips them).
        self.assertIsNone(fb.extend_seq_lens)
        self.assertIsNone(fb.extend_prefix_lens)
        self.assertIsNone(fb.extend_num_tokens)
        self.assertIsNone(fb.extend_logprob_start_lens_cpu)


class TestPrecompileLoops(CustomTestCase):
    """``precompile`` fires one forward per TPU precompile variant:
    EXTEND covers token buckets at the fixed batch cap, and DECODE varies bs
    (tokens == bs)."""

    def test_extend_iterates_token_buckets_at_fixed_batch_cap(self):
        mr = _model_runner()
        policy = _make_policy(mr, token_buckets=(8, 64, 256), bs_buckets=(1, 4))
        _run_cases(policy, ForwardMode.EXTEND)
        self.assertEqual(mr.model.forward.call_count, 3)
        self.assertEqual(
            mr.attn_backend.calls,
            [(8, 4), (64, 4), (256, 4)],
        )

    def test_extend_skips_token_bucket_below_bs(self):
        # A 4-token bucket can't hold the fixed 8-req extend cap.
        mr = _model_runner()
        policy = _make_policy(mr, token_buckets=(4, 64), bs_buckets=(1, 8))
        _run_cases(policy, ForwardMode.EXTEND)
        self.assertEqual(mr.model.forward.call_count, 1)
        self.assertEqual(mr.attn_backend.calls, [(64, 8)])

    def test_extend_raises_when_no_token_bucket_can_hold_bs_cap(self):
        mr = _model_runner()
        policy = _make_policy(mr, token_buckets=(4,), bs_buckets=(8,))
        with self.assertRaisesRegex(ValueError, "No TPU extend compile shapes"):
            _run_cases(policy, ForwardMode.EXTEND)

    def test_decode_loop_iterates_bs(self):
        mr = _model_runner()
        policy = _make_policy(mr, token_buckets=(64,), bs_buckets=(1, 4, 8))
        _run_cases(policy, ForwardMode.DECODE)
        self.assertEqual(mr.model.forward.call_count, 3)
        # Decode: tokens == bs for each call.
        self.assertEqual(mr.attn_backend.calls, [(1, 1), (4, 4), (8, 8)])

    def test_precompile_combines_extend_and_decode(self):
        mr = _model_runner()
        policy = _make_policy(mr, token_buckets=(8, 64), bs_buckets=(1, 4))
        with patch(
            "sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy."
            "get_tensor_model_parallel_rank",
            return_value=1,
        ):
            policy.precompile()
        # EXTEND: (8,4), (64,4) = 2. DECODE: (1,1), (4,4) = 2.
        self.assertEqual(mr.model.forward.call_count, 4)


class TestRunOneCallsCompiledForward(CustomTestCase):
    """``_run_one`` wires the synthesized batch through the same chain
    serving uses: ``prepare_forward_batch`` → ``init_forward_metadata``
    → compiled ``forward``."""

    def test_run_one_calls_prepare_then_metadata_then_forward(self):
        mr = _model_runner()
        policy = _make_policy(mr)
        policy._run_one(num_tokens=64, bs=4, mode=ForwardMode.EXTEND)
        # prepare_forward_batch runs first so model-specific prep hooks
        # (Moss-VL etc.) see precompile too.
        mr.model.prepare_forward_batch.assert_called_once()
        self.assertEqual(mr.attn_backend.calls, [(64, 4)])
        mr.model.forward.assert_called_once()
        # Args: (input_ids, positions, forward_batch). Shapes match bucket.
        args, _ = mr.model.forward.call_args
        self.assertEqual(args[0].shape, (64,))
        self.assertEqual(args[1].shape, (64,))
        self.assertEqual(args[2].batch_size, 4)

    def test_run_one_uses_no_grad_not_inference_mode(self):
        mr = _model_runner()
        policy = _make_policy(mr)
        cm = MagicMock()
        with patch(
            "sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy.torch.no_grad",
            return_value=cm,
        ) as no_grad, patch(
            "sglang.srt.hardware_backend.tpu.compilation."
            "decoder_compile_policy.torch.inference_mode",
            side_effect=AssertionError("inference_mode should not be used"),
        ):
            policy._run_one(num_tokens=64, bs=4, mode=ForwardMode.EXTEND)
        no_grad.assert_called_once_with()
        cm.__enter__.assert_called_once()

    def test_materialize_out_syncs_logits_and_flagged_kv_side_effects(self):
        logits = torch.zeros(1)
        kv0 = torch.ones(1)
        kv1 = torch.ones(1) * 2
        pool = SimpleNamespace(
            start_layer=2,
            layer_num=2,
            get_fused_kv_buffer=MagicMock(side_effect=[kv0, kv1]),
        )
        forward_batch = SimpleNamespace(
            attn_backend=SimpleNamespace(requires_kv_sync_after_compile=True),
            token_to_kv_pool=pool,
        )
        with patch("torch_tpu._internal.sync.sync.synchronize") as sync:
            _materialize_out(SimpleNamespace(next_token_logits=logits), forward_batch)
        sync.assert_called_once()
        tensors, kwargs = sync.call_args
        self.assertEqual(tensors[0], [logits, kv0, kv1])
        self.assertEqual(kwargs, {"wait": True})
        self.assertEqual(pool.get_fused_kv_buffer.call_args_list[0].args, (2,))
        self.assertEqual(pool.get_fused_kv_buffer.call_args_list[1].args, (3,))


def _spec_model_runner(device="cpu", is_draft_worker=False, num_steps=3):
    """ModelRunner stand-in with the spec config SpecDecoderTpuCompilePolicy reads."""
    return SimpleNamespace(
        device=device,
        model=SimpleNamespace(
            forward=MagicMock(),
            prepare_forward_batch=MagicMock(),
        ),
        attn_backend=_FakeAttnBackend(),
        # _run_one seeds req_to_token row 1; needs a real indexable tensor.
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.zeros(4, 4096, dtype=torch.int64)
        ),
        token_to_kv_pool=SimpleNamespace(),
        model_config=SimpleNamespace(hidden_size=16),
        dtype=torch.bfloat16,
        is_draft_worker=is_draft_worker,
        server_args=SimpleNamespace(
            speculative_num_steps=num_steps,
            # topk=1 chain: verify width == num_steps + 1.
            speculative_num_draft_tokens=num_steps + 1,
            speculative_algorithm="EAGLE3",
            speculative_eagle_topk=1,
        ),
    )


def _make_spec_policy(
    model_runner=None, token_buckets=(8, 64), bs_buckets=(1, 4), **mr_kwargs
):
    if model_runner is None:
        model_runner = _spec_model_runner(**mr_kwargs)
    return SpecDecoderTpuCompilePolicy(
        model_runner=model_runner,
        t_bucketer=Bucketer(name="Token", buckets=list(token_buckets)),
        b_bucketer=Bucketer(name="Bs", buckets=list(bs_buckets)),
        owner=model_runner.model,
    )


class TestSpecPrecompileLoops(CustomTestCase):
    """Target spec precompile fires EXTEND + DECODE + TARGET_VERIFY +
    EXTEND(FULL); the draft worker skips precompile entirely."""

    def test_target_precompiles_all_spec_modes(self):
        mr = _spec_model_runner(num_steps=3)  # draft_token_num = 4
        policy = _make_spec_policy(mr, token_buckets=(8, 64), bs_buckets=(1, 4))
        # precompile() spans two modules: the base EXTEND/DECODE warmup lives in
        # decoder_compile_policy (via super()), the spec TARGET_VERIFY/EXTEND(FULL)
        # in spec_decode_compile_policy. Patch the rank helper in both.
        with patch(
            "sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy."
            "get_tensor_model_parallel_rank",
            return_value=1,
        ), patch(
            "sglang.srt.hardware_backend.tpu.compilation.spec_decode_compile_policy."
            "get_tensor_model_parallel_rank",
            return_value=1,
        ):
            policy.precompile()
        # EXTEND (8,4),(64,4)=2  DECODE (1,1),(4,4)=2
        # TARGET_VERIFY buckets bs: (4,1),(16,4)=2  EXTEND(FULL) 2
        self.assertEqual(mr.model.forward.call_count, 8)
        self.assertIn((16, 4), mr.attn_backend.calls)
        self.assertIn((4, 1), mr.attn_backend.calls)

    def test_draft_worker_skips_precompile(self):
        mr = _spec_model_runner(is_draft_worker=True)
        policy = _make_spec_policy(mr)
        with patch(
            "sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy."
            "get_tensor_model_parallel_rank",
            return_value=1,
        ):
            policy.precompile()
        self.assertEqual(mr.model.forward.call_count, 0)
        self.assertEqual(mr.attn_backend.calls, [])

    def test_non_eagle3_algorithm_raises(self):
        # The spec policy enforces the supported algorithm at construction (only
        # EAGLE3 today); an unsupported algorithm fails loudly rather than
        # silently degrading to a non-spec policy.
        mr = _spec_model_runner()
        mr.server_args.speculative_algorithm = "NGRAM"
        with self.assertRaisesRegex(ValueError, "only supports EAGLE3"):
            _make_spec_policy(mr)


class TestSpecDummyForwardBatch(CustomTestCase):
    """Spec dummy batches carry spec_info and slot-1 KV seeding."""

    def test_target_verify_batch_shapes_and_capture_mode(self):
        policy = _make_spec_policy(num_steps=3)  # draft_token_num = 4
        spec_info = policy._make_verify_spec_info()
        fb = policy._make_dummy_forward_batch(
            num_tokens=16,
            bs=4,
            mode=ForwardMode.TARGET_VERIFY,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            spec_info=spec_info,
        )
        self.assertEqual(fb.capture_hidden_mode, CaptureHiddenMode.FULL)
        self.assertIsNotNone(fb.spec_info)
        # Runtime TARGET_VERIFY is decode-like and leaves extend metadata unset.
        self.assertIsNone(fb.extend_seq_lens)
        self.assertIsNone(fb.extend_prefix_lens)
        self.assertIsNone(fb.extend_num_tokens)
        # Slot-1 KV seeding (avoids draft/target pool conflicts): not slot 0.
        self.assertEqual(fb.out_cache_loc.tolist(), [1] * 16)
        self.assertEqual(fb.req_pool_indices.tolist(), [1, 1, 1, 1])
        # Text-only mm_inputs is [None]*bs, not None (Dynamo guard parity).
        self.assertEqual(fb.mm_inputs, [None] * 4)

    def test_plain_extend_carries_idle_spec_stub(self):
        # With spec enabled, non-verify modes still get a non-None spec stub so
        # Dynamo traces both branches of ``if spec_info is not None``.
        policy = _make_spec_policy()
        fb = policy._make_dummy_forward_batch(
            num_tokens=64,
            bs=4,
            mode=ForwardMode.EXTEND,
            spec_info=policy._make_null_spec_info(),
        )
        self.assertIsNotNone(fb.spec_info)


class TestSpecAcceptIndex(CustomTestCase):
    def test_unpadded_indices_gather_logical_rows(self):
        accept_indices = AcceptIndices.from_per_request(torch.tensor([[1, 3, 4]]))

        self.assertFalse(accept_indices.is_padded)
        self.assertEqual(
            accept_indices.gather(torch.arange(6)).tolist(), [1, 3, 4]
        )

    def test_unfinished_subset_is_packed_to_fixed_width(self):
        policy = _make_spec_policy(num_steps=3)
        accept_index_matrix = torch.tensor(
            [[0, 1, -1, -1], [4, -1, -1, -1]], dtype=torch.int32
        )
        accept_indices = policy.build_accepted_indices(accept_index_matrix)

        gathered = accept_indices.gather(torch.arange(8))

        self.assertIs(accept_indices.per_request, accept_index_matrix)
        self.assertEqual(gathered.tolist(), [0, 1, 4, 0, 0, 0, 0, 0])
        self.assertEqual(
            accept_indices.valid_mask.tolist(),
            [True, True, True, False, False, False, False, False],
        )


class TestSpecVerifyPadding(CustomTestCase):
    @staticmethod
    def _spec_info():
        return EagleVerifyInput.create_idle_input(
            topk=1,
            spec_steps=3,
            num_verify_tokens=4,
            device="cpu",
        )

    @staticmethod
    def _inputs(bs=2, draft_token_num=4, vocab_size=8):
        num_tokens = bs * draft_token_num
        return (
            torch.zeros((num_tokens, vocab_size)),
            torch.arange(num_tokens, dtype=torch.int32).reshape(bs, draft_token_num),
            torch.arange(num_tokens, dtype=torch.int32).reshape(bs, draft_token_num),
            torch.empty((num_tokens + 1,), dtype=torch.int32),
            torch.full((bs, draft_token_num), -1, dtype=torch.int32),
            torch.empty((bs,), dtype=torch.int32),
        )

    def test_pads_only_verify_tensors_to_batch_bucket(self):
        policy = _make_spec_policy(bs_buckets=(1, 4))

        logits, candidates, retrieve_index, predict, accept_index, nums = (
            policy.prepare_spec_verify(
                *self._inputs(bs=2), spec_info=self._spec_info()
            )
        )

        self.assertEqual(logits.shape, (16, 8))
        self.assertEqual(candidates.shape, (4, 4))
        self.assertEqual(retrieve_index.shape, (4, 4))
        self.assertEqual(predict.shape, (17,))
        self.assertEqual(accept_index.shape, (4, 4))
        self.assertEqual(nums.shape, (4,))
        self.assertTrue(torch.all(retrieve_index[2:] == -1))
        self.assertTrue(torch.all(accept_index == -1))

    def test_at_bucket_returns_the_original_tensors(self):
        policy = _make_spec_policy(bs_buckets=(1, 4))
        inputs = self._inputs(bs=4)

        outputs = policy.prepare_spec_verify(*inputs, spec_info=self._spec_info())

        self.assertTrue(
            all(output is input_ for output, input_ in zip(outputs, inputs))
        )

    def test_rejects_logits_candidate_shape_mismatch(self):
        policy = _make_spec_policy(bs_buckets=(1, 4))
        inputs = list(self._inputs(bs=2))
        inputs[0] = inputs[0][:-1]

        with self.assertRaisesRegex(ValueError, "one row per candidate"):
            policy.prepare_spec_verify(*inputs, spec_info=self._spec_info())

    def test_records_padding_state(self):
        policy = _make_spec_policy(bs_buckets=(1, 4))
        spec_info = self._spec_info()

        outputs = policy.prepare_spec_verify(
            *self._inputs(bs=2), spec_info=spec_info
        )

        state = spec_info.padding_state
        self.assertEqual(state.logical_batch_size, 2)
        self.assertEqual(state.physical_batch_size, 4)
        self.assertEqual(state.logical_num_tokens, 8)
        self.assertEqual(state.physical_num_tokens, 16)
        self.assertTrue(state.is_padded)
        self.assertTrue(state.outputs_padded)

        policy.finalize_spec_verify(
            outputs[3], outputs[4], outputs[5], 2, spec_info=spec_info
        )
        self.assertFalse(spec_info.padding_state.outputs_padded)

    def test_reuses_deferred_padded_logits(self):
        policy = _make_spec_policy(bs_buckets=(1, 4))
        spec_info = EagleVerifyInput.create_idle_input(
            topk=1,
            spec_steps=3,
            num_verify_tokens=4,
            device="cpu",
        )
        spec_info.record_padding(
            logical_batch_size=2,
            physical_batch_size=4,
            logical_num_tokens=8,
            physical_num_tokens=16,
            outputs_padded=True,
        )
        inputs = list(self._inputs(bs=2))
        padded_logits = torch.zeros((16, 8))
        inputs[0] = padded_logits

        outputs = policy.prepare_spec_verify(*inputs, spec_info=spec_info)

        self.assertIs(outputs[0], padded_logits)
        self.assertEqual(outputs[1].shape, (4, 4))
        self.assertEqual(outputs[2].shape, (4, 4))
        self.assertEqual(outputs[3].shape, (17,))

    def test_rejects_deferred_logits_from_a_different_bucket(self):
        policy = _make_spec_policy(bs_buckets=(1, 4))
        spec_info = EagleVerifyInput.create_idle_input(
            topk=1,
            spec_steps=3,
            num_verify_tokens=4,
            device="cpu",
        )
        spec_info.record_padding(
            logical_batch_size=2,
            physical_batch_size=3,
            logical_num_tokens=8,
            physical_num_tokens=12,
            outputs_padded=True,
        )
        inputs = list(self._inputs(bs=2))
        inputs[0] = torch.zeros((12, 8))

        with self.assertRaisesRegex(ValueError, "do not match the verify bucket"):
            policy.prepare_spec_verify(*inputs, spec_info=spec_info)

    def test_greedy_verify_operation_hides_deferred_padding(self):
        policy = _make_spec_policy(bs_buckets=(1, 4))
        spec_info = EagleVerifyInput.create_idle_input(
            topk=1,
            spec_steps=3,
            num_verify_tokens=4,
            device="cpu",
        )
        spec_info.record_padding(
            logical_batch_size=2,
            physical_batch_size=4,
            logical_num_tokens=8,
            physical_num_tokens=16,
            outputs_padded=True,
        )
        candidates = torch.tensor(
            [[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int32
        )
        logits = torch.zeros((16, 10))
        target_tokens = torch.tensor([2, 3, 4, 9, 6, 7, 8, 9])
        logits[torch.arange(8), target_tokens] = 1
        retrieve_index = torch.arange(8, dtype=torch.int32).reshape(2, 4)

        predict, accept_index, num_correct = verify_tree_greedy_from_logits(
            next_token_logits=logits,
            candidates=candidates,
            retrieve_index=retrieve_index,
            retrieve_next_token=torch.empty(0, dtype=torch.int32),
            retrieve_next_sibling=torch.empty(0, dtype=torch.int32),
            spec_steps=3,
            topk=1,
            compile_policy=policy,
            spec_info=spec_info,
        )

        self.assertEqual(accept_index.tolist(), [[0, 1, 2, 3], [4, 5, 6, 7]])
        self.assertEqual(num_correct.tolist(), [3, 3])
        self.assertEqual(predict[:8].tolist(), target_tokens.tolist())
        self.assertFalse(spec_info.padding_state.outputs_padded)

    def test_deferred_unpad_survives_padding_update(self):
        spec_info = EagleVerifyInput.create_idle_input(
            topk=1,
            spec_steps=3,
            num_verify_tokens=4,
            device="cpu",
        )
        spec_info.record_padding(
            logical_batch_size=2,
            physical_batch_size=2,
            logical_num_tokens=5,
            physical_num_tokens=8,
            outputs_padded=False,
            defer_output_unpad=True,
        )
        state = spec_info.record_padding(
            logical_batch_size=2,
            physical_batch_size=4,
            logical_num_tokens=5,
            physical_num_tokens=16,
            outputs_padded=True,
        )
        self.assertEqual(state.logical_num_tokens, 5)
        self.assertEqual(state.physical_num_tokens, 16)
        self.assertTrue(state.defer_output_unpad)

        spec_info.mark_outputs_unpadded()
        state = spec_info.record_padding(
            logical_batch_size=3,
            physical_batch_size=4,
            logical_num_tokens=12,
            physical_num_tokens=16,
            outputs_padded=True,
        )
        self.assertEqual(state.logical_batch_size, 3)
        self.assertEqual(state.logical_num_tokens, 12)
        self.assertFalse(state.defer_output_unpad)

    def test_mark_outputs_unpadded_clears_defer_intent(self):
        spec_info = EagleVerifyInput.create_idle_input(
            topk=1,
            spec_steps=3,
            num_verify_tokens=4,
            device="cpu",
        )
        spec_info.record_padding(
            logical_batch_size=2,
            physical_batch_size=4,
            logical_num_tokens=8,
            physical_num_tokens=16,
            outputs_padded=True,
            defer_output_unpad=True,
        )

        spec_info.mark_outputs_unpadded()

        self.assertFalse(spec_info.padding_state.outputs_padded)
        self.assertFalse(spec_info.padding_state.defer_output_unpad)


class TestSpecUnpadOutputs(CustomTestCase):
    """Spec output slicing: TARGET_VERIFY logits use real_q; capture-LAST hidden
    uses real_b; everything else matches the base decoder policy."""

    def _out_and_batch(self, forward_mode, capture, real_q, real_b, rows):
        out = SimpleNamespace(
            next_token_logits=torch.zeros(rows, 8),
            hidden_states=torch.zeros(rows, 8),
        )
        fb = SimpleNamespace(
            forward_mode=forward_mode,
            capture_hidden_mode=capture,
            batch_size=rows,
            spec_info=None,
            _tpu_real_q=real_q,
            _tpu_real_b=real_b,
        )
        return out, fb

    def test_target_verify_slices_logits_to_real_q(self):
        policy = _make_spec_policy()
        out, fb = self._out_and_batch(
            ForwardMode.TARGET_VERIFY,
            CaptureHiddenMode.FULL,
            real_q=16,
            real_b=4,
            rows=32,
        )
        policy.finalize_forward_outputs(out, fb)
        # Verify emits one logit per draft token -> real_q, not real_b.
        self.assertEqual(out.next_token_logits.shape[0], 16)
        self.assertEqual(out.hidden_states.shape[0], 16)  # FULL -> real_q
        self.assertEqual(fb.batch_size, 4)

    def test_capture_last_slices_hidden_to_real_b(self):
        policy = _make_spec_policy()
        out, fb = self._out_and_batch(
            ForwardMode.DRAFT_EXTEND,
            CaptureHiddenMode.LAST,
            real_q=8,
            real_b=4,
            rows=16,
        )
        policy.finalize_forward_outputs(out, fb)
        self.assertEqual(out.next_token_logits.shape[0], 4)  # not verify -> real_b
        self.assertEqual(out.hidden_states.shape[0], 4)  # LAST -> real_b

    def test_decode_default_slicing(self):
        policy = _make_spec_policy()
        out, fb = self._out_and_batch(
            ForwardMode.DECODE,
            CaptureHiddenMode.NULL,
            real_q=4,
            real_b=4,
            rows=8,
        )
        policy.finalize_forward_outputs(out, fb)
        self.assertEqual(out.next_token_logits.shape[0], 4)
        self.assertEqual(out.hidden_states.shape[0], 4)

    def test_final_draft_outputs_restore_real_batch_size(self):
        policy = _make_spec_policy()
        fb = SimpleNamespace(_tpu_real_b=2, batch_size=4, spec_info=None)
        outputs = (torch.arange(4), torch.arange(8).reshape(4, 2))

        unpadded = policy.finalize_spec_forward_batch(outputs, fb)

        self.assertEqual([t.shape[0] for t in unpadded], [2, 2])
        self.assertEqual(fb.batch_size, 2)


class TestSpecV1ForwardBatchPreparation(CustomTestCase):
    @staticmethod
    def _policy():
        policy = _make_spec_policy()
        policy._pad_draft_decode = MagicMock()
        policy._pad_draft_extend = MagicMock()
        return policy

    def test_draft_decode_dispatches(self):
        policy = self._policy()
        forward_batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)

        policy.prepare_spec_forward_batch(forward_batch)

        policy._pad_draft_decode.assert_called_once_with(forward_batch)

    def test_draft_extend_dispatches(self):
        policy = self._policy()
        forward_batch = SimpleNamespace(forward_mode=ForwardMode.DRAFT_EXTEND)

        policy.prepare_spec_forward_batch(forward_batch)

        policy._pad_draft_extend.assert_called_once_with(forward_batch)

    def test_idle_is_noop(self):
        policy = self._policy()
        forward_batch = SimpleNamespace(forward_mode=ForwardMode.IDLE)

        policy.prepare_spec_forward_batch(forward_batch)

        policy._pad_draft_decode.assert_not_called()
        policy._pad_draft_extend.assert_not_called()

    def test_target_verify_is_noop(self):
        policy = self._policy()
        forward_batch = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)

        policy.prepare_spec_forward_batch(forward_batch)

        policy._pad_draft_decode.assert_not_called()
        policy._pad_draft_extend.assert_not_called()

    def test_draft_extend_padding_is_internal_to_prepare(self):
        policy = _make_spec_policy(num_steps=3)
        di = EagleDraftExtendInput(
            hidden_states=torch.zeros((5, 8)),
            num_correct_drafts=torch.tensor([2, 1], dtype=torch.int32),
            num_accept_tokens=torch.tensor([3, 2], dtype=torch.int32),
            num_accept_tokens_cpu=[3, 2],
            input_ids=torch.arange(5),
            seq_lens=torch.tensor([8, 7], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([8, 7], dtype=torch.int32),
            req_pool_indices=torch.tensor([0, 1]),
            positions=torch.arange(5),
        )
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.DRAFT_EXTEND,
            spec_info=di,
            input_ids=di.input_ids,
            positions=di.positions,
            out_cache_loc=torch.arange(5),
            seq_lens=di.seq_lens,
        )

        def generic_prepare(fb):
            # bs=2 maps to bucket 4, so draft-extend tokens map directly to
            # 4 * (spec_steps + 1) = 16 without an intermediate 8-token pad.
            fb.input_ids = torch.nn.functional.pad(fb.input_ids, (0, 11))
            fb.positions = torch.nn.functional.pad(fb.positions, (0, 11))
            fb.out_cache_loc = torch.nn.functional.pad(fb.out_cache_loc, (0, 11))
            fb.seq_lens = torch.nn.functional.pad(fb.seq_lens, (0, 2))

        policy.model_runner.model.prepare_forward_batch.side_effect = generic_prepare

        policy.prepare_spec_forward_batch(forward_batch)

        self.assertEqual(forward_batch.input_ids.shape[0], 16)
        self.assertEqual(forward_batch.positions.shape[0], 16)
        self.assertEqual(forward_batch.out_cache_loc.shape[0], 16)
        self.assertEqual(di.hidden_states.shape[0], 16)
        self.assertIs(di.input_ids, forward_batch.input_ids)
        self.assertEqual(di.padding_state.logical_num_tokens, 5)
        self.assertEqual(di.padding_state.physical_num_tokens, 16)
        self.assertEqual(di.padding_state.physical_batch_size, 4)
        policy.model_runner.model.prepare_forward_batch.assert_called_once_with(
            forward_batch
        )


if __name__ == "__main__":
    unittest.main()
