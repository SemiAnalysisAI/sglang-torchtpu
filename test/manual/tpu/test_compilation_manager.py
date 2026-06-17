"""Unit tests for the in-process TPU CompilationManager."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.hardware_backend.tpu.compilation.bucketer import Bucketer
from sglang.srt.hardware_backend.tpu.compilation.compilation_manager import (
    CompilationManager,
    _materialize_out,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from sglang.test.test_utils import CustomTestCase


class _FakeAttnBackend:
    """Records ``init_forward_metadata`` calls so tests can assert shape."""

    def __init__(self):
        self.calls: list[tuple[int, int]] = []  # (num_tokens, bs)

    def init_forward_metadata(self, forward_batch):
        self.calls.append(
            (forward_batch.input_ids.shape[0], forward_batch.seq_lens.shape[0])
        )


def _model_runner(device="cpu", spec_hidden_size=0):
    """Minimal ModelRunner stand-in for unit tests.

    ``prepare_forward_batch`` is included because ``_run_one`` calls it
    before ``init_forward_metadata`` (mirrors the runtime chain installed
    by ``TpuCompiledForwardShim.install()``)."""
    inner_model = SimpleNamespace()
    if spec_hidden_size:
        # EAGLE3 draft models have a ``model.fc`` with ``in_features``.
        inner_model.fc = SimpleNamespace(in_features=spec_hidden_size)
    return SimpleNamespace(
        device=device,
        model=SimpleNamespace(
            forward=MagicMock(),
            prepare_forward_batch=MagicMock(),
            model=inner_model,
        ),
        model_config=SimpleNamespace(hidden_size=4096),
        attn_backend=_FakeAttnBackend(),
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.zeros(4, 64, dtype=torch.int64),
        ),
        token_to_kv_pool=SimpleNamespace(),
    )


def _make_manager(
    model_runner=None,
    token_buckets=(64, 256),
    bs_buckets=(1, 4),
    speculative_num_steps=0,
    speculative_algorithm=None,
    is_draft_worker=False,
    topk=1,
    spec_hidden_size=0,
):
    if model_runner is None:
        model_runner = _model_runner(
            spec_hidden_size=spec_hidden_size if is_draft_worker else 0,
        )
    return CompilationManager(
        model_runner=model_runner,
        compiled_fns={"forward": model_runner.model.forward},
        t_bucketer=Bucketer(name="Token", buckets=list(token_buckets)),
        b_bucketer=Bucketer(name="Bs", buckets=list(bs_buckets)),
        speculative_num_steps=speculative_num_steps,
        speculative_algorithm=speculative_algorithm,
        is_draft_worker=is_draft_worker,
        topk=topk,
    )


class TestMakeDummyForwardBatch(CustomTestCase):
    """``_make_dummy_forward_batch`` shapes valid + ghost slots correctly."""

    def test_extend_shapes_and_ghost_tokens(self):
        mr = _model_runner()
        mgr = _make_manager(mr)
        fb = mgr._make_dummy_forward_batch(num_tokens=64, bs=4, mode=ForwardMode.EXTEND)
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
        # KV writes go to slot 1 (not 0) to avoid stale page-index
        # conflicts on the shared req_to_token_pool.
        self.assertTrue((fb.out_cache_loc == 1).all())
        self.assertTrue((fb.req_pool_indices == 1).all())

    def test_decode_has_no_extend_fields(self):
        mr = _model_runner()
        mgr = _make_manager(mr)
        fb = mgr._make_dummy_forward_batch(num_tokens=8, bs=8, mode=ForwardMode.DECODE)
        # Decode: one query token per running seq.
        self.assertEqual(fb.input_ids.shape[0], 8)
        self.assertEqual(fb.seq_lens.shape[0], 8)
        # Extend fields are None in decode (the path skips them).
        self.assertIsNone(fb.extend_seq_lens)
        self.assertIsNone(fb.extend_prefix_lens)
        self.assertIsNone(fb.extend_num_tokens)
        self.assertIsNone(fb.extend_logprob_start_lens_cpu)


class TestPrecompileLoops(CustomTestCase):
    """``precompile_all`` fires one forward per TPU precompile variant:
    EXTEND covers token buckets at the fixed batch cap, and DECODE varies bs
    (tokens == bs)."""

    def test_extend_iterates_token_buckets_at_fixed_batch_cap(self):
        mr = _model_runner()
        mgr = _make_manager(mr, token_buckets=(8, 64, 256), bs_buckets=(1, 4))
        mgr._precompile_extend()
        self.assertEqual(mr.model.forward.call_count, 3)
        self.assertEqual(
            mr.attn_backend.calls,
            [(8, 4), (64, 4), (256, 4)],
        )

    def test_extend_skips_token_bucket_below_bs(self):
        # A 4-token bucket can't hold the fixed 8-req extend cap.
        mr = _model_runner()
        mgr = _make_manager(mr, token_buckets=(4, 64), bs_buckets=(1, 8))
        mgr._precompile_extend()
        self.assertEqual(mr.model.forward.call_count, 1)
        self.assertEqual(mr.attn_backend.calls, [(64, 8)])

    def test_extend_raises_when_no_token_bucket_can_hold_bs_cap(self):
        mr = _model_runner()
        mgr = _make_manager(mr, token_buckets=(4,), bs_buckets=(8,))
        with self.assertRaisesRegex(ValueError, "No TPU extend compile shapes"):
            mgr._precompile_extend()

    def test_decode_loop_iterates_bs(self):
        mr = _model_runner()
        mgr = _make_manager(mr, token_buckets=(64,), bs_buckets=(1, 4, 8))
        mgr._precompile_decode()
        self.assertEqual(mr.model.forward.call_count, 3)
        # Decode: tokens == bs for each call.
        self.assertEqual(mr.attn_backend.calls, [(1, 1), (4, 4), (8, 8)])

    def test_precompile_all_combines_extend_and_decode(self):
        mr = _model_runner()
        mgr = _make_manager(mr, token_buckets=(8, 64), bs_buckets=(1, 4))
        mgr.precompile_all()
        # EXTEND: (8,4), (64,4) = 2. DECODE: (1,1), (4,4) = 2.
        self.assertEqual(mr.model.forward.call_count, 4)


class TestRunOneCallsCompiledForward(CustomTestCase):
    """``_run_one`` wires the synthesized batch through the same chain
    serving uses: ``prepare_forward_batch`` → ``init_forward_metadata``
    → compiled ``forward``."""

    def test_run_one_calls_prepare_then_metadata_then_forward(self):
        mr = _model_runner()
        mgr = _make_manager(mr)
        mgr._run_one(num_tokens=64, bs=4, mode=ForwardMode.EXTEND)
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
        mgr = _make_manager(mr)
        cm = MagicMock()
        with (
            patch(
                "sglang.srt.hardware_backend.tpu.compilation.compilation_manager.torch.no_grad",
                return_value=cm,
            ) as no_grad,
            patch(
                "sglang.srt.hardware_backend.tpu.compilation.compilation_manager.torch.inference_mode",
                side_effect=AssertionError("inference_mode should not be used"),
            ),
        ):
            mgr._run_one(num_tokens=64, bs=4, mode=ForwardMode.EXTEND)
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


class TestMakeDummyForwardBatchSpeculative(CustomTestCase):
    """Dummy batch construction for speculative decoding modes."""

    def test_target_verify_has_extend_fields_with_draft_token_num(self):
        mr = _model_runner()
        mgr = _make_manager(mr, speculative_num_steps=5)
        spec_info = SimpleNamespace(draft_token_num=6)
        fb = mgr._make_dummy_forward_batch(
            num_tokens=24,
            bs=4,
            mode=ForwardMode.TARGET_VERIFY,
            spec_info=spec_info,
        )
        # TARGET_VERIFY is extend-like, so extend fields must be present.
        self.assertIsNotNone(fb.extend_seq_lens)
        # Each request contributes draft_token_num tokens.
        self.assertEqual(fb.extend_seq_lens.tolist(), [6, 6, 6, 6])
        self.assertEqual(fb.extend_prefix_lens.tolist(), [0, 0, 0, 0])
        self.assertEqual(fb.extend_start_loc.tolist(), [0, 6, 12, 18])
        self.assertEqual(fb.extend_num_tokens, 24)
        self.assertEqual(fb.extend_seq_lens_cpu, [6, 6, 6, 6])
        self.assertEqual(fb.extend_prefix_lens_cpu, [0, 0, 0, 0])
        # Token-level tensors at num_tokens.
        self.assertEqual(fb.input_ids.shape[0], 24)
        self.assertEqual(fb.positions.shape[0], 24)
        # spec_info is passed through.
        self.assertIs(fb.spec_info, spec_info)
        self.assertEqual(fb.forward_mode, ForwardMode.TARGET_VERIFY)

    def test_extend_full_capture_hidden_mode(self):
        mr = _model_runner()
        mgr = _make_manager(mr, speculative_num_steps=5)
        fb = mgr._make_dummy_forward_batch(
            num_tokens=64,
            bs=4,
            mode=ForwardMode.EXTEND,
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )
        self.assertEqual(fb.capture_hidden_mode, CaptureHiddenMode.FULL)
        # Normal EXTEND fields: 1 token per request.
        self.assertEqual(fb.extend_seq_lens.tolist(), [1, 1, 1, 1])
        self.assertIsNone(fb.spec_info)

    def test_default_capture_hidden_mode_is_null(self):
        mr = _model_runner()
        mgr = _make_manager(mr)
        fb = mgr._make_dummy_forward_batch(
            num_tokens=64,
            bs=4,
            mode=ForwardMode.EXTEND,
        )
        self.assertEqual(fb.capture_hidden_mode, CaptureHiddenMode.NULL)


class TestPrecompileTargetVerify(CustomTestCase):
    """``_precompile_target_verify`` iterates bs buckets with
    ``num_tokens = bs * draft_token_num``."""

    def test_iterates_bs_buckets_with_draft_token_num(self):
        mr = _model_runner()
        mgr = _make_manager(
            mr,
            token_buckets=(64,),
            bs_buckets=(1, 4, 8),
            speculative_num_steps=5,
        )
        mgr._precompile_target_verify()
        # draft_token_num = 5 + 1 = 6.
        # Shapes: bs=1 -> tokens=6, bs=4 -> tokens=24, bs=8 -> tokens=48.
        self.assertEqual(mr.model.forward.call_count, 3)
        self.assertEqual(
            mr.attn_backend.calls,
            [(6, 1), (24, 4), (48, 8)],
        )

    def test_forward_batch_has_spec_info_and_full_capture(self):
        mr = _model_runner()
        mgr = _make_manager(
            mr,
            token_buckets=(64,),
            bs_buckets=(1,),
            speculative_num_steps=3,
        )
        mgr._precompile_target_verify()
        # Verify that the forward was called with a batch carrying spec_info
        # and CaptureHiddenMode.FULL (the EAGLE worker sets FULL during verify
        # so the target model captures hidden states for the draft model).
        args, _ = mr.model.forward.call_args
        fb = args[2]
        self.assertEqual(fb.forward_mode, ForwardMode.TARGET_VERIFY)
        self.assertIsNotNone(fb.spec_info)
        self.assertEqual(fb.spec_info.draft_token_num, 4)
        self.assertEqual(fb.capture_hidden_mode, CaptureHiddenMode.FULL)

    def test_extend_fields_populated_for_verify(self):
        mr = _model_runner()
        mgr = _make_manager(
            mr,
            token_buckets=(64,),
            bs_buckets=(2,),
            speculative_num_steps=3,
        )
        mgr._precompile_target_verify()
        args, _ = mr.model.forward.call_args
        fb = args[2]
        # draft_token_num=4, bs=2, so extend_seq_lens=[4, 4].
        self.assertEqual(fb.extend_seq_lens.tolist(), [4, 4])
        self.assertEqual(fb.extend_start_loc.tolist(), [0, 4])
        self.assertEqual(fb.extend_num_tokens, 8)


class TestPrecompileExtendCaptureHidden(CustomTestCase):
    """``_precompile_extend_capture_hidden`` mirrors EXTEND but with
    ``CaptureHiddenMode.FULL``."""

    def test_iterates_same_grid_as_extend(self):
        mr = _model_runner()
        mgr = _make_manager(
            mr,
            token_buckets=(8, 64),
            bs_buckets=(1, 4),
            speculative_num_steps=3,
        )
        mgr._precompile_extend_capture_hidden()
        # Same grid as EXTEND: (8,1), (64,1), (8,4), (64,4).
        self.assertEqual(mr.model.forward.call_count, 4)
        self.assertEqual(
            mr.attn_backend.calls,
            [(8, 1), (64, 1), (8, 4), (64, 4)],
        )

    def test_capture_hidden_mode_is_full(self):
        mr = _model_runner()
        mgr = _make_manager(
            mr,
            token_buckets=(64,),
            bs_buckets=(1,),
            speculative_num_steps=3,
        )
        mgr._precompile_extend_capture_hidden()
        args, _ = mr.model.forward.call_args
        fb = args[2]
        self.assertEqual(fb.capture_hidden_mode, CaptureHiddenMode.FULL)
        self.assertEqual(fb.forward_mode, ForwardMode.EXTEND)

    def test_skips_token_bucket_below_bs(self):
        mr = _model_runner()
        mgr = _make_manager(
            mr,
            token_buckets=(4, 64),
            bs_buckets=(1, 8),
            speculative_num_steps=3,
        )
        mgr._precompile_extend_capture_hidden()
        # Same skip rule as EXTEND: (4,1), (64,1), (64,8).
        self.assertEqual(mr.model.forward.call_count, 3)
        self.assertEqual(mr.attn_backend.calls, [(4, 1), (64, 1), (64, 8)])


class TestPrecompileAllWithSpeculative(CustomTestCase):
    """``precompile_all`` includes speculative paths when configured."""

    def test_includes_spec_paths_when_speculative(self):
        mr = _model_runner()
        mgr = _make_manager(
            mr,
            token_buckets=(8,),
            bs_buckets=(1, 4),
            speculative_num_steps=3,
        )
        mgr.precompile_all()
        # EXTEND: (8,1), (8,4) = 2.
        # DECODE: (1,1), (4,4) = 2.
        # TARGET_VERIFY: bs=1 -> tokens=4, bs=4 -> tokens=16 = 2.
        # EXTEND(FULL): (8,1), (8,4) = 2.
        # Total = 8.
        self.assertEqual(mr.model.forward.call_count, 8)

    def test_skips_spec_paths_when_no_speculative(self):
        mr = _model_runner()
        mgr = _make_manager(
            mr,
            token_buckets=(8,),
            bs_buckets=(1, 4),
            speculative_num_steps=0,
        )
        mgr.precompile_all()
        # EXTEND: (8,1), (8,4) = 2. DECODE: (1,1), (4,4) = 2. Total = 4.
        self.assertEqual(mr.model.forward.call_count, 4)

    def test_is_speculative_property(self):
        mgr_spec = _make_manager(speculative_num_steps=3)
        mgr_none = _make_manager(speculative_num_steps=0)
        self.assertTrue(mgr_spec._is_speculative)
        self.assertFalse(mgr_none._is_speculative)

    def test_draft_token_num_computed_correctly(self):
        mgr = _make_manager(speculative_num_steps=5)
        self.assertEqual(mgr.draft_token_num, 6)  # 5 + 1
        mgr0 = _make_manager(speculative_num_steps=0)
        self.assertEqual(mgr0.draft_token_num, 0)


class TestMakeDummyForwardBatchDraft(CustomTestCase):
    """Dummy batch construction for draft model modes."""

    def test_draft_extend_has_extend_fields_with_draft_token_num(self):
        mr = _model_runner()
        mgr = _make_manager(
            mr,
            speculative_num_steps=5,
            is_draft_worker=True,
        )
        fb = mgr._make_dummy_forward_batch(
            num_tokens=12,
            bs=2,
            mode=ForwardMode.DRAFT_EXTEND,
            capture_hidden_mode=CaptureHiddenMode.LAST,
        )
        # DRAFT_EXTEND is extend-like; draft_token_num = 6.
        self.assertIsNotNone(fb.extend_seq_lens)
        self.assertEqual(fb.extend_seq_lens.tolist(), [6, 6])
        self.assertEqual(fb.extend_start_loc.tolist(), [0, 6])
        self.assertEqual(fb.extend_num_tokens, 12)
        self.assertEqual(fb.forward_mode, ForwardMode.DRAFT_EXTEND)
        self.assertEqual(fb.capture_hidden_mode, CaptureHiddenMode.LAST)


class TestRunOneSkipPrepare(CustomTestCase):
    """``_run_one`` with ``skip_prepare=True`` skips prepare_forward_batch."""

    def test_skip_prepare_does_not_call_prepare_forward_batch(self):
        mr = _model_runner()
        mgr = _make_manager(mr, is_draft_worker=True, speculative_num_steps=3)
        mgr._run_one(
            num_tokens=4,
            bs=4,
            mode=ForwardMode.DECODE,
            skip_prepare=True,
        )
        mr.model.prepare_forward_batch.assert_not_called()
        # But attn_backend.init_forward_metadata IS still called.
        self.assertEqual(len(mr.attn_backend.calls), 1)
        mr.model.forward.assert_called_once()

    def test_no_skip_prepare_calls_prepare_forward_batch(self):
        mr = _model_runner()
        mgr = _make_manager(mr)
        mgr._run_one(num_tokens=64, bs=4, mode=ForwardMode.EXTEND)
        mr.model.prepare_forward_batch.assert_called_once()


class TestPrecompileExtendCaptureLast(CustomTestCase):
    """``_precompile_extend_capture_last`` mirrors EXTEND but with
    ``CaptureHiddenMode.LAST``."""

    def test_iterates_same_grid_as_extend(self):
        mr = _model_runner()
        mgr = _make_manager(
            mr,
            token_buckets=(8, 64),
            bs_buckets=(1, 4),
            is_draft_worker=True,
            speculative_num_steps=3,
        )
        mgr._precompile_extend_capture_last()
        self.assertEqual(mr.model.forward.call_count, 4)
        self.assertEqual(
            mr.attn_backend.calls,
            [(8, 1), (64, 1), (8, 4), (64, 4)],
        )

    def test_capture_hidden_mode_is_last(self):
        mr = _model_runner()
        mgr = _make_manager(
            mr,
            token_buckets=(64,),
            bs_buckets=(1,),
            is_draft_worker=True,
            speculative_num_steps=3,
        )
        mgr._precompile_extend_capture_last()
        args, _ = mr.model.forward.call_args
        fb = args[2]
        self.assertEqual(fb.capture_hidden_mode, CaptureHiddenMode.LAST)
        self.assertEqual(fb.forward_mode, ForwardMode.EXTEND)

    def test_calls_prepare_forward_batch(self):
        # EXTEND(LAST) runs through the normal path (no skip_prepare).
        mr = _model_runner()
        mgr = _make_manager(
            mr,
            token_buckets=(64,),
            bs_buckets=(1,),
            is_draft_worker=True,
            speculative_num_steps=3,
        )
        mgr._precompile_extend_capture_last()
        mr.model.prepare_forward_batch.assert_called_once()


class TestPrecompileDraftDecode(CustomTestCase):
    """``_precompile_draft_decode`` iterates bs buckets with
    ``tokens = bs * topk`` and ``skip_prepare=True``."""

    def test_iterates_bs_with_topk_1(self):
        mr = _model_runner()
        mgr = _make_manager(
            mr,
            token_buckets=(64,),
            bs_buckets=(1, 4, 8),
            is_draft_worker=True,
            speculative_num_steps=5,
            topk=1,
        )
        mgr._precompile_draft_decode()
        # topk=1: tokens=bs for each entry.
        self.assertEqual(mr.model.forward.call_count, 3)
        self.assertEqual(mr.attn_backend.calls, [(1, 1), (4, 4), (8, 8)])

    def test_iterates_bs_with_topk_2(self):
        mr = _model_runner()
        mgr = _make_manager(
            mr,
            token_buckets=(64,),
            bs_buckets=(1, 4),
            is_draft_worker=True,
            speculative_num_steps=5,
            topk=2,
        )
        mgr._precompile_draft_decode()
        # topk=2: tokens = bs * 2.
        self.assertEqual(mr.model.forward.call_count, 2)
        self.assertEqual(mr.attn_backend.calls, [(2, 1), (8, 4)])

    def test_capture_hidden_mode_is_last(self):
        mr = _model_runner()
        mgr = _make_manager(
            mr,
            token_buckets=(64,),
            bs_buckets=(1,),
            is_draft_worker=True,
            speculative_num_steps=3,
        )
        mgr._precompile_draft_decode()
        args, _ = mr.model.forward.call_args
        fb = args[2]
        self.assertEqual(fb.capture_hidden_mode, CaptureHiddenMode.LAST)
        self.assertEqual(fb.forward_mode, ForwardMode.DECODE)
        self.assertIsNotNone(fb.spec_info)

    def test_skips_prepare_forward_batch(self):
        mr = _model_runner()
        mgr = _make_manager(
            mr,
            token_buckets=(64,),
            bs_buckets=(1,),
            is_draft_worker=True,
            speculative_num_steps=3,
        )
        mgr._precompile_draft_decode()
        mr.model.prepare_forward_batch.assert_not_called()


class TestPrecompileDraftExtendAfterDecode(CustomTestCase):
    """``_precompile_draft_extend_after_decode`` iterates bs buckets with
    ``tokens = draft_token_num * bs``."""

    def test_iterates_bs_buckets(self):
        mr = _model_runner()
        mgr = _make_manager(
            mr,
            token_buckets=(64,),
            bs_buckets=(1, 4, 8),
            is_draft_worker=True,
            speculative_num_steps=5,
        )
        mgr._precompile_draft_extend_after_decode()
        # draft_token_num = 6. Shapes: bs=1->6, bs=4->24, bs=8->48.
        self.assertEqual(mr.model.forward.call_count, 3)
        self.assertEqual(mr.attn_backend.calls, [(6, 1), (24, 4), (48, 8)])

    def test_forward_batch_is_draft_extend(self):
        mr = _model_runner()
        mgr = _make_manager(
            mr,
            token_buckets=(64,),
            bs_buckets=(2,),
            is_draft_worker=True,
            speculative_num_steps=3,
        )
        mgr._precompile_draft_extend_after_decode()
        args, _ = mr.model.forward.call_args
        fb = args[2]
        self.assertEqual(fb.forward_mode, ForwardMode.DRAFT_EXTEND)
        self.assertEqual(fb.capture_hidden_mode, CaptureHiddenMode.LAST)
        # draft_token_num=4, bs=2: extend_seq_lens=[4,4]
        self.assertEqual(fb.extend_seq_lens.tolist(), [4, 4])
        self.assertEqual(fb.extend_num_tokens, 8)

    def test_skips_prepare_forward_batch(self):
        mr = _model_runner()
        mgr = _make_manager(
            mr,
            token_buckets=(64,),
            bs_buckets=(1,),
            is_draft_worker=True,
            speculative_num_steps=3,
        )
        mgr._precompile_draft_extend_after_decode()
        mr.model.prepare_forward_batch.assert_not_called()


class TestPrecompileAllDraftWorker(CustomTestCase):
    """``precompile_all`` dispatches to draft paths when is_draft_worker=True."""

    def test_draft_worker_runs_draft_paths(self):
        mr = _model_runner()
        mgr = _make_manager(
            mr,
            token_buckets=(8,),
            bs_buckets=(1, 4),
            is_draft_worker=True,
            speculative_num_steps=3,
        )
        mgr.precompile_all()
        # Draft precompilation skipped (cold-compile on first request).
        self.assertEqual(mr.model.forward.call_count, 0)

    def test_draft_worker_no_spec_is_noop(self):
        # is_draft_worker=True but speculative_num_steps=0 is a
        # degenerate state (no draft model without spec decoding).
        # precompile_all must be a no-op.
        mr = _model_runner()
        mgr = _make_manager(
            mr,
            token_buckets=(8,),
            bs_buckets=(1, 4),
            is_draft_worker=True,
            speculative_num_steps=0,
        )
        mgr.precompile_all()
        self.assertEqual(mr.model.forward.call_count, 0)

    def test_target_worker_does_not_run_draft_paths(self):
        # With is_draft_worker=False (default), draft paths are NOT run.
        mr = _model_runner()
        mgr = _make_manager(
            mr,
            token_buckets=(8,),
            bs_buckets=(1, 4),
            speculative_num_steps=3,
        )
        mgr.precompile_all()
        # Target paths: EXTEND: 2, DECODE: 2, TARGET_VERIFY: 2, EXTEND(FULL): 2.
        # Total = 8. (No EXTEND(LAST), DECODE(draft), or DRAFT_EXTEND.)
        self.assertEqual(mr.model.forward.call_count, 8)


class TestDraftSpecInfoHiddenStates(CustomTestCase):
    """Draft precompilation attaches ``hidden_states`` to ``spec_info``."""

    def test_make_draft_spec_info_shape(self):
        mr = _model_runner(spec_hidden_size=12288)
        mgr = _make_manager(
            mr,
            is_draft_worker=True,
            speculative_num_steps=3,
        )
        info = mgr._make_draft_spec_info(num_tokens=10)
        self.assertEqual(info.hidden_states.shape, (10, 12288))
        self.assertEqual(info.hidden_states.dtype, torch.bfloat16)
        # Must be the real EagleDraftInput type for Dynamo type guards.
        from sglang.srt.speculative.eagle_info import EagleDraftInput

        self.assertIsInstance(info, EagleDraftInput)

    def test_spec_hidden_size_from_fc(self):
        mr = _model_runner(spec_hidden_size=12288)
        mgr = _make_manager(
            mr,
            is_draft_worker=True,
            speculative_num_steps=3,
        )
        self.assertEqual(mgr.spec_hidden_size, 12288)

    def test_spec_hidden_size_fallback_to_model_config(self):
        # No fc layer -> falls back to model_config.hidden_size.
        mr = _model_runner(spec_hidden_size=0)
        mgr = _make_manager(
            mr,
            is_draft_worker=True,
            speculative_num_steps=3,
        )
        # model_config.hidden_size = 4096 (set in _model_runner).
        self.assertEqual(mgr.spec_hidden_size, 4096)

    def test_spec_hidden_size_zero_for_target(self):
        mr = _model_runner()
        mgr = _make_manager(mr, is_draft_worker=False, speculative_num_steps=3)
        self.assertEqual(mgr.spec_hidden_size, 0)

    def test_extend_last_has_hidden_states(self):
        mr = _model_runner(spec_hidden_size=256)
        mgr = _make_manager(
            mr,
            token_buckets=(64,),
            bs_buckets=(1,),
            is_draft_worker=True,
            speculative_num_steps=3,
            spec_hidden_size=256,
        )
        mgr._precompile_extend_capture_last()
        args, _ = mr.model.forward.call_args
        fb = args[2]
        self.assertIsNotNone(fb.spec_info)
        self.assertEqual(fb.spec_info.hidden_states.shape, (64, 256))

    def test_draft_decode_has_hidden_states(self):
        mr = _model_runner(spec_hidden_size=256)
        mgr = _make_manager(
            mr,
            token_buckets=(64,),
            bs_buckets=(4,),
            is_draft_worker=True,
            speculative_num_steps=3,
            spec_hidden_size=256,
        )
        mgr._precompile_draft_decode()
        args, _ = mr.model.forward.call_args
        fb = args[2]
        self.assertIsNotNone(fb.spec_info)
        # topk=1, bs=4 -> num_tokens=4.
        self.assertEqual(fb.spec_info.hidden_states.shape, (4, 256))

    def test_draft_extend_after_decode_has_hidden_states(self):
        mr = _model_runner(spec_hidden_size=256)
        mgr = _make_manager(
            mr,
            token_buckets=(64,),
            bs_buckets=(2,),
            is_draft_worker=True,
            speculative_num_steps=3,
            spec_hidden_size=256,
        )
        mgr._precompile_draft_extend_after_decode()
        args, _ = mr.model.forward.call_args
        fb = args[2]
        self.assertIsNotNone(fb.spec_info)
        # draft_token_num=4, bs=2 -> num_tokens=8.
        self.assertEqual(fb.spec_info.hidden_states.shape, (8, 256))


if __name__ == "__main__":
    unittest.main()
