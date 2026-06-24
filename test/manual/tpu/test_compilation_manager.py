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

# Mock get_tensor_model_parallel_rank for all tests in this module
# (TP group is not initialized in unit tests).
_tp_rank_patcher = None


def setUpModule():
    global _tp_rank_patcher
    _tp_rank_patcher = patch(
        "sglang.srt.hardware_backend.tpu.compilation.compilation_manager"
        ".get_tensor_model_parallel_rank",
        return_value=0,
    )
    _tp_rank_patcher.start()


def tearDownModule():
    if _tp_rank_patcher is not None:
        _tp_rank_patcher.stop()


class _FakeAttnBackend:
    """Records ``init_forward_metadata`` calls so tests can assert shape."""

    def __init__(self):
        self.calls: list[tuple[int, int]] = []  # (num_tokens, bs)

    def init_forward_metadata(self, forward_batch):
        self.calls.append(
            (forward_batch.input_ids.shape[0], forward_batch.seq_lens.shape[0])
        )


def _model_runner(device="cpu", eagle_hidden_size=0):
    """Minimal ModelRunner stand-in for unit tests.

    ``prepare_forward_batch`` is included because ``_run_one`` calls it
    before ``init_forward_metadata`` (mirrors the runtime chain installed
    by ``TpuCompiledForwardShim.install()``)."""
    inner_model = SimpleNamespace()
    if eagle_hidden_size:
        # EAGLE3 draft models have a ``model.fc`` with ``in_features``.
        inner_model.fc = SimpleNamespace(in_features=eagle_hidden_size)
    return SimpleNamespace(
        device=device,
        model=SimpleNamespace(
            forward=MagicMock(),
            prepare_forward_batch=MagicMock(),
            model=inner_model,
        ),
        model_config=SimpleNamespace(hidden_size=4096, dtype=torch.bfloat16),
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
    speculative_eagle_topk=1,
    eagle_hidden_size=0,
):
    if model_runner is None:
        model_runner = _model_runner(
            eagle_hidden_size=eagle_hidden_size if is_draft_worker else 0,
        )
    return CompilationManager(
        model_runner=model_runner,
        compiled_fns={"forward": model_runner.model.forward},
        t_bucketer=Bucketer(name="Token", buckets=list(token_buckets)),
        b_bucketer=Bucketer(name="Bs", buckets=list(bs_buckets)),
        speculative_num_steps=speculative_num_steps,
        speculative_algorithm=speculative_algorithm,
        is_draft_worker=is_draft_worker,
        speculative_eagle_topk=speculative_eagle_topk,
    )


class TestMakeDummyForwardBatch(CustomTestCase):
    """``_make_dummy_forward_batch`` shapes valid + ghost slots correctly."""

    def test_extend_shapes_and_ghost_tokens(self):
        mr = _model_runner()
        mgr = _make_manager(mr)
        fb = mgr._make_dummy_forward_batch(num_tokens=64, bs=4, mode=ForwardMode.EXTEND)
        # Token-axis tensors padded to num_tokens; arange-based content so
        # libtpu's sub-graph cache keys see runtime-like value patterns.
        self.assertEqual(fb.input_ids.shape[0], 64)
        self.assertEqual(fb.positions.shape[0], 64)
        self.assertEqual(fb.out_cache_loc.shape[0], 64)
        self.assertEqual(fb.input_ids[:4].tolist(), [1, 2, 3, 4])
        self.assertEqual(fb.positions[:4].tolist(), [0, 1, 2, 3])
        self.assertEqual(fb.input_ids.dtype, torch.int64)
        self.assertEqual(fb.positions.dtype, torch.int64)
        # Per-request tensors at bs.
        self.assertEqual(fb.seq_lens.shape[0], 4)
        self.assertEqual(fb.seq_lens.dtype, torch.int64)
        self.assertEqual(fb.seq_lens.tolist(), [2, 3, 4, 5])
        self.assertEqual(fb.extend_seq_lens.shape[0], 4)
        self.assertEqual(fb.extend_seq_lens.tolist(), [1, 1, 1, 1])
        self.assertEqual(fb.extend_logprob_start_lens_cpu, [0, 0, 0, 0])
        # KV writes start at slot 1 (not 0) and use distinct slot indices
        # matching real allocator output (arange-based).
        self.assertEqual(fb.out_cache_loc[:4].tolist(), [1, 2, 3, 4])
        self.assertEqual(fb.req_pool_indices.tolist(), [1, 2, 3, 4])

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

    def test_extend_compiles_token_buckets_at_batch_cap(self):
        mr = _model_runner()
        mgr = _make_manager(mr, token_buckets=(8, 64, 256), bs_buckets=(1, 4))
        mgr._precompile_extend()
        # Only the fixed batch cap (max_val=4); every token bucket >= 4.
        self.assertEqual(mr.model.forward.call_count, 3)
        self.assertEqual(
            mr.attn_backend.calls,
            [(8, 4), (64, 4), (256, 4)],
        )

    def test_extend_skips_token_bucket_below_bs(self):
        # A 4-token bucket can't hold bs=8.
        mr = _model_runner()
        mgr = _make_manager(mr, token_buckets=(4, 64), bs_buckets=(1, 8))
        mgr._precompile_extend()
        # Only the batch cap (max_val=8): (64,8); 4 < 8 skipped.
        self.assertEqual(mr.model.forward.call_count, 1)
        self.assertEqual(mr.attn_backend.calls, [(64, 8)])

    def test_extend_raises_when_no_valid_shapes(self):
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
        # EXTEND (batch cap 4): (8,4),(64,4) = 2. DECODE: (1,1),(4,4) = 2. Total = 4.
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

    def test_target_verify_does_not_set_extend_fields(self):
        # ``ForwardBatch.init_new`` takes the decode-or-verify branch for
        # TARGET_VERIFY and leaves ``extend_*`` as None (attention metadata
        # is rebuilt on the fly from ``spec_info.draft_token_num``). The
        # precompile dummy must match that to avoid Dynamo guard mismatches
        # between precompile (tensor) and runtime (None).
        mr = _model_runner()
        mgr = _make_manager(mr, speculative_num_steps=5)
        spec_info = SimpleNamespace(draft_token_num=6)
        fb = mgr._make_dummy_forward_batch(
            num_tokens=24,
            bs=4,
            mode=ForwardMode.TARGET_VERIFY,
            spec_info=spec_info,
        )
        # extend_* fields are None to mirror runtime.
        self.assertIsNone(fb.extend_seq_lens)
        self.assertIsNone(fb.extend_prefix_lens)
        self.assertIsNone(fb.extend_start_loc)
        self.assertIsNone(fb.extend_num_tokens)
        self.assertIsNone(fb.extend_seq_lens_cpu)
        self.assertIsNone(fb.extend_prefix_lens_cpu)
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
    """``_precompile_target_verify`` compiles the fixed batch cap with
    ``num_tokens = max_val * draft_token_num``."""

    def test_compiles_batch_cap_with_draft_token_num(self):
        mr = _model_runner()
        mgr = _make_manager(
            mr,
            token_buckets=(64,),
            bs_buckets=(1, 4, 8),
            speculative_num_steps=5,
        )
        mgr._precompile_target_verify()
        # draft_token_num = 5 + 1 = 6; only the batch cap (max_val=8): tokens=48.
        self.assertEqual(mr.model.forward.call_count, 1)
        self.assertEqual(
            mr.attn_backend.calls,
            [(48, 8)],
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

    def test_extend_fields_are_none_for_verify(self):
        # TARGET_VERIFY runtime takes the decode-or-verify branch in
        # ForwardBatch.init_new and leaves extend_* as None. Precompile
        # must mirror that to keep Dynamo guards aligned.
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
        self.assertIsNone(fb.extend_seq_lens)
        self.assertIsNone(fb.extend_start_loc)
        self.assertIsNone(fb.extend_num_tokens)


class TestPrecompileExtendCaptureHidden(CustomTestCase):
    """``_precompile_extend_full`` mirrors EXTEND but with
    ``CaptureHiddenMode.FULL``."""

    def test_iterates_same_grid_as_extend(self):
        mr = _model_runner()
        mgr = _make_manager(
            mr,
            token_buckets=(8, 64),
            bs_buckets=(1, 4),
            speculative_num_steps=3,
        )
        mgr._precompile_extend_full()
        # Same batch-cap policy as EXTEND (max_val=4): (8,4), (64,4).
        self.assertEqual(mr.model.forward.call_count, 2)
        self.assertEqual(
            mr.attn_backend.calls,
            [(8, 4), (64, 4)],
        )

    def test_capture_hidden_mode_is_full(self):
        mr = _model_runner()
        mgr = _make_manager(
            mr,
            token_buckets=(64,),
            bs_buckets=(1,),
            speculative_num_steps=3,
        )
        mgr._precompile_extend_full()
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
        mgr._precompile_extend_full()
        # Same skip rule as EXTEND, batch cap max_val=8: (64,8); 4 < 8 skipped.
        self.assertEqual(mr.model.forward.call_count, 1)
        self.assertEqual(mr.attn_backend.calls, [(64, 8)])


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
        # EXTEND (batch cap 4): (8,4) = 1.
        # DECODE: (1,1), (4,4) = 2.
        # TARGET_VERIFY (batch cap 4): tokens=16 = 1.
        # EXTEND(FULL) (batch cap 4): (8,4) = 1.
        # Total = 5.
        self.assertEqual(mr.model.forward.call_count, 5)

    def test_skips_spec_paths_when_no_speculative(self):
        mr = _model_runner()
        mgr = _make_manager(
            mr,
            token_buckets=(8,),
            bs_buckets=(1, 4),
            speculative_num_steps=0,
        )
        mgr.precompile_all()
        # EXTEND (batch cap 4): (8,4) = 1. DECODE: (1,1), (4,4) = 2. Total = 3.
        self.assertEqual(mr.model.forward.call_count, 3)

    def test_enable_spec_decoding_property(self):
        mgr_spec = _make_manager(speculative_num_steps=3)
        mgr_none = _make_manager(speculative_num_steps=0)
        self.assertTrue(mgr_spec.enable_spec_decoding)
        self.assertFalse(mgr_none.enable_spec_decoding)

    def test_draft_token_num_computed_correctly(self):
        mgr = _make_manager(speculative_num_steps=5)
        self.assertEqual(mgr.draft_token_num, 6)  # 5 + 1
        mgr0 = _make_manager(speculative_num_steps=0)
        self.assertEqual(mgr0.draft_token_num, 0)


class TestPrecompileDraftExtend(CustomTestCase):
    """``_precompile_draft_extend`` compiles one shape per bs bucket: the
    worst-case ``(bs * draft_token_num, bs)``."""

    def test_compiles_one_shape_per_bs_bucket(self):
        mr = _model_runner(eagle_hidden_size=256)
        mgr = _make_manager(
            mr,
            token_buckets=(8, 16, 64),
            bs_buckets=(1, 4),
            speculative_num_steps=3,
            speculative_algorithm="EAGLE3",
            is_draft_worker=True,
            eagle_hidden_size=256,
        )
        mgr._precompile_draft_extend()
        # draft_token_num = num_steps + 1 = 4.
        # bs=1 -> worst-case q = max(1*4, 1) = 4 -> smallest token bucket >= 4 is 8.
        # bs=4 -> worst-case q = max(4*4, 4) = 16 -> smallest token bucket >= 16 is 16.
        self.assertEqual(mr.model.forward.call_count, 2)
        self.assertEqual(mr.attn_backend.calls, [(8, 1), (16, 4)])

    def test_falls_back_to_max_bucket_when_no_bucket_fits(self):
        # bs=4, draft_token_num=4 -> worst-case q = 16, but largest
        # token bucket is 8. Falls back to max bucket (8); runtime pad
        # also clamps via next_bucket(... ) so shapes still align.
        mr = _model_runner(eagle_hidden_size=128)
        mgr = _make_manager(
            mr,
            token_buckets=(4, 8),
            bs_buckets=(1, 4),
            speculative_num_steps=3,
            speculative_algorithm="EAGLE3",
            is_draft_worker=True,
            eagle_hidden_size=128,
        )
        mgr._precompile_draft_extend()
        # bs=1 -> worst-case q = 4 -> bucket 4.
        # bs=4 -> worst-case q = 16 -> no bucket fits -> fall back to max=8.
        self.assertEqual(mr.model.forward.call_count, 2)
        self.assertEqual(mr.attn_backend.calls, [(4, 1), (8, 4)])

    def test_skips_single_bucket_when_single_bs(self):
        # b_buckets=(4,) -> exactly one DRAFT_EXTEND compile.
        mr = _model_runner(eagle_hidden_size=128)
        mgr = _make_manager(
            mr,
            token_buckets=(8, 16, 64),
            bs_buckets=(4,),
            speculative_num_steps=3,
            speculative_algorithm="EAGLE3",
            is_draft_worker=True,
            eagle_hidden_size=128,
        )
        mgr._precompile_draft_extend()
        self.assertEqual(mr.model.forward.call_count, 1)
        self.assertEqual(mr.attn_backend.calls, [(16, 4)])


class TestMakeDraftExtendSpecInfo(CustomTestCase):
    """``_make_draft_extend_spec_info`` builds a stub ``EagleDraftExtendInput``
    with shapes the EAGLE3 draft model's forward expects."""

    def test_hidden_states_shape_matches_eagle_fc_in_features(self):
        mr = _model_runner(eagle_hidden_size=384)
        mgr = _make_manager(
            mr,
            speculative_num_steps=3,
            speculative_algorithm="EAGLE3",
            is_draft_worker=True,
            eagle_hidden_size=384,
        )
        spec_info = mgr._make_draft_extend_spec_info(num_tokens=32, bs=8)
        # spec_info.hidden_states drives the draft model's fc projection;
        # token dim = num_tokens, feature dim = eagle_hidden_size.
        self.assertEqual(spec_info.hidden_states.shape, (32, 384))
        self.assertEqual(spec_info.hidden_states.dtype, torch.bfloat16)
        # Batch-shaped book-keeping tensors are bs-shaped.
        self.assertEqual(spec_info.num_correct_drafts.shape, (8,))
        self.assertEqual(spec_info.num_accept_tokens.shape, (8,))
        # num_accept_tokens carries the worst-case accept count so the
        # downstream extend kernel exercises the full draft_token_num path.
        self.assertEqual(spec_info.num_accept_tokens.tolist(), [4] * 8)
        self.assertEqual(spec_info.num_accept_tokens_cpu, [4] * 8)
        # Per-request tensors at bs.
        self.assertEqual(spec_info.seq_lens.shape, (8,))
        self.assertEqual(spec_info.req_pool_indices.shape, (8,))
        # input_ids matches the token dim.
        self.assertEqual(spec_info.input_ids.shape, (32,))


class TestDraftWorkerPrecompileAll(CustomTestCase):
    """``precompile_all`` on the draft worker compiles only DRAFT_EXTEND."""

    def test_draft_worker_with_spec_compiles_all_draft_modes(self):
        mr = _model_runner(eagle_hidden_size=256)
        mgr = _make_manager(
            mr,
            token_buckets=(8, 16, 64),
            bs_buckets=(1, 4),
            speculative_num_steps=3,
            speculative_algorithm="EAGLE3",
            is_draft_worker=True,
            eagle_hidden_size=256,
        )
        mgr.precompile_all()
        # Draft worker compiles three modes:
        # * EXTEND   (initial forward_draft_extend, bs=cap=4, tokens in {8,16,64}) = 3
        # * DECODE   (unrolled spec loop, one per bs bucket {1,4})                = 2
        # * DRAFT_EXTEND (one per bs bucket {1,4})                                = 2
        # TARGET_VERIFY and EXTEND(FULL) are target-only.
        self.assertEqual(mr.model.forward.call_count, 7)

    def test_draft_worker_without_spec_compiles_nothing(self):
        mr = _model_runner()
        mgr = _make_manager(
            mr,
            token_buckets=(8,),
            bs_buckets=(1, 4),
            speculative_num_steps=0,
            is_draft_worker=True,
        )
        mgr.precompile_all()
        self.assertEqual(mr.model.forward.call_count, 0)


if __name__ == "__main__":
    unittest.main()
