"""Grouped row padding matches the per-tensor padding it replaces."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import copy
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.nn.functional as F

from sglang.srt.hardware_backend.tpu.compilation.bucketer import Bucketer
from sglang.srt.hardware_backend.tpu.compilation.spec_decode_compile_policy_eagle3 import (
    SpecDecoderTpuCompilePolicy,
    _row_pad_warm_batch_sizes,
)
from sglang.srt.hardware_backend.tpu.row_padding import RowPadder, _pad_rows_key
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.srt.speculative.eagle_info import EagleDraftExtendInput
from sglang.test.test_utils import CustomTestCase

_WIDTH = 4
_HIDDEN = 8
_ROW_PADDING = "sglang.srt.hardware_backend.tpu.row_padding."
_SPEC_POLICY = (
    "sglang.srt.hardware_backend.tpu.compilation.spec_decode_compile_policy_eagle3."
)


def _fake_tpu_tensor(*shape):
    t = MagicMock(spec=torch.Tensor)
    t.shape = torch.Size(shape)
    t.dtype = torch.int32
    t.device = SimpleNamespace(type="tpu")
    t.new_empty.side_effect = lambda new_shape: _fake_tpu_tensor(*new_shape)
    return t


def _fake_compiled():
    """A compiled pad that records each specialization in its cache."""

    def run(tensors, rows, fills):
        compiled.cache[_pad_rows_key(tensors=tensors, rows=rows, fills=fills)] = None
        return list(tensors)

    compiled = MagicMock(side_effect=run)
    compiled.cache = {}
    return compiled


def _shapes(call):
    return [tuple(t.shape) for t in call.kwargs["tensors"]], call.kwargs["rows"]


class TestRowPadder(CustomTestCase):
    def test_pads_each_tensor_with_its_fill(self):
        a = torch.arange(3, dtype=torch.int32)
        b = torch.ones(2, 5, dtype=torch.bfloat16)
        c = torch.arange(6).view(3, 2)
        out = RowPadder().pad(tensors=[a, b, c], rows=[5, 4, 4], fills=[7, 0, -1])
        self.assertEqual(out[0].tolist(), [0, 1, 2, 7, 7])
        self.assertEqual(out[0].dtype, torch.int32)
        self.assertEqual(out[1].shape, (4, 5))
        self.assertTrue(torch.equal(out[1][:2], b))
        self.assertTrue(torch.all(out[1][2:] == 0))
        self.assertEqual(out[2].tolist(), [[0, 1], [2, 3], [4, 5], [-1, -1]])

    def test_passes_through_values_it_does_not_widen(self):
        full = torch.arange(4)
        listed = [1, 2]
        out = RowPadder().pad(tensors=[None, full, listed], rows=[4, 4, 4])
        self.assertIsNone(out[0])
        self.assertIs(out[1], full)
        self.assertIs(out[2], listed)

    def test_pad_attrs_rebinds_only_widened_attributes(self):
        full = torch.arange(4)
        owner = SimpleNamespace(short=torch.arange(2), full=full, absent=None)
        RowPadder().pad_attrs(
            [
                (owner, "short", 4, 9),
                (owner, "full", 4, 0),
                (owner, "absent", 4, 0),
            ]
        )
        self.assertEqual(owner.short.tolist(), [0, 1, 9, 9])
        self.assertIs(owner.full, full)
        self.assertIsNone(owner.absent)

    def test_tpu_tensors_share_one_compiled_call(self):
        compiled = _fake_compiled()
        tpu = _fake_tpu_tensor(1)
        host = torch.arange(1)
        compiled.cache[_pad_rows_key(tensors=[tpu, tpu], rows=(2, 3), fills=(0, 1))] = (
            None
        )
        with patch(_ROW_PADDING + "_compiled_pad_rows", return_value=compiled):
            out = RowPadder().pad(
                tensors=[tpu, host, tpu], rows=[2, 2, 3], fills=[0, 5, 1]
            )
        compiled.assert_called_once_with(tensors=[tpu, tpu], rows=(2, 3), fills=(0, 1))
        self.assertEqual(out[1].tolist(), [0, 5])

    def test_first_call_per_layout_compiles_every_warm_batch_size(self):
        """A layout seen at one batch size must not compile again while serving."""
        compiled = _fake_compiled()
        eager = MagicMock(side_effect=lambda tensors, rows, fills: list(tensors))
        padder = RowPadder(warm_batch_sizes=[(1, 4), (3, 4), (5, 8)])
        with (
            patch(_ROW_PADDING + "_compiled_pad_rows", return_value=compiled),
            patch(_ROW_PADDING + "_pad_rows_eager", eager),
        ):
            padder.pad(
                tensors=[_fake_tpu_tensor(3), _fake_tpu_tensor(6, 2)],
                rows=[4, 8],
                batch_size=(3, 4),
            )
            self.assertEqual(
                [_shapes(call) for call in compiled.call_args_list],
                [
                    ([(1,), (2, 2)], (4, 8)),
                    ([(3,), (6, 2)], (4, 8)),
                    ([(5,), (10, 2)], (8, 16)),
                    ([(3,), (6, 2)], (4, 8)),
                ],
            )
            compiled.reset_mock()
            # Same layout at another warmed batch size: no further compiles.
            padder.pad(
                tensors=[_fake_tpu_tensor(5), _fake_tpu_tensor(10, 2)],
                rows=[8, 16],
                batch_size=(5, 8),
            )
            self.assertEqual(compiled.call_count, 1)
            eager.assert_not_called()

    def test_unwarmed_shapes_pad_eagerly(self):
        """A shape outside the warmed set must not compile while serving."""
        compiled = _fake_compiled()
        eager = MagicMock(side_effect=lambda tensors, rows, fills: list(tensors))
        padder = RowPadder(warm_batch_sizes=[(1, 4)])
        with (
            patch(_ROW_PADDING + "_compiled_pad_rows", return_value=compiled),
            patch(_ROW_PADDING + "_pad_rows_eager", eager),
        ):
            # Rows not a whole multiple of the batch size: nothing to warm.
            padder.pad(tensors=[_fake_tpu_tensor(3)], rows=[5], batch_size=(3, 4))
            # A DP peer can widen past the bucket of the warmed layout.
            padder.pad(tensors=[_fake_tpu_tensor(1)], rows=[4], batch_size=(1, 4))
            compiled.reset_mock()
            eager.reset_mock()
            padder.pad(tensors=[_fake_tpu_tensor(1)], rows=[8], batch_size=(1, 8))
            # Once compilation is frozen a new layout is not warmed either.
            with torch.compiler.set_stance("fail_on_recompile"):
                padder.pad(
                    tensors=[_fake_tpu_tensor(1, 3)], rows=[4], batch_size=(1, 4)
                )
        compiled.assert_not_called()
        self.assertEqual(eager.call_count, 2)

    def test_warm_batch_sizes_cover_every_padded_request_count(self):
        pairs = _row_pad_warm_batch_sizes([16, 64])
        self.assertEqual(len(pairs), 62)
        self.assertEqual(pairs[0], (1, 16))
        self.assertIn((15, 16), pairs)
        self.assertIn((17, 64), pairs)
        self.assertEqual(pairs[-1], (63, 64))
        self.assertNotIn(16, [rows for rows, _ in pairs])


def _draft_extend_batch(real_b, *, optional):
    """A DRAFT_EXTEND_V2 batch; ``optional`` populates every optional field."""
    tokens = real_b * _WIDTH
    seq_lens = torch.arange(real_b, dtype=torch.int64) + 70
    spec_info = EagleDraftExtendInput(
        hidden_states=torch.randn(tokens, _HIDDEN).to(torch.bfloat16),
        num_correct_drafts=torch.arange(real_b, dtype=torch.int32) % _WIDTH,
        num_accept_tokens=torch.arange(real_b, dtype=torch.int32) % _WIDTH + 1,
        num_tokens_per_req=_WIDTH,
        num_tokens_for_logprob_per_req=_WIDTH,
    )
    fields = dict(
        forward_mode=ForwardMode.DRAFT_EXTEND_V2,
        batch_size=real_b,
        input_ids=torch.arange(tokens) + 100,
        positions=torch.arange(tokens) + 60,
        req_pool_indices=torch.arange(real_b) + 3,
        req_pool_indices_cpu=torch.arange(real_b) + 3,
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens.clone(),
        seq_lens_sum=int(seq_lens.sum()),
        out_cache_loc=torch.arange(tokens) + 500,
        extend_seq_lens=torch.full((real_b,), _WIDTH, dtype=torch.int32),
        extend_seq_lens_cpu=[_WIDTH] * real_b,
        extend_prefix_lens=(seq_lens - _WIDTH).to(torch.int32),
        extend_prefix_lens_cpu=(seq_lens - _WIDTH).tolist(),
        extend_start_loc=torch.arange(0, tokens, _WIDTH, dtype=torch.int32),
        extend_num_tokens=tokens,
        spec_info=spec_info,
    )
    if optional:
        fields.update(
            input_embeds=torch.randn(tokens, _HIDDEN),
            encoder_lens=torch.arange(real_b, dtype=torch.int32) + 1,
            mamba_track_indices=torch.arange(real_b) + 1,
            mamba_track_mask=torch.ones(real_b, dtype=torch.bool),
            mamba_track_seqlens=torch.arange(real_b, dtype=torch.int32) + 2,
            mrope_positions=torch.arange(3 * tokens).view(3, tokens),
            extend_logprob_start_lens_cpu=[1] * real_b,
            rids_int=torch.arange(real_b) + 11,
            bootstrap_room_ids_int=torch.arange(real_b) + 21,
            lora_ids=["a"] * real_b,
            sampling_info=SimpleNamespace(rids_int=None, bootstrap_room_ids_int=None),
        )
    return ForwardBatch(**fields)


def _reference_pad_draft_extend_inputs(policy):
    """The shared ForwardBatch padding plus the extend-metadata pad it replaces."""

    def pad(*, forward_batch, num_tokens, bs):
        forward_batch._pad_inputs_to_size(
            model_runner=policy.model_runner, num_tokens=num_tokens, bs=bs
        )
        for field in ("extend_seq_lens", "extend_prefix_lens", "extend_start_loc"):
            value = getattr(forward_batch, field)
            if isinstance(value, torch.Tensor) and value.shape[0] < bs:
                setattr(forward_batch, field, F.pad(value, (0, bs - value.shape[0])))

    return pad


def _state(obj):
    """Every attribute of a batch or spec input, recursing into spec_info."""
    state = {}
    for name, value in vars(obj).items():
        if name in ("spec_info", "sampling_info") and value is not None:
            state.update({f"{name}.{k}": v for k, v in _state(value).items()})
        else:
            state[name] = value
    return state


class _PolicyCase(CustomTestCase):
    def setUp(self):
        super().setUp()
        override = get_context().override_server_args(
            page_size=1,
            dp_size=1,
            disaggregation_mode="null",
            speculative_algorithm="EAGLE3",
            speculative_num_steps=_WIDTH - 1,
            speculative_num_draft_tokens=_WIDTH,
            speculative_eagle_topk=1,
        )
        override.install()
        self.addCleanup(override.restore)

    def _policy(self, bs_buckets=(1, 4, 16, 64)):
        model_runner = SimpleNamespace(
            device="cpu",
            model=SimpleNamespace(forward=MagicMock()),
            # A nonzero fill checks that seq_lens uses the backend's value.
            attn_backend=SimpleNamespace(
                get_cuda_graph_seq_len_fill_value=lambda: 1,
                get_cpu_graph_seq_len_fill_value=lambda: 1,
            ),
            is_draft_worker=False,
        )
        with patch(
            _SPEC_POLICY + "process_model_config",
            return_value=SimpleNamespace(is_multimodal=False),
        ):
            return SpecDecoderTpuCompilePolicy(
                model_runner=model_runner,
                t_bucketer=Bucketer(name="Token", buckets=[64, 256]),
                b_bucketer=Bucketer(name="Bs", buckets=list(bs_buckets)),
                owner=model_runner.model,
            )

    def assert_same_state(self, actual, expected):
        actual, expected = _state(actual), _state(expected)
        self.assertEqual(sorted(actual), sorted(expected))
        for name, value in expected.items():
            with self.subTest(field=name):
                if isinstance(value, torch.Tensor):
                    self.assertEqual(actual[name].dtype, value.dtype)
                    self.assertTrue(torch.equal(actual[name], value))
                else:
                    self.assertEqual(actual[name], value)


class TestDraftExtendPaddingEquivalence(_PolicyCase):
    def _check(self, real_b, *, optional, dp_peer_tokens=None):
        policy = self._policy()
        batch = _draft_extend_batch(real_b, optional=optional)
        reference = copy.deepcopy(batch)
        dp = dp_peer_tokens is not None
        batch.global_num_tokens_cpu = reference.global_num_tokens_cpu = (
            [real_b * _WIDTH, dp_peer_tokens] if dp else None
        )
        with patch(_SPEC_POLICY + "is_dp_attention_enabled", return_value=dp):
            policy._pad_draft_extend(batch)
            with patch.object(
                policy,
                "_pad_draft_extend_inputs",
                side_effect=_reference_pad_draft_extend_inputs(policy),
            ):
                policy._pad_draft_extend(reference)
        self.assertGreater(batch.batch_size, real_b)
        self.assert_same_state(batch, reference)

    def test_matches_shared_padding(self):
        for real_b in (1, 5, 12, 16, 40):
            for optional in (False, True):
                with self.subTest(real_b=real_b, optional=optional):
                    if real_b in (1, 16):
                        # Already a bucket: only a DP peer can widen it.
                        self._check(real_b, optional=optional, dp_peer_tokens=64 * 4)
                    else:
                        self._check(real_b, optional=optional)

    def test_matches_shared_padding_for_dp_peer_width(self):
        self._check(5, optional=False, dp_peer_tokens=12 * _WIDTH)


if __name__ == "__main__":
    sys.exit(unittest.main())
