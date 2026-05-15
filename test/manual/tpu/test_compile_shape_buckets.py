"""Unit tests for ``compile_shape_buckets`` — token-axis bucket padding.

CPU-only; no TPU dependency.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.hardware_backend.tpu.compile_shape_buckets import (
    TOKEN_BUCKETS,
    _next_bucket,
    _pad_token_axis,
    make_token_bucketed_forward,
)
from sglang.test.test_utils import CustomTestCase


class _DummyForwardMode:
    def __init__(self, mode):
        self._mode = mode

    def is_decode(self):
        return self._mode == "decode"


class _DummyOutput:
    """Mimics LogitsProcessorOutput.next_token_logits / hidden_states."""

    def __init__(self, next_token_logits=None, hidden_states=None):
        self.next_token_logits = next_token_logits
        self.hidden_states = hidden_states


class TestNextBucket(CustomTestCase):
    def test_exact_match_returns_same(self):
        for b in TOKEN_BUCKETS:
            self.assertEqual(_next_bucket(b), b)

    def test_rounds_up(self):
        self.assertEqual(_next_bucket(1), 1)
        self.assertEqual(_next_bucket(3), 4)
        self.assertEqual(_next_bucket(5), 8)
        self.assertEqual(_next_bucket(127), 128)
        self.assertEqual(_next_bucket(129), 256)

    def test_clamps_at_max(self):
        # Past the largest bucket falls back to the largest entry.
        self.assertEqual(_next_bucket(99999), TOKEN_BUCKETS[-1])


class TestPadTokenAxis(CustomTestCase):
    def test_1d_no_pad_needed(self):
        x = torch.arange(4, dtype=torch.int32)
        # When length matches, we return the same tensor (no copy).
        self.assertEqual(_pad_token_axis(x, 4).data_ptr(), x.data_ptr())

    def test_1d_pad_with_zeros(self):
        x = torch.tensor([5, 6, 7], dtype=torch.int32)
        out = _pad_token_axis(x, 8)
        self.assertEqual(out.shape, (8,))
        self.assertTrue(torch.equal(out[:3], x))
        self.assertTrue(torch.all(out[3:] == 0))


class TestTokenBucketedForward(CustomTestCase):
    def test_decode_slices_output_back_to_real_n(self):
        calls = []

        def fake_compiled(input_ids, positions, fb, *args, **kwargs):
            calls.append((input_ids.shape[0], positions.shape[0]))
            # Per-token logits, shape [n, vocab].
            logits = torch.arange(input_ids.shape[0] * 4, dtype=torch.float32).reshape(
                input_ids.shape[0], 4
            )
            return _DummyOutput(next_token_logits=logits)

        wrapped = make_token_bucketed_forward(fake_compiled)
        fb = SimpleNamespace(forward_mode=_DummyForwardMode("decode"))
        out = wrapped(
            torch.arange(3, dtype=torch.int32),
            torch.arange(3, dtype=torch.int32),
            fb,
        )
        # Inner call saw padded size (next bucket = 4)
        self.assertEqual(calls[0], (4, 4))
        # Output sliced back to the real-token count
        self.assertEqual(out.next_token_logits.shape, (3, 4))

    def test_extend_does_not_slice_output(self):
        """In extend mode the LM head only emits one row per real sequence
        (selected via ``extend_seq_lens.cumsum() - 1`` inside the model),
        so the wrapper must leave that output alone.
        """
        calls = []

        def fake_compiled(input_ids, positions, fb, *args, **kwargs):
            calls.append(input_ids.shape[0])
            # Per-sequence logits: shape [num_seqs, vocab]. The LM head decides
            # this size; the wrapper must not touch it.
            return _DummyOutput(next_token_logits=torch.zeros(2, 7))

        wrapped = make_token_bucketed_forward(fake_compiled)
        fb = SimpleNamespace(forward_mode=_DummyForwardMode("extend"))
        out = wrapped(
            torch.arange(5, dtype=torch.int32),
            torch.arange(5, dtype=torch.int32),
            fb,
        )
        self.assertEqual(calls[0], 8)  # padded to bucket 8
        self.assertEqual(out.next_token_logits.shape, (2, 7))  # untouched

    def test_call_with_exact_bucket_size_is_no_op(self):
        seen = []

        def fake_compiled(input_ids, positions, fb, *args, **kwargs):
            # Same tensor object — no copy.
            seen.append((input_ids.data_ptr(), positions.data_ptr()))
            return _DummyOutput(next_token_logits=torch.zeros(input_ids.shape[0], 4))

        wrapped = make_token_bucketed_forward(fake_compiled)
        fb = SimpleNamespace(forward_mode=_DummyForwardMode("decode"))
        ids = torch.arange(4, dtype=torch.int32)
        pos = torch.arange(4, dtype=torch.int32)
        wrapped(ids, pos, fb)
        self.assertEqual(seen[0], (ids.data_ptr(), pos.data_ptr()))


if __name__ == "__main__":
    unittest.main()
