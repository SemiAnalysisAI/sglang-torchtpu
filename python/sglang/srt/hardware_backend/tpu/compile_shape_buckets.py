"""Token-axis bucketing wrapper for the TPU-compiled model forward.

The Pallas attention backend pads the *batch* axis of attention metadata
inside ``PallasAttnBackend._build_metadata`` (see
``sglang.srt.layers.attention.pallas_backend.SEQ_BUCKETS``). This module
handles the orthogonal *token* axis — the leading dimension of
``input_ids`` / ``positions`` that flows through embedding, every
transformer layer's matmul, and rotary embedding.

Padding ``input_ids`` and ``positions`` to one of a small set of buckets
keeps the compiled graph's input shape static across requests, so
``torch.compile(backend="tpu", fullgraph=True, dynamic=False)`` recompiles
only ``O(|TOKEN_BUCKETS| * |forward_modes|)`` times instead of once per
distinct request length.

Trailing padded tokens are harmless because:

* The Pallas RPA kernel consumes ``cu_q_lens`` to find each sequence's
  token range — padded tokens lie outside any range.
* The LogitsProcessor uses ``cumsum(extend_seq_lens) - 1`` to find each
  sequence's last token (extend), or just runs the LM head over all tokens
  (decode). In decode mode we slice the result back to the real batch.
"""

from __future__ import annotations

from typing import Callable

import torch


# Bucket set covering the typical SGLang chunked-prefill window
# (``--chunked-prefill-size`` defaults to 4096 on TPU). Exceed the largest
# bucket → a one-off recompile, but the scheduler is configured to cap
# chunks at 4096 tokens.
#
# Bucket 4 is retained as the smallest bucket so single-stream / small-
# concurrency (c<=4) workloads still get a near-tight specialisation
# rather than padding all the way to 16 (4x extra matmul work on each
# decode step). Buckets 1, 2, 8 are dropped: each ratio jump (4→16→32→...)
# wastes at most a factor of ~4 in matmul work for in-between batch
# sizes, which is acceptable, and we save the dynamo specialisations.
#
# We do not include sub-4 buckets and we keep ``>=16`` as the dense set
# above 4 to mirror torchtpu-vllm's ``min_token_size=16`` in
# ``_get_token_paddings`` (see
# ``torchtpu-vllm/tpu_inference/platforms/tpu_platform.py``).
# torchtpu-vllm's reasoning: shapes ``bf16[<8, 5120]`` can trip XLA's
# SparseCore-collective-offload pass on TP=8 / v7x-8
# (``Core index should be smaller than the number of logical devices per
# chip``, offload_collective_factory.cc:724). We have not reproduced
# that trap in our c=64 bench, but matching their bucket pattern avoids
# walking onto it as concurrency scales down.
TOKEN_BUCKETS: tuple[int, ...] = (
    4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096,
)


def _next_bucket(n: int) -> int:
    for b in TOKEN_BUCKETS:
        if n <= b:
            return b
    return TOKEN_BUCKETS[-1]


def _pad_token_axis(t: torch.Tensor, length: int) -> torch.Tensor:
    """Right-pad a 1-D tensor (token axis) with zeros to ``length``."""
    n = t.shape[0]
    if n == length:
        return t
    out = torch.zeros(length, *t.shape[1:], dtype=t.dtype, device=t.device)
    out[:n] = t
    return out


def _maybe_to_list(x):
    """tensor → Python list; pass through Python lists / None."""
    if isinstance(x, torch.Tensor):
        return x.tolist()
    return x


def _tensors_to_lists_nested(seq_lists):
    """Walk a list-shaped prefill-logprob field and .tolist() any tensors.

    The shape produced by LogitsProcessor.process_input_logprobs in PREFILL is
    ``List[<per-seq>]`` where each per-seq entry is either ``[]`` or a
    ``List[Tensor]`` (top-k path) / a single 2D ``Tensor`` (token-ids path).
    When the LogitsProcessor runs inside ``torch.compile`` on TPU, the inner
    ``.tolist()`` is deferred (see logprob.py); this finishes it after the
    compile boundary so downstream consumers see the same Python-list shape
    they get from the eager (GPU) path.
    """
    if seq_lists is None:
        return None
    out = []
    for entry in seq_lists:
        if isinstance(entry, list):
            out.append([_maybe_to_list(x) for x in entry])
        else:
            out.append(_maybe_to_list(entry))
    return out


_PREFILL_LOGPROB_FIELDS = (
    "input_top_logprobs_val",
    "input_top_logprobs_idx",
    "input_token_ids_logprobs_val",
    "input_token_ids_logprobs_idx",
)


def make_token_bucketed_forward(compiled_fn: Callable) -> Callable:
    """Wrap a compiled model.forward with token-axis bucket padding.

    The compiled function is called as
    ``compiled_fn(input_ids, positions, forward_batch, *args, **kwargs)``.

    For decode mode we slice ``next_token_logits`` (and ``hidden_states``
    if present) back to the real-token row count, so downstream samplers
    see the same shape they saw before.
    """

    def _forward(
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
        *args,
        **kwargs,
    ):
        real_n = int(input_ids.shape[0])
        padded_n = _next_bucket(real_n)
        if padded_n != real_n:
            input_ids = _pad_token_axis(input_ids, padded_n)
            positions = _pad_token_axis(positions, padded_n)

        out = compiled_fn(input_ids, positions, forward_batch, *args, **kwargs)

        if padded_n != real_n and forward_batch.forward_mode.is_decode():
            # Decode emits one logit row per token; trim the padded tail.
            if getattr(out, "next_token_logits", None) is not None:
                out.next_token_logits = out.next_token_logits[:real_n]
            if getattr(out, "hidden_states", None) is not None:
                out.hidden_states = out.hidden_states[:real_n]

        # Finish the deferred .tolist() for prefill input-logprob fields.
        for field in _PREFILL_LOGPROB_FIELDS:
            val = getattr(out, field, None)
            if val is not None:
                setattr(out, field, _tensors_to_lists_nested(val))

        return out

    return _forward
