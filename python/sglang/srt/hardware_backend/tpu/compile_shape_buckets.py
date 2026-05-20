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


# Bucket set for the *sequence-count* axis in the LogitsProcessor compiled
# region (``extend_seq_lens.shape[0]``, a.k.a. K — the number of sequences
# whose last-token logits get computed). Without bucketing, dynamo
# specialises ``LogitsProcessor.forward`` per distinct K value (1, 2, 3,
# 5, 6, 27, 32, ...), producing 10+ recompiles in a single c=64 bench.
# torchtpu-vllm avoids the same trap with ``_get_req_paddings`` rooted at
# ``MIN_NUM_SEQS = 8`` doubling to ``max_num_reqs``; mirror that here.
K_BUCKETS: tuple[int, ...] = (8, 16, 32, 64, 128, 256, 512)


def _next_k_bucket(k: int) -> int:
    for b in K_BUCKETS:
        if k <= b:
            return b
    return K_BUCKETS[-1]


def make_k_bucketed_lp_forward(compiled_fn: Callable) -> Callable:
    """Wrap a compiled ``LogitsProcessor.forward`` with K-axis bucket padding.

    The compiled function is dispatched as
    ``compiled_fn(input_ids, hidden_states, lm_head, logits_metadata, ...)``.
    Pads ``logits_metadata.extend_seq_lens`` (and reuses the existing
    in-LP ``last_index`` padding) to the next entry in ``K_BUCKETS``, then
    slices ``next_token_logits`` back to the real K on the way out.

    Why padding with zeros works: when ``extend_seq_lens`` is right-padded
    with zeros, ``cumsum(extend_seq_lens) - 1`` produces duplicate indices
    for every padded slot (each padded slot's cumulative count equals the
    last real slot's). ``hidden_states[last_index]`` therefore replicates
    the final real row into the padded slots — pure redundant compute
    that gets sliced away after the compile boundary. The matmul + TP
    all-gather inside the compiled region see only bucket-aligned shapes.
    """
    import dataclasses

    # Defer imports of sglang internals to first call so a bad import in a
    # helper module doesn't poison module load order during platform init.
    _IMPORTED = {"LogitsMetadata": None, "ForwardBatch": None}

    def _import_once():
        if _IMPORTED["LogitsMetadata"] is None:
            from sglang.srt.layers.logits_processor import LogitsMetadata
            from sglang.srt.model_executor.forward_batch_info import (
                ForwardBatch,
            )

            _IMPORTED["LogitsMetadata"] = LogitsMetadata
            _IMPORTED["ForwardBatch"] = ForwardBatch
        return _IMPORTED["LogitsMetadata"], _IMPORTED["ForwardBatch"]

    def _forward(
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        lm_head,
        logits_metadata,
        *args,
        **kwargs,
    ):
        LogitsMetadata, ForwardBatch = _import_once()

        # Normalize to LogitsMetadata so ``dataclasses.replace`` works.
        # For the rare multi-item-scoring path we fall back to passing the
        # original ForwardBatch through unchanged — Qwen3-32B does not use
        # it, and the path has its own scheduler-side shape handling.
        if isinstance(logits_metadata, ForwardBatch):
            if logits_metadata.multi_item_delimiter_indices is not None:
                return compiled_fn(
                    input_ids, hidden_states, lm_head, logits_metadata,
                    *args, **kwargs,
                )
            lm = LogitsMetadata.from_forward_batch(logits_metadata)
        else:
            lm = logits_metadata

        # Only bucket on the extend path. Decode shape is already bounded
        # by ``TOKEN_BUCKETS`` (decode batch == token count).
        if not lm.forward_mode.is_extend():
            return compiled_fn(input_ids, hidden_states, lm_head, lm,
                               *args, **kwargs)

        # The padded_static_len branch in _get_pruned_states computes
        # ``idx * padded_static_len + extend_seq_lens - 1`` and would
        # produce out-of-bounds indices for zero-padded positions. Sglang
        # chunked prefill leaves padded_static_len == -1 so this is rare,
        # but skip bucketing for safety when it's set.
        if getattr(lm, "padded_static_len", -1) > 0:
            return compiled_fn(input_ids, hidden_states, lm_head, lm,
                               *args, **kwargs)

        esl = lm.extend_seq_lens
        real_k = esl.shape[0]
        padded_k = _next_k_bucket(real_k)
        if padded_k <= real_k:
            return compiled_fn(input_ids, hidden_states, lm_head, lm,
                               *args, **kwargs)

        pad = padded_k - real_k
        zero_pad = torch.zeros(pad, dtype=esl.dtype, device=esl.device)
        padded_esl = torch.cat([esl, zero_pad])
        padded_lm = dataclasses.replace(lm, extend_seq_lens=padded_esl)

        out = compiled_fn(input_ids, hidden_states, lm_head, padded_lm,
                          *args, **kwargs)

        # Slice ``next_token_logits`` back to real K. Other LogitsProcessor
        # output fields are either per-token-bucket (sliced upstream by
        # ``make_token_bucketed_forward``) or not K-shaped, so leave them.
        if (
            out.next_token_logits is not None
            and out.next_token_logits.shape[0] == padded_k
        ):
            out.next_token_logits = out.next_token_logits[:real_k]
        return out

    return _forward


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
