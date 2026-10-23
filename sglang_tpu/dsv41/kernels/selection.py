# SPDX-License-Identifier: Apache-2.0
"""One BF16 score/streaming-selection pipeline for SGLang V4.1 batches.

The scorer, selector and bounded overlap schedule come from EPIC bdb38700.
This adapter supplies SGLang request rows and one candidate publication contract.
"""

import jax.numpy as jnp
from jax import lax

from sglang_tpu.kernels.index_selection.prefill_scores import prefill_index_scores
from sglang_tpu.kernels.index_selection.streaming_topk import streaming_topk

from .cache_rows import split_halves
from .indexer import pipeline_score_topk_bounded


def select_scores(
    scores,
    lengths,
    candidates,
    *,
    topk,
    block_size,
    publish,
    consume,
    longest,
    scheduling_group_id=None,
):
    """Apply causal visibility, publish blocks, and select a compact token set."""
    width = scores.shape[1]
    logical = jnp.arange(width, dtype=jnp.int32)
    scores = jnp.where(logical[None, :] < lengths[:, None], scores, -jnp.inf)
    blocks = (width + block_size - 1) // block_size
    if publish:
        padded = jnp.pad(
            scores, ((0, 0), (0, -width % block_size)), constant_values=-jnp.inf
        )
        maxima = padded.reshape(scores.shape[0], blocks, block_size).max(-1)
        last = (lengths - 1) // block_size
        # The newest partial block must survive candidate pruning.
        maxima = jnp.where(
            (jnp.arange(blocks)[None, :] == last[:, None]) & (lengths[:, None] > 0),
            jnp.inf,
            maxima,
        )
        count = min(candidates.shape[1], blocks)
        chosen = streaming_topk(
            maxima,
            valid_lengths=(lengths + block_size - 1) // block_size,
            k=count,
            exclude_neg_inf=True,
        )
        candidates = jnp.pad(
            chosen, ((0, 0), (0, candidates.shape[1] - count)), constant_values=-1
        )
    if consume:
        mask = jnp.zeros((scores.shape[0], blocks + 1), jnp.bool_)
        destination = jnp.where(
            (candidates >= 0) & (candidates < blocks), candidates, blocks
        )
        mask = mask.at[jnp.arange(scores.shape[0])[:, None], destination].set(True)
        allowed = mask[:, logical // block_size] | (longest <= topk)
        scores = jnp.where(allowed, scores, -jnp.inf)
    selected = streaming_topk(
        scores,
        valid_lengths=lengths,
        k=topk,
        exclude_neg_inf=True,
        scheduling_group_id=scheduling_group_id,
    )
    return selected, candidates


def index_selection(
    q,
    weights,
    pool,
    pages,
    requests,
    lengths,
    candidates,
    *,
    page_records,
    topk,
    block_size,
    publish,
    consume,
):
    """Return logical record IDs. All requests use the same streaming selector.

    Each score tile is bounded by 128 queries. The per-request loop shares
    packed keys across that tile and skips empty request rows. It also covers
    single-query decode and ragged prefill without a second scoring algorithm.
    """
    tokens = q.shape[0]
    tile = min(128, tokens)
    count = (tokens + tile - 1) // tile * tile
    width = max(2048, (pages.shape[1] * page_records + 2047) // 2048 * 2048)
    pages = jnp.pad(
        pages * page_records,
        ((0, 0), (0, width // page_records - pages.shape[1])),
        constant_values=-1,
    )
    split_q = split_halves(q)
    data = tuple(
        jnp.pad(
            x, ((0, count - tokens),) + ((0, 0),) * (x.ndim - 1), constant_values=fill
        )
        for x, fill in (
            (split_q, 0),
            (weights.astype(jnp.float32), 0),
            (requests, 0),
            (lengths, 0),
            (candidates, -1),
        )
    )

    def prepare(i):
        return tuple(lax.dynamic_slice_in_dim(x, i * tile, tile) for x in data)

    def score(values, group=None):
        queries, row_weights, owners, visible, _ = values
        initial = jnp.full((tile, width), -jnp.inf, jnp.float32)

        def request_step(request, result):
            mine = owners == request
            limit = jnp.max(jnp.where(mine, visible, 0), keepdims=True)

            def read(_):
                scores = prefill_index_scores(
                    queries,
                    row_weights,
                    pool,
                    lax.dynamic_slice_in_dim(pages, request, 1),
                    limit,
                    page_records=page_records,
                    query_tile=tile,
                    key_tile=2048,
                    scheduling_group_id=group,
                )
                return jnp.where(mine[:, None], scores, result)

            return lax.cond(limit[0] > 0, read, lambda _: result, None)

        return lax.fori_loop(0, pages.shape[0], request_step, initial)

    def select(i, scores, group=None):
        _, _, _, row_lengths, row_candidates = prepare(i)
        return select_scores(
            scores,
            row_lengths,
            row_candidates,
            topk=topk,
            block_size=block_size,
            publish=publish,
            consume=consume,
            longest=lengths.max(),
            scheduling_group_id=group,
        )

    def write(i, result, destination):
        return tuple(
            lax.dynamic_update_slice_in_dim(dst, src, i * tile, axis=0)
            for dst, src in zip(destination, result)
        )

    output = (
        jnp.full((count, topk), -1, jnp.int32),
        jnp.full((count, candidates.shape[1]), -1, jnp.int32),
    )
    selected, published = pipeline_score_topk_bounded(
        count // tile, prepare, score, select, write, output
    )
    return selected[:tokens], published[:tokens]
