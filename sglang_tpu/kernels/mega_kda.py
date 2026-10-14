# SPDX-License-Identifier: Apache-2.0
"""KDA prefill through the sglang-jax Mega kernel.

The fused GDN v3 kernel runs a KDA prefill as one recurrent pass per sequence
tile; the Mega kernel (mega_kda_kernel.py, from sglang-jax) runs the chunked
form and is about twice as fast at 8k tokens. It covers only the delta-rule
core, so the short causal conv and the state cache traffic are done here, with
the semantics of the fused kernel's PER_SEQ path:

* a sequence with a prefix (seq_len > query_len) resumes from the conv window
  and recurrent state at state_indices[s] + read_offsets[s]; others start
  from zeros;
* the final conv window (last kernel_size - 1 raw inputs) and recurrent state
  are written to state_indices[s].

Unlike the fused kernel, which keeps the conv output in f32, the Mega kernel
takes bf16 q/k/v/g.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from sglang_tpu.kernels.mega_kda_kernel import _chunk_kda_fwd_native_segids

CHUNK = 64


def mega_layout_supported(query_start_loc: jax.Array, num_tokens: int) -> jax.Array:
    """Whether every 64-token tile intersects at most two live sequences.

    The Mega kernel handles the first and last segment of a tile; a third one
    would be dropped. Scalar device bool (upstream is_mega_kda_layout_supported).
    """
    if query_start_loc.shape[0] <= 3:
        return jnp.asarray(True)
    starts = jnp.minimum(query_start_loc[:-1], num_tokens)
    ends = jnp.minimum(query_start_loc[1:], num_tokens)
    tile_starts = jnp.arange(0, num_tokens, CHUNK, dtype=jnp.int32)
    overlaps = (
        (starts < ends)[:, None]
        & (starts[:, None] < tile_starts[None, :] + CHUNK)
        & (ends[:, None] > tile_starts[None, :])
    )
    return jnp.all(jnp.sum(overlaps, axis=0) <= 2)


def kda_prefill_mega(
    qkv: jax.Array,  # [T, 2 * n_kq * d_k + n_v * d_v]
    b: jax.Array,  # [T, n_v]
    a: jax.Array,  # [T, n_v * d_k], raw gate
    conv_state: jax.Array,  # [S, K - 1, dim] or [S, K - 1, 1, dim]
    recurrent_state: jax.Array,  # [S, n_v, d_k, d_v]
    conv_weight: jax.Array,  # [dim, 1, K]
    conv_bias: jax.Array | None,  # [dim]
    a_log: jax.Array,  # [n_v]
    dt_bias: jax.Array,  # [n_v, d_k]
    query_start_loc: jax.Array,  # [N + 1]
    state_indices: jax.Array,  # [N]
    seq_lens: jax.Array,  # [N]
    read_offsets: jax.Array | None,  # [N]
    *,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    gate_lower_bound: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Returns (new_conv_state, new_recurrent_state, output [T, n_v, d_v])."""
    if n_kq != n_v:
        raise ValueError(f"Mega KDA needs n_kq == n_v, got {n_kq} and {n_v}")
    tokens, dim = qkv.shape
    num_seqs = state_indices.shape[0]
    num_slots = recurrent_state.shape[0]
    prev = kernel_size - 1

    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    has_initial = (seq_lens - query_lens) > 0
    read_rows = state_indices + (0 if read_offsets is None else read_offsets)

    # Causal conv: tap k reads token t - prev + k, or the stored window when
    # that falls before the sequence start.
    window_shape = conv_state.shape
    windows = conv_state.reshape(num_slots, prev, dim)[read_rows].astype(jnp.float32)
    windows = jnp.where(has_initial[:, None, None], windows, 0.0)  # [N, prev, dim]
    t = jnp.arange(tokens, dtype=jnp.int32)
    seg = jnp.clip(
        jnp.searchsorted(query_start_loc[1:], t, side="right"), 0, num_seqs - 1
    )
    pos = t - query_start_loc[seg]
    x = qkv.astype(jnp.float32)
    taps = conv_weight[:, 0, :].astype(jnp.float32)  # [dim, K]
    # Static shifted slices fuse into one pass; a tap reaching before its
    # sequence start reads zero here and its stored-window value is added below.
    x_padded = jnp.concatenate([jnp.zeros((prev, dim), qkv.dtype), qkv], axis=0)
    y = functools.reduce(
        jnp.add,
        [
            jnp.where(
                (pos >= prev - k)[:, None],
                x_padded[k : k + tokens].astype(jnp.float32),
                0.0,
            )
            * taps[None, :, k]
            for k in range(kernel_size)
        ],
    )
    # Token j < prev of a sequence reads window[j + k] at taps k < prev - j.
    corr = jnp.stack(
        [
            functools.reduce(
                jnp.add,
                [windows[:, j + k] * taps[None, :, k] for k in range(prev - j)],
            )
            for j in range(prev)
        ],
        axis=1,
    )  # [N, prev, dim]
    j = jnp.arange(prev, dtype=jnp.int32)
    corr_rows = query_start_loc[:-1, None] + j[None, :]
    corr_rows = jnp.where(j[None, :] < query_lens[:, None], corr_rows, tokens)
    y = y.at[corr_rows].add(corr, mode="drop")
    if conv_bias is not None:
        y = y + conv_bias.astype(jnp.float32)[None, :]
    y = jax.nn.silu(y).astype(jnp.bfloat16)

    # New conv windows: the last `prev` raw inputs of each sequence.
    ends = query_start_loc[1:]
    rows = ends[:, None] - prev + jnp.arange(prev, dtype=jnp.int32)[None, :]  # [N, prev]
    before = rows < query_start_loc[:-1, None]
    win_from_tokens = x[jnp.clip(rows, 0, tokens - 1)]
    win_from_old = windows[
        jnp.arange(num_seqs)[:, None],
        jnp.clip(prev + rows - query_start_loc[:-1, None], 0, prev - 1),
    ]
    new_windows = jnp.where(before[..., None], win_from_old, win_from_tokens)
    live = query_lens > 0
    write_rows = jnp.where(live, state_indices, num_slots)  # out of range: dropped
    new_conv_state = (
        conv_state.reshape(num_slots, prev, dim)
        .at[write_rows]
        .set(new_windows.astype(conv_state.dtype), mode="drop")
        .reshape(window_shape)
    )

    # Delta-rule core.
    q = y[:, : n_kq * d_k].reshape(1, tokens, n_kq, d_k)
    k_ = y[:, n_kq * d_k : 2 * n_kq * d_k].reshape(1, tokens, n_kq, d_k)
    v = y[:, 2 * n_kq * d_k :].reshape(1, tokens, n_v, d_v)
    g = a.astype(jnp.bfloat16).reshape(1, tokens, n_v, d_k)
    beta = jax.nn.sigmoid(b.astype(jnp.float32)).reshape(1, tokens, n_v)
    init = recurrent_state[read_rows].astype(jnp.float32)
    init = jnp.where(has_initial[:, None, None, None], init, 0.0)

    padded = (tokens + CHUNK - 1) // CHUNK * CHUNK

    def pad(arr):
        widths = [(0, 0)] * arr.ndim
        widths[1] = (0, padded - tokens)
        return jnp.pad(arr, widths)

    seg_ids = jnp.repeat(
        jnp.arange(1, num_seqs + 1, dtype=jnp.int32),
        query_lens,
        total_repeat_length=padded,
    )
    seg_ids = jnp.where(
        jnp.arange(padded, dtype=jnp.int32) < query_start_loc[-1], seg_ids, 0
    )[None, :]
    out, final, *_ = _chunk_kda_fwd_native_segids(
        q=pad(q),
        k=pad(k_),
        v=pad(v),
        g=pad(g),
        beta=pad(beta),
        segment_ids=seg_ids,
        initial_state=init[None],
        output_final_state=True,
        scale=d_k**-0.5,
        chunk_size=CHUNK,
        store_h=False,
        store_v_new=False,
        disable_recompute=False,
        only_fwd=True,
        safe_gate=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        A_log=jnp.minimum(a_log.astype(jnp.float32), 80.0),
        dt_bias=dt_bias.reshape(-1).astype(jnp.float32),
        lower_bound=gate_lower_bound,
        mini_batch=None,
        N_max=num_seqs,
        residual_chunk_layout=False,
        batch_first=True,
    )
    new_recurrent_state = recurrent_state.at[write_rows].set(
        final[0].astype(recurrent_state.dtype), mode="drop"
    )
    # Rows past the live tokens feed MoE and collectives: zero them, as the
    # fused kernel's zero-initialized output does.
    output = jnp.where(
        (t < query_start_loc[-1])[:, None, None], out[0, :tokens], 0
    ).astype(qkv.dtype)  # [T, n_v, d_v]
    return new_conv_state, new_recurrent_state, output
