# SPDX-License-Identifier: Apache-2.0
"""JAX glue between sglang's TPU decode state and the GPT-OSS decode megakernel.

Everything here runs per rank (inside ``shard_map`` over the 1-D ``"tp"`` mesh, with every
operand passed as this rank's own block). It maps what the stock ``pallas_tpu_rpa_v3``
backend already builds for a decode batch onto the kernel's operands:

==========================  =================================================================
sglang (per decode step)    kernel
==========================  =================================================================
``kv_lens`` [B] i32         KV length *after* appending this step's token (RPA ``kv_lens`` =
                            ``seq_lens``). ``pos = kv_lens - 1`` is the new token's position
                            and the number of cached history tokens. Rows with
                            ``kv_lens == 0`` are bucket padding (the shim zero-pads
                            ``seq_lens``): attended as empty, KV written to reserved page 0.
``page_indices`` [B*PPS]    full-pool page table, rectangular (``PPS`` pages per request,
                            ``req_to_token[req, ::page] // page``) -> ``pt_full [B, MP]``
                            with ``MP = PPS``.
``swa_page_indices``        the same through ``full_to_swa_index_mapping`` -> SWA pool pages;
  [B*PPS]                   the kernel reads only the <= NWP window pages of sliding layers.
rope ``cos_sin_cache``      the stock rotary embedding's table ``[max_pos, D]`` (cos | sin
  [P, D]                    halves, YaRN incl. ``truncate``), gathered at ``pos``. Using the
                            stock table (not ``reference.yarn_tables``) keeps RoPE bitwise
                            equal to the stock path.
per-layer fused pools       ``[num_blocks, page, 1, packing, 128]`` per layer; split into
                            ``L/2`` SWA (even layers) + ``L/2`` full (odd layers) operands
                            (``DecodeConfig.split_pools``).
==========================  =================================================================

The new token's K/V is written where the stock RPA kernel writes it: page
``pt[b, pos // page]`` at offset ``pos % page`` of each pool (with sglang's paged
allocator this is ``out_cache_loc`` and ``full_to_swa[out_cache_loc]``).
"""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang.srt.hardware_backend.tpu.kernels.gptoss_decode_megakernel import (
    megakernel as MK,
)
from sglang.srt.hardware_backend.tpu.kernels.gptoss_decode_megakernel.config import (
    DecodeConfig,
    cdiv,
    rup,
)

F32 = jnp.float32
BF16 = jnp.bfloat16
I32 = jnp.int32


def trace_config(
    c0: DecodeConfig,
    *,
    B: int,
    Q: int = 1,
    pps: int,
    kv_dtype,
    nb_swa: int,
    nb_full: int,
    page: int,
    capture_layers: tuple[int, ...] | None = None,
) -> DecodeConfig:
    """Specialize the load-time geometry to one compiled decode/verify bucket."""
    kv = "fp8" if jnp.dtype(kv_dtype) == jnp.dtype(jnp.float8_e4m3fn) else "bf16"
    npb = max(1, (c0.npb * 64) // page) if page > 64 else c0.npb
    npb_swa = max(cdiv(c0.window - 1, page) + 1, 1) if page > 64 else c0.npb_swa
    c = dataclasses.replace(
        c0,
        B=B,
        Q=Q,
        max_pages=pps,
        kv_dtype=kv,
        nb_swa=nb_swa,
        nb_full=nb_full,
        page=page,
        npb=npb,
        npb_swa=npb_swa,
        split_pools=True,
        w_slots=min(c0.w_slots, 2) if B * Q >= 256 else c0.w_slots,
        kv_bufs=min(c0.kv_bufs, 2) if B * Q >= 256 else c0.kv_bufs,
        capture_layers=c0.capture_layers
        if capture_layers is None
        else tuple(capture_layers),
    )
    c.check()
    return c


def step_inputs(
    c: DecodeConfig, kv_lens, page_indices, swa_page_indices, cos_sin, x, rank=None
):
    """sglang decode/verify state -> kernel scalar/vector operands (all tiny XLA ops)."""
    B, Q, BQ, MP, D_DCP = c.B, c.Q, c.BQ, c.max_pages, c.dcp_size
    kv_lens = kv_lens.astype(I32)
    # In TARGET_VERIFY (Q > 1), the RPA backend sets kv_lens = seq_lens + Q, so
    # bucket-padded rows (seq_lens == 0) have kv_lens == Q, whereas live rows have
    # seq_lens >= 1 => kv_lens > Q.
    live = kv_lens > (Q if Q > 1 else 0)
    pos = jnp.where(live, jnp.maximum(kv_lens - Q, 0), 0)
    pt_full = page_indices.astype(I32).reshape(B, -1)[:, :MP]
    pt_swa = swa_page_indices.astype(I32).reshape(B, -1)[:, :MP]
    ptw, ptf, newpg = MK.page_args(c, pos, pt_swa, pt_full, rank=rank)
    live_tok = live if Q == 1 else jnp.repeat(live, Q)
    pad_pg = -1 if D_DCP > 1 else 0
    newpg = jnp.where(
        jnp.concatenate([live_tok, live_tok]), newpg, pad_pg
    )  # padding rows -> page 0 (or -1 under DCP so non-live rows skip DMA)
    pos_tok = (
        pos
        if Q == 1
        else jnp.where(
            live_tok,
            (pos[:, None] + jnp.arange(Q, dtype=I32)[None, :]).reshape(-1),
            0,
        )
    )
    pos_slot = (pos_tok // D_DCP) if D_DCP > 1 else pos_tok
    off = jnp.where(live_tok, pos_slot % c.page, 0)
    cos, sin = rope_from_cache(c, cos_sin, pos_tok)
    xp = jnp.pad(x.astype(BF16), ((0, c.BR - BQ), (0, c.Hp - c.H)))
    return dict(
        pos=pos, ptw=ptw, ptf=ptf, newpg=newpg, off=off, cos=cos, sin=sin, xp=xp
    )


def rope_from_cache(c: DecodeConfig, cos_sin, pos_tok):
    """Stock ``cos_sin_cache`` rows -> the kernel's [BR, 128] f32 (cos, signed sin) tables
    (``megakernel.rope_tables`` layout: neox halves, two heads per 128 lanes)."""
    h = c.D // 2
    rows = jnp.take(cos_sin, pos_tok, axis=0).astype(F32)  # [B*Q, D] = cos[h] | sin[h]
    cos, sin = rows[:, :h], rows[:, h : 2 * h]
    cos = jnp.concatenate([cos, cos], -1)
    sin_s = jnp.concatenate([-sin, sin], -1)
    rep = 128 // c.D
    pad = ((0, c.BR - c.BQ), (0, 0))
    return jnp.pad(jnp.tile(cos, (1, rep)), pad), jnp.pad(
        jnp.tile(sin_s, (1, rep)), pad
    )


@functools.lru_cache(maxsize=None)
def _mega(c: DecodeConfig, interpret: bool):
    return MK.build(c, "mega", interpret=interpret)


def per_rank_step(
    c: DecodeConfig,
    rank,
    x,
    kv_lens,
    page_indices,
    swa_page_indices,
    cos_sin,
    weights,
    pools,
    *,
    interpret: bool = False,
):
    """One decode/verify step on this rank.

    rank [1] i32 (this rank's TP index: selects the rank-0 down bias in TP MoE and the
    expert range in EP MoE); x [B*Q, H] embedded tokens; weights in ``megakernel.w_names``
    order (code words as int32 or uint32); pools: L/2 SWA then L/2 full per-layer tensors.
    Returns (hidden [B*Q, H] bf16 before the final norm, kvn [L, BR, kv_groups, packing, 128],
    newpg [2*B*Q] i32 (SWA pages then full pages), off [B*Q] i32[, aux [B*Q, NCAP*H] bf16]).
    """
    assert len(pools) == c.L and len(weights) == len(MK.w_names(c))
    rank_i32 = rank.astype(I32).reshape(1)
    s = step_inputs(
        c, kv_lens, page_indices, swa_page_indices, cos_sin, x, rank=rank_i32
    )
    wl = [
        lax.bitcast_convert_type(w, jnp.uint32) if w.dtype == jnp.int32 else w
        for w in weights
    ]
    res_ = _mega(c, interpret)(
        rank_i32,
        s["pos"],
        s["ptw"],
        s["ptf"],
        s["newpg"],
        s["xp"],
        s["cos"],
        s["sin"],
        *wl,
        *pools,
    )
    out, kvn = res_[0], res_[1]
    if c.capture_layers:
        aux_raw = res_[-1]
        aux = jnp.concatenate(
            [aux_raw[i, : c.BQ, : c.H] for i in range(len(c.capture_layers))], axis=-1
        )
        return out[: c.BQ, : c.H], kvn, s["newpg"], s["off"], aux
    return out[: c.BQ, : c.H], kvn, s["newpg"], s["off"]


def per_rank_noop(
    c: DecodeConfig,
    rank,
    x,
    kv_lens,
    page_indices,
    swa_page_indices,
    cos_sin,
    weights,
    pools,
    *,
    interpret: bool = False,
):
    """Timing ablation with ``per_rank_step``'s operands and outputs: the same glue, and a
    trivial Pallas kernel that takes every weight and pool as an HBM operand (so the same
    operand layout constraints, and any marshalling/relayout they cause, apply) but only
    copies x through. Outputs are wrong by design."""
    rank_i32 = rank.astype(I32).reshape(1)
    s = step_inputs(
        c, kv_lens, page_indices, swa_page_indices, cos_sin, x, rank=rank_i32
    )
    wl = [
        lax.bitcast_convert_type(w, jnp.uint32) if w.dtype == jnp.int32 else w
        for w in weights
    ]
    n_hbm = len(wl) + len(pools)

    def kernel(pos_s, xp_ref, *refs):
        refs[n_hbm][...] = xp_ref[...]

    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    out = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((c.BR, c.Hp), BF16),
        in_specs=[smem, vmem] + [hbm] * n_hbm,
        out_specs=vmem,
        interpret=pltpu.InterpretParams() if interpret else False,
        name="gptoss_decode_mk_noop",
    )(s["pos"], s["xp"], *wl, *pools)
    kvn = jnp.zeros((c.L, c.BR, c.kv_groups, c.kv_packing, 128), pools[0].dtype)
    if c.capture_layers:
        aux = jnp.zeros((c.BQ, len(c.capture_layers) * c.H), BF16)
        return out[: c.BQ, : c.H], kvn, s["newpg"], s["off"], aux
    return out[: c.BQ, : c.H], kvn, s["newpg"], s["off"]


# ----------------------------------------------------------------------------- KV write
#
# No `has_side_effects`: torch_tpu compiles the forward as one SPMD program, and the XLA
# SPMD partitioner requires a sharding on side-effecting custom calls (RET_CHECK
# "Side-effect HLO must have sharding", job j-3f74154c). Like the stock in-place kernels
# (RPA v3's cache update, kv_row_write) the call is kept alive by its aliased output,
# which the torch op exposes as a mutation of the pool.


def _write_rows(B, kvn_rows, pool, pg_of, off_s, sem):
    """Start one HBM->HBM DMA per owned live row b < B (row b of ``kvn_rows`` ->
    ``pool[pg_of(b), off[b]]`` when ``pg_of(b) >= 0``), then wait for all of them."""

    def cp(b):
        pg = jnp.maximum(pg_of(b), 0)
        return pltpu.make_async_copy(
            kvn_rows(b), pool.at[pg, pl.ds(off_s[b], 1)], sem.at[0]
        )

    def start(b, carry):
        @pl.when(pg_of(b) >= 0)
        def _():
            cp(b).start()

        return carry

    def wait(b, carry):
        @pl.when(pg_of(b) >= 0)
        def _():
            cp(0).wait()

        return carry

    lax.fori_loop(0, B, start, 0)
    lax.fori_loop(0, B, wait, 0)


def _kv_write_kernel(pg_s, off_s, kvn, pool_in, pool, sem):
    del pool_in
    _write_rows(
        off_s.shape[0],
        lambda b: kvn.at[pl.ds(b, 1)],
        pool,
        lambda b: pg_s[b],
        off_s,
        sem,
    )


def kv_write(pg, off, kvn, pool, *, interpret: bool = False):
    """``pool`` with this step's token K/V written in place (output aliases ``pool``).

    pg/off [B*Q] i32; kvn [BR, 1, packing, 128] (one layer of the megakernel's ``kvn``);
    pool [num_blocks, page, 1, packing, 128]. Not communicating, so XLA honours the alias.
    """
    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    return pl.pallas_call(
        _kv_write_kernel,
        out_shape=jax.ShapeDtypeStruct(pool.shape, pool.dtype),
        in_specs=[smem, smem, hbm, hbm],
        out_specs=hbm,
        scratch_shapes=[pltpu.SemaphoreType.DMA((1,))],
        input_output_aliases={3: 0},
        interpret=pltpu.InterpretParams() if interpret else False,
        name="gptoss_decode_mk_kv_write",
    )(pg.astype(I32), off.astype(I32), kvn, pool)


def kv_write_layers(newpg, off, kvn, *pools, interpret: bool = False):
    """All layers' new K/V in ONE call: ``pools`` in ``pool_layer_order`` (L/2 SWA, then L/2
    full), each aliased to its own output, so every pool is updated in place.

    newpg [2*B*Q] i32 (SWA pages, then full pages); off [B*Q] i32; kvn [L, BR, 1, packing, 128].
    Returns the L updated pools (aliases of the inputs).
    """
    L = len(pools)
    order = pool_layer_order(L)
    BQ = off.shape[0]

    def kernel(pg_s, off_s, kvn_ref, *refs):
        outs, sem = refs[L : 2 * L], refs[2 * L]
        for i, layer in enumerate(order):  # static: one DMA batch per pool
            base = 0 if layer % 2 == 0 else BQ  # sliding (even) layers use the SWA pages
            _write_rows(
                BQ,
                lambda b, layer=layer: kvn_ref.at[layer, pl.ds(b, 1)],
                outs[i],
                lambda b, base=base: pg_s[base + b],
                off_s,
                sem,
            )

    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    return pl.pallas_call(
        kernel,
        out_shape=tuple(jax.ShapeDtypeStruct(p.shape, p.dtype) for p in pools),
        in_specs=[smem, smem, hbm] + [hbm] * L,
        out_specs=tuple([hbm] * L),
        scratch_shapes=[pltpu.SemaphoreType.DMA((1,))],
        input_output_aliases={3 + i: i for i in range(L)},
        interpret=pltpu.InterpretParams() if interpret else False,
        name="gptoss_decode_mk_kv_write_layers",
    )(newpg.astype(I32), off.astype(I32), kvn, *pools)


def kv_write_all(c: DecodeConfig, newpg, off, kvn, pools, *, interpret: bool = False):
    """Reference composition of the per-layer writes (tests)."""
    BQ = c.BQ
    out = []
    for p, layer in zip(pools, pool_layer_order(c.L)):
        pg = newpg[:BQ] if c.sliding[layer] else newpg[BQ:]
        out.append(kv_write(pg, off, kvn[layer], p, interpret=interpret))
    return out


def pool_layer_order(L: int) -> list[int]:
    """Global layer id of each pool operand: SWA (even) layers, then full (odd) layers."""
    return list(range(0, L, 2)) + list(range(1, L, 2))


def smem_bytes_estimate(c: DecodeConfig) -> int:
    """Rough SMEM use of the megakernel's scalar operands/scratch (page tables dominate)."""
    words = (
        c.B
        + c.B * c.NWP
        + c.B * c.max_pages
        + 2 * c.BQ
        + 2 * c.NIMAX
        + 8 * 128
        + c.EL
        + 1
        + 1
    )
    return rup(words, 128) * 4
