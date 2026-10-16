"""Regression: GDN/KDA state loads must not reuse a VMEM slot whose store is in flight.

Runs the fused kernel in Pallas TPU interpret mode with DMA race detection on CPU.
Covers the KDA and radix-snapshot paths and the W = 1 tile clamp; the GDN decode
and verify races are covered in test_gdn_attention_v3.py.
Adapted from vllm-torchtpu tests/kernels/test_gdn_attention_v3.py (d6bbdbc69).
"""

from __future__ import annotations

import os
import sys

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=40, suite="stage-a-test-tpu-unit", chips=1)

# Interpret mode runs on the host CPU; keep JAX off the TPU chip.
if "jax" not in sys.modules:
    os.environ["JAX_PLATFORMS"] = "cpu"

import contextlib
import functools
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

from sglang.test.test_utils import CustomTestCase
from sglang_tpu.attention.pallas.gdn import config, wrapper

_D_K = 128
_D_V = 128
_KERNEL_SIZE = 4
# 12 sequences at the default decode tile of 4 make 3 pipeline tiles,
# so step 2 reuses step 0's slot.
_NUM_DECODE = 12


def _cpu_device():
    return jax.devices("cpu")[0]


def _abstract_tpu7x():
    return use_abstract_mesh(
        AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    )


class _RunResult:
    def __init__(self, outputs, races_per_call, cfgs):
        self.outputs = outputs
        self.races_per_call = races_per_call
        self.cfgs = cfgs

    def batched_cfg(self):
        cfgs = [c for c in self.cfgs if c.mode == config.GDNMode.BATCHED]
        assert len(cfgs) == 1, self.cfgs
        return cfgs[0]


@contextlib.contextmanager
def _record_kernel_cfgs():
    """Record the GDNConfig of every traced kernel (tile size, mode, window)."""
    cfgs = []
    real_outer = wrapper.outer_kernel

    def spy(*args, cfg, **kwargs):
        cfgs.append(cfg)
        return real_outer(*args, cfg=cfg, **kwargs)

    with mock.patch.object(wrapper, "outer_kernel", spy):
        yield cfgs


def _fused_kernel(**static):
    # A fresh jit per call: retraces, so _record_kernel_cfgs sees every kernel.
    return jax.jit(functools.partial(wrapper.fused_conv1d_gdn.__wrapped__, **static))


_STATIC_ARGS = (
    "n_kq",
    "n_v",
    "d_k",
    "d_v",
    "kernel_size",
    "num_spec_tokens",
    "decode_tile_size",
    "mixed_tile_size",
    "batched_only",
    "prefill_only",
    "attention_mode",
    "gate_lower_bound",
)


def _is_empty(rng) -> bool:
    return any(
        isinstance(r, slice)
        and r.start is not None
        and r.stop is not None
        and int(r.stop) <= int(r.start)
        for r in rng
    )


def _run_interpret(kwargs) -> _RunResult:
    """Run fused_conv1d_gdn in TPU interpret mode with DMA race detection.

    Returns the outputs, one race verdict per pallas_call, and the kernel configs.
    """
    static = {k: kwargs[k] for k in _STATIC_ARGS if k in kwargs}
    dynamic = {k: v for k, v in kwargs.items() if k not in static}

    mosaic_interpret = sys.modules[pltpu.reset_tpu_interpret_mode_state.__module__]
    race_state = sys.modules[mosaic_interpret.RaceDetectionState.__module__]
    orig_overlap = race_state._ranges_overlap

    # Workaround: jax race detection treats an empty slice as overlapping;
    # drop once jax skips zero-length ranges.
    def _ranges_overlap(range1, range2):
        if _is_empty(range1) or _is_empty(range2):
            return False
        return orig_overlap(range1, range2)

    races_per_call: list[bool] = []
    orig_clear = mosaic_interpret._clear_shared_memory

    # Workaround: jax 0.10.2 has no InterpretParams.on_race;
    # drop once the jax pin has it.
    def _record_and_clear():
        races_per_call.append(bool(mosaic_interpret.races.races_found))
        orig_clear()

    pltpu.reset_tpu_interpret_mode_state()
    mosaic_interpret._clear_shared_memory = _record_and_clear
    try:
        with (
            jax.default_device(_cpu_device()),
            _abstract_tpu7x(),
            _record_kernel_cfgs() as cfgs,
            mock.patch.object(race_state, "_ranges_overlap", _ranges_overlap),
            pltpu.force_tpu_interpret_mode(
                pltpu.InterpretParams(
                    detect_races=True,
                    dma_execution_mode="eager",
                    skip_floating_point_ops=False,
                    vector_clock_size=128,
                )
            ),
        ):
            outputs = _fused_kernel(**static)(**dynamic)
            outputs = jax.block_until_ready(outputs)
    finally:
        mosaic_interpret._clear_shared_memory = orig_clear
        pltpu.reset_tpu_interpret_mode_state()
    return _RunResult(outputs, races_per_call, list(cfgs))


def _identity_inputs(
    query_lens,
    *,
    n_kq,
    n_v,
    is_kda=False,
):
    """Inputs under which every sequence's states pass through unchanged.

    Zero qkv and conv weights make conv output, q, k and v zero; a_log,
    dt_bias, a and b at -20 make the decay exp(g) == 1 and beta ~ 0 in fp32.
    So each recurrent checkpoint equals the sequence's initial state
    bit-exactly, the conv state shifts in zero tokens and the output is zero.
    Slot 0 is the null block; sequence s owns slot s + 1.
    """
    d_k, d_v, kernel_size = _D_K, _D_V, _KERNEL_SIZE
    num_seqs = len(query_lens)
    num_tokens = int(sum(query_lens))
    dim = 2 * n_kq * d_k + n_v * d_v
    num_blocks = 1 + num_seqs
    gate_dim = n_v * d_k if is_kda else n_v
    rng = iter(jax.random.split(jax.random.key(0), 2))
    # Distinct random state per block, so a leaked state names its source.
    init_conv = jax.random.normal(
        next(rng), (num_blocks, kernel_size - 1, dim), dtype=jnp.float32
    ).astype(jnp.bfloat16)
    init_rec = jax.random.normal(
        next(rng), (num_blocks, n_v, d_k, d_v), dtype=jnp.float32
    )
    kwargs = dict(
        qkv=jnp.zeros((num_tokens, dim), jnp.bfloat16),
        b=jnp.full((num_tokens, n_v), -20.0, jnp.bfloat16),
        a=jnp.full((num_tokens, gate_dim), -20.0, jnp.bfloat16),
        conv_state=init_conv,
        recurrent_state=init_rec,
        conv_weight=jnp.zeros((dim, 1, kernel_size), jnp.bfloat16),
        conv_bias=None if is_kda else jnp.zeros((dim,), jnp.bfloat16),
        a_log=jnp.full((n_v,), -20.0, jnp.float32),
        dt_bias=jnp.full((n_v, d_k) if is_kda else (n_v,), -20.0, jnp.float32),
        query_start_loc=jnp.array(np.cumsum([0, *query_lens]), dtype=jnp.int32),
        state_indices=jnp.arange(1, num_seqs + 1, dtype=jnp.int32),
        # Every sequence continues a 16-token context.
        seq_lens=jnp.array([16 + q for q in query_lens], dtype=jnp.int32),
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
    )
    if is_kda:
        kwargs["attention_mode"] = config.AttentionMode.KDA
    return kwargs, np.asarray(init_conv, np.float32), np.asarray(init_rec, np.float32)


def _expected_conv(init_conv_row, num_tokens):
    """Conv state after `num_tokens` zero tokens: shift left, zero-fill."""
    k1 = init_conv_row.shape[0]
    out = np.zeros_like(init_conv_row)
    keep = max(k1 - num_tokens, 0)
    if keep:
        out[:keep] = init_conv_row[num_tokens:]
    return out


class TestGDNStateWindowRace(CustomTestCase):
    """Interpret-mode DMA race detection over the GDN / KDA state pipeline."""

    def _assert_no_races(self, result, num_calls):
        # An empty list means the detector hook never ran, not "no races".
        self.assertEqual(
            len(result.races_per_call),
            num_calls,
            f"expected {num_calls} race verdicts, got {result.races_per_call}",
        )
        self.assertFalse(
            any(result.races_per_call),
            "DMA race detected in the GDN state pipeline (per pallas_call: "
            f"{result.races_per_call}); state loads and stores must use "
            "separate VMEM windows.",
        )

    def _assert_multi_tile(self, result, num_seqs, min_tiles=3):
        tile = result.batched_cfg().seq_tile_size
        self.assertGreaterEqual(-(-num_seqs // tile), min_tiles, f"tile={tile}")
        return tile

    def _assert_state_rows(self, name, got, want, init, slots, *, tile):
        """Per-sequence check; on failure, name the block whose state leaked."""
        got = np.asarray(got, np.float32)
        for s, slot in enumerate(slots):
            if np.array_equal(got[slot], want[s]):
                continue
            # A slot-reuse race writes the initial state of a sequence two
            # pipeline steps away; report any block whose state matches.
            sources = [
                blk
                for blk in range(init.shape[0])
                if blk != slot and np.array_equal(got[slot], init[blk])
            ]
            self.fail(
                f"{name}: sequence {s} (slot {slot}, decode tile {tile}) "
                f"mismatches; max |err| = "
                f"{float(np.max(np.abs(got[slot] - want[s]))):.3e}; "
                f"matches initial state of blocks {sources}."
            )

    # --- race detection ------------------------------------------------------

    def test_kda_decode_has_no_dma_race(self):
        """KDA mode shares the state pipeline; W = 1 decode over 3 tiles."""
        lens = [1] * _NUM_DECODE
        kwargs, init_conv, init_rec = _identity_inputs(lens, n_kq=2, n_v=2, is_kda=True)
        kwargs["distribution"] = jnp.array([_NUM_DECODE] * 3, dtype=jnp.int32)
        kwargs["batched_only"] = True
        slots = np.asarray(kwargs["state_indices"])
        result = _run_interpret(kwargs)
        tile = self._assert_multi_tile(result, _NUM_DECODE)
        self._assert_no_races(result, num_calls=1)
        (conv_out, rec_out), out = result.outputs
        self._assert_state_rows(
            "kda recurrent", rec_out, init_rec[slots], init_rec, slots, tile=tile
        )
        want_conv = np.stack([_expected_conv(init_conv[s], 1) for s in slots])
        self._assert_state_rows(
            "kda conv", conv_out, want_conv, init_conv, slots, tile=tile
        )
        np.testing.assert_array_equal(np.asarray(out, np.float32), 0.0)

    def test_radix_snapshot_has_no_dma_race(self):
        """PER_SEQ multi-tile sequences next to one-tile ones, with radix snapshots.

        A store at tile p and a load at tile p + 2 hit the same slot of a shared
        window; the radix snapshot store rides the same output window.
        """
        lens = [3, 1, 3, 1, 3, 1]
        num_seqs = len(lens)
        kwargs, init_conv, init_rec = _identity_inputs(lens, n_kq=1, n_v=2)
        num_blocks = init_rec.shape[0]
        # Extra snapshot slots after the active ones.
        pad = ((0, num_seqs), (0, 0), (0, 0), (0, 0))
        kwargs["conv_state"] = jnp.pad(kwargs["conv_state"], pad[:3])
        kwargs["recurrent_state"] = jnp.pad(kwargs["recurrent_state"], pad)
        track = np.arange(num_blocks, num_blocks + num_seqs, dtype=np.int32)
        ends = np.cumsum(lens)
        kwargs["track_indices"] = jnp.asarray(track)
        kwargs["track_end"] = jnp.asarray(ends, dtype=jnp.int32)
        kwargs["distribution"] = jnp.array([0, num_seqs, num_seqs], dtype=jnp.int32)
        kwargs["prefill_only"] = True
        kwargs["mixed_tile_size"] = 1
        slots = np.asarray(kwargs["state_indices"])
        result = _run_interpret(kwargs)
        self._assert_no_races(result, num_calls=1)

        (conv_out, rec_out), out = result.outputs
        want_conv = np.stack(
            [_expected_conv(init_conv[s], q) for s, q in zip(slots, lens)]
        )
        for name, rows in (("active", slots), ("snapshot", track)):
            self._assert_state_rows(
                f"{name} recurrent", rec_out, init_rec[slots], init_rec, rows, tile=1
            )
            self._assert_state_rows(
                f"{name} conv", conv_out, want_conv, init_conv, rows, tile=1
            )
        np.testing.assert_array_equal(np.asarray(out, np.float32), 0.0)

    # --- tile selection -------------------------------------------------------

    def test_w1_decode_tile_counts_split_state_slots(self):
        """W = 1 decode tiles count 1 input + 1 output state slot per sequence.

        TPU7x (64 MiB VMEM per core) at the 0.7 VMEM fraction: a 4 MiB
        (n_v = 64) recurrent state allows T = 1 and a 2 MiB (n_v = 32) one
        T = 2, where one shared slot per sequence would allow T = 2 and
        T = 4 (the default). Small states keep the default tile.
        """
        num_seqs = 8
        cases = [("n_v64", 16, 64, 1), ("n_v32", 8, 32, 2), ("n_v8", 2, 8, 4)]
        for label, n_kq, n_v, expected in cases:
            with self.subTest(case=label):
                dim = 2 * n_kq * _D_K + n_v * _D_V

                def sds(shape, dtype=jnp.float32):
                    return jax.ShapeDtypeStruct(shape, dtype)

                args = dict(
                    qkv=sds((num_seqs, dim), jnp.bfloat16),
                    b=sds((num_seqs, n_v), jnp.bfloat16),
                    a=sds((num_seqs, n_v), jnp.bfloat16),
                    conv_state=sds((num_seqs + 1, _KERNEL_SIZE - 1, dim)),
                    recurrent_state=sds((num_seqs + 1, n_v, _D_K, _D_V)),
                    conv_weight=sds((dim, 1, _KERNEL_SIZE), jnp.bfloat16),
                    conv_bias=sds((dim,), jnp.bfloat16),
                    a_log=sds((n_v,)),
                    dt_bias=sds((n_v,)),
                    query_start_loc=sds((num_seqs + 1,), jnp.int32),
                    state_indices=sds((num_seqs,), jnp.int32),
                    distribution=sds((3,), jnp.int32),
                    seq_lens=sds((num_seqs,), jnp.int32),
                )
                kernel = functools.partial(
                    wrapper.fused_conv1d_gdn.__wrapped__,
                    n_kq=n_kq,
                    n_v=n_v,
                    d_k=_D_K,
                    d_v=_D_V,
                    kernel_size=_KERNEL_SIZE,
                    batched_only=True,
                )
                # Tracing alone picks the tile; nothing is compiled or run.
                with (
                    jax.default_device(_cpu_device()),
                    _abstract_tpu7x(),
                    _record_kernel_cfgs() as cfgs,
                ):
                    jax.eval_shape(lambda kw: kernel(**kw), args)
                (cfg,) = cfgs
                self.assertEqual(cfg.mode, config.GDNMode.BATCHED)
                self.assertEqual(cfg.seq_tile_size, expected)


if __name__ == "__main__":
    unittest.main()
