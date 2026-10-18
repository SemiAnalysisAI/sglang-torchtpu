# SPDX-License-Identifier: Apache-2.0
"""The JAX program of one Kimi-K3 megakernel decode step on the TP32 mesh.

One step decodes rows ``[0, B)`` of per-rank state arrays holding
``max_rows >= B`` rows. Operands, as each rank's torch tensor sees them:

* the weight families (``weight_names`` order), each ``[1, ...]``, sharded
  ``P("tp")`` so the global array leads with the rank;
* the four decode states, each ``[max_rows, ...]`` (``state_shapes``), sharded
  ``P("tp")``, donated and aliased to outputs 1-4;
* ``rank``, int32 ``[1]`` holding the device's TP rank, sharded ``P("tp")``;
* ``tokens`` and ``positions``, int32 ``[B]``, replicated: row ``r`` of both
  belongs to state row ``r``, and ``positions[r]`` is the number of tokens
  already cached for it (-1 for a padding row).

It returns full-vocabulary f32 logits ``[B, vocab]`` (replicated: the kernel's
vocabulary shards are all-gathered here) followed by the four updated states.
JAX is imported lazily so the module loads on hosts without it.
"""

from __future__ import annotations

import inspect
import math

from .config import ROW_BUCKETS

STATE_COUNT = 4
# The options upstream's demo decodes with; they keep its validated numerics.
_KERNEL_OPTIONS = frozenset(("mla_cache_write_in_attend", "mla_pv_mxu_hilo"))


def layer_counts(layers: int) -> tuple[int, int]:
    """(KDA layers, MLA layers) of a ``layers``-deep prefix of Kimi-K3.

    Every fourth layer is MLA, and the 93rd (the last) is MLA too.
    """
    mla = layers // 4 + (layers == 93)
    return layers - mla, mla


def decode_options(batch_size: int, *, tuned: bool) -> dict:
    """Kernel options of upstream's plain and tuned decode recipes."""
    if batch_size not in ROW_BUCKETS:
        raise ValueError(f"batch_size must be one of {ROW_BUCKETS}, got {batch_size}")
    options = {"gate_first_kda_projection": True, "kernel_options": _KERNEL_OPTIONS}
    if not tuned:
        return options
    eight = batch_size == 8
    return {
        **options,
        "residual_reduction_order": "relaxed",
        "mla_cache_batch_tile_size": math.gcd(4, batch_size),
        "alias_expert_scales": batch_size >= 6,
        "attention_token_scatter": eight,
        "moe_token_scatter_output": eight,
        "moe_routed_reduce_scatter": batch_size in (6, 8),
        "sequence_parallel_pre_attention": eight,
    }


def state_shapes(rows: int, context: int, layers: int):
    """Per-rank ``(shape, dtype name)`` of the four decode states.

    conv: KDA short-convolution history (oldest to newest, q/k/v, head,
    channel); recurrent: KDA state (head, key, value); key / value: expanded MLA
    caches per head, the key being ``k_nope 128 | rope 64 | zeros 64``.
    """
    if context % 128 or context < 384:
        raise ValueError(f"context must be a multiple of 128 and >= 384: {context}")
    kda, mla = layer_counts(layers)
    return (
        ((rows, kda, 3, 3, 3, 128), "bfloat16"),
        ((rows, kda, 3, 128, 128), "float32"),
        ((rows, mla, 3, context, 256), "bfloat16"),
        ((rows, mla, 3, context, 128), "bfloat16"),
    )


def make_decode_program(
    mesh, weight_names, *, batch_size: int, layers: int, tuned: bool
):
    """Flat entry point of one decode step plus its bridge metadata.

    Returns ``(function, input_specs, output_specs, donate_argnums,
    input_output_aliases)``. ``function`` takes one ``jax.Array``-annotated
    parameter per operand, as the torch bridge requires.
    """
    import jax
    from jax.sharding import PartitionSpec as P

    from ._vendor.kimi import decode_megakernel

    weight_names = tuple(weight_names)
    count = len(weight_names)
    options = decode_options(batch_size, tuned=tuned)

    def local(*operands):
        weights = {name: value[0] for name, value in zip(weight_names, operands)}
        states = operands[count : count + STATE_COUNT]
        rank, tokens, positions = operands[count + STATE_COUNT :]
        logits, new_states = decode_megakernel.decode_local(
            weights,
            states,
            tokens,
            positions,
            batch_size=batch_size,
            layers=layers,
            sequence_rows=1,
            rank=rank[0],
            return_state_updates=True,
            **options,
        )
        # [1, B, vocab / 32] on each rank -> [B, vocab] everywhere.
        logits = jax.lax.all_gather(logits[0], "tp", axis=1, tiled=True)
        return (logits, *new_states)

    input_specs = (P("tp"),) * (count + STATE_COUNT + 1) + (P(), P())
    output_specs = (P(),) + (P("tp"),) * STATE_COUNT
    sharded = jax.shard_map(
        local,
        mesh=mesh,
        in_specs=input_specs,
        out_specs=output_specs,
        check_vma=False,
    )

    def decode_step(*operands):
        return sharded(*operands)

    decode_step.__signature__ = inspect.Signature(
        [
            inspect.Parameter(
                f"operand{index}",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=jax.Array,
            )
            for index in range(len(input_specs))
        ],
        return_annotation=tuple[(jax.Array,) * len(output_specs)],
    )
    state_operands = tuple(range(count, count + STATE_COUNT))
    aliases = {operand: 1 + index for index, operand in enumerate(state_operands)}
    return decode_step, input_specs, output_specs, state_operands, aliases
