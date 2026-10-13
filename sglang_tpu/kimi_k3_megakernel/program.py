# SPDX-License-Identifier: Apache-2.0
"""The JAX program for one Kimi-K3 megakernel decode step on the TP32 mesh.

One step decodes rows ``[0, B)`` of per-rank state arrays that hold
``max_rows >= B`` rows (see ``slots.py``). Per-rank operands, in order:

* the megakernel weight families (``weight_names`` order), each ``[1, ...]``,
  sharded ``P("tp")`` so the global array leads with the rank;
* the four decode states, each ``[max_rows, ...]`` (``state_shapes``), sharded
  ``P("tp")`` and donated: the kernel updates them in place;
* ``tokens`` and ``positions``, int32 ``[B]``, replicated. Row ``r`` of both
  belongs to state row ``r``; ``positions[r]`` is the number of tokens already
  cached for that row, and must stay below the state's context length.

The step returns full-vocabulary f32 logits ``[B, vocab]`` (replicated; the
kernel's vocabulary shards are all-gathered in the program) followed by the
four updated states.
"""

from __future__ import annotations

import inspect
import math

KERNEL_DEFAULTS = ("mla_cache_write_in_attend", "mla_pv_mxu_hilo")
RECIPES = ("plain", "tuned")
STATE_COUNT = 4


def layer_counts(layers: int = 93) -> tuple[int, int]:
    """(KDA layers, MLA layers) of a ``layers``-deep prefix of Kimi-K3."""
    mla = layers // 4 + (layers == 93)
    return layers - mla, mla


def decode_options(batch_size: int, recipe: str = "tuned") -> dict:
    """Kernel options of the upstream demo's plain and tuned decode recipes."""
    if recipe not in RECIPES:
        raise ValueError(f"recipe must be one of {RECIPES}, got {recipe!r}")
    options = {
        "gate_first_kda_projection": True,
        "kernel_options": frozenset(KERNEL_DEFAULTS),
    }
    if recipe == "plain":
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


def state_shapes(rows: int, context: int, layers: int = 93):
    """Per-rank ``(shape, dtype name)`` of the four decode states.

    conv: KDA short-convolution history (oldest to newest, q/k/v, head, channel);
    recurrent: KDA recurrent state; key/value: expanded MLA caches (key is
    ``k_nope 128 | 64-d rope tail | 64 zeros``).
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


def make_local_step(
    weight_names,
    *,
    batch_size: int,
    layers: int = 93,
    recipe: str = "tuned",
    rank: int | None = 0,
):
    """Per-rank step body for ``jax.shard_map`` over the ``"tp"`` axis."""
    import jax

    from ._vendor.kimi import decode_megakernel

    weight_names = tuple(weight_names)
    options = decode_options(batch_size, recipe)

    def local(*operands):
        count = len(weight_names)
        weights = {
            name: value[0] for name, value in zip(weight_names, operands[:count])
        }
        states = tuple(operands[count : count + STATE_COUNT])
        tokens, positions = operands[count + STATE_COUNT :]
        logits, new_states = decode_megakernel.decode_local(
            weights,
            states,
            tokens,
            positions,
            batch_size=batch_size,
            layers=layers,
            sequence_rows=1,
            rank=rank,
            **options,
        )
        # [1, B, vocab / 32] on each rank -> [B, vocab] everywhere.
        logits = jax.lax.all_gather(logits[0], "tp", axis=1, tiled=True)
        return (logits, *new_states)

    return local


def make_decode_program(
    mesh,
    weight_names,
    *,
    batch_size: int,
    layers: int = 93,
    recipe: str = "tuned",
    rank: int | None = 0,
):
    """Flat jax.Array entry point of one decode step plus its bridge metadata.

    Returns ``(function, input_specs, output_specs, donate_argnums,
    input_output_aliases)``. ``function`` has one jax.Array-annotated
    parameter per operand, as the torch bridge requires.
    """
    import jax
    from jax.sharding import PartitionSpec as P

    weight_names = tuple(weight_names)
    count = len(weight_names)
    input_specs = (P("tp"),) * (count + STATE_COUNT) + (P(), P())
    output_specs = (P(),) + (P("tp"),) * STATE_COUNT
    sharded = jax.shard_map(
        make_local_step(
            weight_names,
            batch_size=batch_size,
            layers=layers,
            recipe=recipe,
            rank=rank,
        ),
        mesh=mesh,
        in_specs=input_specs,
        out_specs=output_specs,
        check_vma=False,
    )

    def decode_step(*operands):
        return sharded(*operands)

    # The bridge reads one annotated parameter per operand from the signature.
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
