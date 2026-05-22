"""Pallas fused-MoE bridge for the TPU torch_tpu platform.

This module wraps the JAX/Pallas ``gmm_v2`` grouped-matmul kernel
(``sglang.srt.layers.moe.pallas.gmm_v2``) and assembles it into a
single-host fused-MoE op exposed to PyTorch via
``torch_tpu._internal.pallas.jax_op``. It mirrors the pattern used by the
ragged-paged-attention v3 backend in
``sglang.srt.layers.attention.pallas_backend``: register one custom
``torch.library`` op whose implementation is a ``jax.jit``-traced JAX
function. Per-shape / per-(topk, activation) specialisation happens
inside the JAX trace cache maintained by ``JaxCallable``; dynamo sees
the op as an opaque tensor-in / tensor-out call.

MoE math (single host, no shard_map)
------------------------------------
For each token, top-``k`` experts are activated. We do the standard
"permute / GMM1 / activation / GMM2 / un-permute / weighted-reduce"
pipeline:

1. Flatten ``topk_ids`` (``[m, topk]``) → ``[m * topk]``.
2. ``argsort`` the flattened expert ids; rows of the LHS become
   contiguous blocks per expert. ``group_sizes[e] = #(topk_ids == e)``.
3. Gather rows of ``hidden_states`` into the sorted order.
4. GMM1: ``x_sorted @ w1[e] -> [m*topk, 2*intermediate]`` (gate || up).
5. Apply gated activation (``silu(gate) * up`` for ``"silu"`` / SwiGLU).
6. GMM2: ``act_out @ w2[e] -> [m*topk, hidden]``.
7. Scatter back and reduce with ``topk_weights``.

Weight layout
-------------
Callers pre-transpose to gmm_v2's ``(g, k, n)`` LHS-RHS convention. The
unquant path supplies::

    w1 = layer.w13_weight.transpose(1, 2).contiguous()  # (g, hidden, 2*intermediate)
    w2 = layer.w2_weight.transpose(1, 2).contiguous()   # (g, intermediate, hidden)

Activation
----------
Only ``"silu"`` (SwiGLU: ``silu(gate) * up``) is exercised today, which
is what Qwen3-MoE uses. ``"gelu"`` is supported for parity with the
mxfp4 path. ``"swiglu_oai"`` is plumbed through but raises at trace
time until the GPT-OSS-specific clamping is wired up.
"""

# NOTE: do *not* add ``from __future__ import annotations`` here —
# ``torch_tpu._internal.pallas.jax_op`` inspects ``param.annotation is
# jax.Array`` at registration time; PEP 563 would turn those into
# strings and validation would reject them.

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# Singleton: one ``torch.library`` op for the whole process. The JaxCallable
# beneath does its own per-(static_arg, input-shape) trace caching, so a
# single op covers every (topk, activation, dtype, shape) combination the
# model will encounter.
_PALLAS_MOE_OP = None


def _build_fused_moe_op():
    """Lazily register the fused-MoE kernel as a torch custom op.

    Mirrors ``_build_pallas_kernel_op`` in pallas_backend.py — registration
    is heavy (``inspect.signature`` on a JAX function + JAX export) and
    cannot run inside a dynamo trace, so we call this eagerly from the
    TPU pre-warm hook and just look the op up from the model code path.
    """
    global _PALLAS_MOE_OP
    if _PALLAS_MOE_OP is not None:
        return _PALLAS_MOE_OP

    import jax  # noqa: F401  (force loud failure if JAX is missing)
    import jax.numpy as jnp
    from torch_tpu._internal import pallas as _torch_pallas

    from sglang.srt.layers.moe.pallas.gmm_v2 import TileSizes, gmm_v2
    from sglang.srt.layers.moe.pallas.tuned_block_sizes import (
        get_tuned_block_sizes,
    )

    op_name = "sglang_tpu_pallas::fused_moe"

    def _lookup_tile_info(lhs, rhs, num_total_groups):
        """Pick a ``TileSizes`` for this gmm_v2 call.

        ``get_tuned_block_sizes`` is a pure-Python lookup keyed on static
        shapes / dtypes, so it runs at JAX trace time (not at device exec
        time). On a hit we pin the kernel's tile sizes to the tuned
        triple; on a miss we hand ``None`` back to gmm_v2, which falls
        through to its built-in ``calculate_tiling`` heuristic.

        ``num_current_groups`` is set equal to ``num_total_groups``
        because this bridge is single-host (no expert parallelism); when
        EP is added the caller will need to pass the local group count
        through.
        """
        m, k = lhs.shape
        _g, _k, n = rhs.shape
        assert _k == k, f"gmm_v2 lhs/rhs k mismatch: {k=} vs {_k=}"
        # Quantization is per-channel (kernel asserts rhs_scale.shape[1] == 1),
        # so quant_block_size == k whenever scales are provided.
        quant_block_size = k
        tm, tk, tn = get_tuned_block_sizes(
            m=int(m),
            k=int(k),
            n=int(n),
            num_total_groups=int(num_total_groups),
            num_current_groups=int(num_total_groups),
            lhs_dtype=str(jnp.dtype(lhs.dtype).name),
            rhs_dtype=str(jnp.dtype(rhs.dtype).name),
            quant_block_size=quant_block_size,
        )
        return TileSizes(tile_m=tm, tile_k=tk, tile_n=tn)

    def _activation(gmm1_out, activation_name):
        """Apply the gated activation to a ``[m*topk, 2*n]`` GMM1 output.

        Returns the ``[m*topk, n]`` post-activation tensor consumed by GMM2.
        """
        n = gmm1_out.shape[-1] // 2
        gate, up = gmm1_out[:, :n], gmm1_out[:, n:]
        if activation_name == "silu":
            return jax.nn.silu(gate) * up
        if activation_name == "gelu":
            return jax.nn.gelu(gate) * up
        if activation_name == "swiglu_oai":
            raise NotImplementedError(
                "swiglu_oai gating in the Pallas fused-MoE op is not wired "
                "up yet; the GPT-OSS path still goes through the mxfp4 "
                "specialised forward."
            )
        raise NotImplementedError(
            f"Unsupported activation '{activation_name}' for the Pallas "
            "fused-MoE op. Supported: silu, gelu."
        )

    def _fused_moe_jax(
        hidden_states,  # [m, k]
        w1,  # [g, k, 2 * n_intermediate]
        w2,  # [g, n_intermediate, k]
        topk_weights,  # [m, topk]
        topk_ids,  # i32[m, topk]
        w1_scale,  # Optional[g, 1, 1, 2 * n_intermediate] f32
        w2_scale,  # Optional[g, 1, 1, k] f32
        w1_bias,  # Optional[g, 1, 2 * n_intermediate] f32
        w2_bias,  # Optional[g, 1, k] f32
        topk,
        activation_name,
    ):
        m, hidden_size = hidden_states.shape
        num_experts = w1.shape[0]
        out_dtype = hidden_states.dtype

        # Routing: build a permutation that puts rows of hidden_states in
        # expert-major order so a single grouped-matmul covers GMM1 / GMM2.
        topk_ids_flat = topk_ids.reshape(-1).astype(jnp.int32)  # [m*topk]
        # argsort is stable enough on TPU for our usage; the gmm kernel
        # only cares about contiguous expert runs.
        argsort_idx = jnp.argsort(topk_ids_flat)
        # Row-of-original-token for each sorted slot.
        token_indices = jnp.repeat(jnp.arange(m, dtype=jnp.int32), topk)
        sorted_token_ids = token_indices[argsort_idx]

        # one_hot+sum is cheaper on TPU than bincount.
        group_sizes = jax.nn.one_hot(topk_ids_flat, num_experts, dtype=jnp.int32).sum(
            axis=0
        )

        # GMM1: gather rows then expert-batched matmul.
        x_sorted = hidden_states[sorted_token_ids]  # [m*topk, k]
        gmm1_out = gmm_v2(
            lhs=x_sorted,
            rhs=w1,
            group_sizes=group_sizes,
            rhs_scale=w1_scale,
            rhs_bias=w1_bias,
            preferred_element_type=out_dtype,
            tile_info=_lookup_tile_info(x_sorted, w1, num_experts),
        )  # [m*topk, 2 * n_intermediate]

        # Gated activation.
        act_out = _activation(gmm1_out, activation_name)

        # GMM2: project back to hidden size.
        gmm2_out = gmm_v2(
            lhs=act_out,
            rhs=w2,
            group_sizes=group_sizes,
            rhs_scale=w2_scale,
            rhs_bias=w2_bias,
            preferred_element_type=out_dtype,
            tile_info=_lookup_tile_info(act_out, w2, num_experts),
        )  # [m*topk, k]

        # Un-permute back to token-major order and reduce across the top-k
        # experts with the routing weights.
        unsort_idx = jnp.argsort(argsort_idx)
        unsorted = gmm2_out[unsort_idx].reshape(m, topk, hidden_size)
        weights = topk_weights.reshape(m, topk, 1).astype(unsorted.dtype)
        return (unsorted * weights).sum(axis=1).astype(out_dtype)

    # ``jax_op`` inspects annotations to decide which positional params
    # are tensors (jax.Array) vs. compile-time constants (int/str/...).
    # Keep annotations spelled out — no Optional aliases — so PEP-563 is
    # not needed.
    def _wrapped(
        hidden_states: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        topk_weights: jax.Array,
        topk_ids: jax.Array,
        w1_scale: Optional[jax.Array] = None,
        w2_scale: Optional[jax.Array] = None,
        w1_bias: Optional[jax.Array] = None,
        w2_bias: Optional[jax.Array] = None,
        topk: int = 1,
        activation: str = "silu",
    ) -> jax.Array:
        return _fused_moe_jax(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            w1_scale,
            w2_scale,
            w1_bias,
            w2_bias,
            topk,
            activation,
        )

    custom_op = _torch_pallas.jax_op(op_name, _wrapped)

    def _fake_impl(
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        w1_scale: Optional[torch.Tensor] = None,
        w2_scale: Optional[torch.Tensor] = None,
        w1_bias: Optional[torch.Tensor] = None,
        w2_bias: Optional[torch.Tensor] = None,
        topk: int = 1,
        activation: str = "silu",
    ):
        # Output is [m, hidden] in the dtype of hidden_states. Using
        # ``empty_like`` on a slice lets ``torch.compile`` thread the
        # dynamic token dimension without invoking the JAX tracer.
        return torch.empty_like(hidden_states)

    custom_op.register_fake(_fake_impl)

    _PALLAS_MOE_OP = custom_op
    return custom_op


def prewarm_fused_moe_op(*, topk: int, activation: str) -> None:
    """Ensure the fused-MoE custom op is registered before dynamo traces it.

    ``topk`` and ``activation`` are accepted for parity with the call site
    (``tpu_platform.py``'s pre-warm loop iterates over distinct combos),
    but the op itself is a singleton — the JAX trace cache inside
    ``JaxCallable`` handles per-(topk, activation, shape, dtype)
    specialisation on first invocation.
    """
    del topk, activation  # used purely to scope the loop in the caller
    _build_fused_moe_op()


def fused_moe_call(
    *,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    topk: int,
    activation: str,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Invoke the Pallas fused-MoE op.

    Inputs use gmm_v2's ``(g, k, n)`` LHS-RHS convention:
      * ``hidden_states``: ``[num_tokens, hidden_size]``
      * ``w1``: ``[num_experts, hidden_size, 2 * intermediate]``  (gate || up)
      * ``w2``: ``[num_experts, intermediate, hidden_size]``
      * ``topk_weights``: ``[num_tokens, topk]``  (routing scores, dtype of x)
      * ``topk_ids``: ``[num_tokens, topk]``  (int32 expert ids)

    Returns ``[num_tokens, hidden_size]`` in ``hidden_states.dtype``.
    """
    op = _build_fused_moe_op()
    return op(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        w1_scale,
        w2_scale,
        w1_bias,
        w2_bias,
        topk,
        activation,
    )
