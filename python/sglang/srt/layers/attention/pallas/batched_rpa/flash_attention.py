# Ported from https://github.com/vllm-project/tpu-inference
# tpu_inference/kernels/experimental/batched_rpa/flash_attention.py
# Commit 1d2527907b8289a13db19973d664788ead4f2392 (2026-07-09)

import jax
import jax.numpy as jnp
from jax import lax

from sglang.srt.layers.attention.pallas.batched_rpa import configs, utils


def flash_attention_qk_softmax(
    q: jax.Array,  # [B, KV, TQ, H]
    k: jax.Array,  # [B, KV, S, H]
    m_prev: jax.Array,  # [B, KV, TQ, 128]
    l_prev: jax.Array,  # [B, KV, TQ, 128]
    *,
    processed_q_len: list[jax.Array],  # [B]
    processed_kv_len: list[jax.Array],  # [B]
    effective_kv_len: list[jax.Array],  # [B]
    cfgs: configs.RpaConfigs,
    bq_start: int,
):
    """Flash attention kernel."""
    b, k_heads, tq, h_size = q.shape
    s = k.shape[2]

    if cfgs.serve.scale_q is not None:
        q = q / cfgs.serve.scale_q
        if jnp.issubdtype(k.dtype, jnp.floating):
            dtype_info = jnp.finfo(k.dtype)
            minval = float(dtype_info.min)
            maxval = float(dtype_info.max)
            q = jnp.clip(q, min=minval, max=maxval)
        q = q.astype(k.dtype)

    # einshape is a hijax op, so pjit cannot cache the compiled executable and
    # re-traces every call. These only merge or split adjacent leading axes,
    # which a plain reshape does at the same layout and no runtime cost.
    qk = lax.dot_general(
        q.reshape(b * k_heads, tq, h_size),
        k.reshape(b * k_heads, s, h_size),
        dimension_numbers=(([2], [2]), ([0], [0])),
        preferred_element_type=jnp.float32,
    ).astype(cfgs.serve.dtype_out)
    qk = qk.reshape(b, k_heads, tq, s)

    qk *= cfgs.model.sm_scale
    if cfgs.serve.scale_k is not None:
        qk *= cfgs.serve.scale_k
    if cfgs.serve.scale_q is not None:
        qk *= cfgs.serve.scale_q

    if cfgs.model.soft_cap is not None:
        qk = cfgs.model.soft_cap * jnp.tanh(qk / cfgs.model.soft_cap)

    qk_masked = []

    int_ty = cfgs.int_ty

    for b_idx in range(cfgs.block.batch_size):
        kv_idx_b = (
            lax.broadcasted_iota(int_ty, (k_heads, tq, s), 2) + processed_kv_len[b_idx]
        )
        q_idx_b = (
            lax.broadcasted_iota(jnp.int32, (k_heads, tq, s), 1)
            // cfgs.model.num_q_heads_per_kv_head
            + bq_start
        ).astype(int_ty) + processed_q_len[b_idx]

        eff_kv_len_b = effective_kv_len[b_idx]
        mask_b = q_idx_b < eff_kv_len_b
        mask_b = jnp.logical_and(mask_b, q_idx_b >= kv_idx_b)

        if (sliding_window := cfgs.model.sliding_window) is not None:
            # kv_idx_b already sits near max_model_len at the tail, so adding
            # the window can pass INT16_MAX even when the context itself fits.
            # Wrapping there flips the predicate and drops valid keys. The
            # compare yields a bool, so widening it costs nothing downstream.
            mask_b = jnp.logical_and(
                mask_b,
                q_idx_b.astype(jnp.int32) < kv_idx_b.astype(jnp.int32) + sliding_window,
            )

        qk_masked.append(jnp.where(mask_b, qk[b_idx], cfgs.model.mask_value))
    qk = jnp.stack(qk_masked, axis=0)

    m_curr = jnp.max(qk, axis=-1, keepdims=True)
    m_next = jnp.maximum(m_prev, m_curr)
    p = jnp.exp(qk - utils.broadcast_minor(m_next, qk.shape))
    p_rowsum = jnp.sum(p, axis=-1, keepdims=True, dtype=cfgs.serve.dtype_out)

    alpha = jnp.exp(m_prev - m_next)
    l_next = alpha * l_prev + p_rowsum

    return p, alpha, m_next, l_next


def flash_attention_pv(
    p: jax.Array,  # [B, KV, TQ, S]
    v: jax.Array,  # [B, KV, S, H]
    alpha: jax.Array,  # [B, KV, TQ, 128]
    o_prev: jax.Array,  # [B, KV, TQ, H]
    cfgs: configs.RpaConfigs,
):
    """Flash attention kernel."""
    b, k_heads, tq, s = p.shape
    h_size = v.shape[3]
    pv = lax.dot_general(
        p.reshape(b * k_heads, tq, s),
        v.reshape(b * k_heads, s, h_size),
        dimension_numbers=(([2], [1]), ([0], [0])),
        preferred_element_type=jnp.float32,
    ).astype(cfgs.serve.dtype_out)
    pv = pv.reshape(b, k_heads, tq, h_size)

    if cfgs.serve.scale_v is not None:
        pv *= cfgs.serve.scale_v

    o_next = utils.broadcast_minor(alpha, o_prev.shape) * o_prev + pv

    return o_next
