# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted for SGLang-TorchTPU from vllm-project/vllm-torchtpu
# `vllm_torchtpu.kernels.gdn.v3.compute_kda` (Apache-2.0).
# Changes: import paths rebased to `sglang.srt.layers.attention.pallas.gdn`;
# the recurrence keeps only the final state (no speculative checkpoints).
"""Compute kernels for KDA (Kimi Delta Attention) linear attention.

The KDA counterpart of `compute_gdn`, sharing its tiling, state layout and
helpers. The structural difference is the gate: KDA's is per-channel, so the
decay is a [kq_head_dim] vector per value head rather than a per-head scalar.
"""

import jax
import jax.numpy as jnp

from sglang.srt.layers.attention.pallas.gdn import compute_gdn, config


def _l2_norm_f32(x: jax.Array, eps: float = 1e-6) -> jax.Array:
    """L2-normalize along the last dim, accumulating in float32.

    Unlike `compute_gdn.l2_norm`, which accumulates in the input dtype: a bf16
    sum of squares loses enough precision to show up against fp64 goldens.
    """
    x_f32 = x.astype(jnp.float32)
    inv_norm = jax.lax.rsqrt(jnp.sum(x_f32 * x_f32, axis=-1, keepdims=True) + eps)
    return (x_f32 * inv_norm).astype(x.dtype)


def activate_gate(
    a: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array,
    cfg: config.GDNConfig,
) -> jax.Array:
    """Turn the raw gate `a` into a per-channel log-decay (always <= 0).

    `a_log` and `dt_bias` arrive already reshaped to broadcast against `a`.
    """
    a_f32 = a.astype(jnp.float32)
    if cfg.gate_lower_bound is None:
        # Unbounded: the decay can reach 0, i.e. a channel forgets entirely.
        return -jnp.exp(a_log) * jax.nn.softplus(a_f32 + dt_bias)
    # Bounded: the log-decay is floored at `gate_lower_bound`, so the
    # per-token decay never drops below exp(gate_lower_bound).
    return cfg.gate_lower_bound * jax.nn.sigmoid(jnp.exp(a_log) * (a_f32 + dt_bias))


def recurrent_kda_per_seq(
    q: jax.Array,  # (num_kq_heads, chunk, kq_head_dim)
    k: jax.Array,  # (num_kq_heads, chunk, kq_head_dim)
    v: jax.Array,  # (num_v_heads, chunk, v_head_dim)
    gating_decay: jax.Array,  # (num_v_heads, chunk, kq_head_dim)
    beta: jax.Array,  # (num_v_heads, chunk, 1)
    state: jax.Array,  # (num_v_heads, kq_head_dim, v_head_dim)
    cfgs: config.GDNConfig,
) -> tuple[jax.Array, jax.Array]:
    """Token recurrence over one sequence's chunk; returns (out, final state).

    Invalid ragged tail rows have q/k/v/beta zeroed and a unit decay, so their
    output is zero and they leave the state unchanged.
    """
    out_list = []
    contract_dk = (((2,), (1,)), ((0,), (0,)))

    for c_idx in range(cfgs.chunk_size):
        # Repeat Q and K for GQA.
        q_heads = jnp.repeat(q[:, c_idx], cfgs.v_per_kq_head, axis=0)[
            :, None, :
        ]  # (num_v_heads, 1, kq_head_dim)
        k_heads = jnp.repeat(k[:, c_idx], cfgs.v_per_kq_head, axis=0)[
            :, None, :
        ]  # (num_v_heads, 1, kq_head_dim)
        k_heads_t = compute_gdn.fused_transpose_broadcast(
            k_heads, src_dim=2, dst_dim=1
        )  # (num_v_heads, kq_head_dim, 1)

        v_heads = v[:, c_idx : c_idx + 1]
        beta_heads = beta[:, c_idx : c_idx + 1]
        decay = gating_decay[:, c_idx, :, None]

        state_updated = state * decay
        v_updated = jax.lax.dot(
            k_heads,
            state_updated,
            dimension_numbers=contract_dk,
            preferred_element_type=jnp.float32,
        ).astype(cfgs.dtypes.compute)

        v_new = beta_heads * (v_heads - v_updated)
        state = state_updated + k_heads_t * v_new

        out = jax.lax.dot(
            q_heads,
            state,
            dimension_numbers=contract_dk,
            preferred_element_type=jnp.float32,
        ).astype(cfgs.dtypes.compute)
        out_list.append(out[:, 0, :])

    return jnp.stack(out_list, axis=0), state


def recurrent_kda(
    real_sizes: jax.Array,
    q_compact: jax.Array,
    k_compact: jax.Array,
    v_compact: jax.Array,
    b_compact: jax.Array,
    a_compact: jax.Array,
    state_prev: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array,
    cfg: config.GDNConfig,
) -> tuple[jax.Array, jax.Array]:
    """Recurrent KDA over compact inputs [seq, num_heads, chunk, 1, head_dim]."""
    mask_dtype = compute_gdn.get_mask_dtype(cfg.dtypes.compute)
    iota = jax.lax.broadcasted_iota(
        mask_dtype, (cfg.seq_tile_size, 1, cfg.chunk_size, 1, 1), 2
    )
    mask = iota < real_sizes.reshape(-1, 1, 1, 1, 1).astype(mask_dtype)

    q = jnp.where(mask, q_compact.astype(cfg.dtypes.compute), 0.0)
    k = jnp.where(mask, k_compact.astype(cfg.dtypes.compute), 0.0)
    v = jnp.where(mask, v_compact.astype(cfg.dtypes.compute), 0.0)

    q = _l2_norm_f32(q) * (cfg.kq_head_dim**-0.5)
    k = _l2_norm_f32(k)

    # Delta rule step-size beta in [0, 1].
    beta = jnp.where(mask, jax.nn.sigmoid(b_compact.astype(jnp.float32)), 0.0)
    beta = compute_gdn.fused_transpose_broadcast(beta, src_dim=4, dst_dim=1)
    beta = beta[:, : cfg.num_v_heads, :, 0, :]

    a_log_kda = a_log.reshape(1, cfg.num_v_heads, 1, 1, 1).astype(jnp.float32)
    dt_bias_kda = dt_bias.reshape(1, cfg.num_v_heads, 1, 1, cfg.kq_head_dim).astype(
        jnp.float32
    )
    gating_log = activate_gate(a_compact, a_log_kda, dt_bias_kda, cfg)
    # Invalid ragged tail tokens must leave the state untouched.
    gating_log = jnp.where(mask, gating_log, 0.0)
    gating_decay = jnp.exp(gating_log)[:, :, :, 0, :]

    # Drop the compact layout's singleton dimension before the token scan.
    q = q[:, :, :, 0, :]
    k = k[:, :, :, 0, :]
    v = v[:, :, :, 0, :]

    out_list = []
    new_state_list = []
    for idx in range(cfg.seq_tile_size):
        out, state = recurrent_kda_per_seq(
            q[idx],
            k[idx],
            v[idx],
            gating_decay[idx],
            beta[idx],
            state_prev[idx],
            cfg,
        )
        out_list.append(out)
        new_state_list.append(state)

    return jnp.stack(out_list, axis=0), jnp.stack(new_state_list, axis=0)
