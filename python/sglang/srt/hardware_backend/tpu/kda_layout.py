# SPDX-License-Identifier: Apache-2.0
"""Layout of the packed KDA kernel weights.

Kimi-K3 packs its conv weight, dt_bias and A_log into one f32 operand after
load (``models/kimi_k3.py``); the GDN/KDA Pallas kernel reads it back
(``layers/attention/pallas/gdn/memory_ref.unpack_kda_weights``). Pure Python, so
the model imports it on hosts without JAX.
"""


def kda_packed_weights_shape(
    kernel_size: int, dim_size: int, num_v_heads: int, kq_head_dim: int
) -> tuple[int, int]:
    """[rows, width] of the packed f32 KDA weights: conv [K, dim] in rows
    [0, K), dt_bias [n_v, d_k] in rows [K, K + n_v), A_log [n_v] in row
    K + n_v. Rows round up to the f32 sublane tile (8), width to the lane
    tile (128)."""
    rows = -(-(kernel_size + num_v_heads + 1) // 8) * 8
    width = -(-max(dim_size, kq_head_dim, num_v_heads) // 128) * 128
    return rows, width
