# SPDX-License-Identifier: Apache-2.0
"""Layout of Kimi-K3's KDA layers on TPU.

Kimi-K3 packs its conv weight, dt_bias and A_log into one f32 operand after
load (``pack_kda_weights``); the GDN/KDA Pallas kernel reads it back
(``layers/attention/pallas/gdn/memory_ref.unpack_kda_weights``). The Pallas KDA
o_norm reads g in place from the fused ``[q, k, v, g]`` projection
(``onorm_g_block``). Pure Python at import, so the model imports it on hosts
without JAX.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    import torch

    from sglang.kernels.ops.attention.fla.fused_norm_gate import FusedRMSNormGated

# Row cap for the Pallas KDA o_norm (SGLANG_TPU_ENABLE_K3_FUSED_ONORM): the
# kernel holds the whole batch in one VMEM block (x, g and the output take under
# 2 MiB at 12 heads per rank, double-buffered). 64, the largest batch measured,
# covers decode batches of up to 64 requests and extend chunks that small.
ONORM_MAX_TOKENS = 64


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


def pack_kda_weights(
    conv_weight: torch.Tensor,
    dt_bias: torch.Tensor,
    a_log: torch.Tensor,
    num_v_heads: int,
    kq_head_dim: int,
) -> torch.Tensor:
    """Host f32 array in the ``kda_packed_weights_shape`` layout from the
    depthwise conv weight [dim, 1, K], dt_bias (n_v * d_k values) and A_log
    (n_v values). Built on the host and moved once: slice-assigning into a
    device tensor leaves a view chain that torch_tpu re-materializes before
    every forward."""
    import torch

    dim, kernel_size = conv_weight.shape[0], conv_weight.shape[-1]
    rows, width = kda_packed_weights_shape(kernel_size, dim, num_v_heads, kq_head_dim)
    packed = torch.zeros(rows, width, dtype=torch.float32)
    with torch.no_grad():
        packed[:kernel_size, :dim] = (
            conv_weight.detach().float().reshape(dim, kernel_size).t().cpu()
        )
        packed[kernel_size : kernel_size + num_v_heads, :kq_head_dim] = (
            dt_bias.detach().float().reshape(num_v_heads, kq_head_dim).cpu()
        )
        packed[kernel_size + num_v_heads, :num_v_heads] = (
            a_log.detach().float().reshape(num_v_heads).cpu()
        )
    return packed


def onorm_g_block(
    *,
    use_full_rank_gate: bool,
    head_dim: int,
    activation: str,
    split_sizes: Sequence[int] | None,
) -> int | None:
    """Column block of g in the fused ``[q, k, v, g]`` projection when
    SGLANG_TPU_ENABLE_K3_FUSED_ONORM is set and the layer can take the Pallas
    o_norm (full-rank gate, 128-lane heads, sigmoid gate), else None. Registers
    the op, which cannot be created inside the dynamo trace."""
    from sglang.srt.environ import envs

    if not envs.SGLANG_TPU_ENABLE_K3_FUSED_ONORM.get():
        return None
    if not (use_full_rank_gate and head_dim % 128 == 0 and activation == "sigmoid"):
        from sglang.srt.utils.common import print_warning_once

        print_warning_once(
            "SGLANG_TPU_ENABLE_K3_FUSED_ONORM is ignored: the Pallas o_norm needs "
            "the full-rank gate, head_dim % 128 == 0 and a sigmoid gate "
            f"(use_full_rank_gate={use_full_rank_gate}, head_dim={head_dim}, "
            f"activation={activation})."
        )
        return None
    # g follows [q, k, v], three g widths in: a whole block index.
    g_offset, g_width = split_sizes
    assert g_offset % g_width == 0, split_sizes
    from sglang.srt.hardware_backend.tpu.gated_rms_norm import get_gated_rms_norm_op

    get_gated_rms_norm_op()
    return g_offset // g_width


def gated_onorm(
    core_attn_out: torch.Tensor,
    qkvg: torch.Tensor,
    o_norm: FusedRMSNormGated,
    g_block: int,
) -> torch.Tensor | None:
    """The KDA gated o_norm through the Pallas kernel: the ``[1, T, H, D]`` KDA
    output, gated by column block ``g_block`` of the fused ``[q, k, v, g]``
    projection ``qkvg`` (read in place: a split view would be copied at the op
    boundary), in the same shape. None past ONORM_MAX_TOKENS rows."""
    num_tokens = qkvg.shape[0]
    if not 0 < num_tokens <= ONORM_MAX_TOKENS:
        return None
    from sglang.srt.hardware_backend.tpu.gated_rms_norm import gated_rms_norm_tpu

    head_dim = o_norm.weight.shape[0]
    out = gated_rms_norm_tpu(
        core_attn_out.reshape(num_tokens, -1, head_dim),
        qkvg,
        o_norm.weight,
        o_norm.eps,
        g_block=g_block,
    )
    return out.view(core_attn_out.shape)
