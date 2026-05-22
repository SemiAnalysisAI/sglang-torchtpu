"""GPT-OSS MXFP4 → BF16 weight rewrite for the TPU torch_tpu fused-MoE path.

The Pallas fused-MoE op (``sglang.srt.layers.moe.pallas.fused_moe_op``)
operates in BF16 with gmm_v2's ``(g, k, n)`` LHS-RHS layout. GPT-OSS
ships MXFP4-packed weights in the HuggingFace
``gate_up_proj_blocks`` / ``down_proj_blocks`` convention:

* ``w13_weight``    : ``(E, 2*N, hidden//2)``  uint8 — packed FP4
* ``w13_weight_scale`` : ``(E, 2*N, hidden//32)`` uint8 — e8m0 scales
* ``w13_weight_bias``  : ``(E, 2*N)``          bf16 — pre-activation bias
* ``w2_weight``     : ``(E, hidden, N//2)``    uint8 — packed FP4
* ``w2_weight_scale``  : ``(E, hidden, N//32)`` uint8 — e8m0 scales
* ``w2_weight_bias``   : ``(E, hidden)``       bf16 — pre-activation bias

The first ``2*N`` axis of w13 is **interleaved**: rows alternate
``[gate_0, up_0, gate_1, up_1, ...]``. The fused-MoE kernel splits its
GMM1 output at the midpoint (``jnp.split(acc, 2, -1)``-style — see
``fused_moe_op._activation``), so this module deinterleaves to
``[gate_0..gate_{N-1}, up_0..up_{N-1}]`` before transposing.

Output layout (all bf16, contiguous, on the input device):

* ``w13``      : ``(E, hidden, 2*N)`` — gate ‖ up
* ``w13_bias`` : ``(E, 1, 2*N)``      — gate ‖ up, broadcast-shape for gmm_v2
* ``w2``       : ``(E, N, hidden)``
* ``w2_bias``  : ``(E, 1, hidden)``

This mirrors the BF16 triage path in torchtpu-vllm's
``layers/vllm/quantization/mxfp4.py``: dequant → deinterleave → pad N
→ transpose to gmm_v2's LHS-RHS convention.
"""

from __future__ import annotations

import torch

from sglang.srt.layers.quantization.mxfp4_tensor import MXFP4QuantizeUtil

MXFP4_BLOCK_SIZE = 32


def _dequant_mxfp4_to_bf16(
    packed: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Dequantize an MXFP4-packed tensor along its last axis to BF16.

    ``packed`` has its last axis halved by the uint4 packing
    (``L_packed = L_unpacked // 2``), and ``scale`` has its last axis
    blocked by 32 (``L_scale = L_unpacked // 32``).  Returns a tensor
    with the last axis equal to ``L_unpacked``.

    Runs in the input tensors' device.  ``MXFP4QuantizeUtil.dequantize``
    is pure PyTorch (no JAX), so it works under torch_tpu without
    crossing the JAX/PyTorch interop boundary.
    """
    assert packed.dtype == torch.uint8, f"expected uint8 packed FP4, got {packed.dtype}"
    assert scale.dtype == torch.uint8, f"expected uint8 e8m0 scale, got {scale.dtype}"
    # The util's dequant flattens the leading dims internally so it does
    # not care whether the input is rank-3 (per-expert) or rank-2.
    return MXFP4QuantizeUtil.dequantize(
        packed,
        dtype=torch.bfloat16,
        scale=scale,
        block_sizes=[MXFP4_BLOCK_SIZE],
    )


def _deinterleave_gate_up_last(t: torch.Tensor) -> torch.Tensor:
    """``[gate_0, up_0, gate_1, up_1, ...]`` → ``[gate_0..gate_{N-1}, up_0..up_{N-1}]``.

    Operates on the last axis. ``t.shape[-1]`` must be even.
    Mirrors the same view-permute-reshape AITER uses in the existing
    mxfp4.py CUDA path (lines 645-658), just with the interleave axis
    pinned to ``-1``.
    """
    assert (
        t.shape[-1] % 2 == 0
    ), f"expected even last dim for deinterleave, got {t.shape}"
    n_pair = t.shape[-1] // 2
    out = (
        t.reshape(*t.shape[:-1], n_pair, 2)
        .permute(*range(t.dim() - 1), t.dim(), t.dim() - 1)
        .contiguous()
        .reshape(*t.shape[:-1], 2 * n_pair)
    )
    return out


def _deinterleave_gate_up_axis_minus_two(t: torch.Tensor) -> torch.Tensor:
    """Same as ``_deinterleave_gate_up_last`` but on axis -2 (the 2*N row axis)."""
    assert (
        t.shape[-2] % 2 == 0
    ), f"expected even axis -2 for deinterleave, got {t.shape}"
    n_pair = t.shape[-2] // 2
    last = t.shape[-1]
    return (
        t.reshape(*t.shape[:-2], n_pair, 2, last)
        .permute(*range(t.dim() - 2), t.dim() - 1, t.dim() - 2, t.dim())
        .contiguous()
        .reshape(*t.shape[:-2], 2 * n_pair, last)
    )


def _pad_last_axis(t: torch.Tensor, multiple: int) -> tuple[torch.Tensor, int]:
    """Zero-pad ``t`` along its last axis up to the next multiple.

    Returns ``(padded, pad_amount)``. ``pad_amount == 0`` if already aligned.
    """
    cur = t.shape[-1]
    pad = (multiple - cur % multiple) % multiple
    if pad == 0:
        return t, 0
    pad_shape = list(t.shape)
    pad_shape[-1] = pad
    zeros = torch.zeros(pad_shape, dtype=t.dtype, device=t.device)
    return torch.cat([t, zeros], dim=-1), pad


def process_gpt_oss_mxfp4_weights_bf16(
    w13_q: torch.Tensor,
    w13_scale: torch.Tensor,
    w13_bias: torch.Tensor,
    w2_q: torch.Tensor,
    w2_scale: torch.Tensor,
    w2_bias: torch.Tensor,
    *,
    align_to: int = 128,
) -> tuple[
    torch.Tensor,
    None,
    torch.Tensor,
    torch.Tensor,
    None,
    torch.Tensor,
]:
    """Rewrite GPT-OSS MXFP4 expert weights for the BF16 Pallas fused-MoE.

    See the module docstring for input/output layouts. Returned
    ``w13_scale`` and ``w2_scale`` are ``None`` because the BF16 path
    folds the e8m0 scales into the weights at load time, and the JAX
    op treats ``None`` scales as "unquantized" (see
    ``fused_moe_op._fused_moe_jax``).

    ``align_to`` controls the per-expert N padding (gate / up halves
    are padded independently with zeros so the kernel's
    midpoint-split convention still works after padding).
    """
    # ---- w13: dequant + deinterleave + transpose ----
    # w13_q: (E, 2*N, hidden//2) -> dequant along last axis -> (E, 2*N, hidden)
    w13 = _dequant_mxfp4_to_bf16(w13_q, w13_scale)
    # Deinterleave [gate, up] pairs along the 2*N axis.
    w13 = _deinterleave_gate_up_axis_minus_two(w13)  # (E, 2*N, hidden), now [gate||up]
    # Cast bias to bf16 if needed and deinterleave on the 2*N axis (last dim of bias).
    w13_bias = w13_bias.to(torch.bfloat16)
    w13_bias = _deinterleave_gate_up_last(w13_bias)  # (E, 2*N), now [gate||up]

    # Pad N (half of the 2*N axis) so each of gate/up halves aligns to
    # ``align_to``. Padding zeros into both weights and bias keeps the
    # kernel's split-at-midpoint convention correct.
    E, two_n, hidden = w13.shape
    assert two_n % 2 == 0
    n_inter = two_n // 2
    pad_n = (align_to - n_inter % align_to) % align_to
    if pad_n > 0:
        gate, up = w13[:, :n_inter, :], w13[:, n_inter:, :]
        zeros_w = torch.zeros(E, pad_n, hidden, dtype=w13.dtype, device=w13.device)
        w13 = torch.cat([gate, zeros_w, up, zeros_w], dim=1)
        gate_b, up_b = w13_bias[:, :n_inter], w13_bias[:, n_inter:]
        zeros_b = torch.zeros(E, pad_n, dtype=w13_bias.dtype, device=w13_bias.device)
        w13_bias = torch.cat([gate_b, zeros_b, up_b, zeros_b], dim=1)
        n_inter += pad_n
        two_n = 2 * n_inter

    # Transpose to gmm_v2's (g, k, n) LHS-RHS layout for the GMM1 call.
    # (E, 2*N, hidden) -> (E, hidden, 2*N)
    w13 = w13.permute(0, 2, 1).contiguous()
    # gmm_v2 expects rhs_bias shape (g, 1, n) for broadcast.
    w13_bias = w13_bias.reshape(E, 1, two_n).contiguous()

    # ---- w2: dequant + transpose (no deinterleave) ----
    # w2_q: (E, hidden, N//2) -> dequant along last axis -> (E, hidden, N)
    w2 = _dequant_mxfp4_to_bf16(w2_q, w2_scale)
    w2_bias = w2_bias.to(torch.bfloat16)
    # Pad N to match the padded N from w13 (so GMM2's k dim matches the
    # post-activation tensor produced after the gate*up multiply).
    E2, hidden2, n_inter_w2 = w2.shape
    assert E2 == E and hidden2 == hidden
    if n_inter_w2 != n_inter:
        # Pad the trailing axis with zeros (the inputs into the kernel
        # are zero for these slots, so the contribution is zero too).
        pad = n_inter - n_inter_w2
        zeros = torch.zeros(E, hidden, pad, dtype=w2.dtype, device=w2.device)
        w2 = torch.cat([w2, zeros], dim=-1)
    # (E, hidden, N) -> (E, N, hidden)
    w2 = w2.permute(0, 2, 1).contiguous()
    w2_bias = w2_bias.reshape(E, 1, hidden).contiguous()

    return w13, None, w13_bias, w2, None, w2_bias
