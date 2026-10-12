# SPDX-License-Identifier: Apache-2.0
"""Per-rank Kimi-K3 megakernel weights as torch tensors on the TPU.

The megakernel consumes one weight dictionary per rank (``selected_local_shapes``
of the vendored loader). Two sources fill it:

* Dense families (attention projections, norms, router, latent MoE front/back,
  shared experts, embedding, lm_head) are read from the Hugging Face checkpoint
  with the vendored reader's own per-rank slicing, so their layout is exactly
  what the kernel was validated with.
* The four routed-expert families reuse SGLang's MXFP4 expert tensors (TP32 with
  ``--ep-size 8``: 112 experts per group of 4 ranks, channel quarter
  ``rank % 4``). The packed bytes are shared; only the E8M0 scales are rebuilt,
  because SGLang keeps them as f32 powers of two.

Every tensor handed to the kernel is a whole base tensor: torch_tpu would
re-materialize a view chain on every compiled call.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

logger = logging.getLogger(__name__)

EXPERT_FAMILIES = ("expert_gu", "expert_gus", "expert_down", "expert_ds")
# Kernel-side dtypes torch stores differently: uint32 words are kept as int32
# (torch_tpu has no uint32 device ops) and bitcast back inside the exported
# JAX program.
TORCH_STORAGE_DTYPE = {
    np.dtype(np.float32): torch.float32,
    np.dtype(np.uint8): torch.uint8,
    np.dtype(np.int32): torch.int32,
    np.dtype(np.uint32): torch.int32,
}


def _numpy_to_torch(value: np.ndarray) -> torch.Tensor:
    """Host numpy array (incl. ml_dtypes bfloat16) -> torch tensor, same bits."""
    if value.dtype.name == "bfloat16":
        return torch.from_numpy(np.ascontiguousarray(value).view(np.uint16)).view(
            torch.bfloat16
        )
    if value.dtype == np.uint32:
        return torch.from_numpy(np.ascontiguousarray(value).view(np.int32))
    if value.dtype not in TORCH_STORAGE_DTYPE:
        raise TypeError(f"unsupported megakernel weight dtype {value.dtype}")
    return torch.from_numpy(np.ascontiguousarray(value))


def load_dense_families(checkpoint_path: str, rank: int, device, *, layers: int = 93):
    """Dense megakernel families for ``rank`` read from the checkpoint.

    Returns ``{name: tensor}`` with a leading rank axis of size 1 (the shape the
    shard_map'd kernel sees per device), each a base tensor on ``device``.
    """
    from ._vendor.kimi.load import Checkpoint, _family_layers, selected_local_shapes

    checkpoint = Checkpoint(checkpoint_path)
    shapes = selected_local_shapes(
        layers, checkpoint.config.dim, vocab=checkpoint.config.vocab
    )
    out = {}
    for name, shape in shapes.items():
        if name in EXPERT_FAMILIES:
            continue
        ids = _family_layers(checkpoint.config, name, layers)
        if ids is None:
            value = checkpoint.dense_selected(name, 0, rank)
        else:
            first = checkpoint.dense_selected(name, ids[0], rank)
            value = np.empty(shape, first.dtype)
            value[0] = first
            with ThreadPoolExecutor(checkpoint.READ_THREADS) as pool:
                for i, layer_value in zip(
                    range(1, len(ids)),
                    pool.map(
                        lambda layer: checkpoint.dense_selected(name, layer, rank),
                        ids[1:],
                    ),
                ):
                    value[i] = layer_value
        if tuple(value.shape) != tuple(shape):
            raise ValueError(
                f"{name}: checkpoint gives {value.shape}, kernel expects {shape}"
            )
        out[name] = _numpy_to_torch(value[None]).to(device)
    return out


def e8m0_from_f32_scales(scales: torch.Tensor) -> torch.Tensor:
    """f32 power-of-two scales -> E8M0 bytes (the f32 exponent field), exactly.

    SGLang stores MXFP4 block scales as f32 ``2**(e - 127)``; the megakernel
    reads the E8M0 byte ``e``. Rejects anything that is not a normal power of
    two, so a silent rounding can never reach the kernel.
    """
    bits = scales.contiguous().view(torch.int32)
    mantissa = bits & 0x7FFFFF
    exponent = (bits >> 23) & 0xFF
    if (
        bool((mantissa != 0).any())
        or bool((bits < 0).any())
        or bool(((exponent == 0) | (exponent == 255)).any())
    ):
        raise ValueError("MXFP4 scales must be positive normal powers of two")
    return exponent.to(torch.uint8)


def expert_families_from_sglang(
    w13: torch.Tensor, w13_scale: torch.Tensor, w2: torch.Tensor, w2_scale: torch.Tensor
):
    """Megakernel routed-expert families from SGLang's per-rank MXFP4 tensors.

    Inputs are stacked over the 92 MoE layers:
      w13       uint8 [L, 112, 1792, 1536]   (gate 768 | up 768 channels, K bytes)
      w13_scale f32   [L, 112, 112, 1, 1536]
      w2        uint8 [L, 112, 384, 3584]
      w2_scale  f32   [L, 112, 24, 1, 3584]
    The packed bytes are passed through unchanged (the kernel bitcasts the HBM
    ref to uint32 words); the scales become E8M0 bytes [L, 112, blocks, N].
    """
    if w13.dtype != torch.uint8 or w2.dtype != torch.uint8:
        raise TypeError("routed experts must be packed MXFP4 bytes")
    gus = e8m0_from_f32_scales(w13_scale).squeeze(-2)
    ds = e8m0_from_f32_scales(w2_scale).squeeze(-2)
    return {
        "expert_gu": w13[None],
        "expert_gus": gus[None].contiguous(),
        "expert_down": w2[None],
        "expert_ds": ds[None].contiguous(),
    }
