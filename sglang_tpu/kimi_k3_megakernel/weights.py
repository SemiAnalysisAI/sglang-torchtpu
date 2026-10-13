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


def _own_storage(tensor: torch.Tensor) -> torch.Tensor:
    """Return a contiguous base tensor (``_base is None``, ``storage_offset == 0``)."""
    tensor = tensor.contiguous()
    if tensor._base is None and tensor.storage_offset() == 0:
        return tensor
    out = torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)
    out.copy_(tensor)
    return out


def _numpy_to_torch(value: np.ndarray) -> torch.Tensor:
    """Host numpy array (incl. ml_dtypes bfloat16) -> torch base tensor, same bits."""
    arr = np.ascontiguousarray(value)
    if arr.dtype.name == "bfloat16":
        t = torch.empty(arr.shape, dtype=torch.bfloat16)
        t.view(torch.uint16).copy_(torch.from_numpy(arr.view(np.uint16)))
        return t
    if arr.dtype == np.uint32:
        t = torch.empty(arr.shape, dtype=torch.int32)
        t.copy_(torch.from_numpy(arr.view(np.int32)))
        return t
    if arr.dtype not in TORCH_STORAGE_DTYPE:
        raise TypeError(f"unsupported megakernel weight dtype {arr.dtype}")
    return _own_storage(torch.from_numpy(arr))


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
        out[name] = _own_storage(_numpy_to_torch(value[None]).to(device))
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

    Inputs are stacked over the 92 MoE layers (either 4D ``[L, 112, ...]`` or
    already 5D ``[1, L, 112, ...]``):
      w13       uint8 [L, 112, 1792, 1536]   (gate 768 | up 768 channels, K bytes)
      w13_scale f32   [L, 112, 112, 1, 1536]
      w2        uint8 [L, 112, 384, 3584]
      w2_scale  f32   [L, 112, 24, 1, 3584]
    The packed bytes are passed through unchanged (the kernel bitcasts the HBM
    ref to uint32 words); the scales become E8M0 bytes [1, L, 112, blocks, N].
    """
    if w13.dtype != torch.uint8 or w2.dtype != torch.uint8:
        raise TypeError("routed experts must be packed MXFP4 bytes")
    gus = e8m0_from_f32_scales(w13_scale).squeeze(-2)
    ds = e8m0_from_f32_scales(w2_scale).squeeze(-2)
    if gus.ndim == 4:
        gus = gus.unsqueeze(0)
    if ds.ndim == 4:
        ds = ds.unsqueeze(0)
    return {
        "expert_gu": w13 if w13.ndim == 5 else w13[None],
        "expert_gus": _own_storage(gus),
        "expert_down": w2 if w2.ndim == 5 else w2[None],
        "expert_ds": _own_storage(ds),
    }


def _prepare_moe_layer_on_cpu(moe) -> None:
    """Move an unprocessed MoE layer's parameters to CPU, freeing TPU storage first,
    then run ``quant_method.process_weights_after_loading`` on CPU."""
    if not (
        getattr(moe, "w13_weight_scale", None) is not None
        and moe.w13_weight_scale.ndim == 3
        and getattr(moe, "quant_method", None) is not None
    ):
        return
    for attr in (
        "w13_weight",
        "w13_weight_scale",
        "w13_weight_bias",
        "w2_weight",
        "w2_weight_scale",
        "w2_weight_bias",
    ):
        param = getattr(moe, attr, None)
        if isinstance(param, torch.Tensor) and param.device.type != "cpu":
            old_data = param.data
            cpu_copy = old_data.detach().cpu()
            param.data = torch.empty(0, dtype=old_data.dtype, device=old_data.device)
            if (
                old_data.untyped_storage().data_ptr()
                != param.data.untyped_storage().data_ptr()
            ):
                old_data.untyped_storage().resize_(0)
            del old_data
            if isinstance(param, torch.nn.Parameter):
                setattr(moe, attr, torch.nn.Parameter(cpu_copy, requires_grad=False))
            else:
                setattr(moe, attr, cpu_copy)
            del param
    moe.quant_method.process_weights_after_loading(moe)


def stack_sglang_moe_experts_in_place(moe_layers, device) -> dict[str, torch.Tensor]:
    """Stack per-layer MXFP4 MoE weights into shared 5D base tensors in-place.

    Each MoE layer's per-layer ``w13_weight`` and ``w2_weight`` storage is freed
    as it is copied to a host staging buffer, then moved back to ``device`` as a
    single 5D base tensor ``(1, L, E, K_packed, N)`` (``_base is None``). Both
    ``layer.w13_weight`` / ``layer.w2_weight`` (consumed by prefill ``gmm_v2``
    via ``layer._moe_layer_idx_tensor``) and the returned megakernel weight dict
    reference the exact same 5D base tensors with zero storage duplication and
    zero tensor views.
    """
    import gc

    moe_layers = list(moe_layers)
    num_layers = len(moe_layers)
    if num_layers == 0:
        raise ValueError("stack_sglang_moe_experts_in_place requires at least 1 layer")

    target_device = torch.device(device)
    first = moe_layers[0]
    _prepare_moe_layer_on_cpu(first)

    w13_shape = tuple(first.w13_weight.shape)
    w2_shape = tuple(first.w2_weight.shape)
    w13_s_shape = tuple(first.w13_weight_scale.shape)
    w2_s_shape = tuple(first.w2_weight_scale.shape)

    gu_cpu = torch.empty((1, num_layers, *w13_shape), dtype=torch.uint8, device="cpu")
    down_cpu = torch.empty((1, num_layers, *w2_shape), dtype=torch.uint8, device="cpu")
    gus_cpu = torch.empty(
        (1, num_layers, w13_s_shape[0], w13_s_shape[1], w13_s_shape[3]),
        dtype=torch.uint8,
        device="cpu",
    )
    ds_cpu = torch.empty(
        (1, num_layers, w2_s_shape[0], w2_s_shape[1], w2_s_shape[3]),
        dtype=torch.uint8,
        device="cpu",
    )

    for layer_idx, moe in enumerate(moe_layers):
        _prepare_moe_layer_on_cpu(moe)

        old_w13 = moe.w13_weight.data
        gu_cpu[0, layer_idx].copy_(old_w13.detach().cpu())
        moe.w13_weight.data = torch.empty(0, dtype=torch.uint8, device=old_w13.device)
        if (
            old_w13.untyped_storage().data_ptr()
            != moe.w13_weight.data.untyped_storage().data_ptr()
        ):
            old_w13.untyped_storage().resize_(0)
        del old_w13

        old_w2 = moe.w2_weight.data
        down_cpu[0, layer_idx].copy_(old_w2.detach().cpu())
        moe.w2_weight.data = torch.empty(0, dtype=torch.uint8, device=old_w2.device)
        if (
            old_w2.untyped_storage().data_ptr()
            != moe.w2_weight.data.untyped_storage().data_ptr()
        ):
            old_w2.untyped_storage().resize_(0)
        del old_w2

        w13_s_cpu = moe.w13_weight_scale.data.detach().cpu()
        w2_s_cpu = moe.w2_weight_scale.data.detach().cpu()
        gus_cpu[0, layer_idx].copy_(e8m0_from_f32_scales(w13_s_cpu).squeeze(-2))
        ds_cpu[0, layer_idx].copy_(e8m0_from_f32_scales(w2_s_cpu).squeeze(-2))
        del w13_s_cpu, w2_s_cpu

    gc.collect()
    if target_device.type == "tpu":
        from sglang.srt.platforms import current_platform

        current_platform.synchronize()

    expert_gu = _own_storage(gu_cpu.to(target_device))
    del gu_cpu
    expert_down = _own_storage(down_cpu.to(target_device))
    del down_cpu
    expert_gus = _own_storage(gus_cpu.to(target_device))
    del gus_cpu
    expert_ds = _own_storage(ds_cpu.to(target_device))
    del ds_cpu

    shared_w13 = torch.nn.Parameter(expert_gu, requires_grad=False)
    shared_w2 = torch.nn.Parameter(expert_down, requires_grad=False)
    for layer_idx, moe in enumerate(moe_layers):
        moe.w13_weight = shared_w13
        moe.w2_weight = shared_w2
        for attr in (
            "w13_weight_scale",
            "w2_weight_scale",
            "w13_weight_bias",
            "w2_weight_bias",
        ):
            param = getattr(moe, attr, None)
            if isinstance(param, torch.Tensor) and param.device != target_device:
                dev_t = _own_storage(param.data.to(target_device))
                if isinstance(param, torch.nn.Parameter):
                    setattr(moe, attr, torch.nn.Parameter(dev_t, requires_grad=False))
                else:
                    setattr(moe, attr, dev_t)
        moe._moe_layer_idx_tensor = _own_storage(
            torch.tensor([layer_idx], dtype=torch.int32, device=target_device)
        )
        if getattr(moe, "quant_method", None) is not None:
            moe.quant_method.process_weights_after_loading = lambda _layer: None

    return {
        "expert_gu": expert_gu,
        "expert_gus": expert_gus,
        "expert_down": expert_down,
        "expert_ds": expert_ds,
    }
