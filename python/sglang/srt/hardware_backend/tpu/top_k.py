# SPDX-License-Identifier: Apache-2.0
"""Exact TPU top-k backed by JAX's native XLA top-k primitive."""

from collections.abc import Callable
from threading import Lock

import torch

_TPU_TOP_K_OP: Callable | None = None
_TPU_TOP_K_LOCK = Lock()


def _get_tpu_top_k_op() -> Callable:
    """Lazily register the TorchTPU custom op once per process."""

    global _TPU_TOP_K_OP
    if _TPU_TOP_K_OP is not None:
        return _TPU_TOP_K_OP

    with _TPU_TOP_K_LOCK:
        if _TPU_TOP_K_OP is not None:
            return _TPU_TOP_K_OP

        import jax

        from sglang.srt.hardware_backend.tpu.pallas_jax_op import cpu_export_jax_op

        def _wrapped(values: jax.Array, k: int) -> tuple[jax.Array, jax.Array]:
            # torch_tpu's aten::topk lowering is currently a full StableHLO
            # sort plus slice. lax.top_k exports XLA's native exact TopK op.
            return jax.lax.top_k(values, k)

        op = cpu_export_jax_op("sglang_tpu_pallas::exact_top_k", _wrapped)

        def _fake(values: torch.Tensor, k: int):
            shape = (*values.shape[:-1], k)
            return (
                values.new_empty(shape),
                values.new_empty(shape, dtype=torch.int32),
            )

        op.register_fake(_fake)
        _TPU_TOP_K_OP = op
        return op


def tpu_top_k(values: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact descending top-k values and int32 indices on TPU."""

    if values.device.type != "tpu":
        raise ValueError(f"tpu_top_k requires a TPU tensor, got {values.device}")
    if values.ndim != 2:
        raise ValueError(f"tpu_top_k expects a 2D tensor, got shape {values.shape}")
    if not 0 < k <= values.shape[-1]:
        raise ValueError(f"k must be in [1, {values.shape[-1]}], got {k}")
    return _get_tpu_top_k_op()(values, k)
