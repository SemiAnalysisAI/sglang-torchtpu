# SPDX-License-Identifier: Apache-2.0
"""Torch custom op for the single-kernel sigmoid top-k router (decode)."""

# PEP 563 must remain disabled so `cpu_export_jax_op` sees `jax.Array` type
# objects rather than stringified annotations at registration time.

from threading import Lock
from typing import Callable, Optional

import jax

# Set by register_router_topk_op() at model init; the op cannot be built
# inside a torch.compile trace.
_OP: Optional[Callable] = None
_LOCK = Lock()


def register_router_topk_op() -> None:
    global _OP
    with _LOCK:
        if _OP is not None:
            return
        from sglang_tpu.kernels.router_topk_small_m import (
            router_topk_small_m,
        )
        from sglang_tpu.pallas_jax_op import cpu_export_jax_op

        def _op(
            logits: jax.Array, bias: jax.Array, topk: int, scale: float
        ) -> tuple[jax.Array, jax.Array]:
            return router_topk_small_m(logits, bias, topk, scale)

        op = cpu_export_jax_op("sglang_tpu_pallas::router_topk_small_m", _op)

        def _fake(logits, bias, topk, scale):
            import torch

            m = logits.shape[0]
            return (
                logits.new_empty((m, topk), dtype=torch.float32),
                logits.new_empty((m, topk), dtype=torch.int32),
            )

        op.register_fake(_fake)
        _OP = op


def router_topk_op() -> Optional[Callable]:
    return _OP
