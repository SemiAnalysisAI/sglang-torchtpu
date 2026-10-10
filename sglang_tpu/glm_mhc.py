# SPDX-License-Identifier: Apache-2.0
"""Torch custom ops for the fused GLM-5.3-Flash mHC site kernel."""

# PEP 563 must remain disabled so `cpu_export_jax_op` sees `jax.Array` type
# objects rather than stringified annotations at registration time.

from threading import Lock
from typing import Callable, Optional

import jax
import torch

from sglang_tpu.kernels.glm_mhc import mhc_site
from sglang_tpu.pallas_jax_op import cpu_export_jax_op

_OPS: Optional[tuple[Callable, Callable, Callable]] = None
_OPS_LOCK = Lock()


def register_glm_mhc_ops() -> None:
    """Register before torch.compile traces the model: building an op inside
    the dynamo trace fails in inspect.signature."""
    global _OPS
    with _OPS_LOCK:
        if _OPS is not None:
            return

        def _pre(
            res: jax.Array,
            fn: jax.Array,
            vec: jax.Array,
            norm_weight: jax.Array,
            hc: int,
            iters: int,
            rms_eps: float,
            hc_eps: float,
            norm_eps: float,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            return mhc_site(
                None,
                res,
                None,
                None,
                fn,
                vec,
                hc=hc,
                iters=iters,
                rms_eps=rms_eps,
                hc_eps=hc_eps,
                norm_weight=norm_weight,
                norm_eps=norm_eps,
            )

        def _seam(
            x: jax.Array,
            res: jax.Array,
            post: jax.Array,
            comb: jax.Array,
            fn: jax.Array,
            vec: jax.Array,
            norm_weight: jax.Array,
            hc: int,
            iters: int,
            rms_eps: float,
            hc_eps: float,
            norm_eps: float,
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
            return mhc_site(
                x,
                res,
                post,
                comb,
                fn,
                vec,
                hc=hc,
                iters=iters,
                rms_eps=rms_eps,
                hc_eps=hc_eps,
                norm_weight=norm_weight,
                norm_eps=norm_eps,
            )

        def _post(
            x: jax.Array,
            res: jax.Array,
            post: jax.Array,
            comb: jax.Array,
            hc: int,
        ) -> jax.Array:
            return mhc_site(
                x,
                res,
                post,
                comb,
                None,
                None,
                hc=hc,
                iters=1,
                rms_eps=0.0,
                hc_eps=0.0,
            )[0]

        def _gates_like(res, hc):
            tokens = res.shape[0]
            return (
                res.new_empty((tokens, res.shape[1] // hc)),
                res.new_empty((tokens, hc), dtype=torch.float32),
                res.new_empty((tokens, hc * hc), dtype=torch.float32),
            )

        pre = cpu_export_jax_op("sglang_tpu_pallas::glm_mhc_pre", _pre)
        pre.register_fake(lambda res, fn, vec, norm, hc, *_: _gates_like(res, hc))
        seam = cpu_export_jax_op("sglang_tpu_pallas::glm_mhc_seam", _seam)
        seam.register_fake(
            lambda x, res, post, comb, fn, vec, norm, hc, *_: (
                torch.empty_like(res),
                *_gates_like(res, hc),
            )
        )
        post = cpu_export_jax_op("sglang_tpu_pallas::glm_mhc_post", _post)
        post.register_fake(lambda x, res, *_: torch.empty_like(res))
        _OPS = (pre, seam, post)


def glm_mhc_ops() -> tuple[Callable, Callable, Callable]:
    assert _OPS is not None, "register_glm_mhc_ops() must run at model init"
    return _OPS
