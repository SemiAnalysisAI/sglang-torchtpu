# Adapted from vllm-project/tpu-inference, the megablox gmm_v2 kernel's
# activation/alignment helpers (``swigluoai`` and ``align_to``):
#   https://github.com/vllm-project/tpu-inference/blob/main/tpu_inference/kernels/megablox/gmm_v2.py
#
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
"""Activation and shape-alignment helpers used by the Pallas fused-MoE
kernel (``gmm_v2.py``).
"""

import inspect
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def make_compiler_params(**kwargs):
    """Safely construct pltpu.CompilerParams filtering out deprecated or unsupported kwargs."""
    sig = inspect.signature(pltpu.CompilerParams.__init__)
    valid_params = set(sig.parameters.keys())
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return pltpu.CompilerParams(**filtered_kwargs)


def get_pallas_indirect():
    """Safely resolve Indirect/Blocked block spec class across JAX/Pallas versions."""
    if hasattr(pltpu, "Indirect"):
        return pltpu.Indirect
    if hasattr(pl, "tpu") and hasattr(pl.tpu, "Indirect"):
        return pl.tpu.Indirect
    if hasattr(pl, "Indirect"):
        return pl.Indirect
    if hasattr(pl, "Blocked"):
        return pl.Blocked
    raise AttributeError("Could not find Indirect or Blocked class in jax.experimental.pallas.")



Indirect = get_pallas_indirect()


# Patch pl.when, lax.select, and lax.cond to handle non-scalar conditions (e.g. bool[16]) in JAX 0.9.2
try:
    from jax import lax
    from jax._src.lax import control_flow

    _orig_select = lax.select
    def _safe_select(pred, on_true, on_false):
        if hasattr(on_true, "shape") and getattr(on_true, "ndim", 0) > 0:
            pred_shape = getattr(pred, "shape", ())
            if pred_shape != on_true.shape:
                if pred_shape == ():
                    pred = jnp.broadcast_to(pred, on_true.shape)
                else:
                    pred = jnp.any(pred)
        elif hasattr(pred, "shape") and getattr(pred, "ndim", 0) > 0:
            pred = jnp.any(pred)
        return _orig_select(pred, on_true, on_false)


    lax.select = _safe_select
    control_flow.select = _safe_select

    _orig_cond = lax.cond
    def _safe_cond(pred, true_fun, false_fun, *operands, **kwargs):
        if hasattr(pred, "shape") and getattr(pred, "ndim", 0) > 0:
            pred = jnp.any(pred)
        return _orig_cond(pred, true_fun, false_fun, *operands, **kwargs)

    lax.cond = _safe_cond
    control_flow.cond = _safe_cond

    from jax._src.pallas import helpers as pallas_helpers
    _orig_when = pallas_helpers.when

    def _safe_when(condition):
        if hasattr(condition, "shape") and getattr(condition, "ndim", 0) > 0:
            condition = jnp.any(condition)
        return _orig_when(condition)

    pallas_helpers.when = _safe_when
    pl.when = _safe_when

    # Patch sc_core.supported_shapes in JAX 0.9.2 to accept (1, 16) 2D vector shapes
    import jax._src.pallas.mosaic.sc_core as sc_core
    _orig_supported_shapes = sc_core.supported_shapes
    def _safe_supported_shapes(dtype):
        res = _orig_supported_shapes(dtype)
        if (16,) in res and (1, 16) not in res:
            res = list(res) + [(1, 16)]
        return res
    sc_core.supported_shapes = _safe_supported_shapes
except Exception:
    pass






def swigluoai(
    gate: jax.Array, up: jax.Array, *, alpha: float = 1.702, limit: float = 7.0
) -> jax.Array:
    """Activation used in some models such as GPT-OSS."""

    gate = jnp.clip(gate, max=limit)
    up = jnp.clip(up, min=-limit, max=limit)
    glu = gate * jax.nn.sigmoid(alpha * gate)
    return (up + 1.0) * glu


def apply_act_fn(acc: jax.Array, fuse_act: str | None):
    """Apply a fused activation function to the accumulator.

    The input ``acc`` is expected to contain concatenated results for
    both the 'gate' and 'up' projections (last dim = ``2 * tile_n``).

    Returns ``acc`` unchanged when ``fuse_act`` is None. Raises
    ``NotImplementedError`` for unsupported activations. Supported:
    ``silu``, ``gelu``, ``swigluoai``.
    """

    if fuse_act is None:
        return acc

    acc_gate, acc_up = jnp.split(acc, 2, -1)
    match fuse_act:
        case "silu":
            return jax.nn.silu(acc_gate) * acc_up
        case "gelu":
            return jax.nn.gelu(acc_gate) * acc_up
        case "swigluoai":
            return swigluoai(acc_gate, acc_up)
        case _:
            raise NotImplementedError(f"Unsupported activation function: {fuse_act}")


def align_to(x, a):
    """Round ``x`` up to the next multiple of ``a``."""
    return pl.cdiv(x, a) * a
