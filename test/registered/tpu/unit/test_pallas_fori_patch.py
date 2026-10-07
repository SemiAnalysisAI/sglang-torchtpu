"""The Pallas fori_loop patch drops only the empty dynamic-remainder loop."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=60, suite="stage-a-test-tpu-unit")

import unittest
from unittest.mock import patch

from sglang.test.test_utils import CustomTestCase

try:
    import jax
    import jax.numpy as jnp
    from jax._src.pallas.mosaic import lowering, pallas_call_registration
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
except ImportError:  # pragma: no cover - CPU-only host
    jax = None


def _tpu_jax_available() -> bool:
    try:
        return bool(jax and jax.devices("tpu"))
    except RuntimeError:
        return False


def _kernel(n_ref, x_ref, o_ref):
    o_ref[...] = jax.lax.fori_loop(
        0, n_ref[0], lambda i, acc: acc + x_ref[...], jnp.zeros_like(o_ref)
    )


def _call(n, x):
    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
    )(n, x)


def _num_mosaic_loops() -> int:
    """Lower for TPU (no device needed) and count scf.for before canonicalize."""
    jax.clear_caches()  # a cached lowering would skip the capture below
    modules = []
    orig = pallas_call_registration.lowering.lower_jaxpr_to_pipelined_module

    def capture(*args, **kwargs):
        module = orig(*args, **kwargs)
        modules.append(str(module))
        return module

    with patch.object(
        pallas_call_registration.lowering, "lower_jaxpr_to_pipelined_module", capture
    ):
        jax.jit(_call).trace(
            jax.ShapeDtypeStruct((1,), jnp.int32),
            jax.ShapeDtypeStruct((8, 128), jnp.float32),
        ).lower(lowering_platforms=("tpu",))
    return modules[0].count("scf.for")


@unittest.skipIf(jax is None, "requires jax with Pallas TPU")
class TestPallasForiPatch(CustomTestCase):
    def test_drops_dead_remainder_loop(self):
        from sglang.srt.hardware_backend.tpu.patches import (
            _patch_jax_pallas_fori_lowering,
        )

        # Restore the unpatched lowering so other tests see stock JAX.
        with patch.dict(lowering.__dict__):
            lowering.__dict__.pop("_sglang_fori_patch_applied", None)
            before = _num_mosaic_loops()
            _patch_jax_pallas_fori_lowering()
            _patch_jax_pallas_fori_lowering()  # idempotent
            after = _num_mosaic_loops()
        if before == 1:
            self.skipTest("this JAX no longer emits the remainder loop")
        self.assertEqual((before, after), (2, 1))

    @unittest.skipUnless(_tpu_jax_available(), "requires a TPU visible to JAX")
    def test_patched_loop_runs_every_iteration(self):
        from sglang.srt.hardware_backend.tpu.patches import (
            _patch_jax_pallas_fori_lowering,
        )

        with patch.dict(lowering.__dict__):
            lowering.__dict__.pop("_sglang_fori_patch_applied", None)
            _patch_jax_pallas_fori_lowering()
            x = jnp.arange(8 * 128, dtype=jnp.float32).reshape(8, 128)
            fn = jax.jit(_call)
            for n in (0, 1, 7):
                out = fn(jnp.array([n], dtype=jnp.int32), x)
                self.assertTrue(bool(jnp.array_equal(out, n * x)), n)


if __name__ == "__main__":
    unittest.main()
