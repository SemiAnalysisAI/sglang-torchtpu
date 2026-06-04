"""Regression tests for the TPU Pallas fused-MoE custom-op registration.

Parallel to :mod:`test.manual.tpu.test_pallas_rpa_v3_dispatch` for the
attention backend -- this file pins the TPU-specific dispatch glue
that turns ``UnquantizedFusedMoEMethod.forward_tpu`` into a usable
torch op.

The behaviour being guarded here:

  * A torch.library custom op named ``sglang_tpu_pallas::fused_moe``
    is registered and reachable via ``torch.ops.sglang_tpu_pallas``.
  * ``_build_fused_moe_op`` is idempotent — torch.library rejects
    double-registration of the same op with a confusing error, so the
    module-level ``_PALLAS_MOE_OP`` singleton must short-circuit on
    subsequent calls.
  * ``apply_tpu_patches`` (TPU backend init) invokes the builder via
    ``_prewarm_fused_moe`` so the heavyweight JAX export happens before
    dynamo tracing; if that hook becomes a no-op, the first ``forward_tpu``
    call pays the full registration cost (and may graph-break inside dynamo).

These tests require torch_tpu + JAX with a TPU backend visible to
``jax.devices()`` because ``_build_fused_moe_op`` does JAX export at
registration time.
"""

import unittest

import torch

from sglang.test.test_utils import CustomTestCase

try:
    import jax  # noqa: F401

    _HAS_JAX = True
except Exception:  # pragma: no cover
    _HAS_JAX = False


def _has_tpu_via_jax() -> bool:
    if not _HAS_JAX:
        return False
    try:
        return any(getattr(d, "platform", "") == "tpu" for d in jax.devices())
    except Exception:
        return False


_SKIP_REASON = "Requires torch_tpu + JAX with a TPU backend visible to jax.devices()"


@unittest.skipUnless(_has_tpu_via_jax(), _SKIP_REASON)
class TestPallasMoeOpRegistration(CustomTestCase):
    """The fused-MoE custom op must be registered under the expected
    qualified name so ``UnquantizedFusedMoEMethod.forward_tpu``'s call to
    ``fused_moe_call`` reaches the right kernel. A typo or namespace
    drift here would silently produce a separate, unregistered op."""

    def test_op_registered_under_expected_qualified_name(self):
        from sglang.srt.layers.moe.pallas.fused_moe_op import _build_fused_moe_op

        _build_fused_moe_op()
        # Reachable via the torch.ops namespace -- this is what dynamo
        # and the production model code path resolve at trace time.
        self.assertTrue(
            hasattr(torch.ops, "sglang_tpu_pallas"),
            "sglang_tpu_pallas namespace missing from torch.ops",
        )
        self.assertTrue(
            hasattr(torch.ops.sglang_tpu_pallas, "fused_moe"),
            "fused_moe op missing from sglang_tpu_pallas namespace",
        )


@unittest.skipUnless(_has_tpu_via_jax(), _SKIP_REASON)
class TestPallasMoeOpIdempotency(CustomTestCase):
    """``_build_fused_moe_op`` is called both at TPU pre-warm time and
    from ``fused_moe_call``. torch.library rejects double-registration
    of the same op, so subsequent calls must return the cached singleton
    instead of re-registering."""

    def test_build_returns_same_singleton(self):
        from sglang.srt.layers.moe.pallas.fused_moe_op import _build_fused_moe_op

        op_a = _build_fused_moe_op()
        op_b = _build_fused_moe_op()
        self.assertIs(op_a, op_b)


@unittest.skipUnless(_has_tpu_via_jax(), _SKIP_REASON)
class TestPallasMoePrewarmHook(CustomTestCase):
    """``apply_tpu_patches`` (TPU backend init) calls ``_prewarm_fused_moe``
    so the JAX export (~hundreds of ms) happens before the first forward.
    The hook must actually invoke ``_build_fused_moe_op`` -- a silent no-op
    would defer registration to first inference and may cause a dynamo
    graph break."""

    def test_prewarm_invokes_builder(self):
        import sglang.srt.layers.moe.pallas.fused_moe_op as fmop
        from sglang.srt.hardware_backend.tpu import patches

        # Spy on ``_build_fused_moe_op`` without re-registering (which
        # would fail because torch.library has already accepted the op
        # from prior test runs in this process). ``_prewarm_fused_moe``
        # imports the builder lazily, so it picks up the patched attribute.
        original = fmop._build_fused_moe_op
        called = []

        def spy():
            called.append(True)
            return original()

        fmop._build_fused_moe_op = spy
        try:
            patches._prewarm_fused_moe()
            self.assertTrue(
                called,
                "_prewarm_fused_moe did not invoke _build_fused_moe_op",
            )
        finally:
            fmop._build_fused_moe_op = original


if __name__ == "__main__":
    unittest.main()
