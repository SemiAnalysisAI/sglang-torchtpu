"""Pallas ``_fused_moe_jax`` (production kernel body) vs a pure-JAX
reference MoE.

This file isolates the Pallas kernel's correctness from the
PyTorch <-> JAX <-> torch_tpu shuttle by calling the **production**
``_fused_moe_jax`` directly with JAX inputs (no torch custom op in the
loop), comparing against a dense JAX reference built from
``jnp.einsum`` + one-hot expert masking.

Both sides run on TPU under JAX, and the kernel under test is the
exact function ``_build_fused_moe_op`` exports to ``jax_op``. Any
production drift (sort order, pad amount, activation gating, tile
selection) lands here.

Run as:
    SGLANG_PLATFORM=tpu python \\
      test/manual/tpu/test_pallas_fused_moe.py
"""

from __future__ import annotations

import unittest

try:
    import jax
    import jax.numpy as jnp

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


_SKIP_REASON = "Requires JAX with a TPU backend visible to jax.devices()"


def _call_production_kernel(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    *,
    activation: str,
    no_combine: bool = False,
):
    """Invoke the production ``_fused_moe_jax`` for the unquant path.

    ``w*_scale`` / ``w*_bias`` are ``None`` -- the kernel signature
    accepts them positionally so a future quantised branch can plug in
    without churning this test.
    """
    # Lazy import so the module loads on hosts without JAX (the
    # @skipUnless decorator below will skip the whole class there).
    from sglang.srt.layers.moe.pallas.fused_moe_op import _fused_moe_jax

    topk = int(topk_ids.shape[-1])
    return _fused_moe_jax(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        None,  # w1_scale
        None,  # w2_scale
        None,  # w1_bias
        None,  # w2_bias
        topk,
        activation,
        no_combine,
    )


def _jax_reference_moe(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    activation: str,
    no_combine: bool = False,
):
    """Pure-JAX MoE reference -- dense einsum over a one-hot expert mask.

    Algebraically equivalent to ``_fused_moe_jax`` but with no Pallas
    grid, no grouped-matmul, no padding -- the diff vs the kernel can
    only be the kernel's own arithmetic.
    """
    m, k = hidden_states.shape
    g, _k, two_n = w1.shape
    assert _k == k
    n = two_n // 2

    hs = hidden_states.astype(jnp.float32)
    w1f = w1.astype(jnp.float32)
    w2f = w2.astype(jnp.float32)
    tw = topk_weights.astype(jnp.float32)

    # one-hot mask: [m, topk, g] picks the routed expert for each
    # (token, slot). Multiplying by it selects w1[e] / w2[e] per slot.
    onehot = jax.nn.one_hot(topk_ids, g, dtype=jnp.float32)  # [m, topk, g]

    # gmm1[m,j,2n] = sum_{e} onehot[m,j,e] * (hs[m,:] @ w1[e])
    gmm1 = jnp.einsum("mjg,mk,gkn->mjn", onehot, hs, w1f)

    gate, up = gmm1[..., :n], gmm1[..., n:]
    if activation == "silu":
        act = jax.nn.silu(gate) * up
    elif activation == "gelu":
        act = jax.nn.gelu(gate) * up
    else:
        raise AssertionError(f"unsupported activation in reference: {activation}")

    gmm2 = jnp.einsum("mjg,mjn,gnk->mjk", onehot, act, w2f)  # [m, topk, k]
    if no_combine:
        valid_slots = topk_weights.reshape(m, topk, 1) != 0
        return jnp.where(valid_slots, gmm2, jnp.zeros_like(gmm2)).astype(
            hidden_states.dtype
        )
    return jnp.einsum("mj,mjk->mk", tw, gmm2)


def _make_inputs(m, k, n, g, topk, seed, dtype=None):
    """JAX-on-TPU inputs with magnitudes chosen to stay inside bf16's
    accurate range (sub-saturated gates, sub-overflow accumulators).
    """
    if dtype is None:
        dtype = jnp.bfloat16
    key = jax.random.PRNGKey(seed)
    kh, kw1, kw2, ktw, kids = jax.random.split(key, 5)
    hidden = (jax.random.normal(kh, (m, k), dtype=jnp.float32) * 0.5).astype(dtype)
    w1 = (jax.random.normal(kw1, (g, k, 2 * n), dtype=jnp.float32) * 0.05).astype(dtype)
    w2 = (jax.random.normal(kw2, (g, n, k), dtype=jnp.float32) * 0.05).astype(dtype)
    topk_ids = jax.random.randint(kids, (m, topk), 0, g, dtype=jnp.int32)
    topk_weights = jax.nn.softmax(
        jax.random.normal(ktw, (m, topk), dtype=jnp.float32), axis=-1
    ).astype(dtype)
    return hidden, w1, w2, topk_weights, topk_ids


@unittest.skipUnless(_has_tpu_via_jax(), _SKIP_REASON)
class TestFusedMoEJaxKernelVsReference(unittest.TestCase):
    """Compare the production ``_fused_moe_jax`` body to a dense JAX
    reference. Any drift in the production kernel's permute / GMM1 /
    activation / GMM2 / un-permute path lands here as a numerics diff,
    independent of the torch_tpu bridge.
    """

    def _run_and_diff(
        self,
        *,
        m,
        k,
        n,
        g,
        topk,
        activation,
        seed,
        tol,
        force_ids=None,
        force_weights=None,
        no_combine=False,
    ):
        hidden, w1, w2, w_tk, ids = _make_inputs(m, k, n, g, topk, seed)
        if force_ids is not None:
            ids = force_ids
        if force_weights is not None:
            w_tk = force_weights

        out = jax.block_until_ready(
            _call_production_kernel(
                hidden,
                w1,
                w2,
                w_tk,
                ids,
                activation=activation,
                no_combine=no_combine,
            )
        )
        ref = _jax_reference_moe(
            hidden, w1, w2, w_tk, ids, activation, no_combine=no_combine
        )

        out32 = jnp.asarray(out).astype(jnp.float32)
        ref32 = ref.astype(jnp.float32)

        self.assertEqual(out32.shape, ref32.shape)
        self.assertTrue(
            bool(jnp.all(jnp.isfinite(out32))),
            "Pallas kernel produced non-finite output",
        )
        max_diff = float(jnp.max(jnp.abs(out32 - ref32)))
        self.assertLess(
            max_diff,
            tol,
            f"Pallas vs dense JAX max diff = {max_diff} "
            f"(activation={activation}, m={m}, k={k}, n={n}, g={g}, topk={topk})",
        )
        return out32, ref32, max_diff

    # ---- shape-aligned (no M padding) cases ---------------------------
    def test_silu_aligned_m_matches_reference(self):
        """m*topk = 64 -- a multiple of the 32-row sublane tile, so the
        kernel's ``pad_m=0`` fast path runs. Pinning this avoids a
        regression in the no-pad branch.
        """
        self._run_and_diff(
            m=32,
            k=64,
            n=64,
            g=4,
            topk=2,
            activation="silu",
            seed=0,
            tol=5e-2,
        )

    def test_gelu_aligned_m_matches_reference(self):
        """Same shape, different gate. Catches drift between the silu /
        gelu branches in ``_activation``."""
        self._run_and_diff(
            m=32,
            k=64,
            n=64,
            g=4,
            topk=2,
            activation="gelu",
            seed=1,
            tol=5e-2,
        )

    # ---- M-padding cases ----------------------------------------------
    def test_silu_unaligned_m_pad_path_matches_reference(self):
        """m*topk = 24 -- exercises the pad-to-32 branch.

        Without the pad, ``gmm_v2.zero_out_start`` does
        ``zero_ref.reshape(-1, size_lhs_sublane, num_lanes)`` and raises
        a Pallas ``ValueError`` at JAX trace time. Either failure mode
        is a clean signal.
        """
        self._run_and_diff(
            m=3,
            k=64,
            n=64,
            g=4,
            topk=8,
            activation="silu",
            seed=2,
            tol=5e-2,
        )

    def test_silu_odd_topk_matches_reference(self):
        """topk=3 -- the m*topk % 32 != 0 path with a less symmetric
        ratio (m=8, topk=3 -> 24 rows). Picks up regressions in the
        pad/un-permute path that aligned shapes would miss."""
        self._run_and_diff(
            m=8,
            k=64,
            n=64,
            g=4,
            topk=3,
            activation="silu",
            seed=3,
            tol=5e-2,
        )

    # ---- expert-coverage cases ----------------------------------------
    def test_silu_full_expert_coverage_matches_reference(self):
        """Round-robin routing -- every expert is hit at least once,
        so ``group_sizes`` has no zero entries. Pair with the next
        test to cover both arms of the gmm_v2 empty-group branch.
        """
        m, topk, g = 64, 2, 4
        flat = jnp.arange(m * topk, dtype=jnp.int32) % g
        ids = flat.reshape(m, topk)
        self._run_and_diff(
            m=m,
            k=64,
            n=64,
            g=g,
            topk=topk,
            activation="silu",
            seed=4,
            tol=5e-2,
            force_ids=ids,
        )

    def test_silu_single_expert_dominates_matches_reference(self):
        """All routing collapses to expert 0 -- ``group_sizes`` is
        ``[m*topk, 0, 0, ...]``. The kernel must still process empty
        expert groups without leaking writes into their output rows.
        """
        m, topk, g = 16, 2, 4
        ids = jnp.zeros((m, topk), dtype=jnp.int32)
        self._run_and_diff(
            m=m,
            k=64,
            n=64,
            g=g,
            topk=topk,
            activation="silu",
            seed=5,
            tol=5e-2,
            force_ids=ids,
        )

    def test_silu_no_combine_matches_reference(self):
        """Post-expert-norm MoE models need raw [T, topk, H] expert slots."""
        self._run_and_diff(
            m=8,
            k=64,
            n=64,
            g=4,
            topk=3,
            activation="silu",
            seed=6,
            tol=5e-2,
            no_combine=True,
        )

    def test_silu_no_combine_zero_weight_slots_are_zeroed(self):
        """EP-masked slots are routed to expert 0 with zero kernel weights.

        The no_combine path must not return expert-0 garbage for those slots,
        because such models apply a post-expert norm before their later weighted sum.
        """
        m, k, n, g, topk = 8, 64, 64, 4, 3
        hidden, w1, w2, w_tk, ids = _make_inputs(m, k, n, g, topk, seed=7)
        del hidden, w1, w2
        w_tk = w_tk.at[0, 1].set(jnp.asarray(0, dtype=w_tk.dtype))
        out, ref, _ = self._run_and_diff(
            m=m,
            k=k,
            n=n,
            g=g,
            topk=topk,
            activation="silu",
            seed=7,
            tol=5e-2,
            force_ids=ids,
            force_weights=w_tk,
            no_combine=True,
        )
        self.assertTrue(bool(jnp.all(out[0, 1] == 0)))
        self.assertTrue(bool(jnp.all(ref[0, 1] == 0)))


@unittest.skipUnless(_has_tpu_via_jax(), _SKIP_REASON)
class TestFusedMoEJaxKernelGuards(unittest.TestCase):
    """Negative tests pinning the kernel's documented error contracts
    (raised via the module-level ``_activation`` helper)."""

    def test_swiglu_oai_raises_not_implemented(self):
        """The GPT-OSS-specific gating is not wired up; the kernel must
        say so loudly rather than silently dispatching to silu/gelu."""
        from sglang.srt.layers.moe.pallas.fused_moe_op import _activation

        # Build a dummy [m*topk, 2n] array — the function only cares
        # about its last dim being even.
        x = jnp.zeros((4, 8), dtype=jnp.bfloat16)
        with self.assertRaises(NotImplementedError) as cm:
            _activation(x, "swiglu_oai")
        self.assertIn("swiglu_oai", str(cm.exception))

    def test_unknown_activation_raises_not_implemented(self):
        """A typo or future activation name shouldn't fall through to a
        default branch -- it must surface as NotImplementedError naming
        the offending value, so a model that requests a new activation
        gets a clear failure mode.
        """
        from sglang.srt.layers.moe.pallas.fused_moe_op import _activation

        x = jnp.zeros((4, 8), dtype=jnp.bfloat16)
        with self.assertRaises(NotImplementedError) as cm:
            _activation(x, "relu2")
        self.assertIn("relu2", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
