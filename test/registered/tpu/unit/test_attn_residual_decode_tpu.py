"""Fused Pallas attention-residual decode kernel vs a float64 reference."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=20, suite="stage-a-test-tpu-unit", chips=1)

import unittest

from sglang.test.test_utils import CustomTestCase

try:
    import jax
    import jax.numpy as jnp
    import numpy as np

    _HAS_TPU = any(d.platform == "tpu" for d in jax.devices())
except Exception:  # pragma: no cover - non-TPU host
    _HAS_TPU = False

EPS, OUT_EPS = 1e-6, 1e-5


def _reference(prefix, bank, nvb, cw, norm_w):
    """score = rms_norm(v) . cw; softmax over [bank[:nvb], prefix]; mix; RMSNorm."""
    rows = np.concatenate([bank[:, :nvb], prefix[:, None]], axis=1).astype(np.float64)
    inv = 1.0 / np.sqrt((rows**2).mean(-1) + EPS)
    scores = (rows * cw).sum(-1) * inv
    p = np.exp(scores - scores.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    mixed = (p[..., None] * rows).sum(1)
    mixed = mixed.astype(jnp.bfloat16).astype(np.float64)
    out = mixed / np.sqrt((mixed**2).mean(-1, keepdims=True) + OUT_EPS)
    return out * norm_w


@unittest.skipUnless(_HAS_TPU, "requires a TPU backend")
class TestAttnResidualDecodeKernel(CustomTestCase):
    H, NB = 1024, 6

    def _case(self, tokens, nvb, addend, seed):
        from sglang_tpu.attn_residual_tpu import attention_residual_decode_jax

        rng = np.random.default_rng(seed)
        bf = lambda *s: jnp.asarray(rng.standard_normal(s), jnp.bfloat16)
        a, b, bank = bf(tokens, self.H), bf(tokens, self.H), bf(tokens, self.NB, self.H)
        cw = jnp.asarray(rng.standard_normal(self.H) * 0.05, jnp.float32)
        norm_w = bf(self.H)
        normed, prefix = attention_residual_decode_jax(
            a, b if addend else None, bank, cw, norm_w, nvb, EPS, OUT_EPS
        )
        f64 = lambda x: np.asarray(jnp.asarray(x, jnp.float32), np.float64)
        normed = normed.reshape(tokens, self.H)
        want_prefix = (f64(a) + f64(b)) if addend else f64(a)
        want_prefix = np.asarray(jnp.asarray(want_prefix, jnp.bfloat16), np.float64)
        np.testing.assert_array_equal(f64(prefix), want_prefix)
        want = _reference(want_prefix, f64(bank), nvb, f64(cw), f64(norm_w))
        err = np.abs(f64(normed) - want).max() / np.abs(want).max()
        self.assertLess(err, 1e-2, f"tokens={tokens} nvb={nvb} addend={addend}")

    def test_matches_reference(self):
        for tokens in (1, 4):
            for nvb in (1, 3, self.NB):
                for addend in (False, True):
                    self._case(
                        tokens, nvb, addend, seed=tokens * 100 + nvb * 2 + addend
                    )


if __name__ == "__main__":
    unittest.main()
