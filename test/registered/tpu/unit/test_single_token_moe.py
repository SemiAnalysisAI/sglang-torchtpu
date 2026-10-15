"""Single-token packed-MXFP4 MoE kernel vs the gmm_v2 path of _fused_moe_jax."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=60, suite="stage-a-test-tpu-unit", chips=1)

import contextlib
import unittest
from unittest import mock

from sglang.test.test_utils import CustomTestCase

try:
    import jax
    import jax.numpy as jnp
    import numpy as np

    _HAS_TPU = any(d.platform == "tpu" for d in jax.devices())
except Exception:  # pragma: no cover - non-TPU host
    _HAS_TPU = False

# Scaled-down Kimi-K3 shape: EP4, 16 local of 64 experts, top-8, SiTU.
H, N, LOCAL, GLOBAL, K = 512, 256, 16, 64, 8
ACT = "situ:4.0:25.0"


@unittest.skipUnless(_HAS_TPU, "requires a TPU backend")
class TestSingleTokenMoe(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        keys = jax.random.split(jax.random.PRNGKey(0), 4)
        u8 = lambda key, shape: jax.random.randint(
            key, shape, 0, 256, jnp.int32
        ).astype(jnp.uint8)
        e8 = lambda key, shape: (
            2.0 ** jax.random.randint(key, shape, -9, -5).astype(jnp.float32)
        )
        cls.w13 = u8(keys[0], (LOCAL, H // 2, 2 * N))
        cls.w2 = u8(keys[1], (LOCAL, N // 2, H))
        cls.s13 = e8(keys[2], (LOCAL, H // 32, 1, 2 * N))
        cls.s2 = e8(keys[3], (LOCAL, N // 32, 1, H))
        cls.x = jnp.asarray(
            np.random.default_rng(1).standard_normal((1, H)) * 0.5, jnp.bfloat16
        )

    def _run(self, ids, offset, kernel_on):
        from sglang_tpu.moe.pallas import single_token_moe
        from sglang_tpu.moe.pallas.fused_moe_op import _fused_moe_jax

        w = np.random.default_rng(len(ids)).random(K).astype(np.float32)
        w /= w.sum()
        # Off: the dispatch sees the kernel as ineligible and takes gmm_v2.
        with (
            contextlib.nullcontext()
            if kernel_on
            else mock.patch.object(single_token_moe, "eligible", return_value=False)
        ):
            out = jax.jit(
                lambda: _fused_moe_jax(
                    hidden_states=self.x,
                    w1=self.w13,
                    w2=self.w2,
                    w1_scale=self.s13,
                    w2_scale=self.s2,
                    topk_weights=jnp.asarray(w[None], jnp.bfloat16),
                    topk_ids=jnp.asarray(np.asarray(ids)[None], jnp.int32),
                    w1_bias=None,
                    w2_bias=None,
                    topk=K,
                    activation_name=ACT,
                    global_num_experts=GLOBAL,
                    expert_group_offset=offset,
                    permute_dense_threshold=512,
                )
            )()
        return np.asarray(out.astype(jnp.float32))

    def test_matches_gmm_path(self):
        rng = np.random.default_rng(0)
        cases = [
            (rng.choice(GLOBAL, K, replace=False), 16),  # some local hits
            (np.arange(K) + 16, 16),  # every slot local
            (np.r_[np.arange(K - 1) + 40, -1], 32),  # a padded (-1) slot
        ]
        for ids, offset in cases:
            ref = self._run(ids, offset, kernel_on=False)
            got = self._run(ids, offset, kernel_on=True)
            self.assertEqual(got.shape, (1, H))
            rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
            self.assertLess(float(rel), 1e-2, f"{ids=} {offset=}")

    def test_no_local_hit_is_zero(self):
        got = self._run(np.arange(K) + 40, 0, kernel_on=True)
        np.testing.assert_array_equal(got, np.zeros((1, H), np.float32))


if __name__ == "__main__":
    unittest.main()
