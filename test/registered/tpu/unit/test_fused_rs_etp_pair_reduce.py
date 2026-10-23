# SPDX-License-Identifier: Apache-2.0
"""Expert-TP fused-RS with and without the on-chip pair reduce.

One Muse Spark MoE call at TP8 on one tpu7x host: 256 experts, hidden 4096,
each core holding its own 512-wide slice of every expert (fp4 weights, bf16
group-16 scales), top-8 routing, post-expert RMS norm. Each core's weights are
distinct, so the two cores of a chip send different partials and the pair sum
is exercised. Pair reduce rounds the chip's pair sum to bf16 once more, so the
outputs agree to bf16 precision rather than bit for bit. It runs at the default
3 output staging slots and at the Muse recipe's 6.
"""

from __future__ import annotations

import functools
import os
import sys
import unittest

import numpy as np

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit", chips=4)

# A standalone raw-JAX kernel test, like test_batched_rpa_kernel.py: select the
# physical backend before the first JAX import.
assert "jax" not in sys.modules
os.environ["JAX_PLATFORMS"] = "tpu"

_ENV = {
    "SGLANG_TPU_FUSED_RS_ETP_LOCAL_ROWS": "1",
    "SGLANG_TPU_FUSED_RS_ETP_TILED_REDUCE": "1",
    "SGLANG_TPU_FUSED_RS_ETP_SKIP_OUT_ZERO": "1",
    "SGLANG_TPU_FUSED_RS_ETP_FUSED_COMBINE": "1",
    "SGLANG_TPU_FUSED_RS_ROW_LOOP_UNROLL": "8",
    "SGLANG_TPU_FUSED_RS_STAGING_SLOTS": "3",
}


def _tpu_devices():
    try:
        import jax

        return [d for d in jax.devices() if d.platform == "tpu"]
    except Exception:
        return []


def _two_core_chips() -> bool:
    devices = _tpu_devices()
    return len(devices) >= 4 and all(
        getattr(d, "core_on_chip", None) is not None for d in devices
    )


@unittest.skipIf(not _two_core_chips(), "Requires TPU chips with two TensorCores")
class TestFusedRsEtpPairReduce(unittest.TestCase):
    T, E, H, I_LOCAL, TOPK = 4096, 256, 4096, 512, 8

    @classmethod
    def setUpClass(cls):
        cls._saved_env = {k: os.environ.get(k) for k in _ENV}
        os.environ.update(_ENV)

    @classmethod
    def tearDownClass(cls):
        for k, v in cls._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _mesh(self, devices):
        from jax.sharding import Mesh

        return Mesh(np.array(devices).reshape(1, len(devices)), ("data", "model"))

    def _chip_ordered_devices(self):
        return sorted(_tpu_devices(), key=lambda d: (tuple(d.coords), d.core_on_chip))

    def test_mesh_pairs_share_chip(self):
        from sglang_tpu.moe.pallas.fused_rs import fused_moe as fm

        devices = self._chip_ordered_devices()
        self.assertTrue(fm._mesh_pairs_share_chip(self._mesh(devices)))
        # Rotating by one core pairs each core with the next chip's.
        rotated = devices[1:] + devices[:1]
        self.assertFalse(fm._mesh_pairs_share_chip(self._mesh(rotated)))

    def test_pair_reduce_matches_per_core_writes(self):
        import jax
        import jax.numpy as jnp
        from jax.sharding import NamedSharding
        from jax.sharding import PartitionSpec as P

        from sglang_tpu.moe.pallas.fused_rs import fused_moe as fm

        T, E, H, inter, K = self.T, self.E, self.H, self.I_LOCAL, self.TOPK
        mesh = self._mesh(self._chip_ordered_devices())
        n = mesh.devices.size
        rep = NamedSharding(mesh, P())
        per_core = NamedSharding(mesh, P("model"))

        @functools.partial(
            jax.jit,
            out_shardings=(rep,) + (per_core,) * 4 + (rep,) * 3,
        )
        def make_inputs(key):
            k = jax.random.split(key, 7)
            hidden = (jax.random.normal(k[0], (T, H)) * 0.5).astype(jnp.bfloat16)
            w1 = (jax.random.normal(k[1], (n * E, H, 2 * inter)) * 2).astype(
                jnp.float4_e2m1fn
            )
            w2 = (jax.random.normal(k[2], (n * E, inter, H)) * 2).astype(
                jnp.float4_e2m1fn
            )
            w1s = jax.random.uniform(
                k[3], (n * E, H // 16, 2 * inter), minval=0.005, maxval=0.02
            ).astype(jnp.bfloat16)
            w2s = jax.random.uniform(
                k[4], (n * E, inter // 16, H), minval=0.005, maxval=0.02
            ).astype(jnp.bfloat16)
            tw, ti = jax.lax.top_k(jax.random.normal(k[5], (T, E)), K)
            tw = jax.nn.softmax(tw, axis=-1).astype(jnp.bfloat16)
            pnw = (jax.random.normal(k[6], (H,)) * 0.1).astype(jnp.bfloat16)
            return hidden, w1, w1s, w2, w2s, tw, ti.astype(jnp.int32), pnw

        hidden, w1, w1s, w2, w2s, tw, ti, pnw = make_inputs(jax.random.PRNGKey(0))
        group_offset = jax.device_put(jnp.arange(n, dtype=jnp.int32) * E, per_core)

        def run(pair_reduce):
            body = functools.partial(
                fm.moe_gmm_local_rs_nodedup,
                activation="silu",
                topk=K,
                ep_size=n,
                ep_axis_name="model",
                has_post_norm=True,
                sp_enabled=False,
                fp8_direct_write=False,
                expert_tp=True,
                pair_reduce=pair_reduce,
            )
            moe = jax.jit(
                jax.shard_map(
                    lambda h, a, a_s, b, b_s, go, w, i, p: body(
                        h, a, a_s, None, b, b_s, None, None, None, go, w, i, p
                    ),
                    mesh=mesh,
                    in_specs=(P(),) + (P("model"),) * 5 + (P(),) * 3,
                    out_specs=P("model"),
                    check_vma=False,
                )
            )
            out = moe(hidden, w1, w1s, w2, w2s, group_offset, tw, ti, pnw)
            return np.asarray(out.astype(jnp.float32))

        base = None
        for slots in ("3", "6"):
            with self.subTest(staging_slots=slots):
                os.environ["SGLANG_TPU_FUSED_RS_STAGING_SLOTS"] = slots
                try:
                    slot_base = run(False)
                    pair = run(True)
                finally:
                    os.environ["SGLANG_TPU_FUSED_RS_STAGING_SLOTS"] = "3"
                if base is None:
                    base = slot_base
                    self.assertTrue(np.isfinite(base).all())
                # The staging depth only changes when a slot is reused.
                np.testing.assert_array_equal(slot_base, base)
                self.assertTrue(np.isfinite(pair).all())
                rel_rms = float(np.linalg.norm(pair - base) / np.linalg.norm(base))
                max_abs = float(np.abs(pair - base).max())
                print(
                    f"pair reduce vs per-core writes, {slots} staging slots: "
                    f"{rel_rms=:.5f} {max_abs=:.4f}"
                )
                # Measured on tpu7x: rel_rms 0.0038, max_abs 0.031 on outputs of
                # mean magnitude 0.30, about one bf16 rounding of the pair sum.
                self.assertLess(rel_rms, 0.01)
                self.assertLess(max_abs, 0.1)


if __name__ == "__main__":
    unittest.main()
