"""CPU interpret-mode checks of the GPT-OSS TP8 decode megakernel against the JAX reference.

Uses 8 virtual CPU devices. Run:
  XLA_FLAGS=--xla_force_host_platform_device_count=8 JAX_PLATFORMS=cpu \
    python -m unittest test.registered.unit.hardware_backend.tpu.test_gptoss_decode_megakernel
"""

import dataclasses
import importlib.util
import os
import unittest

import numpy as np

from sglang.test.ci.ci_register import register_cpu_ci, register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=95, suite="base-a-test-cpu")
register_tpu_ci(est_time=95, suite="stage-a-test-tpu-unit")

_HAS_JAX = importlib.util.find_spec("jax") is not None
if _HAS_JAX:
    os.environ["JAX_PLATFORMS"] = "cpu"
    if "xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):
        os.environ["XLA_FLAGS"] = (
            os.environ.get("XLA_FLAGS", "")
            + " --xla_force_host_platform_device_count=8"
        )

    import jax  # noqa: E402
    import jax.numpy as jnp  # noqa: E402
    from jax.sharding import Mesh  # noqa: E402

    from sglang.srt.hardware_backend.tpu.kernels.gptoss_decode_megakernel import (  # noqa: E402
        config as C,
    )
    from sglang.srt.hardware_backend.tpu.kernels.gptoss_decode_megakernel import (
        megakernel as MK,
    )
    from sglang.srt.hardware_backend.tpu.kernels.gptoss_decode_megakernel import (
        reference as REF,
    )


def _rel(a, b):
    a = np.asarray(jnp.asarray(a).astype(jnp.float32))
    b = np.asarray(jnp.asarray(b).astype(jnp.float32))
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-6))


def _positions(c):
    cap = c.max_pages * c.page - c.Q
    base = [0, 62, 130, cap, 64, 200, 127, 5, 448, 1]
    return np.array([min(base[i % len(base)], cap) for i in range(c.B)], np.int32)


def _setup(c, seed=0):
    rng = np.random.default_rng(seed)
    keys = jax.random.split(jax.random.PRNGKey(seed), c.L + 3)
    layers = [REF.random_layer(c, k) for k in keys[: c.L]]
    if (
        c.w_dtype != "bf16"
    ):  # the reference sees exactly the MXFP4-representable weights
        layers = [REF.mxfp4_roundtrip(c, w) for w in layers]
    x = jax.random.normal(keys[c.L], (c.BQ, c.H), jnp.float32).astype(jnp.bfloat16)
    pos = _positions(c)
    T = c.max_pages * c.page
    kvdt = REF.kv_jdtype(c)
    k_hist = (jax.random.normal(keys[c.L + 1], (c.L, c.B, T, c.NKV, c.D)) * 2.0).astype(
        kvdt
    )
    v_hist = jax.random.normal(keys[c.L + 2], (c.L, c.B, T, c.NKV, c.D)).astype(kvdt)
    assert c.B * c.max_pages <= min(c.nb_swa, c.nb_full) - 1
    pt_swa = (1 + rng.permutation(c.nb_swa - 1)[: c.B * c.max_pages]).reshape(
        c.B, c.max_pages
    )
    pt_full = (1 + rng.permutation(c.nb_full - 1)[: c.B * c.max_pages]).reshape(
        c.B, c.max_pages
    )
    # finite garbage in every slot the step must not read (other pages, future tokens, fp8
    # padding slots, head_dim padding lanes): proves the masking
    garbage = [rng.normal(size=sp.shape).astype(np.float32) for sp in MK.pool_specs(c)]
    pools = MK.build_pools(c, k_hist, v_hist, pos, pt_swa, pt_full, base=garbage)
    return dict(
        layers=layers,
        x=x,
        pos=pos,
        k_hist=k_hist,
        v_hist=v_hist,
        pt_swa=jnp.asarray(pt_swa, jnp.int32),
        pt_full=jnp.asarray(pt_full, jnp.int32),
        pools=pools,
    )


def _run(c, d, xla_ar):
    mesh = Mesh(np.array(jax.devices("cpu")[: c.R]), ("tp",))
    W = MK.pack_weights(c, d["layers"])
    step = jax.jit(MK.make_step(c, mesh, xla_ar=xla_ar, interpret=True))
    return step(
        d["x"], jnp.asarray(d["pos"]), d["pt_swa"], d["pt_full"], W, *d["pools"]
    )


@unittest.skipUnless(_HAS_JAX, "requires JAX")
class TestGptOssDecodeMegakernel(CustomTestCase):
    def _check(self, c, also_fallback=False):
        d = _setup(c)
        margins = []
        ref_res = REF.step(
            c,
            d["x"],
            jnp.asarray(d["pos"]),
            d["k_hist"],
            d["v_hist"],
            d["layers"],
            margins,
        )
        ref, kn_ref, vn_ref = ref_res[:3]
        run_res = _run(c, d, xla_ar=False)
        out, kv_s, kv_f = run_res[:3]
        # Rows whose top-k routing is a near-tie in some layer can legitimately pick a
        # different expert under f32 summation-order noise; compare the other rows.
        tie = np.min(np.stack([np.asarray(m) for m in margins]), axis=0) < 4e-3
        self.assertLessEqual(
            int(tie.sum()), max(1, c.BQ // 4), "too many near-tie rows"
        )
        keep = ~tie
        out_k, ref_k = out[keep], ref[keep]
        rel = _rel(out_k, ref_k)
        print(
            f"{c.moe_mode} w={c.w_dtype} kv={c.kv_dtype} L={c.L} B={c.B} Q={c.Q} cap={c.capture_layers}: "
            f"megakernel vs reference rel {rel:.2e} (near-tie rows skipped: {int(tie.sum())})"
        )
        self.assertLess(rel, 2e-2)
        if c.capture_layers:
            aux_ref = ref_res[3][:, keep]
            aux_out = run_res[3][:, keep]
            rel_aux = _rel(aux_out, aux_ref)
            print(f"  aux capture rel {rel_aux:.2e}")
            self.assertLess(rel_aux, 2e-2)
        self._check_pools(c, d, kv_s, kv_f, kn_ref, vn_ref)
        if also_fallback:
            out_fb, kv_s_fb, kv_f_fb = _run(c, d, xla_ar=True)
            out_fb = out_fb[keep]
            rel_fb = _rel(out_fb, ref_k)
            rel_mf = _rel(out_k, out_fb)
            print(
                f"  fallback vs reference {rel_fb:.2e}, megakernel vs fallback {rel_mf:.2e}"
            )
            self.assertLess(rel_fb, 2e-2)
            self.assertLess(rel_mf, 1e-2)
            np.testing.assert_array_equal(
                np.asarray(kv_s.astype(jnp.float32)),
                np.asarray(kv_s_fb.astype(jnp.float32)),
            )

    def _check_pools(self, c, d, kv_s, kv_f, kn_ref, vn_ref):
        """New K/V landed at (page((pos+q)//dcp), ((pos+q)//dcp) % page) on owner ranks; nothing else moved."""
        before = [np.asarray(p.astype(jnp.float32)) for p in d["pools"]]
        after = [np.asarray(p.astype(jnp.float32)) for p in (kv_s, kv_f)]
        expect = [b.copy() for b in before]
        kn = np.asarray(kn_ref.astype(jnp.float32))  # [L, B*Q, NKV, D]
        vn = np.asarray(vn_ref.astype(jnp.float32))
        for l in range(c.L):
            which, pt = (0, d["pt_swa"]) if c.sliding[l] else (1, d["pt_full"])
            dcp = c.dcp_size if (not c.sliding[l] or c.dcp_swa) else 1
            for b in range(c.B):
                for q in range(c.Q):
                    t_idx = b * c.Q + q
                    p = int(d["pos"][b]) + q
                    pg = int(pt[b, (p // dcp) // c.page])
                    tok = (p // dcp) % c.page
                    if dcp == 1:
                        slot = expect[which][
                            :, l // 2, pg, tok, 0
                        ]  # [R, packing, 128] view
                        slot[...] = (
                            0.0  # padding lanes and fp8 padding slots are written as zeros
                        )
                        slot[:, 0, : c.D] = kn[l, t_idx]
                        slot[:, 1, : c.D] = vn[l, t_idx]
                    else:
                        owner_d = p % dcp
                        for r in range(c.R):
                            if r % dcp != owner_d:
                                continue
                            grp_base = (r // dcp) * dcp
                            slot = expect[which][r, l // 2, pg, tok]
                            slot[...] = 0.0
                            for h in range(dcp):
                                kv_h = grp_base + h
                                if c.dcp_lane_pack and c.kv_dtype == "fp8":
                                    g_idx = 0
                                    k_slot, v_slot = 2 * (h // 2), 2 * (h // 2) + 1
                                    lane_off = 64 * (h % 2)
                                elif c.kv_packing == 4:
                                    g_idx = h // 2
                                    k_slot, v_slot = 2 * (h % 2), 2 * (h % 2) + 1
                                    lane_off = 0
                                else:
                                    g_idx = h
                                    k_slot, v_slot = 0, 1
                                    lane_off = 0
                                slot[
                                    g_idx, k_slot, lane_off : lane_off + c.D
                                ] = kn[l, t_idx, kv_h]
                                slot[
                                    g_idx, v_slot, lane_off : lane_off + c.D
                                ] = vn[l, t_idx, kv_h]
        for a, e, b0 in zip(after, expect, before):
            bad = np.abs(a - e) > 0.02 * np.maximum(np.abs(e), 1.0)
            # tiny K/V differences come only from f32 summation order upstream
            self.assertLess(
                bad.mean(), 1e-3, f"pool mismatch fraction {bad.mean():.2e}"
            )
            changed = np.any(a != b0, axis=-1)
            exp_changed = np.any(e != b0, axis=-1)
            self.assertTrue(
                np.all(changed <= exp_changed), "pool rows written outside new tokens"
            )

    def test_tp_moe_bf16_kv_with_fallback(self):
        self._check(C.tiny(B=4, w_dtype="bf16"), also_fallback=True)

    def test_ep_moe_fp8_kv(self):
        self._check(C.tiny(B=4, moe_mode="ep", kv_dtype="fp8"))

    def test_four_layers_b1(self):
        self._check(C.tiny(B=1, L=4, w_dtype="bf16"))

    def test_long_context_deep_prefetch(self):
        # several full-layer items per sequence, prefetch crossing sequence boundaries
        self._check(
            C.tiny(
                B=4,
                max_pages=8,
                npb=2,
                kv_bufs=4,
                nb_swa=40,
                nb_full=40,
                kv_dtype="fp8",
            )
        )

    def test_mxfp4_tp_moe(self):
        # block-aligned TP split of I: 12 scale blocks over 8 ranks (2,2,2,2,1,1,1,1)
        self._check(C.tiny(B=4, w_dtype="mxfp4"), also_fallback=True)

    def test_mxfp4_ep_moe_fp8_kv(self):
        self._check(C.tiny(B=4, w_dtype="mxfp4", moe_mode="ep", kv_dtype="fp8"))

    def test_rsag_allreduce_b4(self):
        self._check(C.tiny(B=4, ar_mode="rsag", kv_dtype="fp8"), also_fallback=True)

    def test_ar_unfolded_b4(self):
        self._check(C.tiny(B=4, ar_fold=False))

    def test_ablations_run(self):
        # timing-only switches: they must compile/run and stay finite (outputs are wrong)
        c0 = C.tiny(B=2)
        d = _setup(c0)
        full, _, _ = _run(c0, d, xla_ar=False)
        for ab in C.ABLATIONS:
            out, _, _ = _run(dataclasses.replace(c0, ablate=(ab,)), d, xla_ar=False)
            out = np.asarray(out.astype(jnp.float32))
            self.assertTrue(np.all(np.isfinite(out)), ab)
            self.assertGreater(_rel(out, full), 1e-3, ab)

    def test_fp8_experts_tp(self):
        self._check(C.tiny(B=4, w_dtype="fp8"), also_fallback=True)

    def test_fp8_experts_ep_b16(self):
        self._check(
            dataclasses.replace(
                C.tiny(B=16, w_dtype="fp8", moe_mode="ep"), nb_swa=72, nb_full=72
            )
        )

    def test_in_kernel_kv_write(self):
        # the megakernel writes new K/V into its aliased pools (no separate scatter call)
        self._check(C.tiny(B=4, kv_write="kernel", kv_dtype="fp8"))

    def test_b16_npb1(self):
        # B=16 -> ar_mode "auto" picks reduce-scatter + all-gather
        self._check(
            dataclasses.replace(
                C.tiny(B=16, npb=1, w_dtype="bf16"), nb_swa=72, nb_full=72
            )
        )

    def test_target_verify_q4_and_capture_layers(self):
        # EAGLE3 target verify: Q=4 verify tokens per sequence + auxiliary hidden capture
        # Includes pos=62 (crosses 64-token page boundary within the Q=4 tokens) and
        # pos=130 (sliding-window boundary shifts per query q in 0..3).
        self._check(
            C.tiny(
                B=3,
                Q=4,
                w_dtype="fp8",
                kv_dtype="fp8",
                capture_layers=(0, 1, 2),
            ),
            also_fallback=True,
        )

    def test_target_verify_q4_b1_ar_fold(self):
        # B=1, Q=4 -> BQ=4 exercises AR_FOLD with multi-token verify
        self._check(
            C.tiny(
                B=1,
                Q=4,
                w_dtype="fp8",
                kv_dtype="bf16",
                capture_layers=(0, 1, 2),
            )
        )

    def test_dcp2_fp8_kv(self):
        # TP8 + DCP2: 2 KV heads per rank, stock pool layout (kv_groups=1)
        self._check(
            C.tiny(B=4, dcp_size=2, dcp_swa=False, w_dtype="fp8", kv_dtype="fp8"),
            also_fallback=True,
        )

    def test_dcp2_swa_sharded(self):
        # TP8 + DCP2 with SWA pool also DCP-sharded (matches SGLang dcp_shards_every_kv_pool=True)
        self._check(
            C.tiny(B=4, dcp_size=2, dcp_swa=True, w_dtype="fp8", kv_dtype="fp8")
        )

    def test_dcp4_stock_pool_fp8_kv(self):
        # TP8 + DCP4: 4 KV heads per rank, stock pool layout (kv_groups=2, strided_load/store)
        self._check(
            C.tiny(B=4, dcp_size=4, dcp_swa=True, w_dtype="fp8", kv_dtype="fp8"),
            also_fallback=True,
        )

    def test_dcp4_lane_pack_fp8_kv(self):
        # TP8 + DCP4: 4 KV heads per rank, dual-lane compact pool layout (kv_groups=1, dcp_lane_pack=True)
        self._check(
            C.tiny(
                B=4,
                dcp_size=4,
                dcp_swa=False,
                dcp_lane_pack=True,
                w_dtype="fp8",
                kv_dtype="fp8",
            )
        )


if __name__ == "__main__":
    unittest.main()
