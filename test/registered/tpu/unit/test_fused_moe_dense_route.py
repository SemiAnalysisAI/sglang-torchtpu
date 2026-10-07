"""Fused MoE multi-token dense route: local ids, sort-free rank permute."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=480, suite="stage-a-test-tpu-unit")

import unittest
from unittest import mock

import torch

from sglang.test.test_utils import CustomTestCase

try:
    import jax
    import jax.numpy as jnp
    import numpy as np

    _HAS_TPU = any(d.platform == "tpu" for d in jax.devices())
except Exception:  # pragma: no cover - non-TPU host
    _HAS_TPU = False

_SKIP = "requires a TPU backend"

HIDDEN, INTER = 1024, 768
SITU = "situ:4.0:25.0"
DENSE_THRESHOLD = 2048

# Per-rank routing, scaled-down weights: (local experts, global experts, EP
# offset of a middle rank, top-k, token counts on the dense route, activation).
ROUTES = {
    "k3": (28, 896, 13 * 28, 16, (2, 3, 4, 8, 16, 64), SITU),
    "qwen3_coder": (20, 160, 3 * 20, 8, (2, 5, 16, 32, 128), "silu"),  # EP8
    # EP8; decode bucket 20 pads to 24 tokens, 240 routed rows.
    "qwen35": (64, 512, 2 * 64, 10, (2, 9, 20, 24, 100), "silu"),
}


def _silu(gate, up):
    return gate / (1.0 + np.exp(-gate)) * up


def _situ(gate, up, beta=4.0, linear_beta=25.0):
    gate_act = beta * np.tanh(gate / beta) / (1.0 + np.exp(-gate))
    return gate_act * linear_beta * np.tanh(up / linear_beta)


def _moe_ref(hidden, w1, w2, topk_weights, topk_ids, act):
    """Dense float32 MoE over [g, k, 2n] / [g, n, k] weights."""
    n = w1.shape[-1] // 2
    out = np.zeros((hidden.shape[0], w2.shape[-1]), np.float32)
    for t in range(hidden.shape[0]):
        for j in range(topk_ids.shape[1]):
            e = int(topk_ids[t, j])
            gmm1 = hidden[t] @ w1[e]
            out[t] += float(topk_weights[t, j]) * (act(gmm1[:n], gmm1[n:]) @ w2[e])
    return out


def _assert_close(test, out, ref, rtol):
    err = np.max(np.abs(out - ref)) / (np.max(np.abs(ref)) + 1e-9)
    test.assertTrue(np.all(np.isfinite(out)))
    test.assertLess(float(err), rtol)


def _weights(kind, num_experts):
    """(w1, w2, w1_scale, w2_scale) for the kernel, f32 (w1, w2) for the ref."""
    from sglang.srt.layers.quantization.mxfp4_tpu_util import (
        process_mxfp4_moe_weights_bf16,
        process_mxfp4_moe_weights_native,
    )

    E, N, H = num_experts, INTER, HIDDEN
    if kind == "bf16":
        k1, k2 = jax.random.split(jax.random.key(0))
        w1 = jax.random.normal(k1, (E, H, 2 * N)) * 0.05
        w2 = jax.random.normal(k2, (E, N, H)) * 0.05
        w1, w2 = w1.astype(jnp.bfloat16), w2.astype(jnp.bfloat16)
        ref = tuple(np.asarray(t.astype(jnp.float32)) for t in (w1, w2))
        return (w1, w2, None, None), ref
    if kind == "fp8_ch":
        rng = np.random.default_rng(0)
        kernel, ref = [], []
        for shape in ((E, H, 2 * N), (E, N, H)):
            wf = rng.standard_normal(shape).astype(np.float32) * 0.05
            scale = np.abs(wf).max(axis=1, keepdims=True) / 448.0  # [g, 1, n]
            q = jnp.asarray(wf / scale, jnp.float8_e4m3fn)
            kernel.append((q, jnp.asarray(scale[:, None], jnp.float32)))
            ref.append(np.asarray(q.astype(jnp.float32)) * scale)
        (w1, s1), (w2, s2) = kernel
        return (w1, w2, s1, s2), tuple(ref)
    g = torch.Generator().manual_seed(0)
    u8 = lambda *s: torch.randint(0, 256, s, dtype=torch.uint8, generator=g)
    e8 = lambda *s: torch.randint(124, 131, s, dtype=torch.uint8, generator=g)
    hf = (
        u8(E, 2 * N, H // 2),
        e8(E, 2 * N, H // 32),
        torch.zeros(E, 2 * N, dtype=torch.bfloat16),
        u8(E, H, N // 2),
        e8(E, H, N // 32),
        torch.zeros(E, H, dtype=torch.bfloat16),
    )
    w13, w13_s, _, w2, w2_s, _ = process_mxfp4_moe_weights_native(
        *hf, w13_interleaved=False
    )
    w13_bf, _, w2_bf, _ = process_mxfp4_moe_weights_bf16(*hf, w13_interleaved=False)
    to_jax = lambda t: jnp.asarray(t.numpy())
    # bf16-rounded, as the existing MXFP4 reference test does.
    bf = lambda t: t.to(torch.bfloat16).float().numpy()
    return tuple(to_jax(t) for t in (w13, w2, w13_s, w2_s)), (bf(w13_bf), bf(w2_bf))


def _routing(m, route, ep, seed):
    """EP: global ids with 0-8 local slots per token; else ids over local experts."""
    e_local, e_global, offset, topk, _, _ = ROUTES[route]
    rng = np.random.default_rng(seed)
    if not ep:
        ids = np.stack([rng.choice(e_local, topk, replace=False) for _ in range(m)])
    else:
        local = np.arange(offset, offset + e_local)
        remote = np.setdiff1d(np.arange(e_global), local)
        rows = []
        for _ in range(m):
            n_local = int(rng.integers(0, min(8, topk) + 1))
            row = np.concatenate(
                [
                    rng.choice(local, n_local, replace=False),
                    rng.choice(remote, topk - n_local, replace=False),
                ]
            )
            rows.append(rng.permutation(row))
        ids = np.stack(rows)
    w = rng.random((m, topk)).astype(np.float32)
    return ids.astype(np.int32), w / w.sum(-1, keepdims=True)


def _poison_unwritten(gmm_v2):
    """Fill the rows a no-zero-init grouped matmul leaves unwritten with NaN."""

    def wrapped(*args, **kwargs):
        out = gmm_v2(*args, **kwargs)
        if kwargs["zero_initialize"] or kwargs["group_offset"] is not None:
            return out
        written = jnp.arange(out.shape[0])[:, None] < jnp.sum(kwargs["group_sizes"])
        return jnp.where(written, out, jnp.nan).astype(out.dtype)

    return wrapped


@unittest.skipUnless(_HAS_TPU, _SKIP)
class TestFusedMoEDenseRoute(CustomTestCase):
    """_fused_moe_jax with the dense route vs the existing paths and a reference."""

    _weight_cache = {}

    def _weights_for(self, kind, route):
        key = (kind, route)
        if key not in self._weight_cache:
            self._weight_cache[key] = _weights(kind, ROUTES[route][0])
        return self._weight_cache[key]

    def _fused(self, kind, route, m, ep, ids, w, *, use_route=True, threshold):
        import importlib

        from sglang.srt.layers.moe.pallas import fused_moe_op

        gmm_mod = importlib.import_module("sglang.srt.layers.moe.pallas.gmm_v2")

        e_local, e_global, offset, topk, _, act = ROUTES[route]
        (w1, w2, w1_s, w2_s), _ = self._weights_for(kind, route)
        hidden = jnp.asarray(
            np.random.default_rng(m).standard_normal((m, HIDDEN)), jnp.bfloat16
        )
        # Fresh closure per call: the patches apply at trace time.
        fn = jax.jit(
            lambda h, w1, w2, w1_s, w2_s: fused_moe_op._fused_moe_jax(
                h,
                w1,
                w2,
                jnp.asarray(w, jnp.bfloat16),
                jnp.asarray(ids),
                w1_s,
                w2_s,
                None,
                None,
                topk,
                act,
                global_num_experts=e_global if ep else 0,
                expert_group_offset=offset if ep else 0,
                permute_dense_threshold=threshold,
            )
        )
        with (
            mock.patch.object(
                fused_moe_op,
                "_RANK_PERMUTE_MAX_ROWS",
                fused_moe_op._RANK_PERMUTE_MAX_ROWS if use_route else 0,
            ),
            mock.patch.object(gmm_mod, "gmm_v2", _poison_unwritten(gmm_mod.gmm_v2)),
        ):
            out = jax.block_until_ready(fn(hidden, w1, w2, w1_s, w2_s))
        out = np.asarray(out.astype(jnp.float32))
        self.assertTrue(np.all(np.isfinite(out)))
        return out, hidden

    def _reference(self, kind, route, hidden, ids, w, ep):
        e_local, _, offset, _, _, act = ROUTES[route]
        _, (w1, w2) = self._weights_for(kind, route)
        local = ids - (offset if ep else 0)
        owned = (local >= 0) & (local < e_local)
        return _moe_ref(
            np.asarray(hidden.astype(jnp.float32)),
            w1,
            w2,
            np.where(owned, w, 0.0),
            np.where(owned, local, 0),
            _situ if act == SITU else _silu,
        )

    def _check(self, kind, route, ep, *, gather_tol=1e-2, ref_tol=0.03):
        for m in ROUTES[route][4]:
            with self.subTest(kind=kind, route=route, ep=ep, m=m):
                ids, w = _routing(m, route, ep, m + 1)
                run = lambda **kw: self._fused(kind, route, m, ep, ids, w, **kw)
                new, hidden = run(threshold=DENSE_THRESHOLD)
                old, _ = run(threshold=DENSE_THRESHOLD, use_route=False)
                if ep:
                    # Owned rows move from the global window to a prefix, which
                    # can regroup the f32 combine sum: within one bf16 ulp.
                    np.testing.assert_allclose(new, old, rtol=2**-7, atol=1e-6)
                else:
                    # Same rows in the same stable order: bitwise.
                    np.testing.assert_array_equal(new, old)
                gather, _ = run(threshold=0)
                _assert_close(self, new, gather, gather_tol)
                ref = self._reference(kind, route, hidden, ids, w, ep)
                self.assertGreater(np.abs(ref).max(), 0)
                _assert_close(self, new, ref, ref_tol)

    def test_bf16_all_local(self):
        self._check("bf16", "k3", ep=False)

    def test_bf16_ep(self):
        self._check("bf16", "k3", ep=True)

    def test_mxfp4_situ_all_local(self):
        self._check("mxfp4", "k3", ep=False)

    def test_mxfp4_situ_ep(self):
        self._check("mxfp4", "k3", ep=True)

    def test_fp8_channelwise_top8_top10(self):
        # Qwen3-Coder-480B (top-8) and Qwen3.5-397B (top-10) recipe routing.
        # The reference skips the kernel's FP8 activation quantization.
        for route in ("qwen3_coder", "qwen35"):
            for ep in (True, False):
                self._check("fp8_ch", route, ep, gather_tol=3e-2, ref_tol=0.1)

    def test_out_of_range_ids_contribute_nothing(self):
        # -1 and >= global ids land on the sentinel, like non-local ids.
        from sglang.srt.layers.moe.pallas import fused_moe_op

        e_local, e_global, offset, topk, _, _ = ROUTES["k3"]
        (kw1, kw2, _, _), _ = self._weights_for("bf16", "k3")
        hidden = jnp.asarray(
            np.random.default_rng(0).standard_normal((4, HIDDEN)), jnp.bfloat16
        )
        bad_ids, w = _routing(4, "k3", True, 1)
        bad_ids[:, :2] = np.array([-1, e_global + 5])
        out = fused_moe_op._fused_moe_jax(
            hidden,
            kw1,
            kw2,
            jnp.asarray(w, jnp.bfloat16),
            jnp.asarray(bad_ids),
            None,
            None,
            None,
            None,
            topk,
            SITU,
            global_num_experts=e_global,
            expert_group_offset=offset,
            permute_dense_threshold=DENSE_THRESHOLD,
        )
        out = np.asarray(jax.block_until_ready(out).astype(jnp.float32))
        ref = self._reference("bf16", "k3", hidden, bad_ids, w, ep=True)
        _assert_close(self, out, ref, 0.03)


if __name__ == "__main__":
    unittest.main()
