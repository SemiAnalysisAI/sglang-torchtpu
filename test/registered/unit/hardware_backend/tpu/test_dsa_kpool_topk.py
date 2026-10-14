# SPDX-License-Identifier: Apache-2.0
"""The pooled (kpool) DSA indexer step must select what a from-scratch reference selects.

The reference keeps each request's raw keys and gates as plain arrays and, at
every step, recompresses all closed pools, scores them and expands the choice;
the step under test pages its caches and closes pools incrementally across
prefill, decode and speculative-verify steps.
"""

import unittest

import numpy as np

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

try:
    import jax
    import jax.numpy as jnp

    from sglang_tpu.attention.pallas import dsa_kpool_topk as kp

    _HAS_JAX = True
except Exception:  # pragma: no cover - host without JAX
    _HAS_JAX = False

register_cpu_ci(est_time=60, suite="stage-a-test-cpu")

D, HEADS, KPOOL, PAGE, PAGES_PER_SEQ, TOPK = 128, 4, 4, 16, 4, 16
NUM_PAGES = 2 * PAGES_PER_SEQ + 1


def _ref_select(keys, gates, ape, q, w, pos, weight_scale):
    """Token positions the reference selects for one query at sequence position pos."""
    pool_len = (pos + 1) // KPOOL
    pooled = []
    for p in range(pool_len):
        sl = slice(p * KPOOL, (p + 1) * KPOOL)
        c = kp.compress_pools(jnp.asarray(keys[sl]), jnp.asarray(gates[sl]), ape)
        k8, ks, _ = kp.quantize_ue8m0(c[None])
        pooled.append((np.asarray(k8[0].astype(jnp.float32)), float(ks[0, 0])))
    q8, qs, _ = kp.quantize_ue8m0(jnp.asarray(q))
    q8 = np.asarray(q8.astype(jnp.float32))
    ww = np.asarray(w) * np.asarray(qs)[:, 0] * weight_scale
    scores = np.array(
        [np.sum(ww * np.maximum(q8 @ k, 0.0)) * s for k, s in pooled], np.float32
    )
    n_pick = TOPK // KPOOL
    groups = (
        np.argsort(-scores, kind="stable")[:n_pick]
        if pool_len > n_pick
        else range(pool_len)
    )
    toks = {g * KPOOL + j for g in groups for j in range(KPOOL)}
    toks |= {pool_len * KPOOL + j for j in range((pos + 1) % KPOOL)}
    return toks


@unittest.skipUnless(_HAS_JAX, "requires JAX")
class TestKpoolIndexerStep(CustomTestCase):
    def test_matches_reference_over_prefill_decode_verify(self):
        rng = np.random.default_rng(0)
        ape = jnp.asarray(rng.normal(size=(KPOOL, D)).astype(np.float32))
        weight_scale = HEADS**-0.5 * D**-0.5
        step = jax.jit(
            lambda *a: kp.kpool_indexer_step(
                *a, weight_scale=weight_scale, topk=TOPK, kpool=KPOOL
            )
        )
        raw = jnp.zeros((NUM_PAGES, PAGE, 2 * D), jnp.bfloat16)
        pooled = jnp.zeros(
            (NUM_PAGES, PAGE // 4, 4, kp.pooled_record_bytes(D)), jnp.uint8
        )
        # Request r owns pages 1 + r * PAGES_PER_SEQ ...
        page_indices = jnp.asarray(
            [1 + r * PAGES_PER_SEQ + i for r in range(2) for i in range(PAGES_PER_SEQ)],
            jnp.int32,
        )
        hist = {r: ([], []) for r in range(2)}  # committed raw keys / gates
        lens = [0, 0]
        # (tokens per request, committed per request) for each forward.
        plan = [((13, 9), (13, 9)), ((1, 1), (1, 1)), ((1, 1), (1, 1))]
        plan += [((8, 8), (3, 8)), ((1, 1), (1, 1)), ((8, 8), (8, 2)), ((1, 1), (1, 1))]
        plan += [((1, 1), (1, 1))] * 8
        for new, keep in plan:
            n_tok = sum(new)
            pad = 32
            keys = rng.normal(size=(pad, D)).astype(np.float32)
            gates = rng.normal(size=(pad, D)).astype(np.float32)
            query = rng.normal(size=(pad, HEADS, D)).astype(np.float32)
            weights = rng.normal(size=(pad, HEADS)).astype(np.float32)
            kb = np.asarray(jnp.asarray(keys, jnp.bfloat16).astype(jnp.float32))
            gb = np.asarray(jnp.asarray(gates, jnp.bfloat16).astype(jnp.float32))
            kv_lens = jnp.asarray([lens[0] + new[0], lens[1] + new[1]], jnp.int32)
            cu = jnp.asarray([0, new[0], n_tok], jnp.int32)
            out, raw, pooled = step(
                jnp.asarray(query, jnp.bfloat16),
                jnp.asarray(keys, jnp.bfloat16),
                jnp.asarray(gates, jnp.bfloat16),
                jnp.asarray(weights),
                ape,
                raw,
                pooled,
                kv_lens,
                page_indices,
                cu,
                jnp.asarray([0, 0, 2], jnp.int32),
            )
            out = np.asarray(out)
            t = 0
            for r in range(2):
                ks, gs = hist[r]
                win_k = ks + list(kb[t : t + new[r]])
                win_g = gs + list(gb[t : t + new[r]])
                for i in range(new[r]):
                    pos = lens[r] + i
                    row = out[t + i]
                    got = set(row[row >= 0].tolist())
                    want = _ref_select(
                        np.stack(win_k[: pos + 1]),
                        np.stack(win_g[: pos + 1]),
                        ape,
                        query[t + i],
                        weights[t + i],
                        pos,
                        weight_scale,
                    )
                    self.assertEqual(got, want, f"req {r} pos {pos}")
                    # Selected positions form a prefix of the row.
                    n = int((row >= 0).sum())
                    self.assertTrue((row[:n] >= 0).all() and (row[n:] == -1).all())
                # Keep only the accepted tokens, as a verify step would.
                hist[r] = (win_k[: lens[r] + keep[r]], win_g[: lens[r] + keep[r]])
                lens[r] += keep[r]
                t += new[r]
            self.assertTrue((out[n_tok:] == -1).all())


if __name__ == "__main__":
    unittest.main()
