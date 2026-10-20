"""Kimi-K3's KDA speculative-verify call must match token-by-token decode.

Kimi-K3's verify step reaches the GDN op with every option its layers set at
once:
- KDA mode, the fused [q, k, v, g] projection as qkv (the op reads its
  leading conv-dim columns), and the packed conv / dt_bias / A_log operand;
- checkpoint rows: slot ids, spec_window rows per slot, and the pool's
  per-slot read-offset table gathered inside the op;
- several sequences at different read offsets, plus a padded batch row
  (slot -1, clamped to the null slot).

test_gdn_attention_op.py covers the fused / packed decode call, and the
checkpoint arithmetic on the plain GDN op at small widths; this pins the
combination, at the per-rank head count Kimi-K3 runs at TP32 and the DSpark
window (gamma 7, 8 rows), against the same op stepped one token at a time.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=60, suite="stage-a-test-tpu-unit", chips=1)

import unittest

import numpy as np
import torch

from sglang.test.test_utils import CustomTestCase

try:
    import torch_tpu  # noqa: F401

    _HAS_TORCH_TPU = True
except Exception:  # pragma: no cover - non-TPU host
    _HAS_TORCH_TPU = False

# Kimi-K3 KDA (linear_attn_config): 96 heads of 128, short conv 4, gate lower
# bound -5. Per rank: 3 heads at TP32 (4 hosts), 12 at TP8 (one host).
KDA_HEADS = 96
D = 128
K = 4
LOWER = -5.0
WINDOW = 8  # DSpark gamma 7 + the bonus row
CONTEXT = 37


@unittest.skipUnless(_HAS_TORCH_TPU, "requires torch_tpu")
class TestKimiK3KdaVerifyOp(CustomTestCase):
    def test_tp32_rank(self):
        self._check(KDA_HEADS // 32)

    def test_tp8_rank(self):
        self._check(KDA_HEADS // 8)

    def _check(self, n_v):
        from sglang_tpu.attention_gdn_pallas import build_gdn_pallas_op
        from sglang_tpu.kda_layout import pack_kda_weights

        op = build_gdn_pallas_op()
        dev = torch.device("tpu")
        rng = np.random.default_rng(11)
        conv_dim = 3 * n_v * D
        read_offsets = [3, 0]  # per real sequence
        real = len(read_offsets)
        bs = real + 1  # one padded batch row
        # Slot 0 is the null slot; sequence s owns slot s + 1, whose checkpoint
        # rows are [(s + 1) * WINDOW, (s + 2) * WINDOW).
        slots = real + 1
        rows = slots * WINDOW

        def rn(*shape, scale=1.0):
            return (scale * rng.standard_normal(shape)).astype(np.float32)

        def tt(x, dtype=None):
            t = torch.from_numpy(np.ascontiguousarray(x)).to(dev)
            return t if dtype is None else t.to(dtype)

        qkvg = rn(bs * WINDOW, conv_dim + n_v * D, scale=0.5)
        b = rn(bs * WINDOW, n_v)
        a = rn(bs * WINDOW, n_v * D)
        conv_w, dt_bias, a_log = rn(conv_dim, 1, K), rn(n_v, D), rn(n_v)
        packed = pack_kda_weights(
            *(torch.from_numpy(x) for x in (conv_w, dt_bias, a_log)), n_v, D
        ).numpy()
        conv0 = rn(rows, K - 1, 1, conv_dim)
        rec0 = rn(rows, n_v, D, D, scale=0.1)
        weights = (tt(conv_w), None, tt(a_log), tt(dt_bias))
        fused = dict(fused_qkvg=True, packed_weights=tt(packed))

        def call(qkv, bb, aa, conv, rec, qsl, idx, dist, seq_lens, **kw):
            return op(
                tt(qkv, torch.bfloat16),
                tt(bb, torch.bfloat16),
                tt(aa, torch.bfloat16),
                conv,
                rec,
                *weights,
                tt(qsl),
                tt(idx),
                tt(dist),
                tt(seq_lens),
                n_v,
                n_v,
                D,
                D,
                K,
                True,
                LOWER,
                batched_only=True,
                **fused,
                **kw,
            )

        # The verify step, as TpuGDNAttnBackend issues it for TARGET_VERIFY.
        table = np.zeros(slots, np.int32)
        table[1 : real + 1] = read_offsets
        conv_v, rec_v = tt(conv0), tt(rec0)
        out_v = call(
            qkvg,
            b,
            a,
            conv_v,
            rec_v,
            np.arange(bs + 1, dtype=np.int32) * WINDOW,
            np.array(list(range(1, real + 1)) + [-1], np.int32),
            np.full(3, real, np.int32),
            np.full(bs, CONTEXT + WINDOW, np.int32),
            read_offsets=tt(table),
            num_spec_tokens=WINDOW - 1,
            spec_window=WINDOW,
        )
        out_v, conv_v, rec_v = (x.float().cpu().numpy() for x in (out_v, conv_v, rec_v))

        touched = set()
        for s, off in enumerate(read_offsets):
            base = (s + 1) * WINDOW
            # Decode reference: a two-row pool (null + this sequence) seeded
            # from the row the verify read, stepped one token at a time.
            conv_d = tt(np.stack([conv0[0], conv0[base + off]]))
            rec_d = tt(np.stack([rec0[0], rec0[base + off]]))
            for t in range(WINDOW):
                row = s * WINDOW + t
                out_d = call(
                    qkvg[row : row + 1],
                    b[row : row + 1],
                    a[row : row + 1],
                    conv_d,
                    rec_d,
                    np.array([0, 1], np.int32),
                    np.array([1], np.int32),
                    np.array([1, 1, 1], np.int32),
                    np.array([CONTEXT + t + 1], np.int32),
                )
                np.testing.assert_allclose(
                    out_v[row],
                    out_d.float().cpu().numpy()[0],
                    rtol=2e-2,
                    atol=2e-2,
                    err_msg=f"seq {s} token {t} output",
                )
                # Verify reads its inputs at full precision up front and
                # decode loads them compactly: states agree to precision, not
                # bitwise; a wrong slot, offset or column window is O(1) off.
                for name, got, want in (
                    ("recurrent", rec_v[base + t], rec_d.cpu().numpy()[1]),
                    ("conv", conv_v[base + t], conv_d.cpu().numpy()[1]),
                ):
                    rel = np.linalg.norm(got - want) / np.linalg.norm(want)
                    self.assertLess(rel, 1e-2, f"seq {s} {name} checkpoint {t}")
                touched.add(base + t)
        # The null slot (the padded row's target) may be written; every other
        # row the verify did not own must be untouched.
        for r in range(WINDOW, rows):
            if r not in touched:
                np.testing.assert_array_equal(rec_v[r], rec0[r], f"row {r}")
                np.testing.assert_array_equal(conv_v[r], conv0[r], f"row {r}")


if __name__ == "__main__":
    unittest.main()
