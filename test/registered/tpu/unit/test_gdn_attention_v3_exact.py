"""Bit-exactness of the GDN/KDA kernel's layout and fast-path variants.

The TPU state pool stores conv state in the kernel-native rank-4
``[slots, K-1, 1, dim]`` form and Kimi-K3 binds its conv weight as
``[K, 1, dim]``. Both only skip a relayout, so the kernel must produce the
same bits (output, conv state, recurrent state) as with the rank-3 pool and the
depthwise ``[dim, 1, K]`` weight, for GDN (Qwen3.5) and KDA (Kimi), on batches
that read non-zero initial state.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit")

import unittest

from sglang.test.test_utils import CustomTestCase


def _has_jax_tpu() -> bool:
    try:
        import jax
    except Exception:
        return False
    try:
        return any("TPU" in d.device_kind for d in jax.devices())
    except Exception:
        return False


if _has_jax_tpu():
    import jax.numpy as jnp
    import numpy as np

    from sglang.srt.layers.attention.pallas.gdn import fused_conv1d_gdn
    from sglang.srt.layers.attention.pallas.gdn.config import AttentionMode

_K = 4
_D = 128
# (name, lengths, distribution): decode rows first, then prefill rows.
_BATCHES = [
    ("decode", [1] * 8, [8, 8, 8]),
    ("mixed", [1, 1, 1, 40], [3, 4, 4]),
    ("prefill", [24, 40], [0, 2, 2]),
]


def _case(kda: bool, lengths, distribution, seed=0):
    """Numpy inputs for one call; every slot starts from non-zero state."""
    rng = np.random.default_rng(seed)
    n_kq, n_v = (4, 4) if kda else (2, 8)
    dim = 2 * n_kq * _D + n_v * _D
    num_seqs, num_tokens = len(lengths), sum(lengths)

    def rn(*shape, scale=1.0):
        return (scale * rng.standard_normal(shape)).astype(np.float32)

    arrays = dict(
        qkv=rn(num_tokens, dim).astype(jnp.bfloat16),
        b=rn(num_tokens, n_v).astype(jnp.bfloat16),
        a=rn(num_tokens, n_v * _D if kda else n_v).astype(jnp.bfloat16),
        conv_state=rn(num_seqs + 1, _K - 1, dim),
        recurrent_state=rn(num_seqs + 1, n_v, _D, _D, scale=0.1),
        conv_weight=rn(dim, 1, _K),
        conv_bias=None if kda else rn(dim),
        a_log=rn(n_v),
        dt_bias=rn(n_v, _D) if kda else rn(n_v),
        query_start_loc=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32),
        state_indices=np.arange(1, num_seqs + 1, dtype=np.int32),
        distribution=np.array(distribution, np.int32),
        # 5 tokens of context each: has_initial_state is True everywhere.
        seq_lens=(np.array(lengths) + 5).astype(np.int32),
    )
    static = dict(n_kq=n_kq, n_v=n_v, d_k=_D, d_v=_D, kernel_size=_K)
    if kda:
        static.update(attention_mode=AttentionMode.KDA, gate_lower_bound=-5.0)
    return arrays, static


def _call(arrays, static, kernel=None, **overrides):
    """Run the kernel on fresh device copies (the states are donated)."""
    inputs = {**arrays, **overrides}
    inputs = {
        k: jnp.asarray(v) if isinstance(v, np.ndarray) else v for k, v in inputs.items()
    }
    (conv, rec), out = (kernel or fused_conv1d_gdn)(**inputs, **static)
    return np.asarray(conv), np.asarray(rec), np.asarray(out.astype(jnp.float32))


@unittest.skipUnless(_has_jax_tpu(), "Requires JAX with a TPU device")
class TestGDNKernelLayouts(CustomTestCase):
    def assert_same_bits(self, got, want, conv_shape=None):
        conv, rec, out = got
        if conv_shape is not None:
            self.assertEqual(conv.shape, conv_shape)
        np.testing.assert_array_equal(conv.reshape(want[0].shape), want[0])
        np.testing.assert_array_equal(rec, want[1])
        np.testing.assert_array_equal(out, want[2])

    def test_rank4_conv_state_and_kd_weight_are_bit_identical(self):
        for kda in (False, True):
            for name, lengths, dist in _BATCHES:
                with self.subTest(kda=kda, batch=name):
                    arrays, static = _case(kda, lengths, dist)
                    ref = _call(arrays, static)
                    conv = arrays["conv_state"]
                    conv4 = conv.reshape(conv.shape[0], _K - 1, 1, -1)
                    # Rank-4 pool: the state comes back in the pool's shape.
                    self.assert_same_bits(
                        _call(arrays, static, conv_state=conv4), ref, conv4.shape
                    )
                    w_kd = np.ascontiguousarray(
                        arrays["conv_weight"].transpose(2, 1, 0)
                    )
                    self.assert_same_bits(
                        _call(arrays, static, conv_state=conv4, conv_weight=w_kd),
                        ref,
                    )


if __name__ == "__main__":
    unittest.main()
