"""Regress the real producer-layout-sensitive QA RMS rounding boundary."""

import hashlib
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils._pytree import tree_leaves

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=45, suite="stage-a-test-tpu-unit")

try:
    import jax

    _HAS_TPU = any(device.platform == "tpu" for device in jax.devices())
except Exception:
    _HAS_TPU = False


_FIXTURE = Path(__file__).with_name("fixtures") / "glm_fp8_mla_front_qa_bf16.bin"


def load_fixture():
    payload = _FIXTURE.read_bytes()
    metadata = json.loads(_FIXTURE.with_suffix(".json").read_text())
    if (
        len(payload) != 8192
        or hashlib.sha256(payload).hexdigest() != metadata["sha256"]
    ):
        raise ValueError("GLM QA fixture is incomplete or changed")
    # Normalize byte order before viewing as BF16; fixture is little-endian.
    bits = np.frombuffer(payload, dtype="<u2").astype(np.uint16, copy=True)
    values = torch.from_numpy(bits).view(torch.bfloat16)
    return values[:2048].reshape(1, 2048), values[2048:]


def make_inputs(qa, norm_weight):
    x = torch.zeros((1, 6144), dtype=torch.bfloat16)
    x[0, 0] = 1
    a = torch.zeros((6144, 2624), dtype=torch.float32)
    a[0, :2048] = qa.flatten().float().sign()
    a_scale = torch.ones((2624, 1), dtype=torch.float32)
    a_scale[:2048, 0] = torch.where(qa.flatten() != 0, qa.flatten().float().abs(), 1.0)
    b = torch.zeros((2048, 1024), dtype=torch.float32)
    # Expose the affected quantized QA coordinate in both NoPE and RoPE.
    b[649, 0] = 1
    b[649, 192] = 1
    kc = torch.zeros((4, 512, 192), dtype=torch.bfloat16)
    kc[0, 0, 0] = 1
    cache = torch.zeros((2048, 64), dtype=torch.bfloat16)
    cache[:, :32] = 1
    return (
        x,
        a.to(torch.float8_e4m3fn),
        a_scale,
        norm_weight,
        torch.ones(512, dtype=torch.bfloat16),
        b.to(torch.float8_e4m3fn),
        torch.ones((1024, 1), dtype=torch.float32),
        kc,
        torch.tensor([0], dtype=torch.int32),
        cache,
    )


def native_front(x, a, sa, nq, nk, b, sb, kc, positions, cache):
    from sglang.srt.hardware_backend.tpu.fp8_linear import apply_tpu_fp8_linear
    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.layers.rotary_embedding.utils import apply_rotary_emb

    def norm(value, weight):
        attributes = SimpleNamespace(
            weight=weight,
            variance_epsilon=1e-5,
            hidden_size=weight.numel(),
            variance_size_override=None,
            override_orig_dtype=None,
            fp32_residual=False,
            cast_x_before_out_mul=False,
        )
        return RMSNorm.forward_native(attributes, value)

    qkv = apply_tpu_fp8_linear(x, a, sa)
    qa = norm(qkv[:, :2048], nq)
    kv = norm(qkv[:, 2048:2560], nk).unsqueeze(1)
    q = apply_tpu_fp8_linear(qa, b, sb).reshape(1, 4, 256)
    q_nope, q_rope = q[..., :192], q[..., 192:]
    q_latent = torch.bmm(q_nope.transpose(0, 1), kc.transpose(1, 2)).transpose(0, 1)
    k_rope = qkv[:, 2560:].unsqueeze(1)
    cos, sin = cache.index_select(0, positions).chunk(2, -1)
    q_rope = apply_rotary_emb(q_rope, cos, sin, False)
    k_rope = apply_rotary_emb(k_rope, cos, sin, False)
    return tuple(value.contiguous() for value in (q_latent, q_rope, kv, k_rope))


class TestGlmFp8MlaFrontFixture(unittest.TestCase):
    def test_compact_fixture_is_finite_and_has_expected_shapes(self):
        qa, weight = load_fixture()
        self.assertEqual(tuple(qa.shape), (1, 2048))
        self.assertEqual(tuple(weight.shape), (2048,))
        self.assertEqual(qa.dtype, torch.bfloat16)
        self.assertEqual(weight.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(qa).all())
        self.assertTrue(torch.isfinite(weight).all())


@unittest.skipUnless(_HAS_TPU, "requires a TPU backend")
class TestGlmFp8MlaFrontProducerRms(unittest.TestCase):
    def test_native_projection_rms_boundary_matches_fused_front(self):
        from torch_tpu._internal.sync.sync import synchronize

        from sglang.srt.hardware_backend.tpu.compilation.direct_compile import (
            direct_compile,
        )
        from sglang.srt.hardware_backend.tpu.fp8_linear import apply_tpu_fp8_linear
        from sglang.srt.hardware_backend.tpu.glm_fp8_mla_front import (
            glm_fp8_mla_front,
            register_glm_fp8_mla_front_op,
        )

        def compile_fn(fn):
            return direct_compile(
                fn,
                specialization_key=lambda *args: tuple(
                    (value.shape, value.dtype) for value in tree_leaves(args)
                ),
            )

        qa, norm_weight = load_fixture()
        inputs = tuple(value.to("tpu") for value in make_inputs(qa, norm_weight))
        # Keep the real producer shape. A bare RMS over an isolated QA input
        # picks a different reduction layout and does not reproduce this bug.
        produced = compile_fn(apply_tpu_fp8_linear)(*inputs[:3])
        synchronize([produced], wait=True)
        produced = produced.cpu()
        self.assertTrue(torch.equal(produced[:, :2048], qa))
        self.assertEqual(torch.count_nonzero(produced[:, 2048:]).item(), 0)

        register_glm_fp8_mla_front_op()

        def fused(*args):
            return glm_fp8_mla_front(*args, 1e-5, 1e-5)

        expected = compile_fn(native_front)(*inputs)
        actual = compile_fn(fused)(*inputs)
        synchronize([*expected, *actual], wait=True)
        for name, value, target in zip(
            ("q_latent", "q_rope", "kv_latent", "k_rope"), actual, expected
        ):
            with self.subTest(output=name):
                self.assertTrue(torch.equal(value.cpu(), target.cpu()))
        # Original generic Pallas mean yielded 0.0162353515625 at both outputs;
        # the artifact old/new experiment records that failure independently.
        self.assertEqual(expected[0].cpu()[0, 0, 0].item(), 0.01470947265625)
        self.assertEqual(expected[1].cpu()[0, 0, 0].item(), 0.01470947265625)


if __name__ == "__main__":
    unittest.main()
