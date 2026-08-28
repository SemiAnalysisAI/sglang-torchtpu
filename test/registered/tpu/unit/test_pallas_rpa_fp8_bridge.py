"""TorchTPU-to-Pallas FP8 KV-cache correctness tests.

The raw-JAX RPA tests validate the kernel once its inputs are JAX arrays. These
tests cover the production boundary before that kernel: TorchTPU FP8
quantization, conversion through the custom JAX op, and the aliased fused-cache
write. The single-KV-head shape is the local geometry used by Qwen3 under TP=8.
"""

from __future__ import annotations

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit")

import math
import unittest

import torch

try:
    import torch_tpu  # noqa: F401
    from torch_tpu._internal import compile as _torch_tpu_compile  # noqa: F401

    _HAS_TPU = True
except Exception:  # pragma: no cover
    _HAS_TPU = False


def _has_tpu_device() -> bool:
    if not _HAS_TPU:
        return False
    try:
        torch.empty(1, device="tpu")
        return True
    except Exception:
        return False


@unittest.skipUnless(_has_tpu_device(), "Requires torch_tpu + a TPU device")
class TestPallasRpaFp8Bridge(unittest.TestCase):
    NUM_Q_HEADS = 2
    NUM_KV_HEADS = 1
    HEAD_DIM = 128
    PAGE_SIZE = 16
    PAGES_PER_SEQ = 4
    TOTAL_NUM_PAGES = 16
    NUM_TOKENS = 8

    @classmethod
    def setUpClass(cls):
        from sglang.srt.hardware_backend.tpu.attention_pallas_rpa_base import (
            _build_pallas_kernel_op,
        )
        from sglang.srt.hardware_backend.tpu.attention_pallas_rpa_v3 import (
            RPA_V3_SPEC,
        )
        from sglang.srt.hardware_backend.tpu.patches import apply_tpu_patches

        apply_tpu_patches()
        cls.op = _build_pallas_kernel_op(RPA_V3_SPEC)

    def test_fp8_quantization_matches_cpu_bytes(self):
        from sglang.srt.hardware_backend.tpu.attention_pallas_rpa_base import (
            _pallas_fp8_kv_scale,
            _quantize_kv_to_fp8,
        )

        values = torch.linspace(-500, 500, 4096, dtype=torch.float32).reshape(
            16, 2, self.HEAD_DIM
        )
        scale = _pallas_fp8_kv_scale(1.0, torch.float8_e4m3fn)
        expected = _quantize_kv_to_fp8(values, scale, torch.float8_e4m3fn)
        actual = _quantize_kv_to_fp8(values.to("tpu"), scale, torch.float8_e4m3fn).cpu()
        self.assertTrue(
            torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
        )
        self.assertLessEqual(float(actual.float().abs().max()), 240.0)
        torch.testing.assert_close(
            actual.float() * scale,
            values.clamp(-448, 448),
            rtol=0.07,
            atol=0.02,
        )

    def test_custom_op_output_and_cache_write_match_reference(self):
        from sglang.srt.hardware_backend.tpu.attention_pallas_rpa_base import (
            _quantize_kv_to_fp8,
        )

        torch.manual_seed(7)
        q_cpu = torch.randn(
            self.NUM_TOKENS,
            self.NUM_Q_HEADS,
            self.HEAD_DIM,
            dtype=torch.bfloat16,
        )
        k_cpu = torch.randn(
            self.NUM_TOKENS,
            self.NUM_KV_HEADS,
            self.HEAD_DIM,
            dtype=torch.bfloat16,
        )
        v_cpu = torch.randn(
            self.NUM_TOKENS,
            self.NUM_KV_HEADS,
            self.HEAD_DIM,
            dtype=torch.bfloat16,
        )

        q = q_cpu.to("tpu")
        k = _quantize_kv_to_fp8(k_cpu.to("tpu"), 1.0, torch.float8_e4m3fn)
        v = _quantize_kv_to_fp8(v_cpu.to("tpu"), 1.0, torch.float8_e4m3fn)
        cache = torch.zeros(
            (
                self.TOTAL_NUM_PAGES,
                self.PAGE_SIZE,
                1,
                4,
                self.HEAD_DIM,
            ),
            dtype=torch.uint8,
            device="tpu",
        ).view(torch.float8_e4m3fn)

        kv_lens = torch.tensor([self.NUM_TOKENS], dtype=torch.int32, device="tpu")
        page_indices = torch.arange(self.PAGES_PER_SEQ, dtype=torch.int32, device="tpu")
        cu_q_lens = torch.tensor([0, self.NUM_TOKENS], dtype=torch.int32, device="tpu")
        cu_kv_lens = torch.tensor(
            [0, self.PAGES_PER_SEQ * self.PAGE_SIZE],
            dtype=torch.int32,
            device="tpu",
        )
        distribution = torch.tensor([0, 1, 1], dtype=torch.int32, device="tpu")
        sm_scale = 1.0 / math.sqrt(self.HEAD_DIM)

        out = (
            self.op(
                q,
                k,
                v,
                cache,
                kv_lens,
                page_indices,
                cu_q_lens,
                cu_kv_lens,
                distribution,
                None,
                sm_scale,
                None,
                None,
                1.0,
                1.0,
            )
            .cpu()
            .float()
        )

        k_ref = _quantize_kv_to_fp8(k_cpu, 1.0, torch.float8_e4m3fn).float()[:, 0]
        v_ref = _quantize_kv_to_fp8(v_cpu, 1.0, torch.float8_e4m3fn).float()[:, 0]
        q_ref = q_cpu.float()
        scores = torch.einsum("thd,sd->hts", q_ref, k_ref) * sm_scale
        causal_mask = torch.triu(
            torch.ones(self.NUM_TOKENS, self.NUM_TOKENS, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask.unsqueeze(0), float("-inf"))
        ref = torch.einsum("hts,sd->thd", torch.softmax(scores, dim=-1), v_ref)
        torch.testing.assert_close(out, ref, rtol=0.08, atol=0.08)

        cache_cpu = cache.cpu()
        cache_atol = float(torch.finfo(torch.float8_e4m3fn).smallest_normal)
        torch.testing.assert_close(
            cache_cpu[0, : self.NUM_TOKENS, 0, 0].float(),
            k.cpu().float()[:, 0],
            rtol=0,
            atol=cache_atol,
        )
        torch.testing.assert_close(
            cache_cpu[0, : self.NUM_TOKENS, 0, 1].float(),
            v.cpu().float()[:, 0],
            rtol=0,
            atol=cache_atol,
        )
        self.assertEqual(
            float(cache_cpu[0, : self.NUM_TOKENS, 0, 2:].float().abs().sum()),
            0.0,
        )

        compiled_cache = torch.zeros(
            (
                self.TOTAL_NUM_PAGES,
                self.PAGE_SIZE,
                1,
                4,
                self.HEAD_DIM,
            ),
            dtype=torch.uint8,
            device="tpu",
        ).view(torch.float8_e4m3fn)

        def _run_compiled(
            q, k, v, cache, kv_lens, page_indices, cu_q_lens, cu_kv_lens, distribution
        ):
            return self.op(
                q,
                k,
                v,
                cache,
                kv_lens,
                page_indices,
                cu_q_lens,
                cu_kv_lens,
                distribution,
                None,
                sm_scale,
                None,
                None,
                1.0,
                1.0,
            )

        compiled = torch.compile(
            _run_compiled, backend="tpu", fullgraph=True, dynamic=False
        )
        compiled_out = (
            compiled(
                q,
                k,
                v,
                compiled_cache,
                kv_lens,
                page_indices,
                cu_q_lens,
                cu_kv_lens,
                distribution,
            )
            .cpu()
            .float()
        )
        torch.testing.assert_close(compiled_out, ref, rtol=0.08, atol=0.08)
        torch.testing.assert_close(compiled_out, out, rtol=0, atol=0)

        compiled_cache_cpu = compiled_cache.cpu()
        torch.testing.assert_close(
            compiled_cache_cpu[0, : self.NUM_TOKENS, 0, 0].float(),
            k.cpu().float()[:, 0],
            rtol=0,
            atol=cache_atol,
        )
        torch.testing.assert_close(
            compiled_cache_cpu[0, : self.NUM_TOKENS, 0, 1].float(),
            v.cpu().float()[:, 0],
            rtol=0,
            atol=cache_atol,
        )

    def test_fp8_cache_written_by_prefill_is_reused_by_decode(self):
        """Exercise the production TP8 transition across separate op calls."""
        from sglang.srt.hardware_backend.tpu.attention_pallas_rpa_base import (
            _pallas_fp8_kv_scale,
            _quantize_kv_to_fp8,
        )

        torch.manual_seed(11)
        page_size = 64
        prefill_tokens = 38
        padded_tokens = 64
        sm_scale = 1.0 / math.sqrt(self.HEAD_DIM)
        kv_scale = _pallas_fp8_kv_scale(1.0, torch.float8_e4m3fn)
        cache = torch.zeros(
            (1, page_size, 1, 4, self.HEAD_DIM),
            dtype=torch.uint8,
            device="tpu",
        ).view(torch.float8_e4m3fn)
        page_indices = torch.tensor([0], dtype=torch.int32, device="tpu")
        cu_kv_lens = torch.tensor([0, page_size], dtype=torch.int32, device="tpu")

        q_cpu = torch.randn(
            padded_tokens,
            self.NUM_Q_HEADS,
            self.HEAD_DIM,
            dtype=torch.bfloat16,
        )
        k_cpu = torch.randn(
            padded_tokens,
            self.NUM_KV_HEADS,
            self.HEAD_DIM,
            dtype=torch.bfloat16,
        )
        v_cpu = torch.randn(
            padded_tokens,
            self.NUM_KV_HEADS,
            self.HEAD_DIM,
            dtype=torch.bfloat16,
        )
        k = _quantize_kv_to_fp8(k_cpu.to("tpu"), kv_scale, torch.float8_e4m3fn)
        v = _quantize_kv_to_fp8(v_cpu.to("tpu"), kv_scale, torch.float8_e4m3fn)
        self.op(
            q_cpu.to("tpu"),
            k,
            v,
            cache,
            torch.tensor([prefill_tokens], dtype=torch.int32, device="tpu"),
            page_indices,
            torch.tensor([0, prefill_tokens], dtype=torch.int32, device="tpu"),
            cu_kv_lens,
            torch.tensor([0, 1, 1], dtype=torch.int32, device="tpu"),
            None,
            sm_scale,
            None,
            None,
            kv_scale,
            kv_scale,
        )

        intended_k = [k.cpu().float()[:prefill_tokens, 0]]
        intended_v = [v.cpu().float()[:prefill_tokens, 0]]
        cache_atol = float(torch.finfo(torch.float8_e4m3fn).smallest_normal)

        for step in range(1, 5):
            q_decode_cpu = torch.randn(
                1, self.NUM_Q_HEADS, self.HEAD_DIM, dtype=torch.bfloat16
            )
            k_decode_cpu = torch.randn(
                1, self.NUM_KV_HEADS, self.HEAD_DIM, dtype=torch.bfloat16
            )
            v_decode_cpu = torch.randn(
                1, self.NUM_KV_HEADS, self.HEAD_DIM, dtype=torch.bfloat16
            )
            if step == 1:
                # E4M3 byte 0xf8 (-256) is canonicalized to NaN by the
                # single-token Pallas VMEM DMA. The production scale maps this
                # real value into the safe exponent range before the transfer.
                k_decode_cpu[0, 0, 50] = -248
            k_decode = _quantize_kv_to_fp8(
                k_decode_cpu.to("tpu"), kv_scale, torch.float8_e4m3fn
            )
            v_decode = _quantize_kv_to_fp8(
                v_decode_cpu.to("tpu"), kv_scale, torch.float8_e4m3fn
            )
            kv_len = prefill_tokens + step
            out = self.op(
                q_decode_cpu.to("tpu"),
                k_decode,
                v_decode,
                cache,
                torch.tensor([kv_len], dtype=torch.int32, device="tpu"),
                page_indices,
                torch.tensor([0, 1], dtype=torch.int32, device="tpu"),
                cu_kv_lens,
                torch.tensor([1, 1, 1], dtype=torch.int32, device="tpu"),
                None,
                sm_scale,
                None,
                None,
                kv_scale,
                kv_scale,
            )
            intended_k.append(k_decode.cpu().float()[:, 0])
            intended_v.append(v_decode.cpu().float()[:, 0])
            all_k = torch.cat(intended_k)
            all_v = torch.cat(intended_v)
            scores = torch.einsum("thd,sd->hts", q_decode_cpu.float(), all_k * kv_scale)
            scores *= sm_scale
            ref = torch.einsum(
                "hts,sd->thd",
                torch.softmax(scores, dim=-1),
                all_v * kv_scale,
            )
            torch.testing.assert_close(out.cpu().float(), ref, rtol=0.08, atol=0.08)

            persisted = cache.cpu()
            torch.testing.assert_close(
                persisted[0, :kv_len, 0, 0].float(),
                all_k,
                rtol=0,
                atol=cache_atol,
            )
            torch.testing.assert_close(
                persisted[0, :kv_len, 0, 1].float(),
                all_v,
                rtol=0,
                atol=cache_atol,
            )


if __name__ == "__main__":
    unittest.main()
