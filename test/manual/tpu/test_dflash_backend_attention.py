"""Backend-level parity for the non-causal (ENCODER_ONLY) DFlash draft attention.

Unlike the raw-kernel test (``test_ragged_paged_attention_v3.py``), this drives
the real ``TpuPallasRpaV3AttnBackend`` path: ``init_forward_metadata`` /
``_build_metadata`` builds the rectangular page metadata, ``_forward`` selects
``causal=0`` + the fixed-shape all-ones mask for the ENCODER_ONLY layer, writes
the new block K/V into the fused paged cache, and runs the Pallas op. The output
is compared to a dense PyTorch full-attention reference over ``[prefix + block]``
(GQA-aware), covering prefix ``{0,1,17,63,64,65,255}`` x block ``{4,16}``.

Requires JAX + a TPU device.
"""

import types
import unittest

import torch


def _has_jax_tpu() -> bool:
    try:
        import jax
    except Exception:
        return False
    try:
        return any("TPU" in d.device_kind for d in jax.devices())
    except Exception:
        return False


_SKIP_REASON = "Requires JAX with a TPU device"


@unittest.skipIf(not _has_jax_tpu(), _SKIP_REASON)
class TestDflashBackendEncoderOnlyParity(unittest.TestCase):
    PAGE_SIZE = 16
    PAGES_PER_SEQ = 18  # 288 tokens per seq (covers prefix 255 + block 16)
    HEAD_DIM = 128
    NUM_Q_HEADS = 4
    NUM_KV_HEADS = 1
    TOLERANCE = 0.05

    CASES = [
        (0, 4),
        (1, 4),
        (17, 4),
        (63, 4),
        (64, 4),
        (65, 4),
        (255, 4),
        (17, 16),
        (255, 16),
    ]

    def _make_backend(self):
        from sglang.srt.hardware_backend.tpu.attention_pallas_rpa_v3 import (
            TpuPallasRpaV3AttnBackend,
            _build_pallas_kernel_op,
        )

        max_ctx = self.PAGES_PER_SEQ * self.PAGE_SIZE
        req_to_token = (
            torch.arange(max_ctx, dtype=torch.int32, device="tpu")
            .unsqueeze(0)
            .clone()
        )  # single req: req_to_token[0, s] = s
        backend = TpuPallasRpaV3AttnBackend.__new__(TpuPallasRpaV3AttnBackend)
        # Minimal manual init (avoid the model_runner construction graph).
        from sglang.srt.layers.attention.base_attn_backend import AttentionBackend

        AttentionBackend.__init__(backend)
        backend.device = torch.device("tpu")
        backend.speculative_step_id = 0
        backend.page_size = self.PAGE_SIZE
        backend.req_to_token_pool = types.SimpleNamespace(req_to_token=req_to_token)
        backend._is_hybrid_swa = False
        backend.kv_cache_quant_dtype = None
        backend.forward_metadata = None
        backend._distribution_cache = {}
        backend._op = _build_pallas_kernel_op()
        return backend

    def _run_case(self, prefix_len, block_size, seed=0):
        from sglang.srt.layers.attention.pallas.util import get_dtype_packing
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        dev = torch.device("tpu")
        dtype = torch.bfloat16
        P, PS, HD = self.PAGES_PER_SEQ, self.PAGE_SIZE, self.HEAD_DIM
        NKV, NQ = self.NUM_KV_HEADS, self.NUM_Q_HEADS
        kv_len = prefix_len + block_size

        torch.manual_seed(seed)
        # Prefix K/V already resident in the cache; block K/V are the new tokens.
        prefix_k = torch.randn(prefix_len, NKV, HD, dtype=dtype, device=dev)
        prefix_v = torch.randn(prefix_len, NKV, HD, dtype=dtype, device=dev)
        block_q = torch.randn(block_size, NQ, HD, dtype=dtype, device=dev)
        block_k = torch.randn(block_size, NKV, HD, dtype=dtype, device=dev)
        block_v = torch.randn(block_size, NKV, HD, dtype=dtype, device=dev)

        # Pre-write pages hold only the prefix; the kernel writes the block K/V.
        k_pages = torch.zeros(P, PS, NKV, HD, dtype=dtype, device=dev)
        v_pages = torch.zeros(P, PS, NKV, HD, dtype=dtype, device=dev)
        for pos in range(prefix_len):
            k_pages[pos // PS, pos % PS] = prefix_k[pos]
            v_pages[pos // PS, pos % PS] = prefix_v[pos]

        packing = get_dtype_packing(__import__("jax.numpy", fromlist=["bfloat16"]).bfloat16)
        kv_interleaved = torch.stack([k_pages, v_pages], dim=3).reshape(
            P, PS, NKV * 2, HD
        )
        kv_fused = kv_interleaved.reshape(P, PS, (NKV * 2) // packing, packing, HD)

        forward_batch = types.SimpleNamespace(
            seq_lens=torch.tensor([kv_len], dtype=torch.int32, device=dev),
            req_pool_indices=torch.tensor([0], dtype=torch.int64, device=dev),
            forward_mode=ForwardMode.EXTEND,
            extend_seq_lens=torch.tensor([block_size], dtype=torch.int32, device=dev),
            spec_info=None,
            # The DFlash draft block forward passes input_embeds; setting it makes
            # the backend prepare the ENCODER_ONLY mask bucket host-side.
            input_embeds=torch.zeros(1),
            seq_lens_cpu=torch.tensor([kv_len], dtype=torch.int32),
            token_to_kv_pool=types.SimpleNamespace(
                get_fused_kv_buffer=lambda layer_id: kv_fused
            ),
        )
        layer = types.SimpleNamespace(
            attn_type=__import__(
                "sglang.srt.layers.radix_attention",
                fromlist=["AttentionType"],
            ).AttentionType.ENCODER_ONLY,
            tp_q_head_num=NQ,
            tp_k_head_num=NKV,
            tp_v_head_num=NKV,
            qk_head_dim=HD,
            v_head_dim=HD,
            scaling=1.0 / (HD**0.5),
            layer_id=0,
            logit_cap=0,
            sliding_window_size=None,
            k_scale_float=None,
            v_scale_float=None,
        )

        backend = self._backend
        backend.init_forward_metadata(forward_batch)
        out = backend._forward(
            block_q.reshape(block_size, NQ * HD),
            block_k.reshape(block_size, NKV * HD),
            block_v.reshape(block_size, NKV * HD),
            layer,
            forward_batch,
        )
        out = out.reshape(block_size, NQ, HD).float().cpu()

        # Dense reference: full (non-causal) attention over prefix + block.
        full_k = torch.cat([prefix_k, block_k], dim=0).float().cpu()  # [kv_len, NKV, HD]
        full_v = torch.cat([prefix_v, block_v], dim=0).float().cpu()
        q_ref = block_q.float().cpu()  # [block, NQ, HD]
        rep = NQ // NKV
        k_rep = full_k.repeat_interleave(rep, dim=1)  # [kv_len, NQ, HD]
        v_rep = full_v.repeat_interleave(rep, dim=1)
        # [NQ, block, kv_len]
        attn = torch.einsum("qhd,khd->hqk", q_ref, k_rep) * layer.scaling
        attn = torch.softmax(attn, dim=-1)
        ref = torch.einsum("hqk,khd->qhd", attn, v_rep)  # [block, NQ, HD]
        return out, ref

    def setUp(self):
        self._backend = self._make_backend()

    def test_encoder_only_parity(self):
        for prefix_len, block_size in self.CASES:
            with self.subTest(prefix_len=prefix_len, block_size=block_size):
                out, ref = self._run_case(prefix_len, block_size)
                diff = float((out - ref).abs().max())
                self.assertLess(
                    diff,
                    self.TOLERANCE,
                    f"prefix={prefix_len} block={block_size}: backend vs dense "
                    f"reference max diff {diff} >= {self.TOLERANCE}",
                )

    def test_rpa_v3_forward_never_calls_triton_kv_indices(self):
        """Dynamic guard: run a real RPA-v3 backend forward with the
        flashinfer Triton kv-indices kernel monkeypatched to raise. Since RPA-v3
        builds its own metadata (never DFlashVerifyInput.generate_attn_arg_prefill,
        the sole caller of the kernel), the forward must complete untouched.
        """
        # Patch the SOURCE module: dflash_info imports the kernel lazily via
        # `from sglang.srt.layers.attention.utils import
        # create_flashinfer_kv_indices_triton` inside generate_attn_arg_prefill,
        # so it is not a dflash_info attribute — patching the source makes any
        # reach through that lazy import raise.
        import sglang.srt.layers.attention.utils as au

        def _boom(*a, **k):  # pragma: no cover - must never be called
            raise AssertionError(
                "create_flashinfer_kv_indices_triton was reached on the TPU "
                "RPA-v3 verify path"
            )

        saved = au.create_flashinfer_kv_indices_triton
        au.create_flashinfer_kv_indices_triton = _boom
        try:
            out, ref = self._run_case(17, 4)
        finally:
            au.create_flashinfer_kv_indices_triton = saved
        self.assertLess(float((out - ref).abs().max()), self.TOLERANCE)


if __name__ == "__main__":
    unittest.main()
