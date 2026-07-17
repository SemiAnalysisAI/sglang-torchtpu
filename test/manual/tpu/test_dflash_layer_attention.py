"""Layer-path parity for the non-causal (ENCODER_ONLY) DFlash draft attention.

This drives the REAL ``DFlashAttention.forward`` (qkv projection, per-head q/k
RMSNorm, RoPE, ``RadixAttention(ENCODER_ONLY)`` dispatch to the RPA-v3 backend,
and o_proj) and compares it to a dense PyTorch reference that reuses the very
same sub-modules but replaces only the attention backend with full bidirectional
attention over ``[prefix + block]``. If they match, the whole layer path -- not
just the raw kernel or precomputed-qkv backend call -- is correct on TPU.

Covers prefix ``{0,1,17,63,64,65,255}`` x block ``{4,16}``. Requires JAX + TPU.
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


def _make_backend(page_size, pages_per_seq):
    from sglang.srt.hardware_backend.tpu.attention_pallas_rpa_v3 import (
        TpuPallasRpaV3AttnBackend,
        _build_pallas_kernel_op,
    )
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend

    max_ctx = pages_per_seq * page_size
    req_to_token = (
        torch.arange(max_ctx, dtype=torch.int32, device="tpu").unsqueeze(0).clone()
    )
    backend = TpuPallasRpaV3AttnBackend.__new__(TpuPallasRpaV3AttnBackend)
    AttentionBackend.__init__(backend)
    backend.device = torch.device("tpu")
    backend.speculative_step_id = 0
    backend.page_size = page_size
    backend.req_to_token_pool = types.SimpleNamespace(req_to_token=req_to_token)
    backend._is_hybrid_swa = False
    backend.kv_cache_quant_dtype = None
    backend.forward_metadata = None
    backend._distribution_cache = {}
    backend._encoder_only_mask_cache = {}
    backend._op = _build_pallas_kernel_op()
    return backend


@unittest.skipIf(not _has_jax_tpu(), _SKIP_REASON)
class TestDFlashAttentionLayerParity(unittest.TestCase):
    PAGE_SIZE = 16
    PAGES_PER_SEQ = 18  # 288 tokens per seq (covers prefix 255 + block 16)
    HIDDEN = 512
    NUM_Q_HEADS = 4
    NUM_KV_HEADS = 2  # GQA
    HEAD_DIM = 128
    TOLERANCE = 0.05

    CASES = [
        (0, 4),
        (1, 4),
        (17, 4),
        (63, 4),
        (64, 4),
        (65, 4),
        (255, 4),
        (0, 16),
        (1, 16),
        (17, 16),
        (63, 16),
        (64, 16),
        (65, 16),
        (255, 16),
    ]

    def _config(self):
        return types.SimpleNamespace(
            hidden_size=self.HIDDEN,
            num_attention_heads=self.NUM_Q_HEADS,
            num_key_value_heads=self.NUM_KV_HEADS,
            head_dim=self.HEAD_DIM,
            attention_bias=False,
            rms_norm_eps=1e-6,
            rope_theta=1000000.0,
            rope_scaling=None,
            is_neox_style=True,
            max_position_embeddings=40960,
        )

    def _make_layer(self):
        from sglang.srt.models.dflash import DFlashAttention

        torch.manual_seed(0)
        layer = DFlashAttention(self._config(), layer_id=0)
        # Deterministic, small weights for a well-conditioned bf16 comparison.
        with torch.no_grad():
            for p in layer.parameters():
                p.copy_(torch.randn_like(p, dtype=torch.float32) * 0.05)
                p.requires_grad_(False)
        return layer.to("tpu").to(torch.bfloat16)

    def _fused_cache_with_prefix(self, prefix_k, prefix_v, prefix_len):
        from sglang.srt.layers.attention.pallas.util import get_dtype_packing

        dev = torch.device("tpu")
        P, PS, NKV, HD = (
            self.PAGES_PER_SEQ,
            self.PAGE_SIZE,
            self.NUM_KV_HEADS,
            self.HEAD_DIM,
        )
        k_pages = torch.zeros(P, PS, NKV, HD, dtype=torch.bfloat16, device=dev)
        v_pages = torch.zeros(P, PS, NKV, HD, dtype=torch.bfloat16, device=dev)
        for pos in range(prefix_len):
            k_pages[pos // PS, pos % PS] = prefix_k[pos]
            v_pages[pos // PS, pos % PS] = prefix_v[pos]
        packing = get_dtype_packing(
            __import__("jax.numpy", fromlist=["bfloat16"]).bfloat16
        )
        kv_interleaved = torch.stack([k_pages, v_pages], dim=3).reshape(P, PS, NKV * 2, HD)
        return kv_interleaved.reshape(P, PS, (NKV * 2) // packing, packing, HD)

    def _run_case(self, layer, prefix_len, block_size, seed):
        from sglang.srt.models.dflash import apply_qk_norm
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        dev = torch.device("tpu")
        NKV, NQ, HD = self.NUM_KV_HEADS, self.NUM_Q_HEADS, self.HEAD_DIM
        kv_len = prefix_len + block_size
        torch.manual_seed(seed)

        # Arbitrary resident prefix K/V (post-RoPE values already in the cache).
        prefix_k = torch.randn(prefix_len, NKV, HD, dtype=torch.bfloat16, device=dev) * 0.1
        prefix_v = torch.randn(prefix_len, NKV, HD, dtype=torch.bfloat16, device=dev) * 0.1
        hidden = torch.randn(block_size, self.HIDDEN, dtype=torch.bfloat16, device=dev) * 0.1
        positions = torch.arange(prefix_len, kv_len, dtype=torch.int64, device=dev)

        kv_fused = self._fused_cache_with_prefix(prefix_k, prefix_v, prefix_len)
        backend = self._backend
        forward_batch = types.SimpleNamespace(
            seq_lens=torch.tensor([kv_len], dtype=torch.int32, device=dev),
            req_pool_indices=torch.tensor([0], dtype=torch.int64, device=dev),
            forward_mode=ForwardMode.EXTEND,
            extend_seq_lens=torch.tensor([block_size], dtype=torch.int32, device=dev),
            spec_info=None,
            seq_lens_cpu=torch.tensor([kv_len], dtype=torch.int32),
            token_to_kv_pool=types.SimpleNamespace(
                get_fused_kv_buffer=lambda layer_id: kv_fused
            ),
            attn_backend=backend,
        )
        backend.init_forward_metadata(forward_batch)

        # Real layer path (backend attention).
        layer_out = layer.forward(positions, hidden, forward_batch)
        layer_out = layer_out.float().cpu()

        # Dense reference: identical sub-modules, full attention over prefix+block.
        with torch.no_grad():
            qkv, _ = layer.qkv_proj(hidden)
            q, k, v = qkv.split([layer.q_size, layer.kv_size, layer.kv_size], dim=-1)
            q, k = apply_qk_norm(q, k, layer.q_norm, layer.k_norm, HD)
            q, k = layer.rotary_emb(positions, q, k)
            q = q.view(block_size, NQ, HD).float()
            block_k = k.view(block_size, NKV, HD).float()
            block_v = v.view(block_size, NKV, HD).float()
            full_k = torch.cat([prefix_k.float(), block_k], dim=0)  # [kv_len, NKV, HD]
            full_v = torch.cat([prefix_v.float(), block_v], dim=0)
            rep = NQ // NKV
            k_rep = full_k.repeat_interleave(rep, dim=1)
            v_rep = full_v.repeat_interleave(rep, dim=1)
            attn = torch.einsum("qhd,khd->hqk", q, k_rep) * layer.scaling
            attn = torch.softmax(attn, dim=-1)
            ctx = torch.einsum("hqk,khd->qhd", attn, v_rep)  # [block, NQ, HD]
            ref_out, _ = layer.o_proj(
                ctx.reshape(block_size, NQ * HD).to(torch.bfloat16)
            )
            ref_out = ref_out.float().cpu()
        return layer_out, ref_out

    @classmethod
    def setUpClass(cls):
        import os

        from sglang.srt.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )
        from sglang.srt.distributed.parallel_state import model_parallel_is_initialized

        from sglang.srt.server_args import (
            ServerArgs,
            set_global_server_args_for_scheduler,
        )

        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29517")
        try:
            from sglang.srt.server_args import get_global_server_args

            get_global_server_args()
        except Exception:
            set_global_server_args_for_scheduler(
                ServerArgs(model_path="meta-llama/Llama-3.1-8B-Instruct")
            )
        if not model_parallel_is_initialized():
            init_distributed_environment(
                world_size=1, rank=0, local_rank=0, backend="gloo"
            )
            initialize_model_parallel(tensor_model_parallel_size=1)

    def setUp(self):
        self._backend = _make_backend(self.PAGE_SIZE, self.PAGES_PER_SEQ)
        self._layer = self._make_layer()

    def test_layer_parity(self):
        for i, (prefix_len, block_size) in enumerate(self.CASES):
            with self.subTest(prefix_len=prefix_len, block_size=block_size):
                out, ref = self._run_case(self._layer, prefix_len, block_size, seed=i)
                diff = float((out - ref).abs().max())
                scale = max(1.0, float(ref.abs().max()))
                self.assertLess(
                    diff / scale,
                    self.TOLERANCE,
                    f"prefix={prefix_len} block={block_size}: layer vs dense "
                    f"reference rel diff {diff / scale} >= {self.TOLERANCE}",
                )


if __name__ == "__main__":
    unittest.main()
