"""End-to-end test of ``PallasAttnBackend`` driven through the same calling
convention as ``RadixAttention.forward``.

This exercises:

* ``PallasAttnBackend.__init__``
* ``PallasAttnBackend.init_forward_metadata``
* ``PallasAttnBackend.forward_extend`` / ``forward_decode``
* The fused KV cache update done by the Pallas kernel

The "layer" and "forward batch" are minimal duck-typed stubs, but the inputs
match what Qwen3-0.6B-style attention layers actually produce (head_dim=128,
GQA 16/8, etc.). Requires torch_tpu + a TPU device.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

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


_SKIP_REASON = "Requires torch_tpu + a TPU device"


class _DummyForwardMode:
    def __init__(self, mode: str):
        self._mode = mode

    def is_decode(self):
        return self._mode == "decode"

    def is_extend(self):
        return self._mode == "extend"

    def is_idle(self):
        return False

    def is_mixed(self):
        return False

    def is_extend_or_draft_extend_or_mixed(self):
        return self._mode == "extend"


def _make_layer(num_q_heads: int, num_kv_heads: int, head_dim: int, layer_id: int = 0):
    """Duck-typed ``RadixAttention``-like object."""
    return SimpleNamespace(
        layer_id=layer_id,
        tp_q_head_num=num_q_heads,
        tp_k_head_num=num_kv_heads,
        tp_v_head_num=num_kv_heads,
        qk_head_dim=head_dim,
        v_head_dim=head_dim,
        scaling=head_dim ** -0.5,
        sliding_window_size=-1,
        logit_cap=0,
        is_cross_attention=False,
    )


def _make_forward_batch(
    *,
    mode: str,
    seq_lens: list[int],
    extend_seq_lens: list[int] | None,
    req_pool_indices: list[int],
    token_to_kv_pool,
    req_to_token_pool,
    device: str,
):
    return SimpleNamespace(
        forward_mode=_DummyForwardMode(mode),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
        extend_seq_lens=(
            torch.tensor(extend_seq_lens, dtype=torch.int32, device=device)
            if extend_seq_lens is not None
            else None
        ),
        req_pool_indices=torch.tensor(req_pool_indices, dtype=torch.int64, device=device),
        token_to_kv_pool=token_to_kv_pool,
        req_to_token_pool=req_to_token_pool,
    )


@unittest.skipUnless(_has_tpu_device(), _SKIP_REASON)
class TestPallasBackendForwardE2E(unittest.TestCase):
    """Drive the Pallas backend end-to-end through its public forward API."""

    @classmethod
    def setUpClass(cls):
        from sglang.srt.hardware_backend.tpu.kv_pool import TpuFusedMHATokenToKVPool
        from sglang.srt.layers.attention.pallas_backend import PallasAttnBackend

        cls.device = "tpu"
        cls.dtype = torch.bfloat16
        cls.page_size = 16
        cls.max_context_len = 64  # pages_per_seq = 4
        cls.head_dim = 128
        cls.num_q_heads = 16
        cls.num_kv_heads = 8
        cls.layer_id = 0
        cls.num_layers = 1
        cls.kv_pool_size = 256  # 16 blocks of 16 tokens (plus 1 padding block)
        cls.max_reqs = 8

        cls.kv_pool = TpuFusedMHATokenToKVPool(
            size=cls.kv_pool_size,
            page_size=cls.page_size,
            dtype=cls.dtype,
            head_num=cls.num_kv_heads,
            head_dim=cls.head_dim,
            layer_num=cls.num_layers,
            device=cls.device,
            enable_memory_saver=False,
        )

        # Req-to-token table: per-request, per-token cache slot.
        req_to_token = torch.zeros(
            (cls.max_reqs + 1, cls.max_context_len), dtype=torch.int32, device=cls.device
        )
        # For request 1 we assign blocks 1..N (block 0 is reserved as padding).
        # For request 2 we assign blocks 5..N for variety.
        for req_idx, base_block in [(1, 1), (2, 5)]:
            for tok in range(cls.max_context_len):
                req_to_token[req_idx, tok] = base_block * cls.page_size + tok
        cls.req_to_token_pool = SimpleNamespace(req_to_token=req_to_token)

        runner = SimpleNamespace(
            device=cls.device,
            page_size=cls.page_size,
            req_to_token_pool=cls.req_to_token_pool,
        )
        cls.backend = PallasAttnBackend(runner)
        cls.PallasAttnBackend = PallasAttnBackend

    def setUp(self):
        # Reset the KV cache between tests so a write from one doesn't leak into another.
        for buf in self.kv_pool.kv_buffer:
            buf.zero_()

    def _run_forward(self, mode: str, q, k, v, fb, layer):
        if mode == "decode":
            return self.backend.forward_decode(q, k, v, layer, fb)
        return self.backend.forward_extend(q, k, v, layer, fb)

    def test_extend_single_seq_runs_and_writes_kv_cache(self):
        """Extend with one request, validates output shape & KV-cache write."""
        torch.manual_seed(7)
        seq_lens = [16]
        extend_seq_lens = [16]
        req_pool_indices = [1]
        layer = _make_layer(self.num_q_heads, self.num_kv_heads, self.head_dim)

        fb = _make_forward_batch(
            mode="extend",
            seq_lens=seq_lens,
            extend_seq_lens=extend_seq_lens,
            req_pool_indices=req_pool_indices,
            token_to_kv_pool=self.kv_pool,
            req_to_token_pool=self.req_to_token_pool,
            device=self.device,
        )
        self.backend.init_forward_metadata(fb)

        num_tokens = sum(extend_seq_lens)
        q = torch.randn(
            num_tokens, self.num_q_heads * self.head_dim, dtype=self.dtype, device=self.device
        )
        k = torch.randn(
            num_tokens, self.num_kv_heads * self.head_dim, dtype=self.dtype, device=self.device
        )
        v = torch.randn(
            num_tokens, self.num_kv_heads * self.head_dim, dtype=self.dtype, device=self.device
        )

        out = self._run_forward("extend", q, k, v, fb, layer)
        torch.tpu.synchronize()

        self.assertEqual(out.shape, (num_tokens, self.num_q_heads * self.head_dim))
        # Output should not be all-zero (would indicate the kernel never ran).
        self.assertGreater(out.abs().sum().item(), 0)

        # Verify the kernel wrote K/V back into the fused cache. The
        # rectangular layout puts seq 0's pages at page_indices[0..pages_per_seq).
        # For req_pool_idx=1, block 1 is first.
        cache = self.kv_pool.kv_buffer[0]
        # cache[block, slot, k0, v0, ...] should be non-zero for the first 16 tokens.
        first_block = cache[1]  # block index 1 belongs to req_pool_idx=1
        self.assertGreater(first_block.abs().sum().item(), 0)

    def test_decode_single_seq_runs(self):
        """Decode with one new token on top of 16 prefix tokens."""
        torch.manual_seed(8)
        # First, populate KV cache via an extend.
        seq_lens_prefill = [16]
        ext_lens = [16]
        layer = _make_layer(self.num_q_heads, self.num_kv_heads, self.head_dim)
        fb_prefill = _make_forward_batch(
            mode="extend",
            seq_lens=seq_lens_prefill,
            extend_seq_lens=ext_lens,
            req_pool_indices=[1],
            token_to_kv_pool=self.kv_pool,
            req_to_token_pool=self.req_to_token_pool,
            device=self.device,
        )
        self.backend.init_forward_metadata(fb_prefill)
        n = sum(ext_lens)
        q0 = torch.randn(n, self.num_q_heads * self.head_dim, dtype=self.dtype, device=self.device)
        k0 = torch.randn(n, self.num_kv_heads * self.head_dim, dtype=self.dtype, device=self.device)
        v0 = torch.randn(n, self.num_kv_heads * self.head_dim, dtype=self.dtype, device=self.device)
        self._run_forward("extend", q0, k0, v0, fb_prefill, layer)
        torch.tpu.synchronize()

        # Decode 1 new token (seq_lens = prefix + 1 = 17).
        fb_decode = _make_forward_batch(
            mode="decode",
            seq_lens=[17],
            extend_seq_lens=None,
            req_pool_indices=[1],
            token_to_kv_pool=self.kv_pool,
            req_to_token_pool=self.req_to_token_pool,
            device=self.device,
        )
        self.backend.init_forward_metadata(fb_decode)
        q1 = torch.randn(
            1, self.num_q_heads * self.head_dim, dtype=self.dtype, device=self.device
        )
        k1 = torch.randn(
            1, self.num_kv_heads * self.head_dim, dtype=self.dtype, device=self.device
        )
        v1 = torch.randn(
            1, self.num_kv_heads * self.head_dim, dtype=self.dtype, device=self.device
        )
        out = self._run_forward("decode", q1, k1, v1, fb_decode, layer)
        torch.tpu.synchronize()

        self.assertEqual(out.shape, (1, self.num_q_heads * self.head_dim))
        self.assertGreater(out.abs().sum().item(), 0)

    def test_two_seq_extend_runs(self):
        """Two-request extend (different lengths) returns a finite output."""
        torch.manual_seed(9)
        seq_lens = [16, 8]
        ext = [16, 8]
        layer = _make_layer(self.num_q_heads, self.num_kv_heads, self.head_dim)
        fb = _make_forward_batch(
            mode="extend",
            seq_lens=seq_lens,
            extend_seq_lens=ext,
            req_pool_indices=[1, 2],
            token_to_kv_pool=self.kv_pool,
            req_to_token_pool=self.req_to_token_pool,
            device=self.device,
        )
        self.backend.init_forward_metadata(fb)
        n = sum(ext)
        q = torch.randn(n, self.num_q_heads * self.head_dim, dtype=self.dtype, device=self.device)
        k = torch.randn(n, self.num_kv_heads * self.head_dim, dtype=self.dtype, device=self.device)
        v = torch.randn(n, self.num_kv_heads * self.head_dim, dtype=self.dtype, device=self.device)
        out = self._run_forward("extend", q, k, v, fb, layer)
        torch.tpu.synchronize()

        self.assertEqual(out.shape, (n, self.num_q_heads * self.head_dim))
        self.assertTrue(torch.isfinite(out.float()).all().item())

    def test_compiled_extend_matches_eager(self):
        """torch.compile(backend='tpu') wrapping forward_extend produces same output."""
        torch.manual_seed(10)
        seq_lens = [16]
        ext = [16]
        layer = _make_layer(self.num_q_heads, self.num_kv_heads, self.head_dim)
        n = sum(ext)
        q = torch.randn(n, self.num_q_heads * self.head_dim, dtype=self.dtype, device=self.device)
        k = torch.randn(n, self.num_kv_heads * self.head_dim, dtype=self.dtype, device=self.device)
        v = torch.randn(n, self.num_kv_heads * self.head_dim, dtype=self.dtype, device=self.device)

        # Eager pass on a fresh pool.
        from sglang.srt.hardware_backend.tpu.kv_pool import TpuFusedMHATokenToKVPool

        def _fresh_pool():
            return TpuFusedMHATokenToKVPool(
                size=self.kv_pool_size,
                page_size=self.page_size,
                dtype=self.dtype,
                head_num=self.num_kv_heads,
                head_dim=self.head_dim,
                layer_num=self.num_layers,
                device=self.device,
                enable_memory_saver=False,
            )

        runner = SimpleNamespace(
            device=self.device,
            page_size=self.page_size,
            req_to_token_pool=self.req_to_token_pool,
        )

        # Eager
        pool_eager = _fresh_pool()
        backend_eager = self.PallasAttnBackend(runner)
        fb_eager = _make_forward_batch(
            mode="extend",
            seq_lens=seq_lens,
            extend_seq_lens=ext,
            req_pool_indices=[1],
            token_to_kv_pool=pool_eager,
            req_to_token_pool=self.req_to_token_pool,
            device=self.device,
        )
        backend_eager.init_forward_metadata(fb_eager)
        out_eager = backend_eager.forward_extend(q, k, v, layer, fb_eager)
        torch.tpu.synchronize()

        # Compiled — wrap just the inner _forward (avoid torch.compile-ing the
        # eager-only metadata construction).
        pool_compiled = _fresh_pool()
        backend_compiled = self.PallasAttnBackend(runner)
        fb_compiled = _make_forward_batch(
            mode="extend",
            seq_lens=seq_lens,
            extend_seq_lens=ext,
            req_pool_indices=[1],
            token_to_kv_pool=pool_compiled,
            req_to_token_pool=self.req_to_token_pool,
            device=self.device,
        )
        backend_compiled.init_forward_metadata(fb_compiled)

        def _run(q, k, v):
            return backend_compiled.forward_extend(q, k, v, layer, fb_compiled)

        compiled = torch.compile(_run, backend="tpu", fullgraph=True, dynamic=False)
        out_compiled = compiled(q, k, v)
        torch.tpu.synchronize()

        diff = (out_eager.cpu().float() - out_compiled.cpu().float()).abs().max().item()
        self.assertLess(diff, 1e-2, f"eager vs compiled max diff = {diff}")


class TestPallasBackendRegistry(unittest.TestCase):
    """The Qwen3 model picks its attention backend up from the global
    registry. Confirm the wiring chain ``platform default -> registry
    -> PallasAttnBackend`` so the model gets the right backend without
    any model-specific glue.
    """

    def test_pallas_is_default_on_tpu(self):
        from sglang.srt.hardware_backend.tpu.tpu_platform import TpuSRTPlatform

        platform = TpuSRTPlatform()
        self.assertEqual(platform.get_default_attention_backend(), "pallas")

    def test_registry_returns_pallas_backend(self):
        from sglang.srt.layers.attention.attention_registry import (
            ATTENTION_BACKENDS,
        )
        from sglang.srt.layers.attention.pallas_backend import PallasAttnBackend

        self.assertIn("pallas", ATTENTION_BACKENDS)
        # Stub model_runner with just enough attributes to call the
        # constructor — the factory should not introspect anything else.
        runner = SimpleNamespace(
            device="cpu",
            page_size=16,
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.zeros((1, 16), dtype=torch.int32),
            ),
        )
        instance = ATTENTION_BACKENDS["pallas"](runner)
        self.assertIsInstance(instance, PallasAttnBackend)


if __name__ == "__main__":
    unittest.main()
