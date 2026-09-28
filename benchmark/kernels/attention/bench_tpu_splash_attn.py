# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark + numerics check for the TPU Pallas splash attention kernels.

Compares the two builders in
``sglang.multimodal_gen.runtime.layers.attention.backends.tpu_splash_attn``:

* ``_build_fullmask_splash_kernel`` -- stock ``make_splash_mha``, hardcoded
  2048/2048/512 blocks, no ``CompilerParams`` (so Mosaic caps scoped VMEM at
  its 32 MiB default). Timed over the 2048-padded length as a speed baseline
  only.
* ``_build_custom_splash_kernel``   -- the DiffusionServing port: autotuned
  blocks, ``vmem_limit_bytes = 2 x physical VMEM``, exp2 online softmax.

The default shape is Wan2.2-14B T2V at 720p/81f under pure Ulysses=4:
S = 21 latent frames x 45 x 80 = 75600 tokens, 40 heads / 4 shards = 10 local
heads, head_dim 128.

Run on a single TPU chip:

    PYTHONPATH=/workspace/sglang/python python3 \
        benchmark/kernels/attention/bench_tpu_splash_attn.py
"""

from __future__ import annotations

import argparse
import time
import traceback

import torch
import torch.nn.functional as F


def _sync(tensors):
    """Force materialization + completion of lazily scheduled torch_tpu work."""
    from torch_tpu._internal.sync.sync import synchronize

    synchronize(tensors, wait=True)


def _reference_attention(q, k, v, scale, chunk=4096):
    """Chunked fp32 SDPA reference on CPU.

    Chunked over the query axis so the S x S score matrix is never
    materialized in full (75600^2 fp32 would be ~22 GB).
    """
    q32, k32, v32 = q.float(), k.float(), v.float()
    outs = []
    for start in range(0, q32.shape[2], chunk):
        q_blk = q32[:, :, start : start + chunk, :]
        scores = torch.matmul(q_blk, k32.transpose(-2, -1)) * scale
        outs.append(torch.matmul(torch.softmax(scores, dim=-1), v32))
    return torch.cat(outs, dim=2)


def _compare(name, got, want):
    got32, want32 = got.float(), want.float()
    abs_err = (got32 - want32).abs()
    denom = want32.abs().max().clamp_min(1e-6)
    print(
        f"  {name:<10} max_abs={abs_err.max().item():.3e} "
        f"mean_abs={abs_err.mean().item():.3e} "
        f"rel={(abs_err.max() / denom).item():.3e}"
    )


def _under_tpu_info(kernel):
    """Call ``kernel`` under the physical chip's Pallas metadata, as the op does."""
    from sglang.srt.hardware_backend.tpu.pallas_export import tpu_info_for_cpu_export

    def call(*args):
        with tpu_info_for_cpu_export():
            return kernel(*args)

    return call


def _time_kernel(build_fn, q, k, v, scale, warmup, iters, label, prescale=True):
    """Build, warm up, then time a kernel. Returns (seconds_per_call, output)."""
    print(f"\n=== {label} ===")
    try:
        t0 = time.perf_counter()
        kernel, q_padded, kv_padded = build_fn()
        kernel = _under_tpu_info(kernel)
        print(
            f"  build: {time.perf_counter() - t0:.2f}s (q_pad={q_padded} kv_pad={kv_padded})"
        )
    except Exception:
        print(f"  BUILD FAILED:\n{traceback.format_exc()}")
        return None, None

    q_seq, kv_seq = q.shape[2], k.shape[2]
    qs = (q * scale if prescale else q).contiguous()
    if q_padded > q_seq:
        qs = torch.nn.functional.pad(qs, (0, 0, 0, q_padded - q_seq))
    kk, vv = k, v
    if kv_padded > kv_seq:
        kk = torch.nn.functional.pad(k, (0, 0, 0, kv_padded - kv_seq))
        vv = torch.nn.functional.pad(v, (0, 0, 0, kv_padded - kv_seq))
    kk, vv = kk.contiguous(), vv.contiguous()

    try:
        t0 = time.perf_counter()
        out = kernel(qs, kk, vv)
        _sync([out])
        print(
            f"  first call (incl. Pallas/XLA compile): {time.perf_counter() - t0:.2f}s"
        )
    except Exception:
        print(f"  FIRST CALL FAILED:\n{traceback.format_exc()}")
        return None, None

    for _ in range(warmup):
        out = kernel(qs, kk, vv)
    _sync([out])

    t0 = time.perf_counter()
    for _ in range(iters):
        out = kernel(qs, kk, vv)
    _sync([out])
    per_call = (time.perf_counter() - t0) / iters

    # 4 * S * S * D FLOPs per head for QK^T + PV.
    flops = 4 * q_seq * kv_seq * q.shape[3] * q.shape[1] * q.shape[0]
    print(
        f"  steady state: {per_call * 1e3:.2f} ms/call  ({flops / per_call / 1e12:.1f} TFLOP/s)"
    )
    return per_call, out[:, :, :q_seq, :].cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=75600, help="q and kv sequence length")
    ap.add_argument(
        "--heads", type=int, default=10, help="local heads after the Ulysses all-to-all"
    )
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument(
        "--check-numerics",
        action="store_true",
        help="compare both kernels against a chunked fp32 CPU reference (slow at S=75600)",
    )
    args = ap.parse_args()

    import torch_tpu  # noqa: F401  (registers the 'tpu' PrivateUse1 device)

    from sglang.multimodal_gen.runtime.layers.attention.backends import (
        tpu_splash_attn as tsa,
    )

    print(
        f"shape: B=1 H={args.heads} S={args.seq} D={args.head_dim} bf16\n"
        f"autotuned blocks: {tsa.autotune_block_sizes(args.seq, args.seq, args.head_dim)}"
    )

    torch.manual_seed(0)
    shape = (1, args.heads, args.seq, args.head_dim)
    q_cpu = torch.randn(shape, dtype=torch.bfloat16)
    k_cpu = torch.randn(shape, dtype=torch.bfloat16)
    v_cpu = torch.randn(shape, dtype=torch.bfloat16)
    scale = args.head_dim**-0.5
    padded_seq = -(-args.seq // 2048) * 2048

    q = q_cpu.to("tpu")
    k = k_cpu.to("tpu")
    v = v_cpu.to("tpu")
    _sync([q, k, v])

    full_t, full_out = _time_kernel(
        # Built over the 2048-padded length so it needs no dense padding
        # mask; the padded keys make its output inexact, so time it only.
        lambda: tsa._build_fullmask_splash_kernel(
            padded_seq,
            padded_seq,
            args.head_dim,
            block_q=2048,
            block_kv=2048,
            is_causal=False,
        ),
        q,
        k,
        v,
        scale,
        args.warmup,
        args.iters,
        "fullmask make_splash_mha (2048/2048/512, default 32 MiB VMEM, timing only)",
    )

    custom_t, custom_out = _time_kernel(
        lambda: tsa._build_custom_splash_kernel(
            args.seq, args.seq, args.head_dim, scale
        ),
        q,
        k,
        v,
        scale,
        args.warmup,
        args.iters,
        "custom Pallas kernel (autotuned blocks, raised VMEM ceiling)",
        prescale=False,
    )

    sdpa_t, sdpa_out = _time_kernel(
        lambda: (
            lambda q_, k_, v_: F.scaled_dot_product_attention(q_, k_, v_, scale=scale),
            args.seq,
            args.seq,
        ),
        q,
        k,
        v,
        scale,
        args.warmup,
        args.iters,
        "torch SDPA (the torch_sdpa backend)",
        prescale=False,
    )

    if args.check_numerics:
        print("\n=== numerics ===")
        ref = _reference_attention(q_cpu, k_cpu, v_cpu, scale)
        if custom_out is not None:
            _compare("custom", custom_out, ref)

    print("\n=== summary ===")
    for name, t in (("fullmask", full_t), ("custom", custom_t), ("sdpa", sdpa_t)):
        speedup = f"  custom speedup {t / custom_t:.2f}x" if t and custom_t else ""
        print(f"  {name:8s} {t * 1e3:8.2f} ms{speedup}" if t else f"  {name:8s} failed")


if __name__ == "__main__":
    main()
