"""Isolated fp4-vs-bf16 fused-MoE microbench for the TPU NVFP4 path.

Reproduces the reviewer microbench (hidden = intermediate = 4096, top-8) that
flagged the FP4 MoE as 5-8x slower than bf16, and reports the fp4/bf16 latency
ratio per token count so each perf lever (RHS-scale pre-scale / deep matmul,
fused-act GMM1, packed-w2 tuned tiling, narrow scale dtype, block-512
coarsening) can be measured before/after.

Requires torch_tpu + a TPU device. Run on-pod:

    SGLANG_TPU_NVFP4_SCALE_DTYPE=fp32 \\
    python test/manual/tpu/bench_nvfp4_moe.py --block-size 16

The centerpiece A/B (yaochengji's MXU-underutilization fix) toggles the
pre-scale deep-matmul path with an env var (default on); run both:

    SGLANG_TPU_NVFP4_PRESCALE=0 python .../bench_nvfp4_moe.py  # per-16-block baseline
    SGLANG_TPU_NVFP4_PRESCALE=1 python .../bench_nvfp4_moe.py  # single deep matmul

Flags let the same harness cover each lever:
  --block-size {16,512}   per-block scale granularity (512 = coarsened ckpt)
  --scale-dtype {fp32,bf16,fp8}   folded per-block scale storage dtype
  --tokens 8,64,512       routed token counts to sweep
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

# Reuse the on-device parity test's NVFP4 synthetic-layer helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_nvfp4_native_moe import (  # noqa: E402
    _build_fake_layer,
    _dequant_rows_nvfp4,
    _has_tpu_device,
    _make_nvfp4_moe_layer,
)

_SCALE_DTYPES = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp8": torch.float8_e4m3fn,
}


def _inputs(m, hidden, topk, E, seed):
    g = torch.Generator().manual_seed(seed)
    hs = (torch.randn(m, hidden, generator=g) * 0.5).to(torch.bfloat16)
    topk_ids = torch.randint(0, E, (m, topk), generator=g, dtype=torch.int32)
    topk_w = torch.softmax(torch.randn(m, topk, generator=g).float(), dim=-1).to(
        torch.bfloat16
    )
    return hs, topk_ids, topk_w


def _time_call(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.tpu.synchronize()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.tpu.synchronize()
        samples.append((time.perf_counter() - t0) * 1e3)  # ms
    return statistics.median(samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--intermediate", type=int, default=4096)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--scale-dtype", choices=list(_SCALE_DTYPES), default="fp32")
    ap.add_argument("--tokens", default="8,64,512")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    if not _has_tpu_device():
        print("SKIP: requires torch_tpu + a TPU device")
        return

    from sglang.srt.hardware_backend.tpu.fp4_moe import prepare_tpu_nvfp4_moe_weights
    from sglang.srt.layers.moe.pallas.fused_moe_op import fused_moe_call

    E, H, I, topk = args.experts, args.hidden, args.intermediate, args.topk
    scale_dtype = _SCALE_DTYPES[args.scale_dtype]
    tokens = [int(t) for t in args.tokens.split(",")]

    # Build one synthetic NVFP4 layer; derive a bf16 layer from the SAME codes so
    # the only difference is the fp4-packed vs bf16-dequant weight path.
    data = _make_nvfp4_moe_layer(E, H, I, seed=3, block_size=args.block_size)
    gg, gu, g2 = data["globals"]
    w1 = _dequant_rows_nvfp4(
        data["w13_weight"][:, :I], data["w13_weight_scale"][:, :I], gg
    )
    w3 = _dequant_rows_nvfp4(
        data["w13_weight"][:, I:], data["w13_weight_scale"][:, I:], gu
    )
    w13_bf16 = torch.cat([w1, w3], dim=1).permute(0, 2, 1).contiguous().to("tpu")
    w2_bf16 = (
        _dequant_rows_nvfp4(data["w2_weight"], data["w2_weight_scale"], g2)
        .permute(0, 2, 1)
        .contiguous()
        .to("tpu")
    )

    layer = _build_fake_layer(data)
    prepare_tpu_nvfp4_moe_weights(
        layer, group_size=args.block_size, scale_dtype=scale_dtype
    )
    w13_fp4 = layer.w13_weight.to("tpu")
    w2_fp4 = layer.w2_weight.to("tpu")
    w13_s = layer.w13_weight_scale.to("tpu")
    w2_s = layer.w2_weight_scale.to("tpu")

    print(
        f"NVFP4 MoE microbench: hidden={H} inter={I} E={E} topk={topk} "
        f"block_size={args.block_size} scale_dtype={args.scale_dtype}"
    )
    print(f"{'m':>6} {'bf16 ms':>10} {'fp4 ms':>10} {'fp4/bf16':>10}")
    for m in tokens:
        hs, topk_ids, topk_w = _inputs(m, H, topk, E, seed=4)
        hs, topk_ids, topk_w = hs.to("tpu"), topk_ids.to("tpu"), topk_w.to("tpu")

        def bf16():
            return fused_moe_call(
                hidden_states=hs,
                w1=w13_bf16,
                w2=w2_bf16,
                topk_weights=topk_w,
                topk_ids=topk_ids,
                topk=topk,
                activation="silu",
            )

        def fp4():
            return fused_moe_call(
                hidden_states=hs,
                w1=w13_fp4,
                w2=w2_fp4,
                w1_scale=w13_s,
                w2_scale=w2_s,
                topk_weights=topk_w,
                topk_ids=topk_ids,
                topk=topk,
                activation="silu",
            )

        bf16_ms = _time_call(bf16, args.iters, args.warmup)
        fp4_ms = _time_call(fp4, args.iters, args.warmup)
        print(f"{m:>6} {bf16_ms:>10.3f} {fp4_ms:>10.3f} {fp4_ms / bf16_ms:>10.2f}")


if __name__ == "__main__":
    main()
