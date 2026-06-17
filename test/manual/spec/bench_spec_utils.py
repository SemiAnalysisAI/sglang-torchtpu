"""Microbenchmark: old (loop/.item()) vs new (vectorized) spec_utils kernels.

Compares wall-clock time on CPU and TPU for realistic spec-decode batch sizes.

Usage:
    # CPU only (default):
    python test/manual/spec/bench_spec_utils.py

    # TPU only:
    python test/manual/spec/bench_spec_utils.py --device tpu

    # Both:
    python test/manual/spec/bench_spec_utils.py --device cpu tpu
"""

import argparse
import time

import torch

from sglang.srt.speculative.spec_utils import (
    _assign_req_to_token_pool_torch,
    _create_extend_after_decode_spec_info_torch,
)

# ---- Old implementations (from before the branch) ----


def _create_extend_old(accept_tokens, seq_lens, accept_lens, positions, bonus_tokens):
    """Original loop-based implementation with .item() calls."""
    batch_size = seq_lens.numel()
    if batch_size == 0:
        return
    cumsum = torch.cumsum(accept_lens, dim=0)
    bonus_idx = (cumsum - 1).clamp(min=0)
    bonus_tokens.copy_(accept_tokens[bonus_idx].to(bonus_tokens.dtype))

    offset = 0
    for pid in range(batch_size):
        al = int(accept_lens[pid].item())
        sl = int(seq_lens[pid].item())
        positions[offset : offset + al] = torch.arange(
            sl - al, sl, device=positions.device, dtype=positions.dtype
        )
        offset += al


def _assign_req_old(
    req_pool_indices, req_to_token, start_offset, end_offset, out_cache_loc, batch_size
):
    """Original implementation with torch.where redirect to row 0, col 0."""
    if batch_size == 0:
        return
    lengths = end_offset - start_offset
    max_len = min(
        int(req_to_token.shape[1]),
        int(out_cache_loc.numel()) if out_cache_loc.numel() > 0 else 0,
    )
    if max_len == 0:
        return
    offsets = torch.arange(
        max_len, device=req_to_token.device, dtype=start_offset.dtype
    )
    mask = offsets[None, :] < lengths[:, None]
    src_starts = torch.cumsum(lengths, dim=0) - lengths
    src_index = (src_starts[:, None] + offsets[None, :]).clamp(
        max=out_cache_loc.numel() - 1
    )
    dst_index = start_offset[:, None] + offsets[None, :]
    row_index = req_pool_indices[:, None].expand(batch_size, max_len)
    dst_index = torch.where(mask, dst_index, torch.zeros_like(dst_index))
    row_index = torch.where(mask, row_index, torch.zeros_like(row_index))
    vals = torch.where(
        mask,
        out_cache_loc[src_index].to(req_to_token.dtype),
        req_to_token[row_index, dst_index],
    )
    req_to_token[row_index, dst_index] = vals


# ---- Benchmark harness ----


def _sync_device(device):
    """Ensure all pending ops on `device` have completed."""
    if isinstance(device, str):
        backend = device
    else:
        backend = device.type
    if backend == "tpu":
        torch.tpu.synchronize()
    elif backend == "cuda":
        torch.cuda.synchronize()
    # CPU: nothing to sync


def bench(fn, args, device, warmup=20, iters=200, label=""):
    """Time `fn(*args)` with proper device synchronisation."""
    # Warmup
    for _ in range(warmup):
        fn(*args)
        _sync_device(device)

    # Timed region
    _sync_device(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
        _sync_device(device)
    elapsed = time.perf_counter() - t0

    us_per_call = elapsed / iters * 1e6
    print(f"  {label:45s} {us_per_call:8.1f} µs/call  ({iters} iters)")
    return us_per_call


def _get_torch_device(device_str):
    """Return a torch.device, handling 'tpu' via torch_tpu."""
    if device_str == "tpu":
        import torch_tpu  # noqa: F401 — registers the 'tpu' backend

        return torch.device("tpu:0")
    return torch.device(device_str)


# ---- Per-kernel benchmarks ----


def bench_create_extend(batch_size, accept_len, device_str):
    device = _get_torch_device(device_str)
    print(
        f"\n=== _create_extend_after_decode_spec_info  "
        f"bs={batch_size}, accept={accept_len}, device={device_str} ==="
    )
    seq_lens = torch.randint(50, 200, (batch_size,), dtype=torch.int32, device=device)
    accept_lens = torch.full(
        (batch_size,), accept_len, dtype=torch.int32, device=device
    )
    total = batch_size * accept_len
    accept_tokens = torch.randint(0, 32000, (total,), dtype=torch.int32, device=device)

    pos_old = torch.zeros(total, dtype=torch.long, device=device)
    bonus_old = torch.zeros(batch_size, dtype=torch.int32, device=device)
    pos_new = torch.zeros(total, dtype=torch.long, device=device)
    bonus_new = torch.zeros(batch_size, dtype=torch.int32, device=device)

    # On TPU the old implementation's .item() calls act as implicit device
    # barriers inside the loop, which is exactly the pathology we're fixing.
    old_us = bench(
        _create_extend_old,
        (accept_tokens, seq_lens, accept_lens, pos_old, bonus_old),
        device_str,
        label="OLD (loop + .item())",
    )
    new_us = bench(
        _create_extend_after_decode_spec_info_torch,
        (accept_tokens, seq_lens, accept_lens, pos_new, bonus_new),
        device_str,
        label="NEW (vectorized)",
    )
    speedup = old_us / new_us if new_us > 0 else float("inf")
    print(f"  {'Speedup':45s} {speedup:.2f}x")
    return old_us, new_us


def bench_assign_req(batch_size, max_tokens, device_str):
    device = _get_torch_device(device_str)
    print(
        f"\n=== _assign_req_to_token_pool  "
        f"bs={batch_size}, max_tokens={max_tokens}, device={device_str} ==="
    )
    pool_len = 256
    num_rows = batch_size + 10

    # Build inputs on the target device.
    if device_str == "tpu":
        # torch.randperm not supported on XLA; build on CPU then move.
        req_pool_indices = torch.randperm(num_rows)[:batch_size].to(device)
        start_offset = torch.randint(0, pool_len - max_tokens, (batch_size,)).to(device)
        lengths = torch.randint(1, max_tokens + 1, (batch_size,))
        end_offset = (start_offset.cpu() + lengths).to(device)
        out_cache_loc = torch.randint(
            0, 10000, (int(lengths.sum()),), dtype=torch.int64
        ).to(device)
    else:
        req_pool_indices = torch.randperm(num_rows, device=device)[:batch_size]
        start_offset = torch.randint(
            0, pool_len - max_tokens, (batch_size,), device=device
        )
        lengths = torch.randint(1, max_tokens + 1, (batch_size,), device=device)
        end_offset = start_offset + lengths
        out_cache_loc = torch.randint(
            0, 10000, (int(lengths.sum()),), dtype=torch.int64, device=device
        )

    table_old = torch.full((num_rows, pool_len), -1, dtype=torch.int64, device=device)
    table_new = torch.full((num_rows, pool_len), -1, dtype=torch.int64, device=device)

    old_us = bench(
        _assign_req_old,
        (
            req_pool_indices,
            table_old,
            start_offset,
            end_offset,
            out_cache_loc,
            batch_size,
        ),
        device_str,
        label="OLD (3x torch.where + read-back)",
    )
    new_us = bench(
        _assign_req_to_token_pool_torch,
        (
            req_pool_indices,
            table_new,
            start_offset,
            end_offset,
            out_cache_loc,
            batch_size,
        ),
        device_str,
        label="NEW (col_offsets * mask)",
    )
    speedup = old_us / new_us if new_us > 0 else float("inf")
    print(f"  {'Speedup':45s} {speedup:.2f}x")
    return old_us, new_us


# ---- Main ----


def run_suite(device_str):
    print("\n" + "=" * 70)
    print(f"  Microbenchmark: old vs new spec_utils kernels  [{device_str.upper()}]")
    print("=" * 70)

    results = []

    # _create_extend_after_decode_spec_info_torch
    for bs in [1, 4, 16, 32, 64]:
        for accept in [1, 4, 6]:
            old_us, new_us = bench_create_extend(bs, accept, device_str)
            results.append(("create_extend", bs, accept, device_str, old_us, new_us))

    # _assign_req_to_token_pool_torch
    for bs in [1, 4, 16, 32, 64]:
        for mt in [4, 16, 32]:
            old_us, new_us = bench_assign_req(bs, mt, device_str)
            results.append(("assign_req", bs, mt, device_str, old_us, new_us))

    return results


def print_summary(all_results):
    print("\n" + "=" * 70)
    print("  Summary Table")
    print("=" * 70)
    header = f"  {'Kernel':35s} {'bs':>4s} {'param':>5s} {'dev':>4s} {'old µs':>8s} {'new µs':>8s} {'speedup':>8s}"
    print(header)
    print("  " + "-" * len(header.strip()))
    for kernel, bs, param, dev, old_us, new_us in all_results:
        speedup = old_us / new_us if new_us > 0 else float("inf")
        print(
            f"  {kernel:35s} {bs:4d} {param:5d} {dev:>4s} "
            f"{old_us:8.1f} {new_us:8.1f} {speedup:7.2f}x"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Microbenchmark old vs new spec_utils kernels"
    )
    parser.add_argument(
        "--device",
        nargs="+",
        default=["cpu"],
        choices=["cpu", "tpu", "cuda"],
        help="Device(s) to benchmark on (default: cpu)",
    )
    args = parser.parse_args()

    all_results = []
    for dev in args.device:
        results = run_suite(dev)
        all_results.extend(results)

    print_summary(all_results)

    print("\n" + "=" * 70)
    print("  Done")
    print("=" * 70)
