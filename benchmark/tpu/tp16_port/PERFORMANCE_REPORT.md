# MiMo-V2.5 DP-Attention on TPU — Performance Report

**Date:** 2026-07-11
**Status:** DP-attention SparseCore halt RESOLVED; single-host serving validated end-to-end.

---

## 1. Environment

| | |
|---|---|
| Hardware | 1× `tpu7x-8` (Ironwood, 4 chips × 2 cores = 8 devices, single host, 96 GB/core) |
| sglang-torchtpu | `main @ 11059a1d3` (2026-07-10); DP-attention fix `312cb2bf5` **present** |
| torch_tpu | `0.1.1.dev20260707090224` |
| torch | `2.11.0+cpu` |
| libtpu | `0.0.40` |
| sglang (dist) | `0.5.5` |

> The fix `312cb2bf5 "[TPU/DPA] Do not present zero-length sync-padding rows to the RPA kernel as active"` (+ related DP commits) eliminates the SparseCore `e0200 RuntimeUnexpectedCoreHalt` that previously killed any ≥512-token DP-attention prefill.

## 2. Model & serving config

| | |
|---|---|
| Model | MiMo-V2.5 (310B MoE; 48 layers, hybrid SWA+GA, 256 experts, `num_kv_heads=4`) |
| Weights | **dummy** (`--load-format dummy`) — shapes/compute/timing real, generated tokens meaningless |
| Parallelism | tp=8, ep=8, **DP-attention dp=2** (attention_tp = 8/2 = 4 = num_kv_heads) |
| Quant | FP8 e4m3 weights + FP8 e4m3 KV cache |
| torch.compile | ON (`--enable-torch-compile`) |
| Attention backend | `pallas_tpu_rpa_v3` |
| Compile cache | OFF (fresh compile each launch) |

SparseCore-offload disabled via 4 libtpu flags (all_reduce / all_gather / reduce_scatter collective offload + gather offload = false).

## 3. Throughput / Latency — 1k in / 1k out, concurrency 4

`sglang.bench_serving`, random dataset, fixed lengths (`--random-range-ratio 1.0`), 16 prompts, 4 warmup. All 16 requests succeeded; **0 SparseCore halts**, server health 200 throughout.

| Metric | Value |
|---|---|
| Successful requests | 16 / 16 |
| Benchmark duration | 60.82 s |
| Measured concurrency | 4.00 |
| Request throughput | 0.263 req/s |
| Input token throughput | 269.4 tok/s |
| **Output token throughput** | **269.4 tok/s** |
| Total token throughput | 538.8 tok/s |

**Time to First Token (TTFT)** — 1024-token prefill:

| | ms |
|---|---|
| Mean | 391.5 |
| Median | 362.3 |
| P99 | 592.9 |

**Time per Output Token (TPOT)** — excl. first token:

| | ms |
|---|---|
| Mean | 14.47 |
| Median | 14.46 |
| P99 | 14.69 |

**Inter-Token Latency (ITL)** — per decode step:

| | ms |
|---|---|
| Mean | 14.47 |
| Median | 14.25 |
| P95 | 14.90 |
| P99 | 15.36 |
| Max | 416.7 |

**End-to-End Latency** — per 1024-token request:

| | ms |
|---|---|
| Mean | 15196 |
| Median | 15210 |
| P90 | 15269 |
| P99 | 15293 |

**Observations**
- Steady-state decode is very stable: ITL p99 is only ~8% above the median.
- The 417 ms **max ITL** is the single decode step that interleaves an incoming prefill (chunked-prefill contention), not a systemic stall.
- Decode batches ran at `#running-req: 2` — the 4 concurrent requests are split **2-per-DP-replica** across `dp=2`, confirming DP-attention actually distributes the load.

## 4. Long-context capability — same server family, single request

Full MiMo, tp8/ep8/dp2, `context_len=131072`, chunked prefill (each chunk on the 1024 bucket), decode 16 tokens. **0 halts** at every length.

| Context | Prompt tokens | Prefill time | Prefill rate | Decode |
|---|---|---|---|---|
| ~32k | 36,001 | 6.6 s | 5,483 tok/s | OK |
| ~64k | 71,001 | 13.3 s | 5,334 tok/s | OK |
| ~128k | 129,001 | 27.0 s | 4,774 tok/s | OK |

KV pool for 128k tokens = only **1.2 GB** — hybrid SWA means just the ~9 global-attention layers hold the full sequence; the 39 sliding-window layers store window=128. Large headroom for longer contexts (native max 1M).

## 5. Caveats

- **Dummy weights**: throughput/latency reflect real compute and tensor shapes, but output *content* is garbage. Re-run without `--load-format dummy` on a real checkpoint for meaningful generation (and expect a ~5–10 min FP8 requant on load).
- Single-host (tp8) numbers. TP=16 across 2 hosts is scripted but not yet hardware-measured — see `launch_tp16_multihost.sh` / `README.md`.

## 6. Reproduction

Scripts in this directory (`/cache/work/tp16_port/`):
```bash
./launch_single_host.sh          # tp8/ep8/dp2 server (this report's config)
./bench.sh                       # 1k/1k, concurrency 4  (override: ISL= OSL= CONC= NUM_PROMPTS=)
```
Raw bench JSON: `scratchpad/bench_c4_1k1k.json`. See `README.md` for the TP=16 2-host recipe and version-lockstep checks.
