# MiMo-V2.5-Pro — 128k context, concurrency 2 (TPU)

Benchmark run: 2026-07-12 20:22 UTC · `bench_serving.sh` (dataset=random)

## Setup

| | |
|---|---|
| Model | `XiaomiMiMo/MiMo-V2.5-Pro` (1T, native block-fp8) |
| Hardware | 2× tpu7x-8 (TP=16, DP=2, EP=16), DP-attention |
| Stack | torch 2.11.0+cpu, torch_tpu 0.1.1.dev20260707090224, jax/jaxlib 0.10.0, libtpu 0.0.40 |
| fp8 runtime | `channelwise` |
| KV cache dtype | `fp8_e4m3` |
| `mem_fraction_static` | 0.7 |
| `max_running_requests` | 2 |
| `max_total_tokens` | 140288 |
| `context_length` | 131072 |
| attention backend | `pallas_tpu_rpa_v3` |

## Workload

| | |
|---|---|
| Input length (ISL) | 130048 |
| Output length (OSL) | 1024 |
| random_range_ratio | 1.0 (every request exactly 128k context) |
| Max concurrency | **2** |
| Prompts | 8 (+2 warmup) |

## Results

| Metric | Value |
|---|---|
| Completed | 8 / 8 |
| Total input tokens | 1,040,384 |
| Total output tokens | 8,180 |
| Duration | 339.0 s |
| **Request throughput** | 0.0236 req/s |
| **Input (prefill) throughput** | **3,068.8 tok/s** |
| **Output (decode) throughput** | **24.13 tok/s** |
| Total token throughput | ~3,092.9 tok/s |
| **TTFT** mean / median / p99 | **53,258 / 60,514 / 61,930 ms** |
| **TPOT** mean / median | 20.71 / 23.24 ms |
| **ITL** mean / median | 23.65 / 22.91 ms |
| **E2E latency** mean / median | **74,412 / 85,325 ms** |
