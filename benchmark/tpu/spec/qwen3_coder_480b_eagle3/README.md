# Qwen3-Coder EAGLE3 TPU benchmark

## Tested configuration

- Commit: `f2f128a961bf47f6e279c93dd8731061a6819735`
- Hardware: TPU v7x-8
- Target: `Qwen3-Coder-480B-A35B-Instruct-FP8`
- Draft: `SGLang-EAGLE3-Qwen3-Coder-480B-A35B-Instruct-SpecForge-EigenAI`
- Workload: random prompts, ISL 1024, OSL 1024
- Concurrency: 1, 16, and 32
- Requests per lane: concurrency multiplied by 10
- Speculative configuration: EAGLE3 `3-1-4`
- Attention backend: `pallas_tpu_rpa_v3`
- Parallelism: TP 8, EP 8
- Quantization: FP8 weights and FP8 KV cache

The baseline and EAGLE3 variants were run sequentially using the same source
revision and environment.

## Results

| Concurrency | Requests | Baseline output tok/s | EAGLE3 output tok/s | Speedup | Baseline mean TPOT | EAGLE3 mean TPOT | Accept length |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 10 | 72.585 | 138.642 | **1.91x** | 13.615 ms | 6.992 ms | 2.775 |
| 16 | 160 | 585.980 | 868.516 | **1.48x** | 25.946 ms | 17.092 ms | 2.564 |
| 32 | 320 | 943.481 | 1264.267 | **1.34x** | 32.520 ms | 23.538 ms | 2.592 |

All 10/160/320 requests completed successfully in both variants. The EAGLE3
run used the real model acceptance rate; simulated acceptance was not enabled.
There were no post-readiness recompiles, tracebacks, runtime errors, or
assertion failures.

## Reproduce

Use the tested revision:

```bash
git fetch origin
git switch --detach f2f128a961bf47f6e279c93dd8731061a6819735
```

Configure the target and draft model paths:

```bash
export MODEL=gs://pyc-sglang-models/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8
export DRAFT_MODEL=gs://pyc-sglang-models/lmsys/SGLang-EAGLE3-Qwen3-Coder-480B-A35B-Instruct-SpecForge-EigenAI
```

The environment must provide a TPU v7x-8 with eight visible TPU devices and
support direct GCS loading through Run:ai Model Streamer.

Run the non-speculative baseline first:

```bash
VARIANT=baseline \
RUN_ID=final-baseline \
RUN_LABEL=qwen3-final-sequential \
./benchmark/tpu/spec/qwen3_coder_480b_eagle3/run.sh
```

After the baseline command exits, run EAGLE3:

```bash
VARIANT=eagle3-real \
RUN_ID=final-eagle3 \
RUN_LABEL=qwen3-final-sequential \
./benchmark/tpu/spec/qwen3_coder_480b_eagle3/run.sh
```

Each command starts the server, waits through model loading and TPU
precompilation, runs the concurrency sweep, writes its results, and stops the
server. `HEALTH_TIMEOUT` defaults to 3600 seconds because the initial 480B
model compilation is long.

To change the sweep or the number of requests:

```bash
CONCUR_SWEEP="1 16 32" PROMPTS_PER_CONCURRENCY=10 \
  ./benchmark/tpu/spec/qwen3_coder_480b_eagle3/run.sh
```

## Outputs

Without an output-root override, results are written under:

```text
benchmark/tpu/spec/qwen3_coder_480b_eagle3/results/
├── baseline/final-baseline/
└── eagle3-real/final-eagle3/
```

Each run contains `server.log`, `benchmark.log`, server metadata, and one JSONL
summary plus metadata for each concurrency lane.
