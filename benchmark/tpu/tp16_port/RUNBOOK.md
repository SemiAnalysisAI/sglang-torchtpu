# RUNBOOK — MiMo-V2.5-Pro (1T) on 2× tpu7x-8, 128k context

Exact commands used to launch the server and run the 128k perf benchmarks.
See `results_mimo_pro_128k_conc2.md` for the numbers.

## 0. Prerequisites (both hosts, identical)

The environment must match on **both** TPU hosts or the cross-host collective
deadlocks / misbehaves:

| Component | Value | Notes |
|---|---|---|
| torch | `2.11.0+cpu` | ABI-pinned to the torch_tpu wheel; other versions fail to import torch_tpu |
| torch_tpu | `0.1.1.dev20260707090224` | from the private `torch-tpu-virtual-registry` |
| jax / jaxlib | **`0.10.0` / `0.10.0`** | **must be identical on both hosts**; 0.8.1 hits a Pallas MoE StableHLO verifier bug |
| libtpu | `0.0.40` | |
| sglang | editable install (`uv pip install -e python/ --no-deps --no-build-isolation`) | needed so the `tpu` platform entry point registers |
| model | `/gcs/models/MiMo-V2.5-Pro` | 1.03 TB, native block-fp8 (`XiaomiMiMo/MiMo-V2.5-Pro`) |

Verify on each host:
```bash
/cache/work/sglang-torchtpu/.venv/bin/python -c \
  "import importlib.metadata as m; print('torch',m.version('torch'),'jax',m.version('jax'),'jaxlib',m.version('jaxlib'),'libtpu',m.version('libtpu'))"
# expect: torch 2.11.0+cpu jax 0.10.0 jaxlib 0.10.0 libtpu 0.0.40
```

## 1. Launch the server (2-host, ordered)

**fp8 runtime = channelwise** (the script default; do NOT use `--tpu-fp8-runtime
blockwise` — this model's attention qkv shard is 3392, not ÷128, which the block
kernel rejects). `MEM_FRACTION_STATIC=0.7` is now the script default.

`--load-format dummy` is added **only for perf benchmarking** (random weights,
skips the 1 TB checkpoint read; outputs are meaningless). Drop it for real serving.

**Ordering matters:** start rank 0 first and let its coordinator wait at
"Initializing TPU distributed runtime", *then* start rank 1. A freshly started
rank 1 cannot join a group rank 0 already formed with a dead peer.

```bash
cd /cache/work/sglang-torchtpu/benchmark/tpu/tp16_port

# rank 0 (serves HTTP :30000) — start FIRST, wait until it prints
#   "Initializing TPU distributed runtime"
EXTRA_ARGS="--load-format dummy" ./launch_tp16_pro_128k.sh 0

# rank 1 (other host) — start AFTER rank 0 is waiting.
#   GKE tpu7x-16 slice: no arg (rank auto from TPU_WORKER_*):
EXTRA_ARGS="--load-format dummy" ./launch_tp16_pro_128k.sh
#   or explicit rank + RANK0_HOST:
#   RANK0_HOST=<host0-ip> EXTRA_ARGS="--load-format dummy" ./launch_tp16_pro_128k.sh 1
```

Real-weights serving (no dummy) — omit `EXTRA_ARGS`. If loading from `/gcs`, add
`--weight-loader-disable-mmap` to avoid GCS-FUSE mmap SIGBUS:
```bash
EXTRA_ARGS="--weight-loader-disable-mmap" ./launch_tp16_pro_128k.sh 0
```

Server is ready when the log shows `The server is fired up and ready to roll!`
and `curl -s localhost:30000/health` returns 200.

## 2. Warm up compile buckets (once per fresh server)

Compiles the extend/prefill + LP-K shape buckets so TTFT isn't inflated by
cold compiles on the first measured requests.
```bash
cd /cache/work/sglang-torchtpu/benchmark/tpu
PYTHON=/cache/work/sglang-torchtpu/.venv/bin/python \
MODEL=/gcs/models/MiMo-V2.5-Pro ./warmup.sh
```

## 3. Run the benchmark

Wrapper `run_bench_128k.sh` encodes the exact 128k workload
(ISL=130048, OSL=1024, range_ratio=1.0). Argument = concurrency.
```bash
cd /cache/work/sglang-torchtpu/benchmark/tpu/tp16_port
./run_bench_128k.sh 2      # concurrency 2  (8 prompts)
./run_bench_128k.sh 4      # concurrency 4  (16 prompts)
```

Equivalent explicit invocation (what the wrapper runs):
```bash
cd /cache/work/sglang-torchtpu/benchmark/tpu
PYTHON=/cache/work/sglang-torchtpu/.venv/bin/python \
MODEL=/gcs/models/MiMo-V2.5-Pro \
ISL=130048 OSL=1024 RANDOM_RANGE_RATIO=1.0 \
BATCH_SIZE=2 NUM_PROMPTS=8 WARMUP_REQUESTS=2 \
OUTPUT_DIR=./tp16_port/results/mimo_pro_128k_c2 \
  ./bench_serving.sh
```

Results (JSONL summary) land in `OUTPUT_DIR/c<conc>-isl130048-osl1024.jsonl`.

## Notes / gotchas found during bring-up

- **Server saturates at concurrency 2** (`max_running=2`): conc 4 gives the same
  throughput as conc 2 but ~2× the latency (extra requests queue). To raise the
  throughput ceiling, relaunch with higher `MAX_RUNNING` + larger KV budget.
- `common_env.sh` `PYTHON` points at `$SGLANG_TPU_ROOT/.venv/bin/python`.
- Blocker sequence hit during bring-up (all fixed): jax mismatch across hosts →
  `mem_fraction 0.3` negative-KV → channelwise MoE Pallas verifier bug (old jax) →
  blockwise 128-divisibility → **jax 0.10.0 + channelwise + mem 0.7** works.
