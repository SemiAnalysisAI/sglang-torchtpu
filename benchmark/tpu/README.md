# Benchmarking SGLang on TPU (torch_tpu backend)

Launch + serving-benchmark scripts for the **torch_tpu** SGLang backend
(`sglang.launch_server --device tpu`). Parameterized by `MODEL` (and `TP_SIZE`) —
no per-model files. (For the separate sglang-jax engine, see
[`docs/platforms/tpu.md`](../../docs/platforms/tpu.md).)

```
launch_server.sh     # start the server
bench_serving.sh     # run sglang.bench_serving - text workload
run_eval.sh          # run sglang.test.run_eval (e.g. MMLU, GSM8K)
run_lm_eval.sh       # run lm-eval against a live server and check baselines
bench_vl_serving.sh  # run sglang.bench_serving - image workload (MMMU or synthetic)
bench_vl_correctness.sh  # grade VL accuracy on MMMU via bench_vl_correctness.py
bench_vl_correctness.py  # MMMU accuracy helper
warmup.sh            # optional compile-bucket pre-warmup (stopgap)
warmup_buckets.py    # request-sender used by warmup.sh
```

## Usage

Run launch and bench in separate shells; set `MODEL` for both.

```bash
# 1. Launch — blocks while kernels compile; /health -> 200 when ready
MODEL=Qwen/Qwen3-32B TP_SIZE=8 ./launch_server.sh

# 2. Bench — defaults to bs=64, isl=1024, osl=1024
MODEL=Qwen/Qwen3-32B ./bench_serving.sh

# Concurrency/shape sweep ("concurrency:isl:osl" per leg)
MODEL=Qwen/Qwen3-32B SWEEP="16:1024:1024 64:1024:1024" ./bench_serving.sh

# 3. Eval — defaults to EVAL_NAME=mmlu, API=chat, NUM_THREADS=4
MODEL=Qwen/Qwen3-4B ./run_eval.sh
MODEL=Qwen/Qwen3-4B EVAL_NAME=gsm8k ./run_eval.sh
```

Qwen3-Coder-30B FP8 TP8/EP8 quality eval:

```bash
MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
TP_SIZE=8 EP_SIZE=8 \
PAGE_SIZE=128 CONTEXT_LEN=8192 \
CHUNKED_PREFILL_SIZE=4096 MAX_PREFILL_TOKENS=4096 \
MAX_TOTAL_TOKENS=262144 MEM_FRACTION_STATIC=0.30 \
TORCH_COMPILE_MAX_BS=64 \
./launch_server.sh

./run_lm_eval.sh
```

Single-chip smoke (tp=1 auto-pins chiplet 0):

```bash
MODEL=Qwen/Qwen3-0.6B TP_SIZE=1 PAGE_SIZE=32 ./launch_server.sh
MODEL=Qwen/Qwen3-0.6B TP_SIZE=1 ./bench_serving.sh
```

TP8/EP8 MoE throughput run:

```bash
MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
TP_SIZE=8 EP_SIZE=8 \
PAGE_SIZE=128 CONTEXT_LEN=2048 \
CHUNKED_PREFILL_SIZE=1024 MAX_PREFILL_TOKENS=1024 \
MAX_TOTAL_TOKENS=262144 MEM_FRACTION_STATIC=0.30 \
TORCH_COMPILE_MAX_BS=64 \
NUM_CONTINUOUS_DECODE_STEPS=8 \
./launch_server.sh

MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
NUM_PROMPTS=640 \
./bench_serving.sh
```

TP8/EP8 480B FP8 1k/1k c=64 run:

```bash
MODEL=Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
TP_SIZE=8 EP_SIZE=8 \
PAGE_SIZE=128 CONTEXT_LEN=2048 \
CHUNKED_PREFILL_SIZE=1024 MAX_PREFILL_TOKENS=1024 \
MAX_TOTAL_TOKENS=131072 MEM_FRACTION_STATIC=0.05 \
TORCH_COMPILE_MAX_BS=64 \
NUM_CONTINUOUS_DECODE_STEPS=4 \
./launch_server.sh

MODEL=Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
DATASET_NAME=random-ids TOKENIZE_PROMPT=1 \
BATCH_SIZE=64 NUM_PROMPTS=64 ISL=1024 OSL=1024 \
./bench_serving.sh
```

### VL image workloads

`bench_vl_serving.sh` drives image requests against a running VL model. It
defaults to MMMU prompts/images; set `DATASET_NAME=image` for synthetic fixed
resolution inputs when measuring steady-state buckets.

```bash
# 1. Launch a VL model
MODEL=Qwen/Qwen3-VL-30B-A3B-Thinking-FP8 TP_SIZE=2 \
CHUNKED_PREFILL_SIZE=2048 MAX_PREFILL_TOKENS=2048 \
TORCH_COMPILE_MAX_BS=8 MAX_RUNNING=8 MEM_FRACTION_STATIC=0.3 \
./launch_server.sh

# 2a. MMMU serving perf
MODEL=Qwen/Qwen3-VL-30B-A3B-Thinking-FP8 DATASET_NAME=mmmu \
BATCH_SIZE=4 NUM_PROMPTS=16 OSL=256 ./bench_vl_serving.sh

# 2b. Synthetic 448x448 images (784 ViT patches at merge_size=2)
MODEL=Qwen/Qwen3-VL-30B-A3B-Thinking-FP8 DATASET_NAME=image \
IMAGE_RESOLUTION=448x448 IMAGE_COUNT=1 \
SWEEP="1:256:128 4:256:128" ./bench_vl_serving.sh
```

For `DATASET_NAME=mmmu`, the `SWEEP` input-length field is ignored because the
dataset supplies text and images; it is still required so the `c:isl:osl` tuple
parses.

### VL correctness

`bench_vl_correctness.sh` sends MMMU image/question prompts to an already-running
server, parses the multiple-choice answer, and writes `summary.json` plus
per-question `details.jsonl`.

```bash
MODEL=Qwen/Qwen3-VL-30B-A3B-Thinking-FP8 NUM_QUESTIONS=10 \
MAX_NEW_TOKENS=8192 ./bench_vl_correctness.sh
```

Optional warmup — run between launch and bench *only* while the in-server
torch.compile warmup doesn't yet cover all compile-shape buckets:

```bash
MODEL=Qwen/Qwen3-32B ./warmup.sh
```

## Notes

- Needs `torch_tpu` + `sglang` importable by `PYTHON` (default `python3`). For a
  Bazel-built `torch_tpu`, set `JAX_BAZEL=<bazel-output-user-root>/<hash>/external`.
  Run offline against a local cache with `HF_HOME=... HF_HUB_OFFLINE=1`.
- Every knob is an env var — see each script's header comment for the full list.
- `bench_serving.sh` writes JSONL summaries under `./results/<model-slug>-tpu/`
  (gitignored). `bench_vl_serving.sh` writes under
  `./results/<model-slug>-vl-tpu/`; `bench_vl_correctness.sh` writes under
  `./results/<model-slug>-vl-correctness-tpu/`.
