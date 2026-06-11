# Benchmarking SGLang on TPU (torch_tpu backend)

Launch + serving-benchmark scripts for the **torch_tpu** SGLang backend
(`sglang.launch_server --device tpu`). Parameterized by `MODEL` (and `TP_SIZE`) —
no per-model files. (For the separate sglang-jax engine, see
[`docs/platforms/tpu.md`](../../docs/platforms/tpu.md).)

```
launch_server.sh     # start the server
bench_serving.sh     # run sglang.bench_serving — text workload (single run or sweep)
bench_vl_serving.sh  # run sglang.bench_serving — image (VL) workload (mmmu / synthetic)
bench_vl_correctness.sh  # grade VL accuracy on MMMU (concurrency=1) — calls bench_vl_correctness.py
bench_vl_correctness.py  # MMMU accuracy helper (reuses ../mmmu eval/parsing read-only)
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
```

Single-chip smoke (tp=1 auto-pins chiplet 0):

```bash
MODEL=Qwen/Qwen3-0.6B TP_SIZE=1 PAGE_SIZE=32 ./launch_server.sh
MODEL=Qwen/Qwen3-0.6B TP_SIZE=1 ./bench_serving.sh
```

TP8/EP8 MoE throughput run:

```bash
MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 \
TP_SIZE=8 EP_SIZE=8 \
PAGE_SIZE=128 CONTEXT_LEN=2048 \
CHUNKED_PREFILL_SIZE=1024 MAX_PREFILL_TOKENS=1024 \
MAX_TOTAL_TOKENS=262144 MEM_FRACTION_STATIC=0.30 \
TORCH_COMPILE_MAX_BS=64 \
TPU_COMPILE_TOKENS="64 128 256 512 1024" \
TPU_COMPILE_BS="1 2 4 8 16 32 64" \
SGLANG_TPU_DECODE_PROCESS_INTERVAL=8 \
./launch_server.sh

MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 \
NUM_PROMPTS=640 \
./bench_serving.sh
```

### VL (image) workloads

`bench_vl_serving.sh` drives an image workload against a VL model, exercising
the multimodal path (ViT encode + embedding scatter + LLM prefill/decode).
Defaults to the **mmmu** dataset; set `DATASET_NAME=image` for synthetic images
at a fixed resolution.

```bash
# 1. Launch a VL model
MODEL=Qwen/Qwen3-VL-2B-Instruct TP_SIZE=2 ./launch_server.sh

# 2a. MMMU (real VQA prompts/images) — default single run (concurrency=16, osl=512)
MODEL=Qwen/Qwen3-VL-2B-Instruct ./bench_vl_serving.sh

# 2b. Synthetic 448x448 images (784 ViT patches at merge_size=2), concurrency sweep
MODEL=Qwen/Qwen3-VL-2B-Instruct DATASET_NAME=image \
IMAGE_RESOLUTION=448x448 IMAGE_COUNT=1 \
SWEEP="1:256:128 8:256:128 16:256:128" ./bench_vl_serving.sh
```

For `mmmu` the SWEEP `isl` field is ignored (input comes from the dataset) but
must still be present so the `c:isl:osl` tuple parses. On TPU, image resolutions
that don't match the server's precompiled ViT patch-count buckets pay a one-time
compile cost on first hit — prefer `DATASET_NAME=image` with a resolution that
maps to a precompiled bucket for steady-state numbers.

### VL (image) correctness / accuracy

`bench_vl_serving.sh` measures *speed* but does not grade answers.
`bench_vl_correctness.sh` is its companion: it sends MMMU image+question prompts
to the **already-running** server, parses the model's multiple-choice answer, and
reports accuracy (correct / total), writing `summary.json` + per-question
`details.jsonl`. It is a thin wrapper over `bench_vl_correctness.py`, which
reuses the existing MMMU harness in `../mmmu` read-only for sample loading,
prompt/image construction, robust answer parsing (handles long *Thinking*
reasoning before the final choice), and grading.

Concurrency is hard-pinned to 1 (the current TPU server can crash on concurrent
multimodal batches, and concurrency=1 keeps grading deterministic). Image
prefills are slow, so keep `NUM_QUESTIONS` small and `REQUEST_TIMEOUT` generous.

```bash
# Launch a VL model (see above), then grade a small sample:
MODEL=Qwen/Qwen3-VL-30B-A3B-Thinking NUM_QUESTIONS=10 ./bench_vl_correctness.sh

# Thinking models reason for many tokens before the answer — raise the cap so
# the final choice is not truncated:
MODEL=Qwen/Qwen3-VL-30B-A3B-Thinking NUM_QUESTIONS=20 MAX_NEW_TOKENS=8192 \
./bench_vl_correctness.sh

# Full MMMU validation split (slow at concurrency=1):
MODEL=Qwen/Qwen3-VL-2B-Instruct NUM_QUESTIONS=0 ./bench_vl_correctness.sh
```

Key knobs: `NUM_QUESTIONS` (0 = full split), `MAX_NEW_TOKENS`, `TEMPERATURE`,
`REASONING_EFFORT` (none|high), `RESPONSE_ANSWER_REGEX`, `REQUEST_TIMEOUT`. See
the script header for the full list.

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
- `bench_serving.sh` writes JSONL summaries under `./results/<model-slug>-tpu/`;
  `bench_vl_serving.sh` writes under `./results/<model-slug>-vl-tpu/`;
  `bench_vl_correctness.sh` writes under
  `./results/<model-slug>-vl-correctness-tpu/` (gitignored).
