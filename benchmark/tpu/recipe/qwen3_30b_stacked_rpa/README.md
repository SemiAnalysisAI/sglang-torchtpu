# Qwen3-30B skewed long-context decode

This recipe targets the schedule imbalance that stacked or adaptive split-K RPA
is intended to address. The default C16 workload contains two independent
240K-token requests and fourteen 512-token requests. It preloads the long KV
prefixes into the radix cache before profiling so the trace covers steady-state
decode rather than long-prefill work.

Start the compiled TP8/EP8 FP8 server:

```bash
./benchmark/tpu/recipe/qwen3_30b_stacked_rpa/server.sh
```

The recipe enables the expert stacked/adaptive planner gate. Run the matching
regular-RPA control with:

```bash
SGLANG_USE_TPU_STACKED_RPA=0 \
  ./benchmark/tpu/recipe/qwen3_30b_stacked_rpa/server.sh
```

Then run the default profiled workload:

```bash
python benchmark/tpu/recipe/qwen3_30b_stacked_rpa/bench.py
```

For a C32 skewed-decode variant, start the server with
`CONC=32 MAX_TOTAL_TOKENS=900000`, then run:

```bash
python benchmark/tpu/recipe/qwen3_30b_stacked_rpa/bench.py \
  --concurrency 32 --long-requests 3
```

At C1 the planner uses the striped static stacked schedule. At C2 and above it
keeps the regular request scheduler and splits only cached-prefix runs that are
long, skewed, and predicted to save enough schedule rows to pay for their merge
row. Uniform or short batches remain on the ordinary request-parallel schedule.

The validated Qwen3-30B C16 default produced 748 output tok/s with adaptive
split-K versus 612 output tok/s for the exact regular control, with identical
output-token hashes. A uniform C8/64K control remained flat at about 300 output
tok/s because the planner rejected splitting.

To test the 512K regime, use RPA v3 for prefill because the batched mixed-prefill
kernel currently halts while populating a 520K prefix:

```bash
CONTEXT_LEN=524288 \
EXTRA_ARGS="--prefill-attention-backend pallas_tpu_rpa_v3" \
  ./benchmark/tpu/recipe/qwen3_30b_stacked_rpa/server.sh

python benchmark/tpu/recipe/qwen3_30b_stacked_rpa/bench.py \
  --concurrency 16 --long-requests 1 --long-isl 520000 \
  --context-limit 524288 --no-profile
```

With the default 600K token pool, use one 520K request rather than two. In the
validated A/B, the adaptive gain grew from +34.6% at 240K to +66.7% at 520K;
the attention kernel was 4.64 ms adaptive versus 18.67 ms regular at 520K.

The same skewed workload also scales to a populated C64 batch:

```bash
CONC=64 CONTEXT_LEN=262144 TPU_COMPILE_TOKENS="64 32768" \
TPU_COMPILE_BS="1 64" \
  ./benchmark/tpu/recipe/qwen3_30b_stacked_rpa/server.sh

python benchmark/tpu/recipe/qwen3_30b_stacked_rpa/bench.py \
  --concurrency 64 --long-requests 1 --long-isl 240000 \
  --context-limit 262144 --no-profile
```

At C64/240K, adaptive produced 4760 output tok/s versus 3291 for regular RPA
(+44.6%), with identical output hashes. At C32/520K, adaptive produced 2112
output tok/s versus 1074 regular (+96.6%); first-token IDs matched, while full
256-token hashes differed because split-K changes reduction order beyond the
model's native context.
