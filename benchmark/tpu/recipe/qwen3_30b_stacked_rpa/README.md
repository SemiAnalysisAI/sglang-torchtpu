# Qwen3-30B skewed long-context decode

This recipe measures the lane-packed batched-RPA scheduler on skewed decode.
The default C16 workload contains two independent
240K-token requests and fourteen 512-token requests. It preloads the long KV
prefixes into the radix cache before profiling so the trace covers steady-state
decode rather than long-prefill work.

Start the compiled TP8/EP8 FP8 server:

```bash
./benchmark/tpu/recipe/qwen3_30b_stacked_rpa/server.sh
```

The recipe enables distinct C1 and multi-request export labels for profiling.
Current main already places consecutive KV blocks across the lane slots in a
schedule row and chains online-softmax state across them, so disabling the gate
is a control for export/dispatch overhead rather than the pre-#344
request-parallel scheduler:

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

At both C1 and larger batches, current main's scheduler fills a row with
consecutive KV blocks. The kernel carries the online-softmax state across lane
slots and later rows, and resets it when a request's final KV block is reached.
This supersedes the branch's former per-lane scratch plus explicit merge row.

The older 748 versus 612 output tok/s result was measured before that mainline
scheduler rewrite and must not be treated as a current-tree A/B result. Re-run
both arms on the same current commit before publishing a performance delta.

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

With the default 600K token pool, use one 520K request rather than two.

The same skewed workload also scales to a populated C64 batch:

```bash
CONC=64 CONTEXT_LEN=262144 TPU_COMPILE_TOKENS="64 32768" \
TPU_COMPILE_BS="1 64" \
  ./benchmark/tpu/recipe/qwen3_30b_stacked_rpa/server.sh

python benchmark/tpu/recipe/qwen3_30b_stacked_rpa/bench.py \
  --concurrency 64 --long-requests 1 --long-isl 240000 \
  --context-limit 262144 --no-profile
```

Record output hashes as well as throughput because changing KV reduction order
can affect long-context outputs outside the model's native context window.
