# Kimi-K3 decode megakernel (experimental)

`SGLANG_TPU_ENABLE_K3_MEGAKERNEL=1` runs every Kimi-K3 decode step as one Pallas
call covering all 93 layers (KDA, MLA, the latent MoE, attention residuals,
norms and every TP32 collective), using Inferact's megakernel
([`_vendor/`](_vendor/VENDORED.md)). Prefill, chunked prefill, multimodal input
and sampling stay on SGLang's normal path.

Experimental: no registered end-to-end test covers it yet.

## Enabling it

```bash
SGLANG_TPU_ENABLE_K3_MEGAKERNEL=1 \
python -m sglang.launch_server --model-path <Kimi-K3 checkpoint dir> \
  --tp-size 32 --ep-size 8 --enable-torch-compile --disable-radix-cache \
  --context-length 9344 --page-size 128
```

Startup fills in `--max-running-requests 8` (the megakernel's rows),
`--tpu-compile-bs 1 2 4 8`, `--attention-backend pallas_tpu_mla` and
`--kv-cache-dtype bfloat16` when they are unset.

A complete 2x2x4 tpu7x config (4 hosts, real weights) is
[`benchmark/tpu/recipe/configs/kimi_k3/megakernel.py`](../../benchmark/tpu/recipe/configs/kimi_k3/megakernel.py);
`MK=0` serves the same config without the megakernel for A/B. On that slice the
weights plus the megakernel leave no KV room below `--mem-fraction-static` 0.81,
so it uses 0.87.

One tuning knob: `SGLANG_TPU_OPT_K3_MEGAKERNEL_TUNED_RECIPE=0` runs upstream's
plain kernel options instead of the tuned ones (A/B and triage).

## Supported and refused configurations

Required: a Kimi-K3 checkpoint (`KimiK3ForConditionalGeneration` or
`KimiK3LinearForCausalLM`) with MXFP4 routed experts, read from local
safetensors; `--tp-size 32 --ep-size 8`; `--enable-torch-compile` with the
dynamo tracer; `--context-length`; `--max-running-requests` in {1, 2, 4, 8}
(the state rows) with `--tpu-compile-bs` equal to the tiles up to it.

Refused at startup, each with an error naming the flag, because the result
would be silently wrong or the layout differs: `--enable-mixed-chunk`,
`--speculative-algorithm`, decode or attention context parallelism, pipeline
parallelism, `--dp-size`, attention DP, PD disaggregation, the hierarchical
cache, `--enable-pdmux`, an fp8 KV cache, LoRA, `--enable-return-hidden-states`,
`--enable-return-routed-experts`, `--enable-deterministic-inference`, EPLB and
redundant experts, `--load-format` other than auto / safetensors,
`--tpu-compile-tracer direct`, and attention backends other than
`pallas_tpu_mla`. The radix cache must be off (`--disable-radix-cache`):
decoded tokens never reach SGLang's pools, so there is nothing to share. Online
weight updates are already refused for Kimi-K3 on TPU.

At runtime a decode batch past the rows, a position past the context, a padded
batch with no compiled tile, a MIXED batch, or a decoding request appearing in
a non-decode batch raises instead of falling back to SGLang's decode.

Sampling parameters, penalties, grammars, stop strings, aborts and output
logprobs work: the sampler runs on full-vocabulary logits. The vocabulary-sharded
distributed top-k is turned off for the model, so prefill samples from full
logits too.

## How it works

**Handoff.** The megakernel keeps four per-row state arrays: KDA conv
`[rows, 69, 3, 3, 3, 128]` bf16 and recurrent `[rows, 69, 3, 128, 128]` f32,
and expanded per-head MLA caches `[rows, 24, 3, context, 256 | 128]` bf16 (key
`k_nope | rope | 0`, value). When a request first appears in a decode batch it
is given a row, and its prefill state is copied in once (`handoff.py`): the KDA
state from its `TpuMambaPool` slot, and the MLA caches by expanding its latent
rows from the `TpuFusedMLATokenToKVPool` through `m_kb`. From then on only the
megakernel reads or writes that request's state; SGLang's pools are never
updated, which is why everything that would read them is refused.

**Rows.** Live requests always occupy rows `[0, n)`: when one leaves, the
highest row moves into the hole. A request is keyed by
`(req_pool_idx, req_generation)`; the generation changes whenever the slot is
reallocated, so a finished, aborted or retracted request is released at the
next decode batch it is missing from, and a retracted request re-prefills
through SGLang and is handed off again.

**Per step.** The attach re-annotates the language model with a subclass of
its compile policy (`policy.py`). After the base policy pads the batch to a
tile, the runner releases and hands off rows and stages the step's row
permutation and positions (padding rows get position -1, which the kernel
masks out of expert routing). The compiled forward runs the matching tile's
custom op for DECODE and the model's own forward for every other mode.

**Overlap scheduling.** The staged permutation and positions are fresh device
tensors every step, not persistent buffers, so staging step N+1 cannot
overwrite inputs step N still reads. The state arrays are donated to the
compiled step and rebound to its outputs, and the next step's handoff and row
moves are enqueued after it on the same PJRT queue, so they read its results.

**Weights.** The routed experts are restacked once into
`[1, 92, 112, ...]` uint8 tensors that both the megakernel and SGLang's prefill
MoE read (`fused_moe_call(w_layer_index=...)`), and their f32 block scales are
converted back to E8M0 bytes exactly. The dense families (attention, norms,
router, latent and shared experts, embedding, lm_head) are read again from the
checkpoint in the layout the kernel was validated with.

**Attach.** `sglang_tpu/hooks/kimi_k3_megakernel.py` runs it once per rank
after the loader's quant post-processing and before the KV pools are sized, so
the megakernel's memory counts as model memory.

## Memory cost (per rank)

| What | Size |
|---|---|
| MLA state | 55,296 B per token per row: 4.1 GB at 8 rows x 9,344 tokens |
| KDA state | 14.0 MB per row |
| Dense families held twice (SGLang's and the megakernel's) | about 5.2 GiB |
| Expert scales held twice (f32 for prefill, E8M0 for the kernel) | 10.6 + 2.6 GiB |
| Host staging while restacking the experts at load | about 40 GiB (x ranks per host) |

SGLang's own pools are also held twice in practice: a request keeps its prefill
KV pages and Mamba slot for its whole life as handoff sources, and the
allocator still reserves a KV slot for every decoded token, which nothing
writes. Size the KV pool for prefill, and expect the scheduler to retract
earlier than the megakernel's rows would require.

The context ceiling is hard: every row holds the full `--context-length`.

## Tests

- `test/registered/unit/hardware_backend/tpu/test_kimi_k3_megakernel.py`
  (CPU): startup refusals, row bookkeeping, the handoff against the real TPU
  pool classes, expert restacking, the policy step and the attach hook.
- `test/registered/tpu/unit/test_kimi_k3_megakernel_program.py` (stage-a):
  the decode program exports to one TPU custom call with no
  `stablehlo.partition_id`; upstream's component checks run under SGLang's JAX.

## Follow-ups

- **Pool ownership.** Expose the megakernel's layout as SGLang pool classes
  (`get_mla_kv_pool_cls` / `get_mamba_pool_cls`) with rows keyed by request
  slot, so prefill writes the expanded state directly, the handoff and the
  double-held pools disappear, and scheduler accounting is exact.
- **Paged context.** Unpaged full-context rows cap context and concurrency; a
  paged MLA variant would lift both.
- **PD decode role.** KV arrives once on a decode node, so the handoff fits
  there naturally; together with DSpark verification (upstream `kimi/dspark.py`)
  that is the production shape.
- Share the dense weights with SGLang's, let prefill read E8M0 scales, bucket
  the handoff by cached length, file the post-load platform seam that replaces
  the hook, and add a registered multihost end-to-end test.
