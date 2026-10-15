# Kimi-K3 decode megakernel in SGLang-torchtpu (experimental)

Status: **wired into SGLang serving (`SGLANG_TPU_K3_MEGAKERNEL=1`) and validated end-to-end on a 2x2x4 TPU v7x slice (`--tp-size 32 --ep-size 8`) with the 2.7T Kimi-K3 checkpoint.** This branch
(`experimental/kimi-k3-megakernel`) ports Inferact's Kimi-K3 decode megakernel into
SGLang-torchtpu while sharing the 92-layer MXFP4 routed expert weights zero-copy with
SGLang's prefill path.

---

## 1. What the megakernel is

Source: [Inferact/tpu-megakernels](https://github.com/Inferact/tpu-megakernels) at
`4048f08` (Apache-2.0), blog post "700 TPS on Kimi K3: A Case for TPU Megakernels".
A full code analysis is in [`docs/analysis.md`](docs/analysis.md).

- **One Pallas call per decode step for all 93 layers** (`decode_megakernel.py` `stack()`),
  covering KDA, MLA, the latent MoE (896 experts, top-16), attention residuals, norms and
  every collective. Only the embedding gather and the final residual/norm/lm_head run as
  plain XLA around it.
- **Collectives are in-kernel remote DMAs** (`collectives32.py`), one 31-peer barrier per
  step, BF16 payloads, two-level (8 devices in a host, then 4 hosts) or one-shot patterns,
  overlapped with weight streaming.
- **Weights stream HBM→VMEM layer by layer** through a reused 64 MiB pool, prefetching the
  next layer during the current layer's reductions.
- **MoE layout:** 8 expert groups × 4-way channel split (112 experts per group of 4 ranks,
  a quarter of each expert per rank). Router is replicated per chip; the 3584-wide latent is
  sharded 32 ways and all-gathered while top-k runs. Gate/up can be reblocked FP8 (what the
  demo loads, 78.5 GB/device) or packed MXFP4 (51.4 GB/device); the kernel dispatches on dtype
  (`_quantized_expert_dot`).
- **Caches it owns:** MLA expanded per-head K `[rows, 24, 3, ctx, 256]` and V
  `[rows, 24, 3, ctx, 128]` bf16 (contiguous per sequence, no paging, context fixed at
  launch, 55,296 B/token/device); KDA conv bf16 `[rows, 69, 3, 3, 3, 128]` and recurrent f32
  `[rows, 69, 3, 128, 128]`. States are HBM refs aliased in place (`input_output_aliases`).
- **Limits:** batch 1–8 rows per compiled kernel (per-row positions), greedy argmax in the
  demo, no real prefill (token-by-token, or 8 tokens per call through the speculative
  "verify" mode with `sequence_rows=8`).

**Published decode numbers** (16 tpu7x chips, TP32, 1-token prompts, 1,024 outputs,
context 1,152): 249 / 392 / 515 / 865 tok/s at batch 1 / 2 / 4 / 8, i.e.
**4.02 / 5.10 / 7.77 / 9.25 ms per step**.

## 2. Measured TPU v7x Results (2.7T Kimi-K3 on `2x2x4`, 16 chips / 32 cores)

### SGLang + Megakernel Decode (`SGLANG_TPU_K3_MEGAKERNEL=1`, `--tp-size 32 --ep-size 8`, context `9344`, 2026-10-14, job `j-994d97ca-c299-4613-946f`)

| Benchmark / Check | Result | Notes |
|---|---|---|
| **HBM weight memory** | **76.32 GB / rank** (`18.42 GB` free after load) | Zero-copy shared 5D MXFP4 expert weights (`26.43 GiB` `w13` + `13.21 GiB` `w2`) |
| **Decode precompile (`bs=1, 2, 4, 8`)** | **13.5s / 13.5s / 17.5s / 24.4s** | Zero graph breaks (`graph_break=0`), zero Dynamo recompiles at serving time |
| **Validation checks (`capital`, `arith`, `explain`, `code`)** | **4/4 passed** | Coherent reasoning + exact arithmetic (`17 × 23 = 391`) |
| **Greedy generation (`8` prompts)** | **8/8 passed** (`5s` wall time) | Code, math, translation, primes, long context |
| **GSM8K (`64` 5-shot questions)** | **100.0% accuracy** (`64/64`, `0.0%` invalid) | **`147.49 tok/s`** output throughput over `44.22s` (up from `121.30 tok/s`) |
| **Decode: `bs=1`, `isl=1024`, `osl=128`** | **`4.27 ms` mean TPOT / `4.25 ms` median ITL (`234.2 tok/s`)** | TTFT: `344.25 ms`; `p90_itl_ms = 4.30 ms` (down from `9.47 ms` before copy fix) |
| **Decode: `bs=1`, `isl=8192`, `osl=128`** | **`5.29 ms` mean TPOT / `4.98 ms` median ITL (`189.0 tok/s`)** | TTFT: `1876.98 ms`; `p90_itl_ms = 5.10 ms` (**3.01× faster** vs `15.92 ms` baseline) |
| **Decode: `bs=4`, `isl=1024`, `osl=128`** | **`7.95 ms` median ITL / `7.93 ms` steady-state TPOT (`503.1 tok/s`)** | `p90_itl_ms = 8.07 ms`, `495.5 tok/s` last-gen throughput (**2.65× faster** vs `21.05 ms` baseline) |
| **`xprof` decode step (`bs=4`, `isl=1024`)** | **`7.67 ms` total XLA op time (`8.03 ms` launch period)** | `%kimi_decoder_stack`: `7.04 ms` (`91.9%`), `copy`: `46.5 µs` (`0.6%`), state updates: `72.7 µs` (`0.9%`) |

### Comparison Against SGLang Baselines & Pre-Fix Megakernel

SGLang baseline (real 2.7T weights, `2x2x4` tpu7x slice, fused `o_norm`, attention-residual kernel, guard skip enabled):

| Workload | Baseline SGLang | Megakernel (pre-copy fix, `mk_e2e5`) | Megakernel (`return_state_updates=True`, `mk_e2e6`) | Speedup vs Baseline |
|---|---|---|---|---|
| `bs=1`, `1,024`-token prompt (`osl=128`) | `16.84 ms` | `9.47 ms` (`105.6 tok/s`) | **`4.27 ms` TPOT / `4.25 ms` ITL (`234.2 tok/s`)** | **3.94× (`-74.6%` TPOT)** |
| `bs=1`, `8,192`-token prompt (`osl=128`) | `15.92 ms` | `10.44 ms` (`95.8 tok/s`) | **`5.29 ms` TPOT / `4.98 ms` ITL (`189.0 tok/s`)** | **3.01× (`-66.8%` TPOT)** |
| `bs=4`, `1,024`-token prompts (`osl=128`) | `21.05 ms` (median) | `13.98 ms` median ITL (`286.1 tok/s`) | **`7.95 ms` median ITL (`503.1 tok/s`)** | **2.65× (`-62.2%` ITL)** |

## 3. Verified Design & Architecture

1. **The whole decode step exports to one TPU custom call + in-place tile updates under SGLang's JAX 0.10.2**
   (`stack(..., return_state_updates=True)` takes 43 inputs with `input_output_aliases={}` and
   returns 6 compact outputs: final hidden state `[B, 7168]`, attention-residual state `[B, 8, 7168]`,
   updated KDA conv `[B, 69, 3, 3, 3, 128]`, updated KDA recurrent `[B, 69, 3, 128, 128]`,
   and updated 128-token MLA K/V tiles `[B, 24, 3, 128, 256/128]`; outside `stack()`,
   `decode_local` applies in-place `jax.lax.dynamic_update_slice` into the donated `[8, ...]`
   state buffers so `TpuCopyInsertion::AddSpecialCaseCopies` inserts zero full-buffer HBM copies).
2. **It registers and shape-propagates through SGLang's bridge** (`sglang_tpu/pallas_jax_op.py`
   `cpu_export_jax_op`) for `ROW_BUCKETS = (1, 2, 4, 8)` and context up to `9,344`.
3. **Zero-copy 5D routed expert weight sharing**:
   - SGLang loads Kimi-K3 at `--tp-size 32 --ep-size 8` (`moe_tp_size = 4`).
   - `stack_sglang_moe_experts_in_place` stacks all 92 MoE layers' `w13_weight` and `w2_weight`
     into single 5D `uint8` base tensors (`[1, 92, 112, 1792, 1536]` = `26.43 GiB` and
     `[1, 92, 112, 384, 3584]` = `13.21 GiB`, `_base is None`), freeing the per-layer 3D
     storage immediately as each layer is copied.
   - Prefill (`gmm_v2`) indexes `moe_layer_idx` in-kernel via static `pl.BlockSpec` index_map
     closures on the 5D base tensor; decode (`decode_megakernel.py`) bitcasts the `uint8`
     VMEM ref to `uint32` in-kernel via `pltpu.bitcast` (`_bitcast_u8_expert_ref_to_u32`).
   - F32 power-of-two scales convert bit-exactly to `uint8` E8M0 (`2.64 GiB`).
4. **Static-shape prefill-to-decode state handoff (`handoff.py`)**:
   - **MLA handoff:** expanded K/V = SGLang's cached latent row `@ m_kb[layer]` (`[512, 768]`,
     extracted from `kv_b_proj` before weight release); `key = [E[h, 0:128] | latent[512:576] | 0^64]`,
     `value = E[h, 128:256]`, masked to zero past `cached_len`.
   - **KDA handoff:** `conv` = MambaPool conv row `[3, 256, 3]` transposed to `[3, 3, 256]`,
     reshaped to `[3, 3, 3, 128]` and cast `f32 -> bf16`; `recurrent` transposed `[3, 128, 128]`
     to `[3, 128, 128]` (`k, v`).
   - **Row compaction:** `SlotTable` keeps active decode requests dense in rows `[0, B)` and
     emits `RowMove(source, destination)` applied on device via `apply_row_move`. Inactive
     padded rows in a bucket (`row >= num_active`) pass `position = -1`, which suppresses
     expert routing (`safe_selected = jnp.where(valid_rows[:, None], selected, 896)`),
     KV cache scatter writes, and KDA state updates inside the megakernel.

## 4. Design Summary

| Design | Decode Latency (`bs=1 1k` / `bs=1 8k` / `bs=4 1k`) | Verdict |
|---|---|---|
| A. Megakernel-only engine | ~4.0 / ~5.0 / ~7.8 ms (standalone demo) | Rejected for serving: 8k TTFT ~9–12 s, ≤8 concurrent requests, no fallback, no multimodal |
| B. SGLang prefill + full megakernel decode **over SGLang's paged caches** | ~4.7 / ~5.5 / ~9.0 ms (est.) | Rejected: ~10 weeks of kernel surgery verifiable only on TPU |
| **Hybrid A+B (Implemented)** | **`4.27` / `5.29` / `7.95` ms (measured in SGLang!)** | **Matches standalone megakernel decode speed while keeping SGLang chunked/multimodal prefill + zero-copy shared 5D experts** |

## 5. Code in this branch

| Path | What |
|---|---|
| `sglang_tpu/kimi_k3_megakernel/_vendor/` | Upstream `collectives32.py`, `pool_alias.py`, `kimi/__init__.py`, `kimi/decode_megakernel.py`, `kimi/load.py`, Apache-2.0 headers. `SGLANG-PATCH` changes: package-relative imports; rows-subset (`[0, B)` of `R >= B`-row states); `_bitcast_u8_expert_ref_to_u32` for shared 5D `uint8` MXFP4 expert weights; `position < 0` padding row masking for MoE routing, MLA KV cache scatter, and KDA state updates; optional `rank` parameter in `decode_local` to avoid top-level `stablehlo.partition_id` |
| `sglang_tpu/kimi_k3_megakernel/program.py` | Flat JAX decode step for the torch bridge: `shard_map` over `decode_local(rank=0)`, states `[R, ...]` donated and aliased to outputs 1–4, logits all-gathered to `[B, vocab]` |
| `sglang_tpu/kimi_k3_megakernel/slots.py` | Dense row table: request → megakernel row, compaction moves on release, `row_bucket()` |
| `sglang_tpu/kimi_k3_megakernel/weights.py` | In-place 92-layer 5D MXFP4 expert stacker (`stack_sglang_moe_experts_in_place`), dense-family checkpoint loader (`load_dense_families`), exact `f32 -> E8M0` scale conversion |
| `sglang_tpu/kimi_k3_megakernel/handoff.py` | On-device KDA + MLA prefill-to-decode state handoff (`handoff_request_to_row`) and row compaction (`apply_row_move`) |
| `sglang_tpu/kimi_k3_megakernel/config.py` | Launch contract (`validate_megakernel_server_args`, run from `TpuSRTPlatform.apply_server_args_defaults`), `megakernel_context`, `megakernel_max_rows` |
| `sglang_tpu/kimi_k3_megakernel/runner.py` | Custom op registration (`register_megakernel_decode_op`), `KimiK3MegakernelRunner` (slot lifecycle, state buffers, per-rank vocab offset, `forward_decode`, oversize/over-context decode guards), and `maybe_attach_decode_megakernel` (the single call core makes after weight load) |
| `test/registered/unit/hardware_backend/tpu/test_kimi_k3_megakernel_slots.py` | Slot table tests incl. a 5,000-step churn simulation |
| `test/registered/unit/hardware_backend/tpu/test_kimi_k3_megakernel_handoff_and_runner.py` | Unit tests for 5D expert stacking, E8M0 scale roundtrip, KDA/MLA handoff, row compaction, runner lifecycle, non-zero `tp_rank` vocabulary offset, and `return_state_updates=True` parity |
| `test/registered/unit/hardware_backend/tpu/test_kimi_k3_megakernel_config.py` | Launch contract and its platform wiring, runtime decode guards, host-only generation read, env registry, compile-cache key |
| `benchmark/tpu/kimi_k3_megakernel/` | Bridge spike, CPU dry runs, 4-host GKE JobSet harness (`run_variants_v5.sh`, `client_v5.sh`), and xprof breakdown tools |

## 6. Launch contract and configuration

Decode state lives only in the megakernel's buffers: SGLang's paged MLA/Mamba
pools are read once at the prefill->decode handoff and never updated again. The
server therefore refuses, at argument resolution and before the model loads,
anything that would read those pools after the handoff:

| Requirement | Enforced as |
|---|---|
| `--tp-size 32 --ep-size 8` | error |
| `--max-running-requests <= SGLANG_TPU_K3_MEGAKERNEL_MAX_ROWS` | error if larger, declared if unset |
| Radix cache off | declared (`disable_radix_cache=True`) with a warning |
| `--context-length` fits the megakernel context | error if larger; declared from `SGLANG_TPU_K3_MEGAKERNEL_CONTEXT` if unset; error if neither is set |
| No speculative decoding, PP, DP / DP attention, P/D disaggregation, hierarchical cache | error |
| No `--enable-mixed-chunk` (decode rows would run outside the megakernel) | error |

At runtime a real decode batch above the row count, or a position past the
state context, raises instead of falling back to SGLang's decode.

Environment variables (declared in `sglang_tpu/envs.py`):

| Variable | Default | Meaning |
|---|---|---|
| `SGLANG_TPU_K3_MEGAKERNEL` | `false` | Run decode through the megakernel |
| `SGLANG_TPU_K3_MEGAKERNEL_CHECKPOINT` | served model path | Where the dense families load from |
| `SGLANG_TPU_K3_MEGAKERNEL_CONTEXT` | `--context-length` | Decode-state tokens per row (rounded up to 128, min 384) |
| `SGLANG_TPU_K3_MEGAKERNEL_MAX_ROWS` | `8` | Decode rows (1-8); caps `--max-running-requests` |
| `SGLANG_TPU_K3_MEGAKERNEL_RECIPE` | `tuned` | Kernel tuning recipe |

The megakernel sources are part of the torch.compile cache key
(`sglang_tpu/compile_cache.py`), so editing them invalidates cached executables.

## 7. How to run the CPU checks

```bash
# Vendored export (no TPU): one tpu_custom_call, 43 in / 5 out
python -B benchmark/tpu/kimi_k3_megakernel/dryrun/export_flat_cpu.py 1 1152

# Unit tests (slots, 5D weight stacking, handoff, runner lifecycle, launch contract)
PYTHONPATH=python:. pytest test/registered/unit/hardware_backend/tpu/test_kimi_k3_megakernel_slots.py \
  test/registered/unit/hardware_backend/tpu/test_kimi_k3_megakernel_handoff_and_runner.py \
  test/registered/unit/hardware_backend/tpu/test_kimi_k3_megakernel_config.py
```

## 8. Sharp bits learned

- **No `stablehlo.partition_id` outside `pallas_call`**: `torch_tpu`'s `InjectCustomKernel` runs
  `PartitionGlobalToLocal`, unwrapping `sdy.manual_computation` into the main TorchTPU module,
  and `PjRtClient::CompileAndLoad` runs XLA's `SpmdPartitioner` a second time. Any
  `jax.lax.axis_index("tp")` outside `pl.pallas_call` (such as `decode_local`'s embedding shard
  offset `tokens - rank * vocabulary`) emits `stablehlo.partition_id`, which crashes XLA's
  second `SpmdPartitioner` pass. Inside `pl.pallas_call`, Mosaic lowers `axis_index` to
  `tpu.device_id` inside the custom-call payload, which XLA ignores. Pre-subtracting a 1D
  `int32` device tensor `_vocab_offset_dev = [tp_rank * (vocab_size // 32)]` in PyTorch and
  passing `rank=0` to `decode_local` eliminates `stablehlo.partition_id` while keeping the
  `torch.compile` FX graph identical across all 32 ranks.
- **Preserving `m_kb` before `kv_b_proj` release**: On TPU MLA, `KimiK3MLAAttention` splits
  `kv_b_proj` into `w_kc` and `w_vc` and deletes `kv_b_proj.weight` during
  `process_weights_after_loading` to save HBM. We save a contiguous `[512, 768]` (`bf16`)
  `m_kb` projection (`6 KiB/layer`, `9 MiB` total across 24 MLA layers) before `kv_b_proj`
  is freed so prefill-to-decode handoff can expand compressed KV latents without `kv_b_proj`.
- **Bridge annotations & `uint32` handling**: The bridge requires explicit `jax.Array`-annotated
  parameters and a precise tuple return annotation (`tuple[jax.Array, ...]`). `uint32` weight
  families are stored as `int32` in PyTorch and bitcast via `lax.bitcast_convert_type(..., jnp.uint32)`
  inside the JAX wrapper, while shared 5D `uint8` MXFP4 expert weights are bitcast in-kernel
  via `pltpu.bitcast(..., jnp.uint32)` on the VMEM ref.
- **Zero views into compiled graphs**: Views passed into a compiled graph are re-materialized
  per call on `torch_tpu`; every kernel operand must be a storage-owning base tensor
  (`t._base is None`).
- **Avoiding `TpuCopyInsertion::AddSpecialCaseCopies` on communicating custom-calls (`return_state_updates=True`)**:
  In `libtpu.so`, `OutputHasToBeConstant` returns `true` for every output of any `tpu_custom_call`
  with `has_communication=true` (`collective_id=21`). If `stack()` aliases its `[8, ...]` state
  inputs to its `[8, ...]` state outputs and returns them directly in `ROOT %tuple`,
  `CollectParamCopiesForRecv` and `CollectRootCopyForRecv` insert `2 × 3.85 GiB = 7.70 GiB`
  (`5.22 ms/step`) of full-buffer HBM copies before and after `%kimi_decoder_stack`. Returning
  only the compact updated `[B, ...]` KDA state rows and `[B, 24, 3, 128, 256/128]` MLA cache
  tiles from `stack(..., return_state_updates=True)` with `input_output_aliases={}` and applying
  in-place `jax.lax.dynamic_update_slice` on the donated state buffers outside the custom-call
  (split into two halves when `B == max_rows` so `HloAlgebraicSimplifier` does not fold
  `dynamic_update_slice` back into `update`) keeps the live-out `HloBuffer`s owned by
  `kDynamicUpdateSlice` and eliminates all `7.70 GiB` of full-buffer state copies.
