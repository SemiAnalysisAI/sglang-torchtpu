# GPT-OSS-120B TP8 decode megakernel

Correctness is validated in Pallas TPU interpret mode on 8 virtual CPU devices and on TPU v7x
(`tpu7x:2x2x1`). Compilation is checked by ahead-of-time compilation for `tpu7x:2x2x1` on a CPU
host (`70/70` configurations) and by standalone and end-to-end serving benchmarks on TPU v7x.

## What one call does

`make_step(c, mesh)` returns a jittable decode step. It runs under `shard_map` over a 1-D
`"tp"` mesh of `R=8` ranks:

1. `gptoss_decode_mega` is **one `pallas_call` per step** that runs all `L=36` layers. The layers
   go through `fori` over `(sliding, full)` layer pairs; layer `l` is sliding iff `l % 2 == 0`
   (GPT-OSS `layer_types`). Per layer:
   - **A.** RMSNorm, then QKV (`+bias`), then YaRN NeoX RoPE. `q` goes to per-sequence
     `[8, 128]` tiles. The new K/V are clamped (to `+-448` for `fp8`) and rounded to the KV dtype.
   - **B.** Attention over the paged KV pool, read **in place**, run as one flat stream of
     `(sequence, page-block)` work items. The scalar core first lists the items. Page DMAs
     then run `kv_bufs - 1` (default `3`) items ahead of compute, across sequence boundaries.
     All pages of an item share one semaphore, and only pages holding attended tokens are fetched.
     A full layer uses `npb=16` pages (`1024` tokens) per item by default. A sliding layer uses
     one item per sequence with `<= 3` window pages. The current token's K/V come from VMEM.
     `fp8_e4m3fn` KV pages feed the TPU v7x MXU directly without a `bf16` conversion. Includes
     sinks, the 128-token window (SGLang `window_left=127`), and online softmax carried
     across a sequence's items.
   - **C.** `o_proj`, then the **in-kernel all-reduce**, then `+ bo + residual` (rounded to `bf16`).
   - **D.** RMSNorm, then router (`+bias`), then `top-4` over 128 with softmax over the `top-4`
     logits. The active-expert mask goes `VMEM -> SMEM` by DMA, and the scalar core compacts
     it into a list.
   - **E.** For each active local expert chunk: `gate|up` (`+b13`), `swigluoai` (clamp `7`,
     `alpha=1.702`), `down` (`+b2`, added once across ranks), weighted by the router weight. Then
     the **in-kernel all-reduce** (whose completion overlaps with listing layer `l+1`'s attention
     items and starting its first KV page DMAs), then `+ residual`.
2. `gptoss_decode_kv_scatter` is a non-communicating `pallas_call` that writes each
   layer's new K/V token into the pools in place. It aliases the pools and performs one
   `HBM -> HBM` DMA per `(layer, sequence)`.

Embedding, final norm, `lm_head`, and sampling stay outside the kernel. Input is the embedded
token `[B, H]`; output is the last layer's residual stream `[B, H]`.

## Design choices (and why)

| Topic | Choice | Notes |
|---|---|---|
| Attention sharding | 8 Q heads + 1 KV head per rank (`64q / 8kv`) | Matches stock attention TP8 |
| MoE sharding | Default `moe_mode="tp"`: every rank holds an `I/8` slice (`360`, padded to `384`) of all 128 experts. `"ep"` (16 whole experts/rank, `I` padded to `3072`) is also implemented and tested | Stock uses EP8. TP-split is load-balanced at small `B`: at `B=1` every rank streams `4 x 1/8` experts. Under EP, one rank may own `2-4` of the 4 routed experts. At `B >= 32` the streamed bytes are comparable. |
| Batch | `B` static per compile (buckets `1, 2, 4, 8, 16, 32, 64`), padded to `BR = rup(B, 16)` activation rows | Rows `>= B` are masked out of routing and attention |
| Hidden padding | `H=2880 -> Hp=2944` (`rup(H, 128)`) with zero weights/biases; RMSNorm divides by true `H` | `2944 = 23 * 128` suffices for lane tiles without padding to `3072`. |
| All-reduce (`ar_mode="auto"`) | For `B <= 8`, one-shot push over remote DMA: each rank DMAs its partial into its own slot of all 7 peers' receive buffers, waits for the 7 arrivals, and sums the 8 slots in rank order in `f32`. For `B <= 4` (`ar_fold=True`), only the live rows travel, folded as `[B*23, 128]` `f32` (`B=1`: `12 KB/peer` instead of `94 KB`). For `B >= 16`, two-round reduce-scatter + all-gather (`rsag`): round 1 sends AR-dtype `384`-lane column chunks; round 2 all-gathers the `f32` chunk sums (bit-identical to the one-shot sum, cutting per-rank traffic from `BR x 41 KB` to `BR x 16.2 KB`). One barrier at kernel start; sends are waited lazily one layer later; `ar_start` / `finish` overlap MoE AR with layer `l+1`'s initial KV page DMAs. | Rank-order `f32` sum keeps the residual stream and routing identical on all ranks. On TPU v7x, in-kernel all-reduce takes `0.25 ms` total across all 72 ARs at `B=1` (`~3.5 us/AR` vs `~40 us` stock XLA all-reduce) and `1.26 ms` at `B=32`; `rsag` saves an extra `0.40 ms` at `B=32` and `1.11 ms` at `B=64`. |
| XLA fallback | `make_step(..., xla_ar=True)` builds per-layer `attn` / `moe` kernels from **the same phase bodies** with `lax.psum` in between (`2L + 1` calls) | Used to verify the in-kernel AR (bit-identical in interpret mode) and isolate in-kernel AR savings on TPU |
| KV layout & `split_pools` | SGLang's exact fused TPU pool layout per layer: `[pages, 64, 1, packing, 128]` (K slot 0, V slot 1, `fp8` slots 2-3 padding, `head_dim` zero-padded to 128). Supports both stacked pools (`[L/2, ...]` for standalone benchmarks) and `split_pools=True` (`L/2` SWA + `L/2` full separate per-layer pool operands for zero-copy serving integration). | In `split_pools=True` mode, each page-DMA start selects its layer-pair ref under a static `pl.when`; waits only need the destination shape. |
| New K/V write | Separate non-collective scatter kernel (`kv_write="scatter"`). `kv_write="kernel"` (megakernel writes into aliased pools) is also implemented for standalone tests, but XLA inserts pool-sized copies (`4.6 GB` temp) when aliasing inside a `collective_id` `pallas_call`. | Keeping the KV write in a separate non-collective `pallas_call` avoids any pool copies in HLO. |
| Weight streaming | Attention + router weights (`wqkv, wo, wr, vec, sinks`) use a 2-slot buffer (`l+1` prefetched at the start of layer `l`). Expert chunks `(w13 [Hp, 2*tne], w2 [tne, Hp], b13, b2)` stream through a `w_slots`-deep ring (default `3`, so 2 chunks in flight). | Expert chunks depend on layer `l`'s routing; across the layer boundary the stream prefetches layer `l+1`'s attention weights. |
| Weight dtype | Attention/router `bf16`. Experts default to `w_dtype="fp8"`, with `"mxfp4"` and `"bf16"` also supported. **`fp8`** converts MXFP4 offline to `e4m3fn` times a power-of-two per output column (exact when a column's 32-row scale blocks span `<= 13` octaves; `0` inexact blocks out of `18,432` on `openai/gpt-oss-120b`). Runs `bf16 x e4m3fn` natively on the v7x MXU (`3.39 MB` per expert chunk, `~0.6K` VPU ops vs `~11K` VPU ops for in-VMEM MXFP4 dequant). **`mxfp4`** streams `e2m1` codes (`8` per `uint32` along `K`) + `bf16` power-of-two block scales (`1.94 MB` per chunk) and dequantizes in VMEM. | On TPU v7x (`B=32`), native `fp8` experts cut MoE expert time to `3.41 ms` (at the `~10 GB/rank` HBM bandwidth floor), lowering total 36-layer kernel step time from `7.97 ms` (MXFP4) to `5.80 ms` (`fp8` + `rsag`). |

## VMEM budget (analytic, tile-padded; real dims, `fp8` KV, TP MoE, `tne=384`)

| Batch Bucket (`B`) | Total VMEM (`bf16` / `MXFP4` / `fp8` Experts) | Largest Resident VMEM Allocations |
|---|---|---|
| `1, 2, 4` | `38.2 MiB` / `30.8 MiB` / `28.8 MiB` | `w13` 3-slot ring, `wqkv` x2, `w2` 3-slot ring, `wo` x2, folded AR recv buffers |
| `8, 16` | `39.4 MiB` / `32.0 MiB` / `30.0 MiB` | `w13` ring `12.9 MiB`, `wqkv` x2 `8.6 MiB`, `w2` ring `6.5 MiB`, `wo` x2 `5.8 MiB`, `wr` x2 `1.4 MiB`, AR recv `1.4 MiB` |
| `32` | `42.3 MiB` / `34.9 MiB` / `32.9 MiB` | Same weight rings + AR recv `2.9 MiB` |
| `64` | `48.0 MiB` / `40.6 MiB` / `38.7 MiB` | Same weight rings + AR recv `5.8 MiB` |

`vmem_limit_bytes = 60 MiB`. All 70 AOT configurations across `B in {1, 2, 4, 8, 16, 32, 64}` compile cleanly under the 60 MiB limit. Per-rank HBM expert + attention weight footprint is `14.96 GiB/rank` (`fp8`, TP MoE) or `8.66 GiB/rank` (`mxfp4`, TP MoE).

## Validation & TPU v7x benchmark results

### 1. Correctness & AOT Compilation

| Check | Method | Result |
|---|---|---|
| Megakernel vs pure-JAX TP reference (`reference.step`) | Interpret mode on 8 virtual CPU devices, tiny dims (`H=240->256`, real `64q/8kv/D=64` heads, `E=16`, `top-4`, `I=384`, `page=64`, `4 pages/seq`, positions `0, 63, 130, 255`) | Relative error `3-5e-3` across `bf16`/`fp8` KV, `tp`/`ep` MoE, `L=2/4`, `B=1/4/16` |
| In-kernel all-reduce (`oneshot` & `rsag`) vs `lax.psum` | Megakernel vs `xla_ar=True` fallback on identical inputs | Bit-identical outputs and KV pools |
| MXFP4 and FP8 experts | Reference weights snapped to MXFP4; packed codes + scales in kernel (`tp` and `ep`) | Relative error `3.9-4.3e-3`; `fp8` outputs identical to `mxfp4` when column scale range `<= 13` octaves |
| In-place KV scatter | Pools pre-filled with finite sentinels outside attended pages | Only the active sequences' new-token slots change and match reference K/V |
| `tpu7x:2x2x1` AOT compilation | `aot_compile.py` across `B in {1,2,4,8,16,32,64}`, `fp8`/`bf16` KV, `tp`/`ep` MoE, `fp8`/`mxfp4`/`bf16` experts, `max_pages` up to `513` (32K context) | `70/70` configurations compile (`1.4-6 s` each); zero pool-sized HLO copies (`benchmark/kernels/gptoss_decode_megakernel/aot_tpu7x_2x2x1_results.jsonl`) |

### 2. TPU v7x (`2x2x1`, TP=8) Standalone 36-Layer Kernel Step Time (`ms_device`, `fp8` KV, `fp8` Experts, TP MoE)

| Context Length | `B=1` | `B=4` | `B=8` | `B=16` | `B=32` | `B=64` |
|---|---:|---:|---:|---:|---:|---:|
| **1K (`ctx=1024`)** | **0.76 ms** | **1.73 ms** | **2.69 ms** | **3.74 ms** | **5.80 ms** | **9.09 ms** |
| **8K (`ctx=8192`)** | **0.85 ms** | — | **3.29 ms** | — | **8.53 ms** | **14.58 ms** |
| **32K (`ctx=32768`)** | **1.27 ms** | — | **5.88 ms** | — | **18.76 ms** | — |

- **Comparison against XLA-`psum` fallback (`ctx=1024`):** `0.76 / 2.69 / 5.80 / 9.09 ms` (megakernel) vs `3.86 / 5.02 / 8.20 / 11.27 ms` (`xla_ar=True` fallback) — in-kernel all-reduce alone saves `2.2-3.1 ms/step`.
- **Per-component device breakdown (`ctx=1024`):**
  - `B=1` (`0.76 ms`): all-reduce `0.25 ms`, experts `0.23 ms`, attention `0.21 ms`, call/layer floor `~0.22 ms`.
  - `B=8` (`2.69 ms`): experts `1.20 ms`, all-reduce `0.70 ms`, attention `0.70 ms`.
  - `B=32` (`5.80 ms` with `rsag`): experts `3.41 ms`, attention `1.76 ms`, all-reduce `~0.86 ms`.

## Known limitations & future work

- **Long-context attention at large batch (`B >= 32`, `ctx >= 8K`):** Each rank owns 8 query heads (1 GQA group of 8 Q rows per KV head), so attention pushes streamed K and V through the MXU as rhs weights for an 8-row lhs activation tile (`~1.5-2` LLO ops/token). At `B=32`, device step time grows from `5.80 ms` (`1K`) to `8.53 ms` (`8K`) and `18.76 ms` (`32K`).
- **Cross-layer expert prefetch:** Currently attention + router weights are prefetched across layers (`l+1` during `l`), whereas expert chunks start after layer `l`'s router completes.

## Files

- `config.py`: `DecodeConfig` (geometry, sharding mode, KV/expert dtypes, AR mode, tiles, ablations). `tiny()` / `full()`.
- `reference.py`: global model-layout weights and pure-JAX TP reference step.
- `megakernel.py`: `pack_weights`, `packed_specs`, `pool_specs`, `build_pools`, `build` (`mega` / `attn` / `moe` modes), `build_kv_scatter`, `make_step`, `vmem_bytes`.
- `INTEGRATION.md`: opt-in SGLang serving integration (`sglang_tpu/gptoss_decode_mk/`) and end-to-end TPU v7x serving benchmarks.
- `test/registered/unit/hardware_backend/tpu/test_gptoss_decode_megakernel.py`: CPU interpret unit tests.
- `benchmark/kernels/gptoss_decode_megakernel/aot_compile.py`: `tpu7x:2x2x1` AOT compilation check.
- `benchmark/kernels/gptoss_decode_megakernel/bench_decode_step.py`: TPU step microbenchmark (`mk` / `fallback` / `xla`), reporting `ms_median` (host-blocking), `ms_pipelined` (back-to-back dispatch), and `ms_device` (`K` steps in one `fori_loop`).

## Timing ablations

`DecodeConfig.ablate` / bench `--ablate` (timing-only; outputs are intentionally invalid):

- `ar`: pack + local reduce only (no remote DMAs, no barrier, no `collective_id`).
- `attn`: skip the entire attention block, including its all-reduce.
- `attn_kv`: skip only the paged attention core and KV prefetch.
- `moe`: skip router, experts, and the MoE all-reduce.
- `experts`: skip only the expert weight stream and matmuls.

## How to run

```bash
# CPU interpret tests (8 virtual devices)
XLA_FLAGS=--xla_force_host_platform_device_count=8 JAX_PLATFORMS=cpu \
  python -m unittest test.registered.unit.hardware_backend.tpu.test_gptoss_decode_megakernel

# AOT compile for tpu7x:2x2x1 on a CPU host with jax[tpu] and libtpu
JAX_PLATFORMS=cpu python benchmark/kernels/gptoss_decode_megakernel/aot_compile.py \
  --batch 1 2 4 8 16 32 64 --kv fp8

# TPU microbenchmark (one tpu7x 2x2x1 host, single process)
python benchmark/kernels/gptoss_decode_megakernel/bench_decode_step.py \
  --batch 1 4 8 16 32 64 --modes mk fallback xla --ctx 1024 --kv fp8
```
