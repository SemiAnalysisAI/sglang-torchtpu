# Kimi-K3 decode megakernel (Inferact `tpu-megakernels`): analysis for a port into SGLang-torchtpu

Date: 2026-10-12. This is a read-only review of `/home/chandrasekhard/tpu-megakernels` at commit `4048f08`. That repository has four commits: the code drop "add k3 and qwen megakernel implementations" plus documentation. Unless a path is given, line references are to that checkout. SGLang references are to the merged upstream checkout `/home/chandrasekhard/dagent/merged-t5`, written `merged-t5/...`.

What was actually executed on this machine (it has no TPU):
- The megakernel's decode program, lowered for TPU from a CPU host with 32 virtual CPU devices and TPU7x hardware info patched in. This was done with JAX 0.10.2, the version in SGLang's stack (`/home/chandrasekhard/dagent/venv`), and with JAX 0.11.1, the megakernel's pinned version (in a throwaway venv under `/tmp/k3perf/venv111`). The script is at `/tmp/k3perf/scripts/export_check.py` and the logs are `/tmp/k3perf/export_b{1,8}_jax0{102,111}_ckpt.log`.
- The repository's 17 Kimi CPU tests under JAX 0.10.2.

Nothing was compiled by libtpu or run on TPU, so every performance statement below is from the published numbers, from code reading, or an explicitly marked estimate.

---

## 0. Summary

### What the megakernel is
- **One `pl.pallas_call` per decode step.** It is named `kimi_decoder_stack` (`kimi/decode_megakernel.py:5133-5157`) and runs all 93 layers:
  - layer 0, which is KDA plus the dense MLP, is peeled out;
  - layers 1..92 run in an in-kernel loop, `pl.loop(1, layers)` (`:4966-4968`).
  - Each layer does KDA or MLA attention, the attention-residual mixing, RMSNorms, the full latent MoE, and every tensor-parallel collective.
- **Only three things run outside the kernel, as plain XLA:**
  - the embedding gather plus one psum;
  - the final attention residual, final norm and vocab-sharded lm_head;
  - a greedy argmax with one all-gather (`:5216-5249`, `:5296-5329`; `kimi/dspark.py:657-690`).
- **Collectives are in-kernel remote DMA**, using `pltpu.make_async_remote_copy` with `pl.DeviceIdType.MESH` between ranks addressed as `rank XOR offset`.
  - There is one 31-peer barrier per step (`collectives32.py:25-30`, `:3864`).
  - Partial sums travel as BF16 packed two per 32-bit word.
  - The reduction is two-level (8 devices inside a host, then 4 hosts) or one-shot among 16 devices of the same parity.
  - XLA collectives are not used inside the stack.
- **The MoE is not EP32.** Routed experts are partitioned **EP8 x TP4**: 8 groups of 4 ranks, 112 experts per group, and each rank holds a 768-channel quarter of every one of its group's experts (`:1298-1299`, `kimi/load.py:879-880`).
  - The router weights are replicated on every rank, and each chip (a rank pair) computes all 896 scores.
  - Top-16 is computed as 16 sequential argmax passes.
  - The 3584-wide latent is sharded 32 ways and all-gathered by DMA.
  - Routed outputs are reduced in latent space.
- **MLA uses an expanded per-head K/V cache, not the latent cache.** K is 256 wide (192 used) and V is 128 wide, 3 heads per device, contiguous per sequence: `[rows, 24 layers, 3, context, 256|128]` bf16, at 55,296 bytes per token per device. There is no paging, and the context is fixed at launch (`--context`, a multiple of 128 and at least 384).
- **KDA state:**
  - conv history `[rows, 69, 3, 3, 3, 128]` bf16;
  - recurrent state `[rows, 69, 3, 128, 128]` f32 per device;
  - indexed by batch row.
- **Batch size** is a compile-time 1..8 rows. Each row may have its own position. There are no padding masks.
- **There is no prefill kernel.** The demo feeds prompts through the decode kernel one token per launch in target-only mode, or 8 tokens per launch through the speculative-verify variant (`demo_kimi_dspark.py:817-870`).
- **Runtime:**
  - one JAX multi-controller program: 4 processes x 8 TensorCore devices, `jax.distributed.initialize`, `jax.shard_map` over a 1-D 32-device mesh `"tp"`;
  - the decode loop of N steps (64 in the benchmark) runs inside one jitted `fori_loop` with an on-device greedy argmax;
  - the host checks stop conditions once per call.
  - The server batches statically. Target-only mode is greedy only, and every request in a batch must have the same prompt length (`demo_kimi_dspark.py:1236-1237`).

### Published numbers
- Target only: **249 / 392 / 515 / 865 tok/s at batch 1/2/4/8**. That is **4.02 / 5.10 / 7.77 / 9.25 ms per decode step**.
- These come from `scripts/benchmark_kimi.sh`, which sends **1-token prompts and 1,024 output tokens with `CONTEXT=1152`**, and divides decode-only device-loop time by tokens (`scripts/benchmark_kimi.sh:23, 122-188`). So they are short-context numbers.
- The GB200 baseline in the same figure is 127 / 227 / 373 / 636 tok/s.
- Our SGLang decode TPOT is 16.9 ms at batch 1 (8k prompt, 1k output) and 21.9 ms at batch 4. On the face of it the megakernel is 4.2x faster at batch 1 and 2.8x at batch 4.
- The context difference matters, because the megakernel's decode attention computes scores on the vector unit (section 1.6). My rough estimate for an 8k context is +0.4 to 1.4 ms per step at batch 1, and more at batch 4. This must be measured (section 7, item 8, and section 8.1).

### JAX compatibility (measured, without TPU)
- JAX 0.10.2 traces and lowers the full stack at batch 1 and batch 8 (tuned recipe, with dtypes from the real checkpoint) to one `tpu_custom_call`, configured as `has_communication: true, collective_id: 21, scoped VMEM 64 MiB`.
- The Mosaic payload is 497,620 bytes of base64 at batch 1 and 870,456 at batch 8. JAX 0.11.1 gives 500,304 and 873,920 bytes.
- All Pallas TPU APIs and the private hooks the kernel uses exist with identical signatures in 0.10.2 (section 5).
- What remains unproven is whether libtpu 0.0.47 compiles the module and whether it then runs correctly and fast.

### Biggest port blockers
1. **Execution model.** The kernel is a 32-device SPMD Pallas program. It takes the rank from `jax.lax.axis_index("tp")`, addresses peers by mesh device id, uses a barrier semaphore with `collective_id=21`, and claims the whole 64 MiB of TPU7x VMEM.
   - SGLang's torch_tpu bridge does support a `shard_map` over the full world mesh, with `axis_index` equal to the torch rank and remote DMA allowed.
   - But no SGLang kernel combines `shard_map` with buffer donation or aliasing, which the megakernel needs for in-place cache updates.
   - Remote DMA across TP32 and 4 hosts through the bridge has not been demonstrated (section 5).
2. **Cache representation.** The megakernel uses an expanded per-head contiguous MLA cache with no indirection; SGLang uses a replicated, paged latent MLA cache, `[num_pages, 64, 2, 640]` bf16 per layer. KDA state is close: the element order matches, but the conv dtype (bf16 vs f32), slot indirection, and a stacked layer axis vs per-layer tensors all differ.
3. **Weights and HBM.**
   - The megakernel holds 78.5 GB per device in its own layout, 72.2 GB of it experts (FP8 gate/up).
   - SGLang holds roughly 60 GB per device: experts EP32 MXFP4 with f32 scales at about 53 GB, plus several GB of dense weights, including about 4.7 GB for the replicated latent down projection.
   - A TPU7x device has 103 GB (JAX `tpu_info`: 206 GB per chip / 2 TensorCores), so both copies cannot be resident. One expert layout has to serve both prefill and decode.
4. **Scope gaps:**
   - batch ≤ 8;
   - no prefill;
   - no paged KV or continuous batching;
   - target-only mode samples greedily only;
   - no multimodal support;
   - no end-to-end correctness test, and no recorded GSM8K or GPQA scores in the repository.

### Recommendation
1. Run it standalone on our slice first (section 7) to measure TPOT at an 8k context and check accuracy.
2. Spike the bridge: compile and run the stack with synthetic weights as a torch_tpu custom op inside SGLang's process group.
3. Port in stages. Start with a per-layer MoE op, which owns no state, removes most collectives and the EP imbalance, and reuses the same bridge mechanics.
4. Treat the whole-step custom op (option a) as the end state once the bridge, a shared expert layout and paged-latent MLA exist.

The separate JAX engine (option b) is only viable with a disaggregated prefill slice. Section 8 has the details.

---

## 1. Architecture

### 1.1 Kernel granularity: one `pallas_call` per decode step
- `stack()` (`:3618-5157`) builds one `pl.pallas_call(body, name="kimi_decoder_stack", ...)` (`:5133-5157`). It has no grid, so it is a single invocation per device.
- `PrefetchScalarGridSpec(num_scalar_prefetch=2)` puts `positions[B]` (int32) and `state_row` into SMEM (`:5138-5139`, `:5157`).
- **Inputs:** `prefix` and `embedding` sit in VMEM; every weight family and the four state arrays stay in HBM (`pl.BlockSpec(memory_space=tpu.VMEM if index < 2 else tpu.HBM)`, `:5140-5143`).
- **Outputs:** prefix and attention-residual blocks go to VMEM. The KDA conv state, KDA recurrent state, MLA K cache and MLA V cache are HBM outputs aliased to their inputs (`input_output_aliases={state_offset + i: i + 2}`, `:5137`), so they are updated in place. The aux-hidden output is used only by DSpark.
- **Compiler parameters** (`:5151-5156`): `collective_id=21`, `vmem_limit_bytes=64 MiB`, `disable_bounds_checks=True`, `shape_invariant_numerics=True`.
- **Wrapping:** `decode_local()` (`:5162-5338`) calls `stack()`. `make_decode()` (`:5341-5435`) wraps it in `jax.jit(jax.shard_map(local, mesh, in_specs=(P("tp"), P("tp"), P(), P()), out_specs=(P("tp"), P("tp")), check_vma=False), donate_argnums=(1,))`.
  - The demo passes `donate=False` (`demo_kimi_dspark.py:391-394, 522-525`). The states are loop-carried inside the multi-step jitted loop, so the aliasing should be in place within a call. I infer, without having measured it, that XLA copies the non-donated input states once per call.
  - Weights and states are global arrays `[32, ...]` sharded on the leading rank axis.
  - Logits come out as `[32, B, 5120]`, sharded by vocabulary.
- **Measured export** (JAX 0.10.2, batch 1): one `tpu_custom_call` plus one `stablehlo.all_reduce` (the embedding psum), 43 inputs and 5 outputs.

### 1.2 Layer iteration and weight streaming
- **Layer loop.** Layer 0 is peeled out as straight-line code: KDA plus a dense MLP streamed in 1024-row and 384-row slabs (`:4803-4899`, `:4966-4967`). Then `pl.loop(1, layers)(layer)` (`:4968`).
- **Layer type.** `jax.lax.cond((layer % 4 == 3) | (layer == 92), run_latent_attention, run_delta_attention)` (`:4612-4616`), which gives 24 MLA layers (3, 7, ..., 91, 92) and 69 KDA layers.
- **Indexing.** Families are stacked along a leading layer axis and indexed dynamically: KDA index `layer - layer // 4`, MLA index `layer // 4`, MoE index `layer - 1` (`:4138`, `:4281`, `:4738-4785`).
- **One VMEM weight pool.** `projection_expert_pool_vmem` is sized to `max(4 expert slots, KDA projections, MLA projections)` (`:5064-5084`, `:5100`). It is reinterpreted per use through `pool_alias.view` (`:3811-3848`), with the same pool alternately holding attention projections and routed-expert slots.
- **Double buffering.** Norms, router, KDA auxiliary weights and state, and MoE metadata are double-buffered.
- **Cross-layer prefetch.** After layer L's routed experts finish, `after_experts()` prefetches layer L+1's norms, attention projections, KDA aux/state, metadata and router half (`:4704-4730`). The next layer's weight DMA therefore overlaps the MoE reductions.
- **VMEM budget.** The limit is 64 MiB. JAX's TPU7x `tpu_info` gives VMEM as 64 MiB per core, so the kernel claims all of it.

### 1.3 Collectives
- **Barrier.** `collectives32.barrier()` (`collectives32.py:25-30`) signals the barrier semaphore on all 31 peers and waits for 31 signals. It runs once at kernel entry (`:3864`). The per-op barriers are disabled inside the stack (`synchronize_devices=False` at `:4200`, `:4342`, `:4792`).
- **Assumed topology** (`_logical_mesh`, `:5438-5455`, "preserving chip-pair bit 0"):
  - rank bit 0 selects one of the two TensorCores of a chip;
  - bits 1-2 select the chip within a host;
  - bits 3-4 select the host.
  - Peers are `rank ^ 1..7` inside a host and `rank ^ 8k` across hosts.
  - Correctness does not depend on this mapping; performance does.
- **Primitive.** `tpu.make_async_remote_copy(src_vmem, dst_vmem, send_sem, recv_sem, device_id=(rank ^ offset,), device_id_type=pl.DeviceIdType.MESH)`, VMEM to remote VMEM.
  - Several receivers complete with one aggregate wait whose descriptor is sized to the sum of the incoming messages (for example `:1110-1120`, `:1528-1535`, `:3063-3065`). This relies on DMA semaphores counting bytes.
- **Patterns:**

  | Pattern | Where | What |
  |---|---|---|
  | Attention output projection | `_project_attention_2d`, `:3045-3202` | Exchange the 384 attended features with `rank^1` (same chip). Multiply 768 features by this rank's `[768, 3584]` output half. One-shot all-reduce among the 16 same-parity ranks (send a BF16-packed partial to 15 peers, sum locally). Exchange halves with `rank^1`. B4/B8 have a token-scatter variant (`:3068-3171`). |
  | Router scores | `:142-158` | Paired ranks exchange 512-wide score halves (on-chip). |
  | Latent all-gather | `:1517-1547` | Each rank computes 128 latent columns and DMAs them to all 31 peers. Overlapped with top-k: it starts after the 4th selection (`:1610-1615`). |
  | Shared and routed reductions | `:1784-1848`, `:1944-2045` | 7 local peers, then 3 hosts. BF16 values are packed two per u32 (`encode_*`/`decode_*`, `:1715-1782`). B6/B8 use reduce-scatter variants (`:1704-1713`, `:1958-2020`). |
  | MoE output | `_gather_output_shards`, `:1093-1120` | Each local lane computes 1024 output columns, then gathers 7 local shards. |
  | Dense layer-0 MLP | `_all_reduce_rows`, `:3234-3283` | Same 7+3 scheme. |
- **Unused code.** Only `barrier()` from `collectives32.py` is used by the Kimi decoder. `start_local_piece`, `encode_partial` (int16 host packets), `reduce` and `scratch` are not; nor is `_gather_latent_host` (`:3204`).
- **Precision.** Partials are rounded to BF16 before transport and summed in FP32 after receipt. The result is not bitwise equal to an FP32 all-reduce.
- **Per-layer counts:**
  - MLA or KDA layer: 3 communication phases (pair exchange, 15-peer one-shot, pair exchange).
  - MoE layer: about 5 (router pair exchange, 31-peer latent all-gather, shared 7+3 reduction, routed 7+3 reduction, 7-peer output gather).
  - All of these are in-kernel DMAs that overlap weight DMA.
- **Compare SGLang today:** 280 XLA all-reduces per step (1 embedding + 93 `o_proj` + 1 dense + 92 MoE latent + 92 MoE shared, plus a logits all-gather; `merged-t5/python/sglang/srt/distributed/parallel_state.py:719-724`).

### 1.4 MoE (896 experts, top-16, latent 3584, 2 shared)
- **Ownership.**
  - `local_expert_count = 112` and `expert_tensor_parallel = 32*112/896 = 4` (`:1298-1299`).
  - Expert e belongs to the 4 ranks `4*(e // 112) .. +3`: `owned_locally = winners // 112 == rank // 4` (`:191-194`).
  - Within a group, `rank % 4` picks a 768-channel quarter of the 3072-wide intermediate. That is gate/up output columns and down contraction rows (`kimi/load.py:172-184`, `:879-880`).
- **Router.**
  - `routing_projection [7168, 896]` is BF16 and replicated per rank.
  - Rank pairs split the columns 512 / 384 (plus 128 zero padding) (`:1153-1182`), so each chip computes all 896 scores.
  - Selection uses sigmoid plus `e_score_correction_bias`. The mixture weights are the uncorrected sigmoids renormalized over the 16 selected experts (`:160-216`; reference `kimi/__init__.py:97-102`). The routed scaling factor is 1.0.
- **Top-16.** 16 sequential `jnp.argmax` passes over `[B, 896]` f32 with -inf masking (`:177-215`). All ranks see the same data, so they agree with each other. Tie order is Mosaic's, which the SGLang doc says does not pick the lowest index.
- **Latent down projection.** `latent_down [7168, 128]` per rank; ranks 28-31 hold zeros, since 28 x 128 = 3584 (`kimi/load.py:381-391`). The latent is all-gathered into `[B, 4096]`.
- **Routed experts** consume the 3584-dim latent and produce 3584-dim outputs, which are summed across all 32 ranks.
- **Latent tail.**
  - RMSNorm over 3584 features with `routed_expert_norm` (padded to 4096) (`:2079-2088`).
  - `latent_up [3584, 1024]` per rank computes output columns `(rank % 8)*1024`. It is replicated across the 4 hosts, and lane 7 is zero padding (`kimi/load.py:448-451, 464-472`).
  - Add this rank's shard of the shared-expert output, then gather 7 local shards (`:2097-2154`).
- **Routed expert stream** (`_run_routed_expert_stream`, `:543-1038`):
  - 4 VMEM slots; experts run in pairs, with one MXU pass covering both experts' gate/up (`_paired_expert_gate_up`, `:408-540`), and the next pair is prefetched.
  - Fast path when all rows share a route (`:718-831`).
  - Otherwise the "divergent" path processes the minimum pending expert id across rows, so each expert's weights are loaded once per step for every row that needs them (`:833-1038`).
  - B8 has a compact singleton path (`:934-1024`).
  - Activation (SiTU): `4*tanh(g/4)*sigmoid(g)*25*tanh(u/25)` (`:102-110`).
- **Shared experts.** 2 x 3072 = 6144 channels at TP32 gives 192 per rank: `shared_gu [7168, 384]` and `shared_down [192, 7168]` (`kimi/load.py:473-488`).
- **Dense layer 0.** 33792 / 32 = 1056 channels per rank, padded to 1152: `dense_gu [7168, 2304]` and `dense_down [1152, 7168]` (`kimi/load.py:346-371`).

### 1.5 KDA (69 layers, 96 heads, 3 per rank, K = V = 128)
- **Weights per rank** (`kimi/load.py:392-421, 457-463`):

  | Family | Shape | Contents |
  |---|---|---|
  | `k_gate` (= `f_a_proj`) | `[7168, 128]` | replicated |
  | `k_projection` | `[7168, 1664]` | q, k, v (3 x 384) \| g (384) \| beta (3, padded to 128) |
  | `k_fb` | `[128, 384]` | |
  | `k_o` | `[768, 3584]` | paired-rank layout |
  | `k_conv` | `[4, 3, 3, 128]` | (tap, q/k/v, head, channel), f32 |
  | `A_log` | `[3, 1]` | f32 |
  | `dt_bias` | `[3, 128]` | f32 |
  | `o_norm` | `[8, 128]` | f32, padded |

- **Step** (`kda()`, `:2204-2407`):
  1. decay = `-5 * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias))`.
  2. Short conv over a 3-step history followed by SiLU.
  3. L2-normalize q (scaled by 128^-0.5) and k.
  4. Decay the state: `state *= exp(decay)`.
  5. Delta rule with beta = `sigmoid(b)`.
  6. `out = q . state`, then RMSNorm with `o_norm`, times `sigmoid(g)`.
  7. Output projection with the paired all-reduce.
- **State per rank:**
  - conv history `[B, 3 (time), 3 (q/k/v), 3 (heads), 128]` bf16;
  - recurrent `[B, 3, 128 (K), 128 (V)]` f32;
  - stacked as `[B, 69, ...]` (`demo_kimi_dspark.py:407-415`).
- **State movement.**
  - B ≤ 4: the state is prefetched with the layer's aux weights into persistent VMEM.
  - B = 6 or 8: scoped VMEM with overlapped reads and write-backs (`:3718-3720`, `:4137-4278`, `:4704-4722`).
  - Either way it is written back in place through the aliased outputs.

### 1.6 MLA (24 layers, NoPE, 3 heads per rank)
- **No RoPE.** The 64 "rope" dims are used without rotation, and the softmax scale is 192^-0.5 (`:2772`, `:2944`). SGLang's K3 MLA also skips rope (`merged-t5/.../kimi_k3.py:2300`).
- **Weights per rank** (`kimi/load.py:422-436, 457-463`):
  - `q_a [7168, 1536]` and `kv_a [7168, 640]` (576 plus 64 padding) are **replicated on all 32 ranks**;
  - `q_b [1536, 768]`: 3 heads x (192 + 64 zero padding);
  - `kv_b [512, 768]`: 3 heads x (128 k_nope + 128 v);
  - `g_proj [7168, 384]`;
  - paired `o_proj [768, 3584]`.
- **New K/V.** K = `[k_nope 128 | the un-normalized 64-dim tail of the kv_a output | 64 zeros]` and V = 128 (`:2667-2689`).
- **Decode attention** (`attend_cache_tile_body`, `:2701-2839`):
  - double-buffered DMA of 128-token K/V tiles from HBM (`:2598-2634`, `:2759-2762`);
  - online softmax;
  - the current token's K/V is substituted into the tile by a mask instead of being read back;
  - per-row causal mask `token <= position`;
  - the loop runs to the maximum position in the tile of rows, so cost tracks the longest row.
  - **Scores are an elementwise multiply plus a lane reduction on the vector unit** (`:2764-2773`). P.V uses the MXU in "hilo" mode (two BF16 passes, about 16 mantissa bits) when the demo default `mla_pv_mxu_hilo` is set (`:73-99`, `demo_kimi_dspark.py:89-94`).
  - The speculative-verify path uses MXU scores instead (`_attention_scores`, `:2943`).
- **Cache write.** A separate commit pass, per row and per MLA layer, reads the 128-token tile that holds the position, merges the new row, and writes it back to the aliased cache (`:4350-4595`).
- **Output.** `sigmoid(g) * attention`, then `_project_attention_2d`.
- **Context-length implication (estimate).** Per 128-token block, the vector-unit work is a few multiply-adds on a `[3*rows, 128, 256]` f32 tile plus lane reductions. That is roughly 0.3-1 µs per block per tile of rows.
  - 8.7k context at batch 1: 68 blocks x 24 layers ≈ 1,600 blocks per step, or about +0.4-1.4 ms over the 1,152-token benchmark.
  - Batch 4 with a 4-row tile (tuned recipe) has 4x the vector-unit work per block.
  - Moving the scores onto the MXU through `_attention_scores` would be a small change.

### 1.7 Attention residuals (Block AttnRes, block size 12)
- **Storage.** `blocks_vmem [B, 8, 7168]` f32 (8 = ceil(93/12)) plus `prefix [B, 7168]` bf16.
- **Before attention:** a softmax over the active blocks plus the prefix. The per-layer folded weight `attn_res` (= `self_attention_res_norm ⊙ self_attention_res_proj`, f32, `kimi/load.py:321-325, 339-340`) supplies the scores, normalized by RMS (`_residual_mixture_row`, `:3457-3514`).
- **Block boundaries.** `layer % 12 == 0` stores the prefix into a new block (`:4111-4123`).
- **After attention:** prefix = attention output (new block) or prefix + attention output (`:4690-4694`).
- **MLP side:** `mlp_res` mixing (`:4697-4702`), then `prefix += MoE output` (`:4962`).
- **Final:** mixing with `output_res`, then the final RMSNorm, in XLA outside the kernel (`:5311-5321`).
- The "tuned" recipe uses `residual_reduction_order="relaxed"`, which changes the FP32 association order (`:3473-3487`).

### 1.8 Output head and sampling
- `lm_head [7168, 5120]` per rank (163840 / 32) runs in XLA after the kernel (`:5322-5326`).
- **Target-only mode** is greedy only. `make_sharded_argmax` takes a local argmax and does one all-gather of (max, id) (`kimi/dspark.py:657-690`), and the loop feeds the token straight back on the device (`demo_kimi_dspark.py:532-543`).
- **DSpark mode** offers greedy acceptance inside the fused draft kernel, or exact top-p/temperature rejection sampling (`make_row_sampler`, `kimi/dspark.py:738-770`; `nucleus_accept`, `:693`).

---

## 2. Data layouts

### 2.1 Weights per rank (selected layout, `kimi/load.py:492-544`, plus the FP8 reblock at `:623-657`)

Dtypes come from the checkpoint headers at `/mnt/kimik3-ssd/kimik3`. That checkpoint matches the loader's tensor naming (`language_model.model.` prefix, `weight_packed` / `weight_scale`).

| Family | Per-rank shape | Dtype | Partitioning |
|---|---|---|---|
| `router` | `[92, 7168, 896]` | bf16 | replicated; each rank copies a 512 or 384-column half to VMEM |
| `router_bias` | `[92, 1, 896]` | f32 | replicated |
| `latent_down` | `[92, 7168, 128]` | bf16 | column shard; ranks 28-31 zero |
| `latent_up` | `[92, 3584, 1024]` | bf16 | output block `rank % 8` (block 7 zero), replicated across hosts |
| `latent_norm` | `[92, 1, 4096]` | f32 | replicated (3584 real) |
| `shared_gu` / `shared_down` | `[92, 7168, 384]` / `[92, 192, 7168]` | bf16 | TP32 |
| `expert_gu` | `[92, 112, 3584, 1536]` | **fp8 e4m3** | EP8 x TP4; gate‖up quarter-width |
| `expert_gus` | `[92, 112, 14, 1536]` | bf16 power-of-two | one scale per 256-row contraction block (`K3_FP8_BLOCK`, `:623`) |
| `expert_down` | `[92, 112, 96, 3584]` | uint32 (packed MXFP4, low nibble first) | EP8 x TP4 |
| `expert_ds` | `[92, 112, 24, 3584]` | uint8 E8M0 | |
| `k_gate`, `k_projection`, `k_fb`, `k_o` | `[69,7168,128]`, `[69,7168,1664]`, `[69,128,384]`, `[69,768,3584]` | bf16 | heads TP32; `k_o` paired |
| `k_conv`, `k_a_log`, `k_dt`, `k_norm` | `[69,4,3,3,128]`, `[69,3,1]`, `[69,3,128]`, `[69,8,128]` | f32 | |
| `m_qa`, `m_ka` | `[24,7168,1536]`, `[24,7168,640]` | bf16 | replicated |
| `m_qb`, `m_kb`, `m_gate`, `m_o` | `[24,1536,768]`, `[24,512,768]`, `[24,7168,384]`, `[24,768,3584]` | bf16 | heads TP32; `m_o` paired |
| `m_qnorm`, `m_knorm` | `[24,1,1536]`, `[24,1,512]` | bf16 | replicated |
| `attn_res`, `mlp_res` / `attn_norm`, `ffn_norm` | `[93,1,7168]` | f32 / bf16 | replicated |
| `embedding` / `lm_head` | `[5120, 7168]` / `[7168, 5120]` | bf16 | vocab TP32 |
| `dense_gu` / `dense_down` | `[7168, 2304]` / `[1152, 7168]` | bf16 | TP32 (1056 real + 96 padding) |

- **MXFP4 handling.** The checkpoint has MXFP4 experts (`mxfp4-pack-quantized`, group 32, uint8 E8M0).
  - **Gate/up are re-encoded at load** from MXFP4 to FP8 e4m3 with coarse BF16 power-of-two scales (`_reblock_one`, `kimi/load.py:626-646`). The loader asserts that the round trip is bitwise exact and refuses to start otherwise (`:781-782`, `:903-904`).
  - **Down projections stay MXFP4.**
  - The kernel can still consume MXFP4 gate/up (`_paired_expert_gate_up` and `_quantized_expert_dot`, `:285-540`), but the loader always reblocks. The docstring states the trade: "a larger HBM representation for removing MXFP4 unpack" (`:295-301`).
- **Totals per rank** (computed from the shapes; `/tmp/k3perf/scripts/bytes.py`):
  - **78.48 GB of weights**: 72.23 GB experts and 6.24 GB everything else.
  - With MXFP4 gate/up the experts would be 45.20 GB.
  - TPU7x HBM is 103 GB per device (JAX `tpu_info`: `hbm_capacity_bytes=206e9 // 2`), leaving about 24.5 GB for caches, state and the runtime.
- **Bytes streamed per step per rank at batch 1:**

  | Component | Bytes |
  |---|---|
  | MoE fixed weights | 24.8 MB per layer |
  | each routed-expert slice | 7.01 MB |
  | KDA layer | 31.3 MB |
  | MLA layer | 45.4 MB |
  | **Step total** | **6.9-8.2 GB** (2-4 expert slices per group per layer) |

  At 3.70 TB/s per device that is **1.9-2.2 ms**. The measured 4.02 ms is therefore about 50% of a weight-streaming roofline, and the remainder is collective latency and serialization.

### 2.2 Sharding across 32 devices
- **Attention** is head-parallel with 3 heads per device for both KDA and MLA, plus replicated low-rank projections. Output projections use a paired-rank layout: rank r holds rows `768*(r//2)..+768` for output half `r % 2`.
- **MoE**: router per chip, latent down 32-way, routed experts EP8 x TP4, shared experts TP32, latent up 8-way per host and replicated over the 4 hosts.
- **Vocabulary**: TP32.

### 2.3 KV cache and KDA state (megakernel vs SGLang)

| | Megakernel (`demo_kimi_dspark.py:407-415`) | SGLang (agent survey, spot-checked) |
|---|---|---|
| MLA cache | expanded per head, **contiguous per sequence**: K `[B, 24, 3, context, 256]` and V `[B, 24, 3, context, 128]` bf16 per device; row b = sequence b, no indirection | **latent, paged**, replicated: per layer `[num_pages, 64, 2, 640]` bf16 (page 128, two tokens per row; 512 latent + 64 pe + padding), as a Python list of 24 tensors (`merged-t5/sglang_tpu/kv_pool.py:670-808`) |
| Bytes per token per device | 55,296 | 30,720 |
| Page / tile size | 128-token DMA tiles over a contiguous buffer | 128-token pages; page table built on the host and passed through SMEM scalar prefetch (`attention_pallas_rpa_base.py`) |
| Maximum context | fixed `--context` (≥ 384, multiple of 128) per launch | `--context-length`; pool of about 550k tokens (CI test comment) |
| KDA conv | `[B, 69, 3, 3, 3, 128]` bf16 (time, q/k/v, head, channel) | per layer `[slots+1, 3, 1, 1152]` **f32**, with 1152 = q 384 \| k 384 \| v 384. **Same element order.** (`merged-t5/sglang_tpu/mamba_pool.py:116-191`) |
| KDA recurrent | `[B, 69, 3, 128, 128]` f32 | per layer `[slots+1, 3, 128, 128]` f32. **Identical per slot.** |
| Indexing | batch row, stacked layer axis | per-layer tensors; `state_indices` gathered from `req_to_mamba` each step; slot 0 is the null slot for padded rows |

### 2.4 Batch limits, ragged lengths, compile-time constants
- **Batch.** `stack()` and `make_decode()` accept B = 1..8 (`:3653`, `:5382`); MLA allows at most 8 rows (`:2515`); MoE and `_dot` support up to 16 (the native MXU row tile, `:44-58`, `:1316`).
- **Compile-time constants.** Each batch size is its own compile. So are context length, `sequence_rows` (1 for decode, 8 for verify), and kernel options. Recipe-specific code paths exist for B4/B6/B8 (`demo_kimi_dspark.py:97-117`).
- **Ragged lengths.** Each row has its own position, with per-row masks in MLA and loop bounds set by the maximum position in a row tile.
- **No padding mask.** A padded row still routes experts, which costs bandwidth, and still writes its cache row.
- **No bounds checks.** `disable_bounds_checks=True` means a row whose position reaches `context` performs an out-of-bounds DMA.
  - The server reserves only `rows` (8) tokens of slack (`demo_kimi_dspark.py:979`), while target-only decode advances every row in 64-step calls.
  - Generation stops only at a 64-step call boundary, so a batch whose `max_tokens` is within about 55 tokens of that budget writes KV up to about 54 positions past the end of the cache (worst case: last written position = context + 53).
  - This is an edge-case bug in their server, not in the kernel.
- **Context.** `--context` default is 2048 (`:290`); the benchmark uses 1152 (`scripts/benchmark_kimi.sh:23`), GSM8K 32768, GPQA 65536.
- **Prompt cap.** `PROMPT_CAPACITY = 8192` tokens per broadcast job (`demo_kimi_dspark.py:43`).

---

## 3. Prefill

There is no prefill kernel for Kimi (`grep -i prefill kimi/` finds none; only Qwen has one).

- **Target-only (`--target-only`):** `greedy_prefill_program` (`demo_kimi_dspark.py:862-870`) runs the batch-B decode program once per prompt token, `prompt_tokens[:, index]` at position `index`.
  - It is a host loop of `len(prompt)` launches.
  - All prompts in a batch must have the same length (`:1236-1237`).
  - For an 8k prompt that is 8,192 sequential decode launches, about 35-45 s at roughly 4-5 ms each (estimate).
- **DSpark:** `prefill()` (`:817-860`) runs the speculative-verify program (`sequence_rows=8`, all 8 rows from one sequence, MXU scores, KDA post-state for every row) once per 8-token chunk, plus the fused draft step to fill the draft's context cache.
  - The KDA input row is chained through `state_row`.
  - An 8k prompt is 1,024 launches; estimated TTFT is about 7-10 s, versus 1.9 s in SGLang.
- **Conversion to the decode layout:** none is needed, because prefill writes the decode caches directly.
  - No path exists to import caches from another engine.
  - Importing SGLang's prefill output would need:
    - MLA: latent → expanded. Recompute per layer `[kv_c_normed (512)] @ kv_b (512→768)` for k_nope and v, then append the raw 64-dim pe and 64 zeros. SGLang's cache stores exactly the normalized latent and raw pe, so this is exact up to matmul rounding.
    - KDA: f32 → bf16 for conv, then a transpose from per-layer `[slot, 3, 1, 1152]` to stacked `[row, 69, 3, 3, 3, 128]`. The recurrent state copies across unchanged.

---

## 4. Runtime model

- **Processes.** One JAX multi-controller program across 4 processes x 8 local devices. `distributed_entry.py` reads `SLURM_PROCID`, sets `TPU_WORKER_ID` and `CLOUD_TPU_TASK_ID`, and calls `jax.distributed.initialize(coordinator_address=$TP32_COORDINATOR, num_processes=4, process_id=rank, initialization_timeout=180)` (`distributed_entry.py:8-10, 38-43`). It then `runpy`s the demo.
- **Mesh.** `_logical_mesh("0,1,2,3,4")` sorts the 32 devices by `(process_index, id)` and builds `Mesh(devices, ("tp",))` (`:5438-5455`, `demo_kimi_dspark.py:369`). Every program is `jax.jit(jax.shard_map(...))`.
- **Start-up** (`demo_kimi_dspark.py:616-760`):
  - Programs are compiled in a background thread against abstract weights while the weights stream in.
  - A "pacer" runs a tiny psum every 2 s during loading, to drain the runtime's deferred work and report progress (`:585-614`).
  - The launcher says loading takes "about 3 minutes warm (up to 20 cold)" (`scripts/demo_kimi_dspark.sh:8`).
- **Decode loop, target-only** (`:520-543`, `:1225-1321`):
  - `greedy_decode` is `jax.lax.fori_loop` over `--steps-per-call` steps (default 4; 64 in the benchmark). Each step calls the decode program, takes the argmax on the device, and feeds the token back.
  - The host receives `[steps, B]` tokens once per call, applies stop rules (end tokens, `max_tokens`, stop strings, which need a tokenizer decode per row), and streams the result.
  - Rows that finish early keep computing until every row is done.
- **Requests.** `next_job()` (`:962-1021`): process 0 reads the HTTP or terminal job, then `multihost_utils.broadcast_one_to_all` sends a fixed 12 + 8192 + 2048 int32 buffer so that all hosts run in lockstep.
- **Batching is static.** The server waits for exactly `--target-batch-size` jobs (`:1323-1340`).
- **Sampling.** Target-only is greedy; sampling exists only in DSpark mode.
- **What the published per-step numbers include:** almost no host work. One host round trip per 64 steps, and no scheduler, ForwardBatch or eager sampler per token. Inside SGLang each step would add the scheduler, metadata, embedding, lm_head and sampling (about 0.1-0.5 ms; estimate).

---

## 5. Dependencies on JAX 0.11.1 and libtpu 0.0.46.1

The megakernel pins `jax[tpu]==0.11.1`, `jaxlib 0.11.1` and `libtpu 0.0.46.1` (`pyproject.toml:5`, `uv.lock:635-719`), with Python ≥ 3.12. SGLang pins jax/jaxlib 0.10.2, libtpu **0.0.47** (newer than the megakernel's), torch 2.11.0 and torch-tpu 0.1.1+release.2026.9.22 (`merged-t5/python/pyproject_tpu.toml:80-90`).

**APIs the megakernel uses, all present in 0.10.2.** Checked by diffing the published wheels in `/tmp/k3perf/src/jax0102` and `/tmp/k3perf/src/jax0111`.

- **Pallas core:** `pl.pallas_call(name=...)`, `pl.BlockSpec(memory_space=...)`, `pl.ds`, `pl.multiple_of`, `pl.when`, `pl.loop`, `pl.run_scoped`, `pl.semaphore_signal`, `pl.semaphore_wait`, `pl.DeviceIdType.MESH`. The kernel uses the `pl.*` forms; the deprecated `pltpu.*` aliases that 0.11 removed are not used.
- **Pallas TPU:** `make_async_remote_copy`, `make_async_copy`, `get_barrier_semaphore`, `VMEM`, `SMEM`, `HBM`, `SemaphoreType.DMA`, `PrefetchScalarGridSpec`, `bitcast` (also to `jnp.float4_e2m1fn`), plus ref `.bitcast` / `.reshape` / `.at`.
  - Remote `.wait()` semantics (wait_send then wait_recv) are identical in both versions.
- **`CompilerParams`:** `collective_id`, `vmem_limit_bytes`, `disable_bounds_checks` and `shape_invariant_numerics` exist with the same defaults in both versions (`jax/_src/pallas/mosaic/core.py` around line 127).
- **`pool_alias.py` private hooks:**
  - It subclasses `jax._src.state.types.ReshapeTransform` and builds `TransformedRef`.
  - It monkeypatches `jax._src.pallas.mosaic.lowering._reshape_memref` (`pool_alias.py:12-39`), whose signature `(ref, reshaper, ref_aval, ref_block_shape)` is identical in both.
  - It calls Mosaic dialect ops `tpu.reinterpret_cast` and `tpu.erase_memref_layout`, both present in jaxlib 0.10.2's `_tpu_ops_gen.py`.
  - Of the other ops it emits (`memref_bitcast`, `enqueue_dma`, `wait_dma2`, `sem_signal`, `sem_barrier`, `reduce_index`, `bitcast`), none is new in 0.11.1. The ops 0.11.1 added are `conv`, `logistic`, `matmul_lhs_fifo`, `matmul_pop_fifo` and `reduce`, and the kernel uses none of them.
- **Argmax:** f32 argmax lowering (`reduce_index`) is supported in 0.10.2.
- **Other:** `jax.shard_map(..., check_vma=False)` exists in 0.10.2.

**Measured.**
- `jax.export(..., platforms=["tpu"])` of the full stack succeeds under 0.10.2 and gives nearly the same Mosaic payload size as 0.11.1 (section 0).
- The repository's Kimi CPU tests pass under JAX 0.10.2: `tests/test_decode_megakernel_cpu.py` and `tests/test_kimi_reference.py`, 17 passed in interpret mode.

**Not provable here.**
- Whether libtpu 0.0.47's Mosaic compiler accepts and compiles the kernel. It is about 370 KB of bytecode at batch 1 and 650 KB at batch 8, and the compile time is unknown.
- The same question for the jaxlib 0.10.2 serialization of it. SGLang already ships other Pallas kernels through that combination.

**Features that must work in SGLang's torch_tpu bridge.** These come from the bridge survey. Here `TS` is the torch_tpu checkout at `/home/chandrasekhard/torch_tpu`. The existing SGLang remote-DMA kernels are `merged-t5/sglang_tpu/kernels/distributed_topk.py` (wrapped by `merged-t5/sglang_tpu/distributed_topk_pallas.py`) and `merged-t5/sglang_tpu/moe/pallas/fused_rs/gmm_fused_rs_nodedup.py`; the latter has a barrier at `:460-489` and `CompilerParams(collective_id=0)` at `:1855-1861`.
1. Mosaic remote DMA inside a `shard_map` over the full world mesh. Supported: the mesh is torch_tpu's mock-device mesh in torch rank order; `axis_index` equals the torch rank, which torch_tpu tests (`TS/tests/pallas/pallas_communication_test.py:362-410`); MESH device ids become partition ids through the executable's device assignment.
2. A barrier semaphore with `collective_id`. torch_tpu passes it through untouched; SGLang's fused reduce-scatter MoE uses `collective_id=0` with a barrier, at EP16 across hosts.
3. **In-place aliasing or donation of large state tensors from a `shard_map` kernel.** No existing SGLang kernel combines the two, so this is unverified.
4. Scoped VMEM of 64 MiB, the whole core. SGLang kernels use up to 0.95 x VMEM.
5. A single mesh device assignment for all Pallas kernels in one compiled graph.
6. The axis name: SGLang meshes use `"model"` while the kernel hardcodes `"tp"` (about six places). Either rename or build a mesh named `"tp"`.

---

## 6. Correctness evidence in the repository

**Tests.**
- `tests/test_decode_megakernel_cpu.py` (tests at `:19`, `:35`, `:63`, `:78`, `:129`) checks individual pieces only:
  - MXU padding in `_dot`, `_probability_value_mxu`, `_attention_scores`;
  - the residual-mixture row;
  - the interpreted `_quantized_expert_dot` for MXFP4 and reblocked FP8 against dense dequantization.
- `tests/test_kimi_reference.py` (tests at `:12`, `:23`, `:35`, `:57`, `:112`) checks:
  - router bias semantics;
  - MXFP4 unpack;
  - the attention-residual formula;
  - the KDA step against a direct delta rule;
  - BF16 sigmoid rounding.
- **No test executes `kda()`, `mla()`, `moe()` or `stack()`.** None compares against the Hugging Face or PyTorch model, and there are no golden logits. The 32-device `cpu32` test covers DSpark only (`tests/test_dspark_cpu.py:35`).
- The README says these tests "validate numerical behavior, not TPU-specific Mosaic lowering, layouts, VMEM capacity, DMA scheduling, collectives, or performance."
- `kimi/load.py:493` calls the layout "golden-validated", but no golden validation ships with the repository.

**Eval scripts.**
- `scripts/eval_kimi_gsm8k.sh`: lm-eval `gsm8k_cot`, 5-shot, chat template, think mode, `max_gen_toks 32000`, context 32768, concurrency 1, **DSpark by default** (it does not pass `--target-only`).
- `scripts/eval_kimi_gpqa_diamond.sh` with `gpqa_diamond_reasoning.yaml`: 0-shot, greedy, think mode, `max_gen_toks 64000`, context 65536.

**Recorded scores:** none in the repository. There are no result files, and the README has no accuracy numbers; it points to the blog post for analysis.

For comparison, SGLang's real-weight validation on 2026-10-08 scored 184/198 on GPQA and 295/300 on GSM8K.

**Numerics that will differ from SGLang:**
- BF16-rounded transport inside every reduction;
- the relaxed residual association order in the tuned recipe;
- expanded vs absorbed MLA;
- hilo P.V;
- Mosaic argmax tie order;
- the FP8 reblock, which is lossless.

---

## 7. Running it standalone

**What the scripts do.**
- `scripts/demo_kimi_dspark.sh` takes a 4-node Slurm job. It runs `srun --nodes=4 --ntasks=4 --ntasks-per-node=1` with these environment variables (`:37-42`):
  - `TPU_PROCESS_BOUNDS=1,1,4`, `TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1`, `TPU_VISIBLE_CHIPS=0,1,2,3`;
  - `TPU_WORKER_HOSTNAMES=<4 hosts>`, `TP32_COORDINATOR=<host0>:12377`;
  - `JAX_PLATFORMS=tpu`, `HF_HUB_OFFLINE=1`, `JAX_COMPILATION_CACHE_DIR=<repo>/.jax_cache`;
  - `XLA_FLAGS=--xla_allow_excess_precision=false`, `PYTHONPATH=<repo>`;
  - then `.venv/bin/python distributed_entry.py demo_kimi_dspark.py "$@"`.
- So the topology is 4 processes stacked along z, each owning a 2x2x1 block of chips (4 chips = 8 TensorCore devices). That is the same 2x2x4 slice our 4-host tpu7x runs use.
- `scripts/benchmark_kimi.sh` starts the server in the background with `--serve 0.0.0.0:$PORT --context 1152 --max-tokens 1024 --chat none --target-only --target-batch-size $B --steps-per-call 64 --progress log` (`:93-104`).
  - It drives it with `vllm-bench` (random dataset, `--random-input-len 1 --random-output-len 1024 --ignore-eos --max-concurrency B`, 4 x B prompts) (`:123-144`).
  - It parses the server's `target_batch: ...` lines to compute decode tok/s = Σ(generated − B) / Σ seconds (`:153-188`).
  - It needs `squeue`, `scontrol`, `curl` and `vllm-bench`.

**Environment.**
- `uv sync` provides Python 3.12+, `jax[tpu]==0.11.1` and `libtpu 0.0.46.1`, separate from SGLang's image.
- `.env` may set:
  - `HF_HOME`;
  - `KIMI_PRESHARDED_WEIGHTS`, which is a reader only; the repository has no writer for the target model's pre-sharded format;
  - `KIMI_CHECKPOINT_REVISION`.
- `--checkpoint` accepts a local directory (`model_paths.py:13-15`), so `--checkpoint /mnt/kimik3-ssd/kimik3` works. Its tokenizer files (`tiktoken.model`, `tokenization_kimi.py`, `tokenizer_config.json`) are present.
- The loader reads the Hugging Face shards directly and reblocks gate/up on the device, one layer at a time (`kimi/load.py:830-922`). Each host reads its 8 ranks' slices; each host covers 224 experts, about a quarter of the expert bytes (around 360 GB).
- **Gotcha:** even with `--target-only`, `--draft` (default `RedHatAI/Kimi-K3-speculator.dspark`) is resolved before the target-only branch (`demo_kimi_dspark.py:347-348` vs `:373`). With `HF_HUB_OFFLINE=1` it fails unless that repo is cached. Passing `--draft /mnt/kimik3-ssd/kimik3` or any existing directory avoids it.
- The compile cache path defaults to the repository directory; on GKE point `JAX_COMPILATION_CACHE_DIR` at a writable volume.

**Running on a 4-host GKE JobSet instead of Slurm.** One pod per tpu7x host, 4 chips (8 TensorCore devices) each. The code's assumptions are 4 processes, 8 devices per process and exactly 32 devices (`:5443-5445`).
1. Set `SLURM_PROCID=$JOB_COMPLETION_INDEX` (0..3), or patch `distributed_entry.py:8` to read the index.
2. Set `TP32_COORDINATOR=<pod-0 DNS name>:12377`, using the JobSet headless service name.
3. Set `TPU_WORKER_HOSTNAMES` to the 4 pod DNS names.
4. Keep `TPU_PROCESS_BOUNDS=1,1,4`, `TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1`, `TPU_VISIBLE_CHIPS=0,1,2,3`, unless the GKE TPU webhook already injects matching values. Check that each pod sees 8 local devices and 32 global (the entry script prints `ready rank=... local=... global=...`).
5. Run on every pod: `python -B distributed_entry.py demo_kimi_dspark.py --serve 0.0.0.0:8000 --target-only --target-batch-size B --steps-per-call 64 --context C --max-tokens 1024 --chat none --progress log --checkpoint /mnt/kimik3-ssd/kimik3 --draft /mnt/kimik3-ssd/kimik3`.
6. Benchmark against pod 0 port 8000 with `vllm-bench` or `bench_serving` using random ids, and compute decode tok/s from the `target_batch:` log lines as the script does. `vllm-bench` can use the local checkpoint directory as `--tokenizer`.
7. Every pod must mount the checkpoint read-only, for example a ReadOnlyMany disk or a FUSE bucket. The loader's coalesced 8-thread reads are designed for FUSE (`kimi/load.py:107-116`).
8. **8k-context measurement.** Use `--context 9344` (73 x 128) and pass `--random-input-len 8192` to `vllm-bench`; `PROMPT_CAPACITY` allows exactly 8192.
   - Prefill is token by token, about 40 s per batch (estimate).
   - The budget check `9344 − 8192 − 8 = 1144 ≥ 1024` passes.
   - This gives the comparable 8k/1k TPOT that the published numbers lack.

---

## 8. Port options for SGLang-torchtpu

### 8.0 SGLang facts that drive the choice (from the survey agents, spot-checked)

**Model and collectives.**
- `merged-t5/python/sglang/srt/models/kimi_k3.py` uses attention TP32 with no DP attention, absorbed MLA (`w_kc [3,128,512]`, `w_vc [3,512,128]`), and EP32 MoE (28 full experts per device).
- Every collective is an XLA all-reduce, 280 per step.
- Replicated per device:
  - router;
  - latent down projection: 7168 x 3584 bf16, **51 MB per layer read by every device every step**;
  - MLA a-projections.
- The latent up projection is row-sharded behind `SGLANG_TPU_K3_SHARD_LATENT_UP`.

**Experts on the device.**
- w13 `(28, 1792, 6144)` u8 packed fp4 with **f32** scales `(28, 112, 1, 6144)`.
- w2 `(28, 1536, 3584)` u8 with f32 scales.
- About 49.5 GiB per device. The gmm_v2 kernel runs them as W4A16.

**Profile of the 18.1 ms step**, from `/home/chandrasekhard/sglang-torchtpu/docs/kimi_k3_tpu_architecture_and_benchmarks.md:474-478` (the doc is on the `kimi-k3-perf` branch; the TPOT has since dropped to 16.9 ms):

| Item | Time |
|---|---|
| All-reduce, 279 collectives | about 7.5 ms, of which 2.9 ms is the MoE combine waiting for the busiest EP rank |
| HBM-bound GEMVs | about 5 ms, including 2.1 ms for the replicated front GEMV |
| Host / eager graphs | about 0.1 ms |

**Compilation.** The whole forward is one `torch.compile(fullgraph=True)` program per decode bucket. Pallas kernels are inlined as `tpu_custom_call`s through `jax.export` (`merged-t5/sglang_tpu/pallas_jax_op.py`).
- In-place cache updates use `donate_argnums` plus `input_output_aliases` plus a torch custom op with `mutates_args`.
- A donated tensor must be a whole tensor, not a view (`attention_gdn_pallas.py:163-168`).
- Caches are Python lists of per-layer tensors.
- There is no multi-step decode on the device for K3.

### 8.0b Where the megakernel's advantage comes from (by component, mapped to SGLang's profile)
1. **Collectives.** It uses in-kernel DMA, one barrier per step, BF16-packed payloads and two-level or one-shot patterns, overlapped with weight DMA. SGLang runs 280 separate XLA all-reduces, about 7.5 ms.
2. **No replicated front GEMV.**
   - The megakernel shards the latent down projection 32 ways (1.8 MB per rank per layer) and all-gathers the 3584-wide latent by DMA while top-k runs. SGLang reads 51 MB per layer on every device, part of its 2.1 ms front GEMV.
   - SGLang's doc records that an XLA version of this idea lost (24.7 µs vs 19.2 µs, `/home/chandrasekhard/sglang-torchtpu/docs/kimi_k3_tpu_architecture_and_benchmarks.md:469-470`). The difference is that the megakernel's all-gather is in-kernel and overlapped.
   - The router is split per chip (7.3 or 5.5 MB per rank per layer instead of 12.8 MB).
3. **Load balance.** EP8 x TP4 spreads each expert over 4 ranks, which shrinks the busiest-rank tail. That tail is SGLang's 2.9 ms combine wait.
4. **Expert math.** FP8 gate/up skips the FP4 unpack, and two experts share one MXU pass (`_paired_expert_gate_up`).
5. **Cross-layer prefetch.** The next layer's weights stream during the current layer's reductions. Separate ops cannot do this.
6. **Host.** 64 decode steps run per device call. This matters less for SGLang, whose host and eager overhead is already about 0.1 ms.

Items 1-4 can move into SGLang as separate ops (Rank 1 below). Item 5 needs a single kernel (Rank 2), and item 6 needs scheduler changes.

### Option ranking (best risk-adjusted first)

#### Rank 1 — (c) Port fused pieces in stages, starting with a per-layer MoE op
**What.**
- Wrap the megakernel's `moe()` (`:1276-2155`) as one torch_tpu custom op per MoE layer (hidden `[B, 7168]` → output `[B, 7168]`) via `cpu_export_jax_op` with a full-world `shard_map`.
- It contains the router (split per chip), the 32-way-sharded latent down projection with a DMA all-gather overlapped with top-k, the shared expert, the EP8 x TP4 routed stream with 4-slot prefetch, the 7+3 reductions, the latent norm, and the per-host latent up projection.
- It **owns no state**, so it avoids the unverified "shard_map plus donation" combination.
- Next steps:
  - an attention-tail op: paired-rank `o_proj` plus the one-shot 16-peer all-reduce, also stateless;
  - optionally per-layer attention ops that update their own layer's whole cache tensors, which fits SGLang's per-layer pool lists.

**Blockers.**
1. The bridge must carry Mosaic remote DMA plus a barrier at TP32 across 4 hosts. The pattern exists for EP16 fused reduce-scatter but has not been shown at TP32.
2. **Expert layout.** Both copies cannot fit in 103 GB. SGLang's prefill MoE must consume the same EP8 x TP4 layout: 112 local experts at quarter width; MXFP4 down with E8M0 scales; gate/up as FP8 (72 GB) or MXFP4 (45 GB).
   - Mathematically this is the same sum. gmm_v2 needs a new weight format and a group-of-4 ownership rule.
   - Keeping gate/up in MXFP4 saves 27 GB per device at some cost in decode kernel speed, which is unmeasured.
3. The `"tp"` axis name, and batch ≤ 16 rows per op. Decode buckets above 8-16 would loop or fall back.
4. Each separate op pays a barrier (31 signals) and cannot prefetch the next layer's weights.

**Effort:** medium for the MoE op plus a weight converter plus the prefill MoE format change, about 3-6 engineer-weeks. Attention-tail ops are small after that.

**Correctness risk:** low to medium.
- Continuous batching, paged KV, chunked prefill and multimodal are untouched, because caches stay SGLang's.
- Numerics change through BF16 transport and argmax tie order, so it needs GSM8K and GPQA reruns.
- Padded rows need a valid-row mask, otherwise they waste expert bandwidth (a small kernel change).

**Expected gain (estimate).** The MoE block holds 184 of the 280 collectives, the 2.9 ms imbalance (EP8 x TP4 cuts the expected busiest-rank excess roughly 3-4x versus EP32 at 16 picks), and most of the 2.1 ms replicated front GEMV.
- About −5 to −8 ms at batch 1, so 16.9 → about 9-12 ms.
- Adding the attention tails: about 7-9 ms.

#### Rank 2 — (a) The whole megakernel as one torch_tpu custom op per decode step, with SGLang owning the caches
**What.** SGLang computes the embedding, calls the stack op with `[B]` positions and slot indices, then runs its own final attention residual, norm, lm_head and sampler. Prefill stays SGLang's.

**Cache adapters needed.**
1. **MLA:**
   - Preferred: rewrite `mla()` as absorbed-latent attention over SGLang's paged pool.
     - One 128-token page `[64, 2, 640]` per DMA tile, page ids scalar-prefetched.
     - Scores `[3*rows, 576] x [576, 128]` and P.V `[.,128] x [128, 512]` on the MXU, then `w_vc`.
     - The new latent row is written into the page at `seq_len − 1`.
     - This reads about 45% fewer bytes per token than the expanded cache and matches SGLang's absorbed numerics.
   - Alternative: keep the expanded cache as a second, megakernel-owned paged pool, filled by a latent→expanded conversion after each prefill chunk. That costs about 1.8x more cache memory than the latent pool and an extra copy.
2. **Per-layer tensors:**
   - Either stack the 24 KV tensors and the 2 x 69 KDA tensors along a layer axis (SGLang's own KDA and MLA kernels would then take the stacked buffer plus a layer offset and donate the whole buffer);
   - or pass 162 operands and select by layer with `pl.when` branches around the DMA issue code.
3. **KDA:**
   - add slot indirection: scalar-prefetched `state_indices`, with slot 0 for padded rows;
   - read and write f32 conv state, or store bf16; the values are bf16-exact either way;
   - the recurrent layout already matches.
4. **Weights:** megakernel layout for everything. Experts must be shared with prefill (as in Rank 1); dense weights can be duplicated (6.2 GB).

**Blockers.**
- Shard_map plus donation in the bridge (unverified).
- Batch ≤ 8 per kernel call. SGLang's decode buckets go up to 64, so larger batches need multiple calls (each re-streams all weights) or SGLang's own path.
- A long Mosaic compile per bucket (unknown).
- Rank-to-chip ordering must give "rank^1 on the same chip" for the published speed.
- `disable_bounds_checks`, so positions must be validated.
- Every previous blocker from Rank 1.

**Effort:** high, about 2-4 engineer-months. That covers the MLA rewrite, cache plumbing, weight loader, bridge work, scheduler integration and TPU debugging of a 93-layer communicating kernel.

**Risk:** medium to high.
- Continuous batching: fine with slot and page indirection.
- Chunked prefill: unaffected, because prefill stays in SGLang.
- Padding: needs a mask.
- Multimodal: fine for decode, since prefill handles images.
- Speculative decoding: the verify variant exists (`sequence_rows=8`), but SGLang's KDA path rejects target-verify today.

**Expected gain (estimate).** The kernel at an 8k context is about 4.4-5.4 ms at batch 1 (4.02 ms plus the attention estimate), plus SGLang's per-step overhead of about 0.3-0.5 ms.
- **Batch 1: about 5-6 ms TPOT (2.8-3.4x).**
- Batch 4: about 9-14 ms (1.6-2.4x). Moving MLA decode scores to the MXU could recover several ms at batch 4.
- A layer-granular variant (one op per layer receiving that layer's whole cache tensors) avoids stacking. It costs 93 barriers and no cross-layer prefetch: about +1-2 ms, so roughly 6-8 ms at batch 1.

#### Rank 3 — (b) Run the megakernel's JAX decode loop as a separate engine fed by SGLang's prefill
**What.** Keep the JAX program as is: the multi-controller process group owning the 32 devices. Hand it caches produced by SGLang's prefill.

**Blockers.**
- A TPU chip can be owned by only one process and runtime. torch_tpu and JAX must not share the PJRT plugin in one process (torch_tpu warns of deadlocks). The weights also cannot co-reside (78.5 + about 60 GB > 103 GB).
- So this only works as **prefill/decode disaggregation onto separate slices**: SGLang prefill on one slice, the megakernel decode on another. That needs a KV/state transfer path plus latent→expanded conversion (section 3), or the megakernel's own token-by-token or 8-token prefill, with a TTFT of about 35-45 s or 7-10 s for 8k prompts.
- The decode engine also needs a new scheduler. Today it batches statically, requires equal prompt lengths, is greedy only, caps batch at 8, runs one batch at a time and has no slot reuse. The kernel already supports per-row positions, so slot-based continuous batching between device calls is a driver change.
- No multimodal support.

**Effort:** medium for a benchmark-only deployment (run it as is). High for production: transfer, conversion, scheduler, sampling, a second slice.

**Risk:** low to the SGLang code base, high operationally.

**Expected gain.** It keeps the full kernel speed and the on-device multi-step loop: about 4.0 ms at 1k context, and an estimated 4.4-5.4 ms at 8k for batch 1. But it doubles the hardware for disaggregation, or pays a terrible TTFT.

### 8.1 Common first steps, which decide between (a) and (c)
1. Standalone measurement on our slice (section 7): TPOT at 1k and 8k context for batch 1 and 4, plus a GSM8K and GPQA subset in target-only or DSpark mode.
2. Bridge spike:
   - From one SGLang worker per device, build `get_pallas_mesh` with axis `"tp"` (or rename) and register `stack()` with synthetic weights in the `abstract_weights` shapes. Use the bridge's in-place pattern for the four states: `custom_jax_kernel(..., donate_argnums, input_output_aliases)` plus a torch custom op with `mutates_args`, as in `merged-t5/sglang_tpu/kv_page_write.py:42-83`. The simpler `cpu_export_jax_op` supports only `donate_argnums`. The kernel must also sit inside a full-world `shard_map`.
   - Confirm that libtpu 0.0.47 compiles it, that it runs without hanging, and that its rank-to-device mapping keeps `rank^1` on the same chip.
   - Time it against the standalone run.
3. If the spike works, implement the Rank 1 MoE op on the shared EP8 x TP4 expert layout. Keep the code ready to grow into (a).

---

## Appendix A — numbers used
- **Per-step times** from the README figure: 1000/249 = 4.02 ms; 2000/392 = 5.10; 4000/515 = 7.77; 8000/865 = 9.25.
- **GB200 baseline:** 7.87 / 8.81 / 10.72 / 12.58 ms.
- **TPU7x per device** (JAX `tpu_info`): 103 GB HBM, 3.70 TB/s, 64 MiB VMEM, 1 MiB SMEM, 1.155 bf16 PFLOP/s.
- **Weight totals per device:** megakernel 78.48 GB (FP8 gate/up) or 51.4 GB (MXFP4 gate/up); SGLang roughly 60 GB (49.5 GiB experts per the agent survey, plus dense).
- **Cache per token per device:** megakernel MLA 55,296 B; SGLang latent 30,720 B.
- **KDA state per sequence:** megakernel 14.0 MB; SGLang 14.5 MB.
- **Export logs:** `/tmp/k3perf/export_b1_jax0102_ckpt.log`, `/tmp/k3perf/export_b8_jax0102_ckpt.log`, `/tmp/k3perf/export_b1_jax0111_ckpt.log`, `/tmp/k3perf/export_b8_jax0111_ckpt.log`.
