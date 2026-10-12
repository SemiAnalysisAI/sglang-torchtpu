# Plan for porting Inferact's Kimi-K3 decode megakernel into SGLang-torchtpu

**Path conventions.**
- `merged-t5/…` means `/home/chandrasekhard/dagent/merged-t5/…`. This is SGLang main with all Kimi-K3 PRs merged. Five of its files are byte-identical to the tree that produced today's baseline TPOT: `kimi_k3.py`, `pallas_jax_op.py`, `gmm_v2.py`, `unquant.py` and `runtime_context.py`.
- `tpu-megakernels/…` means `/home/chandrasekhard/tpu-megakernels/…` at commit 4048f08. Six key files are byte-identical to the snapshot `/mnt/workspace/dagent/snap/megakernels` that the prepared run scripts use.
- `torch_tpu/…` means a file inside the torch_tpu 0.1.1+release.2026.9.22 wheel. An extracted copy is at `/mnt/kimik3-ssd/dryrun/site/torch_tpu`.
- `vllm-torchtpu/…` means `/home/chandrasekhard/vllm-torchtpu/…` at f2a9e2536.

Nothing in this plan was run on a TPU. Every TPOT number other than the measured baseline is an estimate.

## 0. Baseline and what was re-checked for this plan

**Baseline (measured).** The four-host real-weight benchmark on 2026-10-12 ran current main plus the five open Kimi-K3 PRs, with the fused o_norm, attention-residual kernel and unsafe guard-skip flags on. Results are in `/mnt/workspace/dagent/results/e2e/e2e-p1/*/allon/bench/`.

| Workload | TPOT | TTFT |
|---|---|---|
| Batch 1, 8,192-token prompt, 1,024 output tokens | 15.92 ms | 2.27 s |
| Batch 4, 1,024-token prompts, 1,024 output tokens | 21.05 ms (median) | 1.58 s (median) |

**Re-checked in this pass (CPU only).**
1. **Standalone MoE op export.** `/tmp/designc_scratch/moe_op_export.py` wraps the megakernel's `moe()` in its own `pallas_call` with SGLang's uint8 expert tensors, bitcast to uint32 inside the kernel. Under JAX 0.10.2 at batch 1 and batch 8 it exports to:
   - exactly one `tpu_custom_call`, with no `all_reduce` or `all_gather`;
   - `has_communication`, `collective_id` 21 and 64 MiB scoped VMEM;
   - 12 inputs and 1 output.
2. **The prepared bridge spike passes its CPU dry run.** I ran `/mnt/kimik3-ssd/dryrun/dryrun_bridge.py 1 1152` with torch 2.11 on CPU, the torch_tpu wheel and JAX 0.10.2. It printed `DRYRUN_OK`: op registration, export and FakeTensor shapes all pass, with 43 inputs and 78.6 GB of per-rank inputs.
3. **The libtpu 0.0.47 binary contains both hang-diagnostic flags:** `xla_tpu_debug_sflag_wait_timeout_ms` and `xla_tpu_debug_sflag_wait_shalt_on_detection`. I checked with a string search of `/mnt/kimik3-ssd/dryrun/site/libtpu/libtpu.so`.
4. **The real four-host rank-to-device map exists.** The 2026-10-12 server logs printed `rank_to_device = [0,1,4,5,6,7,2,3, 8,9,12,13,14,15,10,11, 16,17,20,21,22,23,18,19, 24,25,28,29,30,31,26,27]`, topology `2,2,4,2`, via `merged-t5/sglang_tpu/compile_cache.py:105-140`. So:
   - ranks 2k and 2k+1 sit on the device pair (2m, 2m+1);
   - ranks 8h..8h+7 sit on devices 8h..8h+7.

   Earlier analyses said no four-host map existed locally.
5. **No current decode profile exists.** The 2026-10-12 "decode" profile captured only three launches of 183-307 ms each. That matches prefill chunks: 2.27 s TTFT over 8 chunks of 1,024 tokens. Its flags-off variant then died with "No space left on device" on `/mnt/workspace`.
6. **Defects in the prepared spike script** are listed in §4.1.

---

## 1. Recommendation and why

**Recommendation: port the megakernel in stages, one layer-level op at a time, inside SGLang.**
- The first deliverable is a *stateless per-layer MoE-block op*. It is the megakernel's `moe()` as its own communicating Pallas custom call, one call per MoE layer, inside SGLang's existing compiled decode graph.
- It runs on SGLang's own expert bytes after switching the expert layout to 8 expert groups × 4-way channel split (`--ep-size 8` at TP32).
- Later steps grow the op:
  1. a per-layer op covering everything between two attention cores;
  2. a four-layer group op that runs the attention cores in-kernel over SGLang's caches;
  3. optionally, one op per decode step.

**Why this design:**
1. **Serving semantics are untouched.**
   - SGLang keeps its paged latent MLA cache, per-layer KDA state pools, scheduler, chunked prefill, multimodal path and sampler.
   - Buckets above the op limit, and all prefill, fall back to SGLang's MoE on the same bytes.
   - A flag turns the op off.
2. **It fits in HBM.**
   - SGLang's packed-FP4 bytes are uint8 `[112,1792,1536]` and `[112,384,3584]`. Under the HBM-ref bitcast that gmm_v2 already performs in production (`merged-t5/sglang_tpu/moe/pallas/gmm_v2.py:997-1008`), they are exactly the megakernel's uint32 words `[112,448,1536]` and `[112,96,3584]`. The earlier analysis checked word order and nibble order.
   - The new per-rank copies add only 5.45 GB per device (table in §2.3).
3. **No donation is needed.** The first two deliverables own no state, so they avoid the mesh-plus-donation path that has never run on device.
4. **It goes after the biggest measured cost.**
   - On TPU, each MoE layer issues two XLA all-reduces: the latent at `merged-t5/python/sglang/srt/models/kimi_k3.py:1020` and the shared experts at `:1261`.
   - Each MoE layer also reads replicated latent down/up projections (`:649-667`; the merged front is CUDA/HIP-only at `:715`). That is 9.46 GB per step per device.
   - The older 18.1 ms profile put all-reduce at about 7.5 ms, about 2.9 ms of it busiest-expert-rank wait, and the replicated front GEMV at 2.1 ms (`/home/chandrasekhard/sglang-torchtpu/docs/kimi_k3_tpu_architecture_and_benchmarks.md:475-477`).
5. **Most of the work is checkable on CPU.** The TPU-only questions fit into one run per milestone.

**Rejected: replace the language model with the megakernel only.**
- **Prefill is slow and blocks decode.**
  - Prefill would run through the 8-row speculative-verify kernel at about 8 tokens per 9-12 ms. An 8k prompt would take about 9-12 s to first token, against 2.27 s today.
  - SGLang runs prefill before decode (`merged-t5/python/sglang/srt/managers/scheduler.py:4031-4033`), so every arrival stalls all running decodes for the whole prefill.
- **Concurrency is capped at 8 with no fallback.**
  - The two weight sets cannot co-reside: 78.5 GB with FP8 gate/up (or 51.4 GB with MXFP4) plus about 64 GB, on a 103 GB device.
  - The 64-request accuracy recipe (`merged-t5/benchmark/tpu/recipe/configs/kimi_k3/accuracy.py`) cannot run.
- **Context is fixed per row at launch**, reserving 55,296 B per token per row.
- **No correctness evidence.** Every token goes through a kernel whose repository has no end-to-end test or golden logits.
- **Untested in-place path.** 15-18 GB of state would have to stay in place through the never-run mesh-plus-donation path.

**Rejected: SGLang prefill plus a full megakernel decode over SGLang's caches.**
- **The critical path is about 10 weeks of kernel surgery that only a TPU can verify.**
  - Paged absorbed-latent MLA with a 24-way per-layer buffer choice, inside a kernel that already claims all 64 MiB of VMEM (`tpu-megakernels/kimi/decode_megakernel.py:5153`).
  - Per-row KDA DMAs. That is the pattern SGLang found corrupting KDA state at TP32 on v7: several DMA starts drained by one aggregate wait (`merged-t5/sglang_tpu/attention/pallas/gdn/wrapper.py:597-608`).
- **Untested in-place path.** 26 aliased operands (about 17 GB) go through the never-run mesh-plus-donation path, and a dropped alias copies the whole pool every step.
- **Prefill changes first.** It moves all prefill MoE to a new stacked layout with a new gmm_v2 path before any decode gain.
- **Late and under-budgeted validation.** The first integrated test would come around week 8, and three runs are not credible.
- Its paged-latent MLA core and per-row slot indirection are kept as the core of the later four-layer group op.

**What would change the recommendation.** If the next run shows all three of the following, skip the per-layer op between attention cores and go to the group or whole-step op sooner:
- the whole-step bridge op is within about 10-15% of the native kernel;
- states stay in place (HBM flat, no state copies in the HLO dump);
- a per-layer MoE call costs more than about 60 µs at batch 1.

---

## 2. Architecture of the chosen design

### 2.1 Invariants for every stage
- **Execution model is unchanged.**
  - 32 single-device workers at TP32.
  - One `torch.compile(fullgraph)` program per decode bucket via `KimiK3VLDecoderTpuCompilePolicy` (`kimi_k3.py:3836-3843`).
  - Each op is a torch custom op that lowers to one `tpu_custom_call` inside that program.
- **Mesh.**
  - `torch_tpu get_pallas_mesh(("tp",), (32,))` in torch-rank order. Assert at build time that its ids equal `get_device_assignment()` (precedent: `compile_cache.py:105-140`).
  - `axis_index("tp")` is then the torch rank, so the kernel's hard-coded `"tp"` axis needs no rename.
  - This deliberately diverges from vllm-torchtpu, which orders its EP mesh by ascending device id for SparseCore all-gather speed (`vllm-torchtpu/src/vllm_torchtpu/distributed/ep_mesh.py:151-163`). The megakernel derives expert ownership from `axis_index` (`decode_megakernel.py:191-194`) and SGLang loads experts by torch rank, so an ascending-id mesh would need vllm's `ep_rank_order` translation inside the kernel.
- **Placement.** The logged map (§0) keeps rank r and r^1 on device pair (2m, 2m+1) and ranks 8h..8h+7 on one host. That is the megakernel's own assumption (`decode_megakernel.py:5438-5455`). Same-chip status still needs confirming from `core_on_chip` in the next run.
- **Every operand must be a genuine base tensor of its exact dtype and final shape.** Build it on the host and move it once, or produce it with a whole-tensor op. Dtype views, slice-assigned buffers and rank-changing views are re-materialized or copied on every compiled call (`merged-t5/sglang_tpu/kv_pool.py:52-63`, `sglang_tpu/kda_layout.py:48-51`, `python/sglang/srt/layers/quantization/mxfp4_tpu_util.py:298-299`).
- **Vendored code lives in `sglang_tpu/`**, because that package is hashed into the compile-cache key (`sglang_tpu/compile_cache.py:30-36`). `pool_alias.py` must come along: `moe()` calls `pool_alias.view` (`decode_megakernel.py:1810`). Its monkeypatch defers to JAX's original for every other transform (`tpu-megakernels/pool_alias.py:15-41`).
- **All guards are static per compiled graph.** Forward mode and padded batch are part of the graph key, and recompiles after warmup are fatal (`merged-t5/sglang_tpu/tpu_platform.py:1078-1087`).

### 2.2 Expert layout: 8 expert groups × 4-way channel split (`--tp-size 32 --ep-size 8`)
- **Rank arithmetic.** `moe_ep_rank = tp_rank // 4`, `moe_tp_rank = tp_rank % 4` and `moe_tp_size = 4` (`merged-t5/python/sglang/srt/runtime_context.py:128-133, 154`). This is exactly the megakernel's ownership: experts `[112·(r//4), +112)`, channel quarter `r%4`.
- **Loading.** `FusedMoE` narrows w1/w3/w2 by `moe_tp_rank` (`python/sglang/srt/layers/moe/fused_moe_triton/layer.py:732-880`).
- **TPU forward.** It passes global expert ids with `expert_group_offset = 112·moe_ep_rank` (`python/sglang/srt/layers/quantization/unquant.py:1373-1374`, reached from `mxfp4.py:1615-1627`).
- **Per-rank tensors.** `process_mxfp4_moe_weights_native` (`mxfp4_tpu_util.py:227-369`) produces the tensors below. K3's gate/up is not interleaved (`:403-406`). The total bytes equal today's 32-way expert parallelism.

  | Tensor | Shape and dtype |
  |---|---|
  | `w13_weight` | uint8 `[112,1792,1536]`, gate 768 then up 768 |
  | `w13_weight_scale` | f32 `[112,112,1,1536]` |
  | `w2_weight` | uint8 `[112,384,3584]` |
  | `w2_weight_scale` | f32 `[112,24,1,3584]` |
- **Partial sums** across the 4 ranks of a group are added by the existing 32-rank latent all-reduce (`_routed_needs_reduce`, `kimi_k3.py:758-760`; `:1015-1020`).
- **Side effects:**
  - gmm_v2 walks 112 quarter-width groups, using untuned tiles (the lookup falls back to a heuristic).
  - Expert-parallel-local reads (`sglang_tpu/ep_local_expert_reader.py:57-100`) load whole experts per rank, so the 4 ranks of a group read the same 112 experts.
  - Kimi-K3 has never been run with `--ep 8`.

### 2.3 Per-layer MoE-block op (first shippable speedup)

**Placement.** The op hooks into `KimiK3MoE.forward` (`kimi_k3.py:1482-1532`), immediately before the `_forward_unfused` call at `:1514`. It is used only when all of these hold, statically:
- TPU, with the flag on;
- `forward_batch.forward_mode.is_decode()`;
- padded rows ≤ `SGLANG_TPU_K3_MOE_BLOCK_OP_MAX_BS` (default 8). `moe()` itself allows 1-16 rows (`decode_megakernel.py:1316`);
- TP 32 and expert-parallel size 8;
- none of: DP attention, pipeline parallelism, sequence-parallel MoE, deterministic inference, the expert-distribution recorder, routed-expert capture.

Consecutive MoE ops are always separated by the next layer's o_proj XLA all-reduce (`kimi_k3.py:2868-2872`, and `self.mlp(...)` at `:2936`). That all-reduce is a global ordering point.

**Per-rank signature.** P() partition specs, local shapes, explicit `jax.Array` parameters. Torch schema inference rejects `*args`, so use a synthesized `inspect.Signature`.

| Input | Shape / dtype | Source |
|---|---|---|
| `hidden` | bf16 `[B,7168]` (VMEM) | post-attention-layernorm output |
| `live_rows` | int32 `[1]` (SMEM) | `forward_batch.moe_num_token_non_padded()` (`kimi_k3.py:1519-1521`) |
| `router` | bf16 `[7168,896]` | transpose of `gate.weight` (new copy) |
| `router_bias` | f32 `[1,896]` (VMEM) | `e_score_correction_bias` |
| `latent_down` | bf16 `[7168,128]` | columns 128r.., zero for r ≥ 28 (new copy) |
| `latent_up` | bf16 `[3584,1024]` | output block r%8; block 7 zero (new copy) |
| `latent_norm` | f32 `[1,4096]` (VMEM) | `routed_expert_norm`, first 3584 real |
| `shared_gu` / `shared_down` | bf16 `[7168,384]` / `[192,7168]` | transposes of SGLang's TP32 shard |
| `expert_gu` | uint8 `[112,1792,1536]` | **SGLang `w13_weight`, used in place** |
| `expert_gus` | uint8 E8M0 `[112,112,1536]` | from f32 scales (new copy) |
| `expert_down` | uint8 `[112,384,3584]` | **SGLang `w2_weight`, used in place** |
| `expert_ds` | uint8 E8M0 `[112,24,3584]` | new copy |
| **Output** | bf16 `[B,7168]` | routed plus shared output, identical on all ranks; the hook adds `prefix_sum` |

**Kernel.**
- The vendored `moe()` (`decode_megakernel.py:1276-2155`) runs in its own `pallas_call`. The prototype is `/tmp/designc_scratch/moe_op_export.py`.
  - `synchronize_devices=True` gives a 31-peer barrier at entry (`:1498-1500`).
  - `routing_projection_is_prefetched=False` (`:1511`).
- It contains:
  - router halves computed per chip;
  - top-16 as sequential argmax;
  - the 128-column latent down projection plus a 31-peer DMA all-gather (`:1517-1547`);
  - the routed-expert stream (EP8 × TP4 ownership) with 4 VMEM prefetch slots;
  - shared experts;
  - reductions over 7 local peers, then 3 hosts;
  - the latent norm, the per-host latent up projection, and a 7-peer output gather (`:1093-1120`).
- **One local kernel change: a live-row mask.** AND `row < live_rows` into `owned_locally` (`:191-194`) so padded bucket rows stream no experts. SGLang already drops padded rows this way (`kimi_k3.py:1515-1518`).
- **Compiler parameters:**
  - `disable_bounds_checks` stays.
  - `vmem_limit_bytes` is set to the measured need (about 48 MiB) rather than 64 MiB. A SparseCore halt in sglang-torchtpu issue 485 went away when a global scoped-VMEM flag was removed.
  - **collective_id policy is decided by the next run.** Either one fixed id (one Mosaic kernel per bucket) or layer-derived ids, as vllm-torchtpu's K3 prefill rings use (`vllm-torchtpu/src/vllm_torchtpu/models/vllm/kimi_k3/collective_ops.py:316-324`).
- **Config guards at prepare time.** The kernel hard-codes the model constants, so assert they match the checkpoint config (checked against `/mnt/kimik3-ssd/kimik3/config.json`):
  - 896 experts, top-16, latent 3584, sigmoid router, renormalization;
  - routed scaling 1.0, rms eps 1e-5;
  - SiTU constants 4 / 25 (`decode_megakernel.py:100-108`).

**Bridge registration.**
- Use `cpu_export_jax_op` (`merged-t5/sglang_tpu/pallas_jax_op.py:208-279`) with the mesh and P() specs. It is stateless: no donation, and `mutates_args=()` at `:259` is correct here.
- Return a single array, not a one-tuple. vllm-torchtpu's `sharded_jax_op` docstring explains that a one-tuple schema makes torch `guard_int` every input dimension (`vllm-torchtpu/src/vllm_torchtpu/distributed/sharded_jax_op.py:150-155`).
- Replace the default fake, which re-exports the whole module on every fake call (`pallas_jax_op.py:261-278`), with a hand-written fake via `op.register_fake(...)`, as vllm-torchtpu's `collective_ops.py:41-89` does.
- Register from post-load, after `torch.distributed` init and before Dynamo traces.

**Weights.** Build them in the TPU post-load loop (`kimi_k3.py:3676-3694`), next to `_warm_tpu_forward_properties` (`:746-755`). Each one is a whole base tensor, built on the host and moved once, as `pack_kda_weights` does.

| New per-rank copy | Size per device |
|---|---|
| router `[in,out]` | 1.18 GB |
| latent_down shard | 0.17 GB |
| latent_up block | 0.68 GB |
| shared experts `[in,out]` | 0.76 GB |
| E8M0 scales | 2.66 GB |
| **Total** | **+5.45 GB** (weights 68.8 → 74.2 GB) |

- At a fixed `--mem-fraction-static` the KV budget shrinks by that amount, because the budget is free HBM after load (`python/sglang/srt/mem_cache/kv_cache_configurator.py:2521-2530`).
- The existing "copies built at load; restart to change weights" guard (`kimi_k3.py:~3665-3672`) must cover these copies.

**Numerics.** Greedy text will diverge from today's, so GSM8K and GPQA must be re-gated.
- Partial sums are BF16-packed in transport.
- Top-16 ties follow Mosaic's argmax order.
- EP8 × TP4 partial sums are grouped differently.
- The op is not batch-invariant, so it is off under deterministic inference.

**Flags.** Add these to `merged-t5/python/sglang/srt/environ.py` next to `SGLANG_TPU_ENABLE_K3_ATTN_RES_KERNEL` (`:1972`), following the env-var-conventions skill:
- `SGLANG_TPU_ENABLE_K3_MOE_BLOCK_OP = EnvBool(False)`
- `SGLANG_TPU_K3_MOE_BLOCK_OP_MAX_BS = EnvInt(8)`

The TPU debt ratchet limits `kimi_k3.py` to 21 TPU branches and 6 `sglang_tpu` imports (`tools/plugin/tpu_debt_baseline.json`). Raise those ceilings by the two hooks and justify it in the PR.

### 2.4 Freeing HBM after the op lands
- Teach gmm_v2 to decode uint8 E8M0 `[G,K/32,N]` scales in-kernel as `bitcast(uint32(e) << 23)`. Today non-float scales are cast to f32 by value (`gmm_v2.py:1700-1702`); SGLang avoids this only by storing 10.64 GB of decoded f32 scales.
- Have prefill use the op's sharded latent_down/latent_up plus XLA all-gathers, then drop the 9.45 GB of replicated latent projections.
- Reuse the `[in,out]` router and shared-expert copies, then drop the originals.
- Result: weights fall to about 52 GB, about 16 GB below today, and that space goes to KV.

### 2.5 Later: one op per layer covering everything between two attention cores
- **What the op covers:**
  - layer L's post-core tail: KDA gated o_norm, or MLA w_vc plus the sigmoid gate;
  - row-parallel o_proj with an in-kernel reduction, reusing `moe()`'s 7+3 reduction helpers;
  - the pending prefix add, the attention-residual mixes and norms;
  - the MoE, or the dense MLP for layer 0;
  - layer L+1's attention input projections.
- **What stays outside:** SGLang's KDA op (`gdn_attention`, per-layer state pools) and MLA v2 op run between ops. Cache ownership, metadata, retraction and prefill are untouched.
- **What it changes:** 94 communicating ops per step and no XLA all-reduce between layers. That needs a composition protocol chosen from the next run's barrier-chain data: default device barrier, alternating ids, or an exit barrier.

### 2.6 Later: four-layer group op over SGLang's caches
- **Layout:**
  - one op per K3 period (KDA, KDA, KDA, MLA): 23 groups, plus layer-0 and layer-92 variants;
  - layers unrolled, each with its own operands, so there is no traced layer index and no stacking;
  - 24 communicating ops per step.
- **KDA core:** in-kernel over SGLang's per-layer `[slots+1,3,1,1152]` f32 conv and `[slots+1,3,128,128]` f32 recurrent tensors (`sglang_tpu/mamba_pool.py:150-190`), indexed through scalar-prefetched `state_indices`, with one semaphore per row.
- **MLA core:** a new paged absorbed-latent core over SGLang's `[num_pages,64,2,640]` bf16 buffers (`sglang_tpu/kv_pool.py:769, 805-808`), with scores on the MXU.
- **In-place updates:** 7 cache operands per op via donation, explicit `input_output_aliases` and `out_shardings` on the jit (the earlier CPU analysis showed this gives explicit `tf.aliasing_output`), inside `torch.library.custom_op(mutates_args=...)`. This is the GDN op pattern (`sglang_tpu/attention_gdn_pallas.py:163-184`) with a mesh added.

### 2.7 Files for the first two deliverables

| File | Change |
|---|---|
| `merged-t5/sglang_tpu/kernels/kimi_k3_megakernel/{__init__,decode_megakernel,collectives32,pool_alias,reference}.py` | Vendored from 4048f08 (Apache-2.0) with package-relative imports and a header listing local changes: the live-row mask and the axis-name constant |
| `…/kimi_k3_megakernel/moe_block.py` | Standalone `pallas_call` around `moe()`, from the prototype |
| `merged-t5/sglang_tpu/kimi_k3_moe_op.py` | Mesh plus assertion, op registration, weight builders, forward helper, eligibility. No `from __future__ import annotations` in this file |
| `kimi_k3.py` | Two thin lazy-import hooks: `:1514` and `:3676-3694` |
| `environ.py`, `tpu_debt_baseline.json` | Flags and ratchet ceilings |
| `mxfp4_tpu_util.py` (optional) | Keep the checkpoint's E8M0 uint8 scales instead of converting from f32 |
| `test/registered/unit/hardware_backend/tpu/test_kimi_k3_moe_block_export.py`, `test_kimi_k3_moe_weights.py`; `test/registered/unit/models/test_kimi_k3_moe_block_routing.py` | CPU tests |
| `test/manual/tpu/test_kimi_k3_moe_block_op.py` | 32-process harness. The existing template asserts `world_size == 8` (`test_fp8_fused_rs_kernel.py:82-85`); start from `/mnt/workspace/dagent/mkspike/mk_bridge_spike.py` instead |
| new `benchmark/tpu/recipe/configs/kimi_k3/` perf recipe | `--ep-size 8`, buckets 1 4 16 (64), flag on. Read the tpu-nightly-dashboard skill before adding it to nightly |

---

## 3. Milestones

**Rules for every milestone:**
- At most one TPU run per milestone, and at most 2 runs in any rolling 24 hours, counting every CDK job including single-host unit runs.
- Each run needs the user's explicit go-ahead.
- A contingency rerun happens only after a failed exit criterion, in the next free slot.

**Measurement and feasibility.**
- **CPU work:** fix the prepared spike (§4.1), CPU dry runs of every phase, and optionally the extra probes in §4.4.
- **TPU:** the combined run in §4. That is 1 run.
- **Effort:** 2-4 days, plus 2-3 days for the extra probes.

**Expert layout (8 groups × 4-way split).**
- **CPU checks:**
  - extend `test_ep_local_expert_reader.py` with EP8-at-TP32 ranges;
  - build a tiny K3 with a mocked parallel state (TP 32, expert-parallel 8) and check the per-rank shapes from §2.2;
  - `jax.export` gmm_v2 at the quarter-width shapes;
  - startup log of the resolved ownership;
  - argument validation.
- **Mitigation for 4× host reads:** read only each rank's contiguous w1/w3 row quarter.
- **TPU:** no run of its own. It is the extra variant of the measurement run, or the flag-off variant of the MoE-op run.
- **Exit:** launches; TPOT within +2% and TTFT within +10% of the baseline; smoke prompts pass.
- **Effort:** about 1 week, including tile tuning after measurement.

**Per-layer MoE-block op.**
- **CPU checks:**
  1. **Export structure** at B ∈ {1,2,4,8,16}: one custom call, the chosen `collective_id`, scoped VMEM, no XLA collectives, 13 inputs / 1 output.
  2. **Host-side word and nibble mapping** of SGLang's uint8 bytes against `tpu-megakernels/kimi/load.py` packing. Per the earlier analysis, JAX 0.10.2 interpret mode cannot DMA through a bitcast ref, so feed host-converted words.
  3. **Bit-exact weight builders** against `kimi/load.py` readers on `/mnt/kimik3-ssd/kimik3`, for ranks {0,5,27,28,31} and layers {1,3,46,92}.
  4. **A pure-torch reference** of the op's math against `_forward_unfused` on a tiny K3, with tolerances documented.
  5. **The live-row mask** in interpret mode.
  6. **Hook routing** with a CPU stub op under `torch.compile(fullgraph, backend="aot_eager")`:
     - no graph breaks;
     - one graph per bucket;
     - no recompile when positions or live rows change;
     - every guard falls back.
  7. **A debug flag** that computes both MoE paths in the same graph and writes per-layer max-abs differences to a device buffer (the in-launch A/B).
- **TPU (1 run, real weights, `--ep-size 8`, buckets 1 4 16):**
  - op off then op on, as two launches with a warm page cache;
  - per-layer A/B on 200 prompts;
  - TPOT at batch 1 (8k prompt / 1k output), batch 4 (1k / 1k) and batch 8;
  - GSM8K 300 at concurrency 8 (reference 295/300) and a 64-question GPQA Diamond subset (reference rate 184/198);
  - a 2-hour soak with `SGLANG_TEST_RETRACT=1`, staggered arrivals and twin prompts, to catch transient corruption from shared-semaphore DMAs;
  - flat HBM, no recompiles, and a decode profile.
- **Exit:** batch-1 TPOT ≤ 11 ms; batch-4 TPOT ≤ 14 ms; GSM8K ≥ 290/300; no halt.
- **Effort:** 5-7 engineer-weeks.

**Freeing HBM.**
- **CPU:** gmm_v2 E8M0 path checked in interpret mode against a dense reference; sharded-latent prefill reference.
- **TPU (1 run):** TTFT, prefill numerics, recovered KV capacity, GSM8K.
- **Effort:** 2-3 weeks.

**Per-layer op between attention cores.**
- **CPU:**
  - export of the seven variants (prologue, KDA→KDA, KDA→MLA, MLA→KDA, MLA→MLA, layer-0 dense tail, final);
  - a pure-torch reference of a whole segment;
  - checks that the outputs feed SGLang's KDA and MLA ops unchanged.
- **TPU (2 runs):**
  1. synthetic-weight chain plus a dummy-weight server;
  2. real weights.
- **Effort:** 6-9 weeks.

**Four-layer group op.**
- **CPU:**
  - the paged-latent MLA core against SGLang's MLA v2 in interpret mode, at page boundaries (p%128 ∈ {0,1,126,127}) and contexts up to 64k;
  - KDA slot indirection against `TpuMambaPool`;
  - export with 7 explicit aliases;
  - an extended `cpu_export_jax_op` with an `input_output_aliases` pass-through and mutating registration;
  - a churn emulator.
- **TPU (2-3 runs):**
  1. bring-up: HLO with no cache copies, flat HBM, cache correctness;
  2. real weights;
  3. spare.
- **Effort:** 8-12 weeks.

**Optional: one op per decode step.** Only if the measurement run shows the whole-step bridge op is near native and stays in place. 2 runs, 6-10 weeks.

---

## 4. The next combined TPU run (exact content)

A combined run was already prepared earlier today:
- driver `/mnt/workspace/dagent/e2e/scripts/combo_driver.sh`;
- plan `/mnt/workspace/dagent/e2e/plans/plan-combo1.sh`;
- phase plans `plan-combo1-sglang.sh`, `plan-mkspike1.sh`, `plan-mk1.sh`.

Keep it, with the changes below.

### 4.1 Preconditions before asking for the go-ahead

**Fixes to `/mnt/workspace/dagent/mkspike/mk_bridge_spike.py`:**
1. **Weight fill (`:154`).** `t[0, i].copy_(block)` slice-assigns into a device tensor, and torch_tpu re-materializes that view chain before every forward (`kda_layout.py:48-51`). On 78.6 GB of weights this would swamp the timing or run out of HBM.
   - Fix: produce each family with one whole-tensor op. Best is a tiny bridge op (`jnp.broadcast_to`, or a deterministic integer-hash fill) whose output is a fresh base tensor.
   - Assert `t._base is None`.
2. **Recipe parity (`:69`).** The spike passes only `gate_first_kda_projection` and `kernel_options` (the "plain" recipe). The standalone server defaults to `--recipe tuned` (`tpu-megakernels/demo_kimi_dspark.py:298`), whose `mla_cache_batch_tile_size`, residual order and so on differ (`:97-117`).
   - Fix: pass `**decode_options(B, "tuned")`.
3. **uint32 tensors (`:121`, `:145`).** torch.uint32 device ops are unproven.
   - Fix: store the families as int32 and `jax.lax.bitcast_convert_type` them to uint32 inside the shard_map body.
4. **Timing (`:177`).** The current loop syncs every step.
   - Fix: report wall time of 100 back-to-back steps with one final sync, plus the per-step synced latency.
   - Record HBM in use after the fill, after the first call and after 100 steps with `torch.accelerator.get_memory_info` (precedent: `tpu_platform.py:1120-1126`).
5. **Positions.** Set them to match the standalone run's average decode positions: 512 at 1k context, 8,704 at 8k context, 1,536 for batch 4.
6. **Topology dump on rank 0.** Use torch_tpu `_gather_global_device_info("tpu")` (`torch_tpu/_internal/distributed/device_mesh.py:69-105`) to record torch rank, device id, coords, `core_on_chip` and `process_index`. Compare against the SGLang map in §0.
7. **Stack-only variant.** Add one that calls `stack(emb, emb, w35, states, positions, include_layer_zero=True)`. It has no XLA collective inside the op.

**Environment and logistics:**
- **Same libtpu flags as SGLang.** Run the spike with SGLang's `LIBTPU_INIT_ARGS`: `merged-t5/benchmark/tpu/common_env.sh:65-74` plus the additions from `sglang_tpu/env_override.py`. Also add `--xla_tpu_debug_sflag_wait_timeout_ms=120000 --xla_tpu_debug_sflag_wait_shalt_on_detection=true`.
- **Disk.** Send bulky artifacts (xprof, HLO dumps) to `/mnt/kimik3-ssd`. The client already sends profiles there (`client_v5.sh:70`). Confirm at least 10 GB free on `/mnt/workspace` for logs; the root disk has 7.2 GB free now.
- **Timeouts.** Use `DEADLINE_S=21600`.

  | Phase | Current timeout | New timeout |
  |---|---|---|
  | SGLang (with the extra variant) | `SG_TIMEOUT_S=6600` | about 8,400 s |
  | Bridge spike | `SPIKE_TIMEOUT_S=4800` | about 7,200 s |
  | Megakernel standalone | `MK_TIMEOUT_S=6000` | about 7,800 s |

  The standalone phase needs room for one cold load (35-45 min) and two warm loads. Today's server took 3,844 s to come up.
- **CPU dry runs:**
  - `dryrun_bridge.py` for every spike config, including the stack-only and MXFP4 variants;
  - the fake-server combo dry run in `/mnt/kimik3-ssd/dryrun/combo`.
- **Run count.** Confirm fewer than 2 runs in the last 24 h with `cdkx job list`.

**Verified, no change needed:** the spike's default `torch.compile` backend resolves to the TPU backend through torch_tpu's monkeypatch (`torch_tpu/_loader.py:104-117`).

### 4.2 Phase order and budget

The standalone megakernel phase goes last because it installs JAX 0.11.1 and libtpu 0.0.46.1.

| # | Phase | Est. time | Priority |
|---|---|---|---|
| 1 | SGLang real-weight baseline, today's flags, EP32: validate, TPOT sweep, decode profiles | ~70 min | must |
| 2 | Same server with `EP_SIZE=8` (→ `--ep-size 8`, `launch_server.sh:483`), warm load | ~45 min | if time |
| 3 | Bridge spike inside torch_tpu workers, synthetic weights | 60-90 min | must |
| 4 | Megakernel standalone, real weights | 80-120 min | must |

### 4.3 Phase contents

**SGLang baseline and decode profiles.**
- Configuration: `plan-combo1-sglang.sh` as prepared: `CONTEXT_LEN=9344`, buckets 1 4 16 64, chunk 1024, flags as in the baseline.
- TPOT sweep: batch 1 with 8k/1k and batch 4 with 1k/1k.
- Profiles: 20 decode steps at batch 1 (8k) and batch 4 (1k), started 8 s after the requests, so after prefill (`client_v5.sh:89-91`).
- Analysis: `/tmp/k3perf/prof/k3_xprof.py` on rank 0 and rank 8, broken down into:
  - all-reduce count and time, including the MoE combine wait;
  - gmm_v2, route metadata and gate;
  - latent, router and shared GEMV fusions;
  - MLA and KDA kernels;
  - attention-residual kernels;
  - copies and `tt_jit_as_strided` re-materializations.
- The `EP_SIZE=8` variant repeats the sweep and the batch-1 profile, and records load time, HBM and recompiles.

**Bridge spike.** SGLang's stack: torch_tpu 0.1.1, JAX 0.10.2, libtpu 0.0.47, launched as `torchrun` with 4×8 processes.

| Config | Op | Batch | Context | Position | Gate/up |
|---|---|---|---|---|---|
| topology | rank and device dump | – | – | – | – |
| stack-only, 1k | stack only | 1 | 1152 | 512 | FP8 |
| whole-step, 1k | embedding psum + stack + epilogue | 1 | 1152 | 512 | FP8 |
| whole-step, 8k | same | 1 | 9344 | 8704 | FP8 |
| whole-step, batch 4 | same | 4 | 2176 | 1536 | FP8 |
| MXFP4 batch 1 / batch 4 (if time) | whole-step | 1 / 4 | 1152 / 2176 | 512 / 1536 | MXFP4 (51.4 GB) |

For each config, record:
- Mosaic compile time;
- steady-state step time;
- HBM before, after the first call and after 100 steps;
- whether logits and states are finite;
- any hang, with the sflag halt report if one occurs.

For the 8k whole-step config, also take an HLO dump on rank 0 only (`--xla_dump_to=/mnt/kimik3-ssd/… --xla_dump_hlo_module_re=.*tt_jit.*`, as in `launch_server.sh:335-340`). Check it for copies of the `[1,1,24,3,9344,256]` state parameters and for surviving `output_operand_aliases`.

**Megakernel standalone.** `plan-mk1.sh` via `mk_driver.sh`: target-only greedy mode, tuned recipe, its own JAX 0.11.1 / libtpu 0.0.46.1.

| Config | Batch | Context | Prompt / output tokens | Purpose |
|---|---|---|---|---|
| batch 1, 1k | 1 | 1152 | 1 / 1024 | reproduce the published 4.02 ms; the published setup is context 1152 with 1-token prompts (`tpu-megakernels/scripts/benchmark_kimi.sh:23, 93-104`) |
| batch 1, 8k | 1 | 9344 | 8192 / 1024 | long-context cost of the vector-unit MLA scores |
| batch 4, 1k | 4 | 2176 | 1024 / 1024 | compare with the published 7.77 ms and SGLang's 21.05 ms |

- Record the load time, compile time per program and greedy sanity prompts per config.
- If time runs short, merge the two batch-1 configs into one server at context 9344. Decode cost follows actual positions, not cache capacity (`decode_megakernel.py:2705-2711`).

### 4.4 Add-ons, included only if their CPU dry run passes before launch

These belong in the bridge-spike phase, in this order:
1. **MoE-op chain.**
   - Register the stateless op through `cpu_export_jax_op` with the hand-written fake.
   - Use 4 synthetic per-layer weight sets in SGLang's EP8 × TP4 dtypes (§2.3).
   - Measure:
     - per-call latency at B ∈ {1,4,8,16} over 1,000 calls;
     - a 92-op chain with a bf16 `[B,7168]` XLA all-reduce between ops, which mimics o_proj;
     - the same chain with no all-reduce between ops;
     - one fixed `collective_id` against layer-derived ids, comparing compile time and step time;
     - `vmem_limit` at 48 MiB against 64 MiB;
     - 2,000 steps without a halt;
     - outputs bitwise identical across ranks.
2. **Barrier chain.** Chains of 1, 16 and 92 kernels, each a barrier plus a 1 KB remote copy, compared across:
   - the same id, alternating ids and distinct ids;
   - `skip_device_barrier` (`jax/_src/pallas/mosaic/core.py:109`);
   - with and without an XLA all-reduce between kernels.
3. **Native synthetic-weight megakernel run** in the standalone phase. It uses the same deterministic fill as the spike, which isolates bridge overhead from routing differences.

### 4.5 What each result decides

| Result | Decision |
|---|---|
| **Profile:** MoE-attributable time T_moe at batch 1 and 4 | Estimated TPOT with the op ≈ TPOT − T_moe + 92·t_call. If the estimated batch-1 saving is under 3 ms, start with the op between attention cores instead of the MoE-only op |
| **Spike compile under libtpu 0.0.47** | If it fails while the standalone (libtpu 0.0.46.1) works, every megakernel-derived op is blocked: escalate to torch_tpu/libtpu before porting. Precedent: the FP8 fused reduce-scatter MoE regressed on libtpu ≥ 0.0.46 with corrupt output or core halts (sglang-torchtpu issue 1065, open) |
| **Spike step time** vs native at the same batch and position | Within 15% of native real-weight (or 10% of native synthetic): bridge and launch overhead are small and the whole-step op is a credible end state. Larger: inspect the HLO for copies and reassess per-op launch cost |
| **HBM flat and no state copies in the HLO** | The functional donation path keeps multi-GB states in place, so later stateful ops can use it. Otherwise they need the explicit-alias path, to be tested in the following run |
| **Stack-only vs whole-step** (hang or speed difference) | Whether an XLA collective may sit inside a communicating module. This is the known mixed-collective halt class: issues 308 and 485, and the workaround at `sglang_tpu/moe/pallas/fused_rs/fused_moe.py:157-171` |
| **Topology dump** | Confirms that r and r^1 share a chip and that ranks 8h..8h+7 share a host. If not, correctness is unaffected but pair exchanges cross chips; adjust the speed estimates |
| **Standalone batch 1 at 1k** | ≤ 4.3 ms: the published speed transfers to our slice. Higher: scale every megakernel-derived estimate down by the same ratio |
| **Standalone 8k minus 1k** | Above 1.5 ms: keep SGLang's MXU-based MLA in every staged op and do not adopt the megakernel's MLA without MXU scores |
| **`--ep-size 8` variant** | Passes the layout exit criteria: the layout milestone is config plus tile tuning. Otherwise fix the layout before the op |
| **MXFP4 vs FP8 spike** | Size of the MXFP4 penalty the op will pay until the HBM step frees room for FP8 |
| **MoE chain** | t_call ≤ 45 µs at batch 1 and ≤ 65 µs at batch 4 with no halt: go. Above 60 µs at batch 1: skip to the op between attention cores. The id comparison sets the collective-id policy; the no-all-reduce chain sets that op's protocol |
| **Barrier chain** | Per-op fixed cost and whether to set `skip_device_barrier` |

---

## 5. Expected TPOT and main risks

| Stage | Batch 1, 8k prompt / 1k output | Batch 4, 1k / 1k | Basis |
|---|---|---|---|
| Today (measured 2026-10-12) | 15.92 ms | 21.05 ms | real weights |
| Expert layout only | 14.5-16.5 | 18.5-21.5 | Smaller busiest-rank wait (about 3.4× less excess at batch 1, uniform-routing simulation), but 112 untuned quarter-width groups; may not improve |
| + per-layer MoE op | 8.5-11 (central ~9.5) | 11-13.5 | Removes 184 MoE all-reduces (~16.5 µs each), the busiest-rank wait, 9.46 GB/step of replicated GEMVs and gmm glue; adds 92 calls at about 25-45 µs (batch 1) or 45-65 µs (batch 4) |
| + freeing HBM | ≈ same | ≈ same | Prefill TTFT and KV capacity, about +16 GB |
| + op between attention cores | 6.5-8.5 | 9-11 | Removes 93 o_proj all-reduces and the residual/norm glue |
| + four-layer group op | 5-6.5 | 7.5-9 | 24 sync points; weights prefetched across layers within a group |
| Whole-step ceiling | ~4.6-5.5 | ~8-9 | Published 4.02 / 7.77 ms + 8k attention + SGLang per-step work; to be measured in the next run |

**Main risks**, most severe first:
1. **libtpu 0.0.47 may not compile or run the communicating megakernel code.** The megakernel was built on 0.0.46.1, and issue 1065 shows a communicating MoE kernel breaking on libtpu ≥ 0.0.46 with a "Semaphore … nonzero value upon exit" halt. Settled by the bridge spike.
2. **92 communicating kernels in one executable.**
   - This stack already had halts when communicating kernels shared an executable: issue 308 (large-M fused reduce-scatter with paged attention) and issue 485 (distributed top-K plus fused reduce-scatter on the SparseCore).
   - vllm-torchtpu's opt-in K3 prefill path does run five communicating ring ops per layer with layer-derived ids, which is encouraging but on a different path.
   - Settled by the MoE chain, in the next run if ready, otherwise in the MoE-op run.
3. **Per-op sync cost** (31-peer barrier plus the default device barrier). Settled by the barrier chain.
4. **Shared-semaphore aggregate waits** in `moe()`: 31 latent copies drained by one wait (`decode_megakernel.py:1528-1547`) and 7 output copies (`:1100-1120`). The same pattern corrupted KDA state at TP32 (`gdn/wrapper.py:597-608`). For the stateless op this would show up as transient wrong outputs; the in-graph A/B and the twin-prompt soak look for it.
5. **MXFP4 gate/up speed at batch 4-8.** The megakernel authors re-encode to FP8 to avoid the unpack, and FP8 is unaffordable until HBM is freed.
6. **Kimi-K3 has never run with `--ep 8`.** Load time, with 4× expert reads per host, untuned gmm_v2 tiles and prefill TTFT are all unknown.
7. **Numerics change**, so accuracy must be re-gated. The op is also off under deterministic inference.
8. **HBM:** +5.45 GB until the HBM step lands.
9. **Deprecation.** torch_tpu marks `input_output_aliases` as deprecated (`torch_tpu/_internal/pallas/pallas.py:538-539, 567-585`). Only the later stateful ops depend on it; keep a functional fallback (donate, return new state, copy).
10. **Maintenance.** A vendored 5.5k-line kernel plus a private JAX monkeypatch.
11. **Operations.** The full root disk (it already lost a variant on 2026-10-12) and spot preemption, which restarts the job.

**Settled in this pass:**
- the four-host rank map keeps chip pairs and hosts together;
- both hang-diagnostic flags exist in libtpu 0.0.47;
- the standalone `moe()` exports with SGLang's uint8 experts at batch 1 and 8;
- SGLang's EP8 ownership arithmetic equals the megakernel's;
- the K3 config constants match the kernel's hard-coded values;
- the prepared bridge spike passes its CPU dry run;
- no usable current decode profile exists, so the next run must take one.
