# Kimi-K3 decode megakernel in SGLang-torchtpu (experimental)

Status: **work in progress, not wired into serving yet.** This branch
(`experimental/kimi-k3-megakernel`) collects everything done so far towards one goal:

> Port Inferact's Kimi-K3 decode megakernel into SGLang-torchtpu and keep the
> megakernel's promised decode gains.

Nothing here has run on a TPU yet. Every TPOT number below that is not marked
"measured" is an estimate.

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

## 2. Baselines (measured)

SGLang main + the open Kimi-K3 PRs, real 2.7T weights, one 2x2x4 tpu7x slice, all three
opt-in flags on (fused o_norm, attention-residual kernel, guard skip), 2026-10-12:

| Workload | TPOT | TTFT |
|---|---|---|
| batch 1, 8,192-token prompt, 1,024 outputs | **15.92 ms** | 2.27 s |
| batch 4, 1,024-token prompts, 1,024 outputs | **21.05 ms** (median) | 1.58 s |

Flags off on the same stack: 16.84–16.86 ms (8k/1k batch 1), 21.8–21.9 ms (1k/1k batch 4).

The megakernel's own numbers have **not** been reproduced on our slice yet, and they were
measured at ~1k context; at 8k context its vector-unit decode attention adds an estimated
0.3–1.5 ms.

## 3. What is verified so far (CPU only)

1. **The whole decode step exports to one TPU custom call under SGLang's JAX 0.10.2**
   (43 inputs: the weight dict + 4 states + token + position; 5 outputs: logits + 4 states;
   `has_communication`, `collective_id 21`, 64 MiB scoped VMEM, `disable_bounds_checks`).
   Checked for batch 1 and 8 by the analysis, and for the vendored copy in this branch
   (`benchmark/tpu/kimi_k3_megakernel/dryrun/export_flat_cpu.py`).
2. **It registers and shape-propagates through SGLang's bridge** (`sglang_tpu/pallas_jax_op.py`
   `cpu_export_jax_op`) inside a CPU torch_tpu install, for batch 1 / context 1,152,
   batch 1 / context 9,344 and batch 4 / context 2,176 (`dryrun/dryrun_bridge.py`,
   `DRYRUN_OK`; 78.6 GB of per-rank inputs). Two bridge requirements were found this way:
   the jax function must have **explicitly `jax.Array`-annotated parameters** (no `*args`)
   and a **`tuple[jax.Array, ...]` return annotation**.
3. **Synthetic weight fill** (one random block per family, NaN-free FP8, sane E8M0) is
   correct for every family's real per-layer shape (`dryrun/dryrun_hostfill.py`).
4. **The TPU run harness works end to end against fake servers** (combined-run phase
   orchestration with barriers/cleanup, the megakernel client batching, the SGLang
   mid-decode profile capture) — see §7.
5. Facts gathered by the design review (see [`docs/staged_port_plan.md`](docs/staged_port_plan.md)):
   - SGLang's logged four-host rank→device map keeps ranks `2k, 2k+1` on one chip pair and
     ranks `8h..8h+7` on one host — the megakernel's placement assumption.
   - SGLang can shard experts the megakernel's way with `--tp-size 32 --ep-size 8`
     (`moe_tp_size = 4`); its per-rank MXFP4 tensors are uint8 `[112,1792,1536]` /
     `[112,384,3584]` with f32 power-of-two scales `[112,112,1,1536]` / `[112,24,1,3584]`.
     Under the in-HBM uint8→uint32 ref bitcast that gmm_v2 already uses, those bytes are the
     megakernel's MXFP4 words `[112,448,1536]` / `[112,96,3584]`; scales convert to E8M0 bytes
     exactly (f32 exponent field). **Not yet re-verified independently; Kimi-K3 has never run
     at EP 8.**
   - libtpu 0.0.47 has `xla_tpu_debug_sflag_wait_timeout_ms` /
     `xla_tpu_debug_sflag_wait_shalt_on_detection` for diagnosing a hung communicating kernel.
6. Premise check (adversarial, CPU), four of eight answers in, all "true with conditions":
   - **MLA handoff:** expanded K/V = SGLang's cached latent row `@ m_kb[layer]` (no extra
     norm); key = `[E[h, 0:128] | latent[512:576] | 0^64]`, value = `E[h, 128:256]`. Needs a
     bf16 KV cache (no fp8), DCP 1, conversion of exactly the committed length, and the
     request diverted before the overlap scheduler puts it in an SGLang decode batch.
   - **KDA handoff:** bit-exact; conv = MambaPool conv row reshaped to `[3,3,3,128]` and cast
     f32→bf16 (lossless), recurrent copied as is; layer map `l = i + i // 3`.
   - **EP8 load:** `--tp-size 32 --ep-size 8` needs no code change (pallas_tpu MoE backend,
     no DP attention / EPLB); never run before.
   - **Rows subset:** 10 ref sites + a guard, not 3 (see §5). Missing the persist-mode drain
     waits passes every CPU check and would only fail on TPU. With the full patch, the
     Mosaic module for R = 8 rows at B in {1, 2, 4} equals the native B-row module after
     renaming the state memref types; R = B is byte-identical to upstream.
   Still pending: expert-byte equivalence, scheduler seams, memory, in-place donation.

## 4. Design

Three ways to get the megakernel into SGLang were designed and judged (details and judge
verdicts in [`docs/staged_port_plan.md`](docs/staged_port_plan.md)):

| Design | Decode TPOT (est.) bs1 8k / bs4 1k | Verdict |
|---|---|---|
| A. Megakernel-only engine (SGLang keeps scheduler/API/sampler; megakernel weights and caches only; prefill through the 8-row verify kernel) | ~5.2 / ~8.3 ms | Rejected by the panel for serving: 8k TTFT ~9–12 s (prefill runs before decode, stalling it), ≤8 concurrent requests, no fallback, no multimodal |
| B. SGLang prefill + full megakernel decode **over SGLang's paged caches** | ~4.7 / ~9.0 ms | Rejected: ~10 weeks of kernel surgery verifiable only on TPU (paged absorbed-latent MLA, per-row KDA slot DMAs) |
| C. Staged per-layer ops (MoE-block op first, growing to group ops) | 8–10 → 4.8–6 ms at the end | Panel's choice for risk, but the first stages do **not** keep the megakernel's gains |

**Chosen direction for the stated goal ("keep the megakernel's gains"): a hybrid of A and B.**

- SGLang loads Kimi-K3 at `--tp-size 32 --ep-size 8` and keeps its own prefill
  (chunked, multimodal) into its own caches.
- When a request's prefill finishes, a one-time **handoff op** fills that request's row in
  **megakernel-owned decode caches**: MLA expanded K/V computed from SGLang's cached latents
  with the megakernel's own weights, KDA conv/recurrent states copied from the MambaPool.
- Decode batches of ≤ 8 requests run **one full-step megakernel op per step** over rows
  `[0, B)` (requests kept dense in the lowest rows, `B ∈ {1, 2, 4, 8}`; `slots.py`), with
  logits handed to SGLang's sampler.
- Routed experts are **shared** with SGLang (packed MXFP4 bytes + E8M0 scale copies, 2.66 GB);
  the other megakernel weights are layout copies (~5 GB) read from the checkpoint with the
  vendored reader.
- A megakernel-decoded request never returns to SGLang's decode path; max running requests
  ≤ 8 in this mode. Larger batches keep SGLang's normal path.

**Open problem found while writing `weights.py`:** the megakernel expects each expert family
as **one tensor stacked over the 92 MoE layers**, while SGLang holds one parameter per layer.
Stacking copies 49.5 GB (does not fit), and passing views is not an option (torch_tpu
re-materializes views on every compiled call). Resolution options, in order of preference:
1. Store Kimi-K3's routed experts as stacked base tensors in SGLang and let its MoE kernels
   (gmm_v2 / fused MoE) index the layer inside the kernel (static index_map offset), so
   neither path ever passes a view.
2. Pass per-layer expert tensors to the megakernel and select the layer's ref in-kernel
   (~368 extra operands; operand-count limits unknown).
An adversarial premise check of this whole design (EP8 loading, expert-byte equivalence, MLA
and KDA handoff exactness, rows-subset patch sites, scheduler seams, memory, in-place
donation through the bridge) was running when this snapshot was taken.

## 5. Code in this branch

| Path | What |
|---|---|
| `sglang_tpu/kimi_k3_megakernel/_vendor/` | Upstream `collectives32.py`, `pool_alias.py`, `kimi/__init__.py`, `kimi/decode_megakernel.py`, `kimi/load.py`, Apache-2.0 headers. `SGLANG-PATCH` changes: package-relative imports; rows-subset (the step reads/writes rows `[0, B)` of `R >= B`-row states: KDA state reads, write starts and drain waits, MLA cache refs, `make_decode` checks relaxed to `<`, plus a guard in `stack()`). Exported once on CPU at B = 1, R = 8 (one Mosaic module); the module-equality test is not written yet |
| `sglang_tpu/kimi_k3_megakernel/program.py` | The step as one flat jax function for the torch bridge: shard_map over `decode_local`, states `[R, ...]` per rank donated and aliased to outputs 1–4, logits all-gathered to `[B, vocab]`; `decode_options` (upstream recipes), `state_shapes`, `layer_counts`. **Written, never run** |
| `sglang_tpu/kimi_k3_megakernel/slots.py` | Dense row table: request → megakernel row, compaction moves on release, `row_bucket()` |
| `sglang_tpu/kimi_k3_megakernel/weights.py` | Dense-family loader from the checkpoint (per rank, base tensors), exact f32→E8M0 scale conversion, SGLang→megakernel expert family mapping (**blocked on the stacking problem above**) |
| `test/registered/unit/hardware_backend/tpu/test_kimi_k3_megakernel_slots.py` | Slot table tests incl. a 5,000-step churn simulation |
| `benchmark/tpu/kimi_k3_megakernel/spike/mk_bridge_spike.py` | Bridge spike: the full decode step as a torch custom op inside 32 torch_tpu workers (torchrun), synthetic weights, timing |
| `benchmark/tpu/kimi_k3_megakernel/dryrun/` | CPU dry runs: export checks, bridge registration + FakeTensor propagation, host fill, fake megakernel and SGLang servers |
| `benchmark/tpu/kimi_k3_megakernel/harness/` | 4-host GKE JobSet harness: `combo_driver.sh` (phases), `run_variants_v5.sh` + `client_v5.sh` (SGLang TPOT + mid-decode xprof), `mk_spike_driver.sh`, `mk_driver.sh` + `mk_client.py` (megakernel standalone), `plans/` |
| `benchmark/tpu/kimi_k3_megakernel/k3_xprof.py`, `planes.py` | xplane.pb breakdown by category / per decode step |
| `sglang_tpu/kimi_k3_megakernel/docs/analysis.md` | Full code analysis of the megakernel and port options |
| `sglang_tpu/kimi_k3_megakernel/docs/staged_port_plan.md` | Multi-agent design review: integration maps, three designs, judge verdicts, staged plan |

## 6. How to run the CPU checks

```bash
# Vendored export (no TPU): one tpu_custom_call, 43 in / 5 out
python -B benchmark/tpu/kimi_k3_megakernel/dryrun/export_flat_cpu.py 1 1152

# Bridge registration + FakeTensor shapes, with a CPU torch_tpu + libtpu wheel install
# in $SITE (pip install --no-deps --target $SITE torch_tpu*.whl libtpu==0.0.47 absl-py frozendict)
MK_SRC=<tpu-megakernels or vendored copy> \
PYTHONPATH=$SITE:python:.:benchmark/tpu/kimi_k3_megakernel/spike \
  python -B benchmark/tpu/kimi_k3_megakernel/dryrun/dryrun_bridge.py 1 9344   # DRYRUN_OK

# Slot table
python -m pytest test/registered/unit/hardware_backend/tpu/test_kimi_k3_megakernel_slots.py
```

The dry run sets `TORCH_DEVICE_BACKEND_AUTOLOAD=0`, renames PrivateUse1 to `tpu` and stubs
`torch.tpu`, so torch_tpu imports without a TPU.

## 7. TPU validation plan

Rules: **at most 2 TPU runs per 24 h**, and **no TPU run until the port is complete**.
Every run is one job carrying all its measurements, dry-run on CPU first.

The prepared single combined run (`harness/plans/plan-combo1.sh`, driver `combo_driver.sh`):
1. **SGLang phase:** TPOT 8k/1k bs1 and 1k/1k bs4 on the same slice, then xprof of 20 decode
   steps started *after* prefill, written to the attached SSD (a 32-rank capture is ~5 GB;
   writing it to the root-disk workspace crashed a server on 2026-10-12).
2. **Bridge spike:** the decode step inside torch_tpu workers via the bridge.
3. **Megakernel standalone** (jax 0.11.1, libtpu 0.0.46.1): bs1 at context 1,152 and 9,344,
   bs4 at 2,176.

Known fixes still to apply to the spike before it runs (from the design review): fill weights
with whole-tensor ops (no slice-assignment views), store uint32 families as int32 and bitcast
in JAX, use the demo's `tuned` recipe options, time 100 back-to-back steps plus synced
latency, set positions to the standalone run's averages, dump rank→device/core topology on
rank 0, add a stack-only variant, and run with SGLang's `LIBTPU_INIT_ARGS` plus the sflag
hang-diagnostic flags.

## 8. Sharp bits learned

- The bridge wants explicit `jax.Array`-annotated parameters and a precise tuple return
  annotation; `*args` and `-> tuple` fail at registration.
- `torch.uint32` device ops are unproven on torch_tpu: keep uint32 words as int32 and bitcast
  inside the exported JAX program.
- Views passed into a compiled graph are re-materialized per call on torch_tpu; every kernel
  operand must be a whole base tensor.
- `/mnt/workspace` and `/tmp` share the 968 GB root disk, which runs at ~99% full; profiles
  and large artifacts go to the attached SSD.
- Never edit a script that a queued job will read: a job that starts mid-edit reads a torn
  file.
- PR TPU CI is on demand; "NOT ADMITTED" is a slice-queue timeout, not a test failure.

## 9. Next steps

1. Finish the premise check and fix the design where it fails (expert stacking first).
2. Vendored kernel patches: rows `[0, B)` of `max_rows`-row state tensors (**applied**;
   add the CPU test: export R = 8 vs R = B for B in {1, 2, 4}, compare Mosaic text after
   renaming `memref<8x(69|24)x` to `memref<Bx..>`, and assert no `.at[:, state_index]`
   remains); in-kernel uint8→uint32 expert ref bitcast; live-row mask so padded rows stream
   no experts.
2b. Torch op for `program.py`: register with the bridge plus a custom op that marks the four
   states mutated (GDN pattern: `donate_argnums` + `input_output_aliases`, fake returns
   logits only); confirm on CPU the export carries `tf.aliasing_output` on the states.
   Open choice: mesh in torch-rank order (shares SGLang's EP8 experts as is) differs from
   upstream's physical order inside a host; measure on TPU.
3. Handoff op (SGLang latent + MambaPool → megakernel row), verified on CPU against the
   vendored reference math.
4. Model-side integration: decode-mode routing for buckets ≤ 8 in `kimi_k3.py`, slot
   lifecycle hooks (prefill done → handoff; finish/abort/retract → release + compaction move),
   logits → SGLang sampler.
5. Weight path: stacked experts shared with SGLang's MoE, dense families from the checkpoint.
6. Then one TPU run: megakernel standalone baseline + the integrated SGLang mode
   (accuracy smoke + TPOT at bs1 8k and bs4 1k).
