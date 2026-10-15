# GLM-5.3-Flash on TPU: prefix cache, KV offload, PD — readiness (2026-10-14)

Code review of branch `glm-dsa-tpu-rebase` @ 4eb37186f8. Boot-level confirmation runs
are pending (both slices are on AIME26 overnight). Flash is a hybrid model: 34 KDA
(recurrent-state) layers + 11 MLA (latent KV) layers, served with DFlash or MTP.

Benchmarks never exercised any of these: `benchmark/tpu/recipe/configs/base.py` sets
`disable_radix_cache: True` for every FP8 recipe.

## 1. Prefix cache (radix) — not supported for KDA models on TPU

- `TpuPlatform.supports_mamba_prefix_cache` returns True only for GDN models:
  "KDA shares GDN metadata setup, but does not pass snapshot tracking to its kernel yet."
- `attention_kda.py` (`TpuKDAAttnBackend._run`) never passes `track_indices` / `track_end`
  to the fused kernel, so no reusable KDA state checkpoints are written.
- With spec decoding there are two further blocks:
  - the fused kernel rejects radix snapshots during a speculative window
    ("Radix snapshots require dense, non-speculative states");
  - the TPU read-offset verify commit asserts tracking is off
    (`hybrid_linear_attn_backend.update_mamba_state_after_mtp_verify`:
    "read-offset commit runs with radix off").

What it takes:
1. Non-spec KDA tracking: pass `track_indices`/`track_end` through `attention_kda.py`
   (the kernel's KDA mode already supports them for non-spec batches); flip
   `supports_mamba_prefix_cache` for KDA. Small.
2. Spec (DFlash/MTP) + tracking: snapshot the state at the track boundary inside a
   verify window and commit it with the accepted length (the read-offset commit would
   pick the checkpoint row at the boundary). Kernel + commit work. Medium/large.
3. The MLA full-attention layers and the drafter's KV need nothing special.

## 2. KV offload (HiCache / Raiden) — not supported for this model

- `TpuPlatform.validate_mamba_prefix_cache` refuses `--enable-hierarchical-cache` for
  hybrid recurrent models: host stores expect GPU-layout stacked states; TPU keeps one
  kernel-layout tensor per layer. (This applies even to GDN.)
- `TpuRaidenHostKVCache` mirrors a fused **MHA** pool (`TpuFusedMHATokenToKVPool`) and
  raises for spec draft pools: "MTP draft KV offload is not supported on the TPU raiden path."
- The on-device Raiden tests cover MHA + SWA + an EAGLE draft sidecar
  (`test/manual/tpu/test_tpu_raiden_hybrid_spec_offload_ondevice.py`), not MLA latent
  pools or recurrent state.

What it takes: prefix cache (1) first; then a Raiden host pool for the fused MLA pool,
a host layout/transfer for per-layer KDA state, and draft-pool offload (DFlash/MTP).
Large.

## 3. PD disaggregation (Raiden) — MLA part close, KDA state missing

- Raiden PD (`sglang_tpu/disaggregation/raiden/`) transfers full-attention KV plus
  sliding-window (`StateType.SWA`) pages only; there is no recurrent-state
  (`StateType.MAMBA`) path, so the prefill rank cannot hand the KDA state to decode.
- The fused MLA pool exposes `kv_buffer` / `get_fused_kv_buffer`, which is what
  `RaidenKVCacheGroups` needs, so the MLA latent transfer should mostly work.
- Spec decoding under PD also needs the drafter's state (DFlash/MTP hidden/KV) at
  decode start.

What it takes: a MAMBA state type in the Raiden PD connector (register the per-layer
KDA conv + recurrent tensors, send one slot per request after prefill), plus drafter
state handoff. Medium/large.

## Progress

- 1.1 done, untested on device: `0994287ea4` passes the tracking plan through
  `attention_kda.py` and reports KDA prefix-cache support; spec decoding + radix is
  refused at startup with a clear error until 1.2.
- 1.2 design (needs a decision): a verify already writes one state checkpoint per
  window position (rows `slot * W + t`). When the radix track boundary lands inside the
  *accepted* prefix, copy checkpoint row `slot * W + step` (conv + recurrent, all 34
  layers) into the track slot after acceptance. A per-layer copy is ~68 launches per
  verify round and would undo the decode host-time work; do it as one fused
  all-layer copy (masked no-op when no boundary was crossed), or defer it to just
  before the next verify overwrites those rows.

## Proposed order

1. Confirm each with a simple boot/run test once a slice is free (expected: refusal at
   startup for 1-2, KDA state missing for 3).
2. Implement prefix cache (1.1 then 1.2) — prerequisite for offload and useful alone.
3. PD MAMBA state type (3), then KV offload (2).
Validation order per the plan: DFlash first, then MTP; simple -> corner cases -> AgentX
(<= 1 h; AgentX has no reference answers, so correctness comes from greedy on/off
identity and scored sets).

## Debt: frozen-compiler robustness (found 2026-10-14)

After warmup the TPU server freezes torch.compile (`fail_on_recompile`), so any graph
signature traffic produces that warmup did not — shape, stride, or input aliasing —
kills the scheduler. Batched AIME with MTP crashed this way: in `_fused_draft_steps`
the draft's `positions` and `seq_lens` were sometimes one tensor and sometimes two
(padding to a bucket creates fresh tensors; an exact-bucket batch does not), and a
`[steps, 1]` `out_cache_loc` arrived with two stride layouts. Production traffic
(concurrent, mixed prefill/decode) would hit this.

- Instance fixed: `f8587d4ffe` (alias positions to seq_lens when they agree; canonical
  out_cache_loc strides).
- To do: (1) canonicalize strides of size-1 dims and make aliasing deterministic for
  every compiled call in the compile shim; (2) on an unseen signature, log and
  recompile instead of crashing (targeted, keeping skip-guard otherwise); (3) a
  concurrency stress test (c1 -> c32, mixed lengths, staggered arrivals, aborts) for
  DFlash and MTP.
- AIME eval configs set `SGLANG_TPU_ALLOW_SPEC_RECOMPILE=1` meanwhile (full guards,
  recompile on a miss; costs up to the skip-guard gain, ~0.36 ms TPOT at c1).
- DFlash at 32+ concurrent: a recompiled RPA verify graph failed with
  `Expected cu_q_lens.shape=(33,) to be (32,)` (metadata one sequence longer than the
  compiled batch). DFlash AIME runs at concurrency 8 (eval-proven) meanwhile.

## Smoke tests on device (2026-10-15, HEAD 65fba312f6, DSA, no spec, psdr + x5kk)

| Feature | Result |
|---|---|
| Prefix cache (radix, `dsa_nospec_radix`, 16 KDA state slots) | **Boots and hits.** 6571-token prompt: cold 0 cached / 0.65 s; repeat 6528 cached / 0.38 s; shared-prefix question 6528 cached / 0.35 s. **Suspect:** the greedy repeat's first token differs from the cold run (`" Station 5."` vs `"</think>Station 5."`), so restored state may not match a fresh prefill. Needs a logprob comparison on several prompts. |
| KV offload (Raiden HiCache, `dsa_nospec_hicache`) | **Refused at startup** by `TpuPlatform.validate_mamba_prefix_cache`: `--enable-hierarchical-cache is not supported for TPU GDN prefix caching`. Needs a host store for per-layer KDA state and Raiden support for the split DSA pool. |
| PD (Raiden, `dsa_pd`, prefill psdr / decode x5kk) | Upstream refuses `--language-model-only` with PD (`model_hook.py`); with that relaxed (smoke-only patch, not committed) both engines fail in `init_disaggregation`: `get_contiguous_buf_infos is not supported on the TPU split MLA pool`. The DSA split pool exposes no token-addressable buffers for transfer; KDA state has no Raiden state type either. |

Radix needs about mamba_ratio (5) KDA state slots per running request: with the c1 recipe's 3 slots the server refuses to start.

## Update 2026-10-15 (later): prefix cache verified, PD working, large-pool fix

**Prefix cache (no spec):** works and is correct. KL consistency (kl_test_utils, same server,
24-32 LongBench prompts, 512 tokens): DSA no-cache 0.0063 vs prefill-from-cache-hit 0.0078;
dense 0.0062 vs 0.0070 (decode-on-warm-cache 0.0034). Prompts with decoded tokens hit too
(3072 of 3300 cached). Sizing: radix needs KDA state slots for running requests plus one
snapshot per cached prefix; undersized pools drop hits silently (both test failures were this).
The earlier "chunk-split logprob gap" is MoE routing noise, not a bug (no split pattern agrees
with any other; KL level matches dense).

**PD (Raiden, no spec, same TP):** works end to end on two 2x2x1 slices. Commits f03ec6b319
(split DSA pool page buffers), 4efe5a26f8 (KDA state as Raiden's second namespace, one slot per
request on the closing read), 14dabab821 (language-model-only + PD), 5fe5a0c2bc (no separate
DSA state component). Pods need tpu-sync-torch==0.0.1.dev20261008082921 (base image has the
Sep 30 build without poll_stats_with_details; bootstrap-port-pod.sh installs it). 5-prompt check
vs single server: 2 identical, 3 same answers in different words (incl. two 3.3k-token prompts).

**DSA with KV pools > 131072 tokens:** the SparseCore cache writer halts the SparseCore once a
layer's NoPE buffer passes 64 MiB. 0243899b72 uses the XLA writer above that size.

**KV offload:** still refused at startup for KDA hybrids (next).
