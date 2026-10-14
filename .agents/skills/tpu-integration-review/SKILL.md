---
name: tpu-integration-review
description: Use this when you write or review any change that adds or alters TPU behavior in sgl-project/sglang-torchtpu. That covers a Pallas kernel, model enablement, compile policy, attention or MoE backend, KV/state pool, scheduler or spec-decode path, P/D (Raiden) transport, distributed or multihost launch, env var or CLI flag, TPU test or CI job, or a perf or dependency bump. It also covers any edit to shared `python/sglang/` code motivated by TPU. Each rule says where TPU code must live and which seams it must go through. It lists the compile, host-device, memory, state-lifecycle and SPMD invariants reviewers enforce, and the tests and evidence that get a PR merged. The rules come from the repo's own review history, ranked by how often they were raised and how severe the failure is.
---

# TPU Integration Review Rules (sglang-torchtpu)

## 1. Purpose & how to use

These rules explain how new code reaches SGLang on TPU, and what reviewers check. The stack is `torch_tpu`: PyTorch PrivateUse1 (`device="tpu"`), lowered to XLA HLO, executed by PJRT/libtpu.

- **Authors:** walk §4 before opening a PR, then attach the evidence listed in references/rules.md §5.
- **Reviewers:** use §4 as the walk order, cite rule IDs in comments, and use severity as defined in `AGENTS.md`:
  - **P0:** functional, correctness, reliability, critical performance or critical design problem.
  - **P1:** reachable wrong behavior, possible bottleneck, debatable choice, readability or extensibility problem.
  - **P2:** nit.
- **Corpus:** 1,203 PRs, of which 357 had human review discussion. 2,071 inline review comments and 226 review summaries. Mined 2026-10-14.
  - Mined as 8 chronological shards (#2 → #1345). Older shards cite `python/sglang/srt/hardware_backend/tpu/...` paths; that code now lives in `sglang_tpu/`.
  - Every rule traces to cited PRs.
- **Freq:** the number of distinct PRs where reviewers raised or enforced the rule, summed across shards.
- **Ordering:** within a section, rules are sorted by freq × severity weight (P0=3, P1=2, P2=1).
- **Policy precedence:** where older practice conflicts with `AGENTS.md` or `sglang_tpu/README.md`, current policy wins. An *Evolution* note records the change.

---

## 2. Integration Map: the seams code must go through, in preference order

TPU-only code lives in `sglang_tpu/` and is installed as the `sglang-tpu` plugin next to the fork in `python/`.

- Core must not import `sglang_tpu`, and must not grow new `is_tpu()` branches.
- `tools/plugin/check_tpu_branches.py` (pre-commit) ratchets both counts against `tools/plugin/tpu_debt_baseline.json`.
- `test/registered/unit/plugins/test_tpu_oot_platform_contract.py` must stay green.

| # | Seam (try in this order) | What goes there | Concrete modules/examples from the corpus | Rules |
|---|---|---|---|---|
| 1 | **Existing platform method / capability flag.** The default keeps upstream behavior. | Device capability differences and TPU defaults. | `current_platform` → `TpuSRTPlatform` (`sglang_tpu/tpu_platform.py`, mode A) / `TpuOOTPlatform` (`sglang_tpu/platform.py`, mode B).<br>Examples: `supports_overlap_streams()` (#23), `get_compile_backend()` / `supports_torch_compile()` (#66), `weight_loading_context()` (#105), `get_mha_kv_bytes_per_token()` (#956), `supports_safetensors_device_load()` (#1225), `on_warmup_complete()` (#1178), `get_attn_backend_cls_str()` (#1223), `get_hicache_explicit_size_context()` (#1283), `apply_server_args_defaults()` (#48, #224, #1160). | PLC-2, PLC-6, XBE-1, CFG-5 |
| 2 | **Registry filled from `init_backend()`** | New attention backends, fused-op forwards, factories, stack strategies, runtime patches applied at activation. | `register_attention_backend`, `BaseFusedOp.register_oot_forward` (README), `register_stack_strategy()` (#1046), TPU dsv4 factory registered from `sglang_tpu/tpu_platform.py` (#1215), PD queue-limit guard applied from `init_backend()` in `sglang_tpu/tpu_platform.py` (#1191). | PLC-11 |
| 3 | **`register_resolution_hook`** | Server-arg resolution that needs TPU knowledge (pool choice, layouts). | HiCache pool registered from TPU classes instead of `hicache_hook.py` importing `tpu_platform` (#890). | PLC-11 |
| 4 | **Subclass-and-override via a `get_*_cls()` factory** | TPU pools, host pools, quant methods, samplers, workers, attention-backend hooks. | `get_mha_kv_pool_cls()` (#15, #44), `get_mha_kv_pool_host_cls()` (#148), `get_mamba_pool_cls()` (#271), `HiCacheController.create_for_backend` (#44), `TpuModelOptNvFp4FusedMoEMethod` (#186), `TpuSampler` backend (#364), `TpuDFlashWorker` / `DFlashBackendState` (#812), `DSV4TPUAttnBackend.prepare_mqa_qkv` (#1215), parent-class `input_patch_count()` hook (#1209). | PLC-5, KVS-2, KVS-6 |
| 5 | **Upstream seam PR, carried as a `sglang_tpu/hooks/` entry until it merges** | A seam core doesn't have yet. Needs a linked upstream seam issue. | `hooks.register` entry points (`tpu_hooks` / `tpu_oot_hooks`, README).<br>`tools/plugin/inventory.py` lists every hook and patch.<br>New `current_platform.<m>()` seams need an issue link (#827). | PLC-6, PLC-3 |
| 6 | **Model-class capability attributes** | Per-model opt-ins read by plugin code; no `isinstance`. | `relays_pp_aux_hidden_states` (#935), `supports_tpu_ep_local_expert_reads` (#1084), `decode_embeds_in_language_model` (#1085), `allow_tpu_distributed_top_k` (#364). | PLC-13 |
| 7 | **Compile policy, bucketers, compile cache** | Compile boundaries, warmup grids, padding, specialization keys. | `support_torch_compile(cls, policy=...)` (#66, #360), `DecoderTpuCompilePolicy` / `SpecDecodeCompilePolicyMixin` / `_extend_compile_cap()` (#196), `VisionTpuCompilePolicy`, `ViTTpuCompileRunner` (#654), `CompiledForwardShim`, `direct_compile` + `_direct_compile_specialization_key` (#306, #872), `Bucketer.next_bucket()` (#168), `policy.max_padded_tokens()` (#661), plugin-owned `sglang_tpu/compilation/core_compile_annotations.py` (#1162), `compile_cache.py` kernel fingerprint list (#909, #1190), `RecompilationChecker` / `SGLANG_TPU_CHECK_RECOMPILATION[_SITES]` (#83, #1178). | CMP-* |
| 8 | **Pallas export: `cpu_export_jax_op`** | Every Pallas/JAX kernel exposed as a torch op. | `cpu_export_jax_op` (#237, #257), `tpu_info_for_cpu_export()` (#394), `register_fake` (#1223), `pallas_export.argnums(fn, *names)` (#1190).<br>Kernels go in `sglang_tpu/kernels/<name>/` (+ `_jax_entrypoint.py`, #515), attention in `sglang_tpu/attention/pallas`, MoE in `sglang_tpu/moe/pallas`.<br>Missing aten kernels go in `sglang_tpu/patches.py` (#1227). | KRN-*, PLC-1 |
| 9 | **Pool builders / configurators** | KV, SWA, mamba and host pools, their layout, and sizing. | `KVCacheConfigurator` (#319), `TpuFusedMHATokenToKVPool` (#38, #152), `kv_pool_host.py` + pool-declared `kv_offload_io_backends` (#853), `get_req_to_token_extra_context_len()` (#198), shared RPA kernel spec for layout/page size (#474). | KVS-*, MEM-4 |
| 10 | **Config registries** | Env vars, CLI flags, validation. | New TPU-only env vars go in `sglang_tpu/envs.py`; existing `SGLANG_TPU_*` stay in `sglang.srt.environ` (`Envs`, `EnvBool`, `.get()` / `.override()`). Never declare one name in both.<br>`ServerArgs` validation / `_resolve_layout_io_compatibility` (#113), `apply_server_args_defaults()`. | CFG-* |

**Guardrails a reviewer runs**
- `python tools/plugin/check_tpu_branches.py` (or `pre-commit run check-tpu-branches --all-files`).
- `pre-commit run check-registered-tests --all-files`.
- `test_tpu_oot_platform_contract.py`.
- `tools/plugin/scoreboard.py` for debt counters.

---

## 3. Top 25 rules (by frequency × severity)

Full rule text, evidence, and checks for all 120 rules: [references/rules.md](references/rules.md).

| ID | Sev | Freq | Rule (abridged) |
|---|---|---|---|
| EVD-1 | P1 | 55 | Perf claims need a matched before/after giving: exact server and client commands (via `benchmark/tpu/launch_server.sh`); model, chip, TP/EP/DP, ISL/OSL, concurrency, num prompts; torch_tpu wheel and main vs PR SHAs; repeat counts, with warmup and compile excluded (report cold vs warm… |
| CMP-1 | P0 | 26 | Zero serving-time compiles. Every shape, dtype, mode, sampler specialization, spec phase and calling convention reachable in serving must be in the precompile/warmup set. Gate optional variants on capability (e.g. `is_multimodal`). Alternate paths (EAGLE with `skip_attn_backend_init`, split… |
| SYN-1 | P0 | 23 | No D2H reads on device tensors in per-step prefill, decode, scheduler, offload or spec paths: `.item()`, `.tolist()`, `.cpu()`, `.to("cpu")`, `.numpy()`, or Python conditions on tensors.; Use the `*_cpu` mirrors; if data should already be on host, assert it.; Vectorize per-request Python loops… |
| CFG-1 | P0 | 20 | Reject unsupported or incompatible combinations at `ServerArgs` validation or backend `__init__`, before the model loads. Use an actionable, model-specific message (the unmet requirement, and whether the cause is untested or a hard limit), and keep the README consistent. Never raise… |
| TST-1 | P0 | 20 | Tests must fail on revert: exercise the load-bearing production path (call `prepare_tpu_w4a16_moe_weights` rather than re-implementing packing; drive a real `Req`); never mock away the function under test; cover the default configuration, not only the fallback; name the guarding test in the PR,… |
| SCH-1 | P0 | 17 | Padded and ghost rows are inert: kernel active counts (`distribution`, `num_active`) use non-empty real rows; padded ids go to sentinels no kernel computes (valid expert id with weight 0, null slot), never `-1` into Pallas DMA; initialize padding buffers (no 0·NaN leakage); derive counts… |
| PLC-1 | P1 | 25 | Put new TPU-only code in `sglang_tpu/`. That includes helpers, kernels (`kernels/<name>`), attention backends (`attention/`), runners, patches and `*_tpu` modules. Never put it inline in shared `managers/`, `models/`, `speculative/`, `mem_cache/` or `layers/`, and never in the re-export shims. Keep… |
| XBE-1 | P0 | 16 | Where a TPU change touches a multi-backend function, CUDA/ROCm/NPU/CPU must compute exactly what they did before. Check: early returns, `elif` chains and indentation; unconditional transforms (`round_up`) and guards; that base-class and platform defaults remain the old behavior.; Either gate on… |
| TST-2 | P0 | 16 | Device-dependent changes need registered TPU coverage. Kernel numerics run eager *and* compiled (`backend="tpu"`), multi-rank where relevant, and drive the real impl above the kernel threshold. Spec, DP, PP and P/D changes get an e2e PR-gate test. CPU mocks or selection-logic tests alone are not… |
| EVD-2 | P1 | 23 | Accuracy evidence: an eval (GSM8K/MMLU/GPQA/MMMU) beyond a smoke prompt, with a statistically meaningful sample; isolate the new path when thresholds split traffic; compare against the baseline backend or main; bug fixes include an ablation (without fix → fail); collective/MoE/SparseCore… |
| PLC-2 | P1 | 22 | Express TPU differences as a `current_platform` method or capability whose default is today's upstream behavior. Don't add inline `_is_tpu` / `device == "tpu"` / `torch.tpu.*` in generic code. Hook names must not hide side effects (a `get_*` must not mutate global config). |
| PLC-3 | P1 | 22 | A workaround for a torch_tpu, libtpu, PJRT, JAX or PyTorch gap must: be fixed in the right layer where possible; otherwise carry a `TODO(<upstream issue>)` with a removal condition and be reverted when the pin lands.; Use only public APIs of dependencies (no `torch_tpu._internal`, raiden `_impl`).… |
| TST-3 | P1 | 21 | New kernels, dispatch wrappers, weight transforms and helpers get reference-comparison tests (vs pure torch/JAX, or the naive implementation) covering boundaries: sentinel ids, chunk sizes, activations; non-aligned dims or the raise; partially filled buckets, `active_rows == 0`; exact threshold… |
| SYN-2 | P0 | 13 | Under overlap scheduling, order every DMA that touches KV pages against in-flight forwards: Raiden H2D/D2H, PD transfer, HiCache load/backup, prealloc.; Prefer fencing the *free* over fencing the *load*; don't disable a feature to dodge a race.; With buffer donation, only the forward-dispatch… |
| PLC-4 | P1 | 19 | Reuse before you write: Use existing helpers, fields and predicates (`next_power_of_2`, `copy_or_rebind_param`, `is_dp_attention_enabled()`, `seq_lens_cpu`, `get_layer_id`).; Don't vendor a second copy of a kernel, mesh, tuning table or bridge.; Keep one canonical backend per capability.; Reuse… |
| CFG-2 | P1 | 19 | Declare env vars centrally: new TPU-only vars go in `sglang_tpu/envs.py`; existing `SGLANG_TPU_*` stay in `sglang.srt.environ`; never both; typed `Env*` descriptors, read via `.get()` and overridden in tests via `.override()`, with no raw `os.environ` / `os.getenv`; `SGLANG_TPU_[ENABLE_]*` names… |
| CMP-2 | P0 | 12 | Warmup and precompile must call the real serving functions, through the parent `_make_dummy_forward_batch` / `_run_one`, not a hand-written prewarm copy. They must reproduce serving's global state: grad mode, `requires_grad`, default dtype, dummy dtypes derived from the allocator/pool… |
| TST-4 | P1 | 18 | TPU test conventions: `test_tpu_<model>_<topology>_<feature>_<eval>.py` under `e2e/` (not `integration/`), with `_smoke` when there's no eval, `multihost` not `mh`, and the `MODEL_CONFIG` / `model_config_main` pattern; shared P/D helpers; weights from the `/weights` PVC via… |
| PLC-5 | P1 | 17 | When a shared class needs several TPU differences, choose one of: a TPU subclass selected by a factory or backend; an override (`_weight_loader_impl`, `_accept_block`); an optional injection point (`sampling_fn`); a hook on the parent interface.; Don't add `if _is_tpu` inside the shared method.… |
| TST-5 | P1 | 17 | CI reliability and economy: graceful teardown (class fixtures + `addClassCleanup`); curl and per-file timeouts derived from `est_time`; propagate worker failures; checkpoint logs; Bazel TPU tests `exclusive`; k8s name-length limits; reuse nightly workflows and the smallest topology that exercises… |
| EVD-3 | P1 | 17 | Scope: split infrastructure with a large blast radius from the feature that uses it; split independent fixes, formatting or pre-commit churn, cherry-picks and rebase noise; explain non-obvious cross-subsystem coupling; refactor PRs are behavior-neutral (order, exceptions, early returns, log… |
| KRN-1 | P0 | 11 | Register Pallas/JAX ops through the repo's `cpu_export_jax_op`: no raw `pallas.jax_op`, no `Mesh(jax.devices())`; late JAX imports, `keep_unused=True`, `register_fake`; static args in the op key; register before tracing via an idempotent cached helper; on partial failure, mark the feature… |
| CFG-3 | P0 | 11 | Fail loudly: never silently mutate user config after the fact (`disable_radix_cache`, layouts); never accept inputs and ignore them (EPLB mapping, ignored args); never warn-and-continue on a violated invariant; assert impossible states; make error policy explicit (`missing_ok`); keep… |
| KVS-1 | P0 | 10 | A TPU subclass of a pool, host pool or controller must honor the full parent contract or raise `NotImplementedError`: `get_cpu_copy` / `load_cpu_copy`, `move_kv_cache`, `get_contiguous_buf_infos`, `kv_buffer`.; Always call `super().__init__`; assert on ignored constructor args.; Methods the new… |
| DST-1 | P0 | 10 | Every SPMD rank traces an identical program. Rank-dependent values (rank, local pad, shard offset) enter as device data: an eager 0-D tensor or a non-persistent buffer. Never Python ints, `lru_cache`d helpers or rank branches. Fed shards share shape, stride and storage offset (clone views to offset… |

## 4. Integration Review Checklist (walk order)

- [ ] **Placement:** TPU-only code is in `sglang_tpu/`, not shared files or shims (PLC-1). Kernel and integration code are separate (PLC-1, KRN-2).
- [ ] **Seam:** an existing platform method, registry, resolution hook, factory subclass or `hooks/` entry was used, in that order (PLC-2, PLC-5, PLC-11).
- [ ] No core import of `sglang_tpu`. `check_tpu_branches.py` passes, the baseline is unchanged or justified with a seam issue, and the OOT contract test is green (PLC-6).
- [ ] No monkeypatches or proxies. Unavoidable branches are explicit `is_tpu()` and visible (PLC-9, PLC-12, XBE-2).
- [ ] TPU/JAX imports are lazy in shared and CPU-CI modules (PLC-7). There are no TPU fields or params on shared structures (PLC-14).
- [ ] **Other backends:** non-TPU control flow and defaults are unchanged, or a GPU run is attached and called out (XBE-1, XBE-4). Blast radius is enumerated for `mem_cache`, process groups and disagg common (XBE-3).
- [ ] **Compile:** every reachable shape, mode and calling convention is precompiled through the real path, and state matches warmup (CMP-1, CMP-2).
- [ ] Specialization and persistent cache keys cover everything baked into the artifact, and new kernel files are fingerprinted (CMP-7, CMP-9).
- [ ] No mutable Python state, graph-breaking ops or eager per-shape ops on the serving path. Over-bucket inputs fall back explicitly (CMP-3, CMP-5, CMP-6, CMP-12).
- [ ] Buckets come from the policy's single source and cover `max_running_requests ÷ dp` (CMP-4).
- [ ] **Syncs:** no new D2H on device tensors in hot paths; mirrors are used and shapes are batch-proportional (SYN-1, SYN-9).
- [ ] Overlap: KV DMAs are fenced (free over load), donation ownership is respected, and mirrors are fresh (SYN-2, SYN-8).
- [ ] Async copies: real events, propagated errors, buffers that outlive the DMA, bounded polls (SYN-5).
- [ ] **Memory:** no whole-buffer copies or relayouts; old weights freed safely; pool bytes accurate (MEM-1, MEM-2, MEM-3, MEM-5).
- [ ] **State lifecycle:** pool subclasses honor the parent contract; guard removal is verified (KVS-1).
- [ ] Abort and free have exactly one owner, with nothing released under in-flight native IO. Every hybrid pool is handled (KVS-2, KVS-3).
- [ ] Page tables include the spec headroom; guards exist on both host and device branches (KVS-4). Shortfalls fail the request, not the scheduler (KVS-7, SCH-4).
- [ ] **Scheduler:** padded rows are inert, the logical batch size is used, and outputs are trimmed once (SCH-1, SCH-2). Parallel fields and mirrors are padded together (SCH-6).
- [ ] Every ForwardMode is checked; unsupported request features are rejected at intake (SCH-3, SCH-9).
- [ ] **Distributed:** rank-invariant programs; idle and uneven ranks run identical collectives (DST-1, DST-2).
- [ ] Collectives go through registered group APIs; subgroup and topology mapping are correct; EP/PP gating is in place (DST-3, DST-5, DST-8).
- [ ] **Kernels:** registered via `cpu_export_jax_op` with fakes; input contracts asserted; VMEM modeled; int32 indices and f32 carries (KRN-1, KRN-4, KRN-5, KRN-6).
- [ ] **Config:** unsupported combos rejected at startup; no silent overrides; exact predicates (CFG-1, CFG-3, CFG-6).
- [ ] Env vars live in `sglang_tpu/envs.py` (new) or `environ.py` (existing), read through `Envs`, and are minimized (CFG-2, CFG-4). TPU defaults are applied in `apply_server_args_defaults` (CFG-5).
- [ ] **Tests:** a test that fails on revert exercises the production path (TST-1).
- [ ] TPU coverage exists, eager and compiled, with an e2e PR gate for spec, DP or PD (TST-2). Boundary cases are covered (TST-3, TST-10).
- [ ] Tests are registered and ran on the head, follow e2e conventions and `/weights`, and strictness wasn't weakened (TST-4, TST-6, TST-9).
- [ ] **Evidence:** a matched perf A/B with commands and warmup excluded (EVD-1), an isolated accuracy eval plus ablation (EVD-2), and recompile stats (EVD-12).
- [ ] The validation statement includes the wheel and what wasn't run (EVD-7). A 480B run is attached for pins, syncs or shared backends (EVD-9).
- [ ] Scope is tight and behavior-neutral where claimed, deferrals are tracked, and comments and docs are accurate (EVD-3, EVD-6, EVD-8).


## 5. Reference files

- [references/rules.md](references/rules.md): all 120 rules (PLC, XBE, CMP, SYN, MEM, KVS, SCH, DST, KRN, CFG, TST, EVD) with evidence and checks; anti-patterns (§4); blockers and expected evidence (§5); contested items (§6); reviewer focus map (§7).
- Repo policy that takes precedence: `AGENTS.md`, `sglang_tpu/README.md`.
- Guardrails: `tools/plugin/check_tpu_branches.py`, `tools/plugin/inventory.py`, `tools/plugin/scoreboard.py`, `test/registered/unit/plugins/test_tpu_oot_platform_contract.py`.
