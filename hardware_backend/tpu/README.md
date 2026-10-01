# sglang-torchtpu (wave 0 skeleton)

The out-of-tree TPU platform plugin. Wave 0 creates the boundary; **no TPU code has moved
yet**. See `plugin-refactor-proposal.md` sections 5, 6 and 14 in the launch-plan repo.

## What this is today

| Piece | State |
|---|---|
| `sglang_torchtpu.platform.TpuOOTPlatform` | subclasses the in-tree `TpuSRTPlatform`, declares `PlatformEnum.OOT`, adds the methods the OOT gates require |
| `sglang_torchtpu._legacy` | re-exports the in-tree TPU package so new code can import from a stable path |
| `sglang_torchtpu.hooks` | registration point for `sglang.srt.plugins`; empty, by design |
| `sglang_torchtpu.compat` | version-keyed shims across sglang releases |
| entry point `tpu_oot` | **opt-in only** -- `activate()` returns `None` unless `SGLANG_PLATFORM=tpu_oot` |

## Why the entry point is opt-in

Declaring `PlatformEnum.OOT` is not behaviour-neutral. The fork's core tree has 27
`is_out_of_tree()` gates; 8 already read `is_out_of_tree() or is_tpu()` and are unaffected,
but the rest change which branch they take. Two are known to need work before the flip can
be the default -- see "Known divergences" below. Until those are closed, the default path
(`tpu`, the in-tree entry point) is untouched and this package is selected explicitly.

## CI modes

| Mode | Install | `SGLANG_PLATFORM` | Proves |
|---|---|---|---|
| A | `-e python/` | unset (`tpu`) | the fork is not regressed |
| B | `-e python/ -e hardware_backend/tpu/` | `tpu_oot` | the OOT platform drives the same code |
| C | upstream wheel + this package | `tpu_oot` | the real target; needs relocated subsystems |

Mode C is only meaningful for code that has actually moved, so it stays red until wave 4.

## Known divergences (must close before `tpu_oot` becomes the default)

Branch-routing changes, which need a hardware run to clear:

1. **`mem_cache/kv_cache_configurator.py:1318` vs `:1352`.** The generic OOT branch precedes
   the TPU branch in the same `elif` chain. Under `PlatformEnum.OOT` the OOT branch wins and
   hybrid-SWA models would route to `_build_oot_mha_kv_pool` instead of `_build_tpu_swa_kv_pool`.
   Fix: fold the TPU branch's SWA handling into the OOT branch, then delete the TPU branch.
2. **`model_executor/model_runner.py:250`.** `init_backend()` is called only on the OOT path.
   Today TPU reaches it another way; under OOT it would also fire here. Verify there is no
   double initialisation before flipping.

Gate-reachable platform methods nobody answers, found by the gate audit. Each is listed in
`KNOWN_UNANSWERED` in the contract test with its reasoning, and that list may only shrink:

3. **`get_torch_profiler_activity` / `..._str`.** `[Planned]` stubs on `DeviceMixin` that no
   platform implements, reached from pure-OOT gates in `utils/profile_utils.py`. Torch-profiler
   startup raises under `tpu_oot`. Needs a `torch_tpu` ProfilerActivity decision on hardware.
4. **`get_device_core_count`.** Defined on no platform class at all, yet called behind
   `is_out_of_tree() or is_tpu()` at `utils/common.py:1012` -- so it is already broken on the
   in-tree platform, not a regression from the flip. Its callers (triton/wave backends,
   batch-invariant ops) are all unreachable on TPU, which is why nobody has hit it. Upstream's.

`tools/plugin/gate_audit.py` regenerates this inventory; run it after every upstream sync.

## Tests

The contract test lives at `test/registered/unit/plugins/test_tpu_oot_platform_contract.py`,
**not** in this package: `test/run_suite.py` discovers registered tests by globbing
`test/registered/**`, so a `register_cpu_ci` marker anywhere else is collected by no job. It
runs on CPU, imports this package from the worktree when it is not pip-installed, and recomputes
the gate inventory from source on every run -- an upstream sync that adds a gate calling a new
platform method fails it.

`tools/plugin/check_tpu_branches.py` is the other half: a per-file ratchet on `is_tpu` branches
in shared core and on TPU modules outside the in-tree package. It runs as a pre-commit hook.

## Verification status

Everything above is verified on CPU. Mode B -- this platform driving the full TPU suite on real
hardware -- **has not been run**, and is the definition of done for wave 0.
