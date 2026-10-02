# Handoff — wave 0, the out-of-tree plugin boundary

**Branch:** `wave0/plugin-boundary` · **Base:** `main` @ `11eb70e4ce` · **Repo:** `sgl-project/sglang-torchtpu`

You have TPU hardware. I did not. Everything below that is still open needs a machine that can
actually run TPU code, which is why this stops where it does.

This document is self-contained: the full proposal lives outside the repo, so every number and
decision you need is repeated here.

---

## 1. What wave 0 is, and what it is not

The repo is being reshaped so that TPU support becomes an out-of-tree plugin of sglang instead of a
fork. Wave 0 creates the **boundary** and moves **no code**. Its point is that new TPU work has a
home with the right dependency direction, so the pile stops growing while the existing pile is
migrated later.

What wave 0 deliberately does *not* do:

- It does not move any of the ~57k lines of TPU-only code. Those files exist only in the fork, so
  they never conflict during an upstream sync; relocating them saves zero merge cost. All sync pain
  is in the ~152 modified core files, and that is the seam work of waves 1-3.
- It does not change the default platform. See §3.

---

## 2. What is on the branch

Everything lives under `hardware_backend/tpu/`, a new top-level directory holding one Python
distribution.

| Path | What it is | State |
|---|---|---|
| `hardware_backend/tpu/pyproject.toml` | dist `sglang-torchtpu`, package `sglang_torchtpu`, entry points `tpu_oot` (platform) and `tpu_oot_hooks` (plugins) | done |
| `hardware_backend/tpu/README.md` | package state, CI modes, known divergences | done |
| `sglang_torchtpu/__init__.py` | `activate()`; returns `None` unless `SGLANG_PLATFORM=tpu_oot` | done |
| `sglang_torchtpu/platform.py` | `TpuOOTPlatform(TpuSRTPlatform)`: `_enum = OOT`, `is_tpu() -> True`, plus the 6 methods the OOT gates need | done, **unverified on hardware** |
| `sglang_torchtpu/_legacy.py` | stable import path into the in-tree TPU package, with a migration-aware error | done |
| `sglang_torchtpu/compat/__init__.py` | `first_available()` for name drift (`dsa` vs `nsa`) | done |
| `sglang_torchtpu/hooks/__init__.py` | `sglang.srt.plugins` registration; empty by design, with the debt-header policy | done |
| `hardware_backend/tpu/tests/` | directory exists, **empty** | TODO |
| `tools/plugin/` | directory exists, **empty** | TODO |

Nothing under `python/` was touched. `git diff main --stat -- python/` is empty, by design.

---

## 3. The one design decision you should agree with before continuing

**The new platform is opt-in, and the existing `tpu` entry point is untouched.**

I originally planned to flip the in-tree platform's `_enum` to `PlatformEnum.OOT` and rename the
existing entry point to `tpu_intree`. The gate audit says that is not safe yet. So instead
`activate()` declines unless `SGLANG_PLATFORM=tpu_oot` is set explicitly. Default behaviour on this
branch is bit-identical to `main`.

Why it is not safe yet — two known divergences:

1. **`python/sglang/srt/mem_cache/kv_cache_configurator.py:1318` shadows `:1355`.** Both are arms of
   the same `elif` chain. 1318 is `elif current_platform.is_out_of_tree() and not self.mambaish_config:`
   and 1355 is the TPU arm, whose own comment says "TPU is in-tree (PlatformEnum.TPU), so the
   is_out_of_tree() branch never fires". Under `OOT` the generic arm wins and **hybrid-SWA models
   would route to `_build_oot_mha_kv_pool` instead of `_build_tpu_swa_kv_pool`**. Fix: fold the SWA
   handling into the OOT arm, delete the TPU arm, then validate on a hybrid-SWA model.
2. **`python/sglang/srt/model_executor/model_runner.py:250`.** `current_platform.init_backend()` is
   called only on the OOT path. TPU reaches initialisation another way today. Under OOT this would
   also fire. Confirm there is no double initialisation before flipping.

Closing both is the precondition for making `tpu_oot` the default and renaming the old entry point.

---

## 4. Measured facts, so you do not re-derive them

Baseline for every fork-vs-upstream number: the upstream snapshot the fork actually contains,
**`1f60ddef5d` (2026-09-18)**. Do **not** use `git merge-base`: the 2026-09-23 sync (#835) was
squash-merged, so the merge base is stuck at `a874b83c` (2026-08-31) and diffs against it overcount
by ~1,100 upstream files.

- Fork vs that snapshot: 200 modified core files (152 carrying TPU logic, +10.4k/−1.1k; 48 generic,
  +601/−252), 143 new files (+57.5k), 126 of them in TPU-only directories.
- 303 `is_tpu()` sites in ~80 core files.
- **27 `is_out_of_tree()` gates** in 14 core files. 8 already read `is_out_of_tree() or is_tpu()`
  and are unaffected by the flip; the other 19 change which branch they take.
- Methods the in-tree `TpuSRTPlatform` does **not** implement that OOT gates call:
  `get_dsa_kv_pool_cls` (the fork defines `get_nsa_kv_pool_cls`; the base raises),
  `get_full_graph_backend_cls` (base returns `None`, safe), and the four speculative hooks from
  upstream #38740 — `supports_speculative_algorithm`,
  `supports_speculative_draft_attention_backend`,
  `get_default_speculative_draft_attention_backend`, `get_speculative_cache_locs_fn`. **The #38740
  gates are not in the fork's core tree yet** (merged upstream 2026-09-20, two days after our
  snapshot), so they are inert today and will become live at the next sync. `platform.py` implements
  them ahead of time.
- Open PRs on this repo: ~194. `python/sglang/srt/hardware_backend/tpu` alone accounts for 362 of
  the 3,567 open-PR file touches, which is why nothing moves in wave 0.

---

## 5. What is left, in order

§5.1-5.3 are **done** (see §5.6 for what they turned up). §5.4 and §5.5 still need hardware.

### 5.1 Contract test — DONE
`test/registered/unit/plugins/test_tpu_oot_platform_contract.py`, `register_cpu_ci(suite="base-a-test-cpu")`.

Not `hardware_backend/tpu/tests/` as originally planned: `test/run_suite.py` discovers registered
tests by globbing `test/registered/**`, so a `register_cpu_ci` marker anywhere else is collected by
no job and the gate would have been silently empty. That empty directory is gone.

30 cases, all green locally. Each of the load-bearing ones was mutation-checked — the fix reverted,
the test confirmed red, the fix restored — because a contract test that cannot fail is worse than none.

### 5.2 Gate audit tool — DONE
`tools/plugin/gate_audit.py`. AST, not grep: it classifies each gate pure-OOT vs `or is_tpu()`-guarded,
tracks negation (`if not (is_out_of_tree() or is_tpu())` takes the *other* arm), and resolves one hop
through same-class `self._helper()` calls, which is the only way `get_dsa_kv_pool_cls` is reachable at all.

It reports **23 gates in 14 files: 14 pure-OOT, 8 `or is_tpu()`-guarded, 1 unscoped.** The "27" in §4
was grep counting four comment lines; the 8 guarded match exactly, which is the number that matters.

### 5.3 Lint — DONE
`tools/plugin/check_tpu_branches.py`, wired as the `check-tpu-branches` pre-commit hook.

Per-file ceilings in `tools/plugin/tpu_debt_baseline.json`, so a removal in one module cannot pay for
an addition in another. Baseline today: **320 `is_tpu` references in 71 shared-core files, and 22 TPU
modules outside the in-tree package.** Imports and the symbol's own definition are not counted; "tpu"
is matched as a path word, so `auxiliary_output.py` is not a TPU module.

The filename rule is narrower than first stated — only `python/sglang/**`, not the whole repo. Workflows,
Dockerfiles, bazel patches and docs legitimately carry `tpu` in their names and will keep being added;
a lint that cries wolf on those gets disabled within a week.

### 5.4 CI wiring — DONE on this box, one environmental blocker left
Mode B runs. `pip install -e hardware_backend/tpu/ --no-deps` (plain install would resolve
`sglang` from PyPI and clobber the editable fork), then `SGLANG_PLATFORM=tpu_oot`.

Platform resolution verified three ways: unset -> `TpuSRTPlatform`, `tpu` -> `TpuSRTPlatform`,
`tpu_oot` -> `TpuOOTPlatform` (`is_tpu` and `is_out_of_tree` both True). **Installing the plugin
does not change the default**, which is the whole point of the opt-in design.

The existing TPU suites hardcoded `os.environ["SGLANG_PLATFORM"] = "tpu"` in four places, which
pinned every run to mode A. Now `setdefault`, so mode B can drive the same suites. `ci_utils.py`
compared `== "tpu"` for its cache-clearing; now `startswith("tpu")`.

**Mode A and mode B reach byte-identical server startup** on Qwen3-0.6B: same KV cache
(688576 tokens, K/V 18.39 GB each), same `max_total_num_tokens`, same memory pool. Both then die
in the same place for the same environmental reason — see below. 48 TPU unit tests pass
identically under both platforms; 14 Pallas attention kernel tests pass on hardware.

**Blocker, environmental, not ours:** this box has torch_tpu `0.1.1.dev20260707090224` while
`python/pyproject_tpu.toml` pins `dev20260813160135`.
`hardware_backend/tpu/result_copy_worker.py:77` calls `self.device_module.device(...)`, and
`TpuDeviceModule.device` exists in neither the installed build nor the newer cached wheel
(`dev20260804175944`), so **e2e server tests cannot complete in either mode**. Fixing it needs
the pinned wheel from the private registry, which 403s for the service account — a human
`gcloud auth login` is the unblock. Until then mode B is verified up to the first forward pass,
not through it.

### 5.5 Close the divergences in §3 and §5.6 — TODO
Then, and only then, make `tpu_oot` the default and rename the in-tree entry point to `tpu_intree`.

### 5.6 What the audit turned up that §3 and §4 did not

**`get_dsa_kv_pool_cls` was a no-op dressed as a fix, now fixed.** §4 says the fork defines
`get_nsa_kv_pool_cls` and the base raises, implying a rename to bridge. It is not a rename:
`TpuSRTPlatform.get_nsa_kv_pool_cls` is itself `raise NotImplementedError("NSA is not supported on
TPU.")`. The shipped `first_available(super(), "get_dsa_kv_pool_cls", "get_nsa_kv_pool_cls")` also
picked the *first* name, which is the base class's raising stub — so the override changed nothing and
DSA models would still have died with a bare `NotImplementedError`. TPU has no DSA pool at all, so the
method now refuses outright with a message naming TPU. `compat.first_available` has no users left; its
docstring no longer claims one.

**Three gate-reachable methods nobody answers.** Recorded in `KNOWN_UNANSWERED` in the contract test,
a list that may only shrink — two tests enforce that, one failing if an entry gets implemented and
another if a gate stops reaching it.

- `get_torch_profiler_activity` and `get_torch_profiler_activity_str` are `[Planned]` stubs on
  `DeviceMixin` that **no platform in the tree implements**, reached from pure-OOT gates at
  `utils/profile_utils.py:251` and `:332`. Torch-profiler startup raises under `tpu_oot`. This is a real
  mode-B blocker and wants a `torch_tpu` ProfilerActivity decision made on hardware — the runtime logs
  that native TPU profiling is off below torch 2.12, so it is not guessable from a CPU box.
- `get_device_core_count` is defined on **no platform class anywhere**, yet `utils/common.py:1012` calls
  it behind `is_out_of_tree() or is_tpu()`. That gate is already true for the in-tree platform, so this
  is a live AttributeError on `main`, not something the flip introduces. Nobody has hit it because its
  only callers are the triton and wave backends and batch-invariant ops, none of which TPU reaches.
  It belongs upstream.

**`get_piecewise_backend_cls` is fine as-is** and is the model for the above: it raises
`"TPU piecewise backend not yet implemented"`. Under OOT, `compilation/backend.py:55` reaches it and
fails loudly, where the in-tree path silently fell through to `CUDAPiecewiseBackend`. A refusal that
names TPU is a decision; an inherited bare `raise NotImplementedError` is nobody having looked. That
distinction is what the contract test actually enforces.

**Divergence 2 was a hard blocker, and worse than described — now closed.** §3 called it a
possible double initialisation. It is not: `model_runner.py:251` calls
`current_platform.init_backend()` at **module import**, an arm the in-tree platform never took.
That import also runs in the tokenizer and detokenizer processes, where `WORLD_SIZE` is unset, so
the in-tree `init_backend` raises `"requires worker WORLD_SIZE"` and **no server starts at all**
under `tpu_oot` — observed, mode B dying in 10s.

Fixed in the plugin, not in core (wave-0 rule 3): `TpuOOTPlatform.init_backend()` returns early
when `WORLD_SIZE` is unset. That is safe rather than merely quiet — `scheduler.py:6234` calls
`initialize_torch_tpu_worker`, which calls `init_backend()` again inside the worker, and the base
class's `_backend_initialized` guard makes that second call the one that does the work. The
import-time call is redundant for TPU. Two regression tests cover it, the first verified red
before the fix.

**Divergence 1 is untested, not cleared.** Qwen3-0.6B is not a hybrid-SWA model, so the
`_build_tpu_swa_kv_pool` vs `_build_oot_mha_kv_pool` routing bug never fires in this smoke test.
The identical KV-pool numbers above exercise the MHA arm only. Closing divergence 1 still needs a
hybrid-SWA model on hardware.

## 6. How to verify

```bash
pip install -e python/ -e hardware_backend/tpu/

# mode A — unchanged default, must stay green
python -m sglang.launch_server --model-path <m> --device tpu ...

# mode B — the new platform drives the same code
SGLANG_PLATFORM=tpu_oot python -m sglang.launch_server --model-path <m> --device tpu ...
```

In mode B, confirm at startup that `current_platform` is `TpuOOTPlatform` and that
`is_out_of_tree()` is True. Then run the TPU unit and e2e suites. Expect trouble first in
KV-pool construction (§3 item 1) and in anything DSA or sparse-attention, which is the
`get_dsa_kv_pool_cls` path that previously could not be reached at all.

**Nothing in mode B has been run.** CPU-side verification is now real — 30 contract cases, the gate
audit, and the debt ratchet, all green, all mutation-checked — but every statement about this platform
driving actual TPU execution remains untested.

---

## 7. Rules for continuing

1. **Move no code in wave 0.** Relocation is wave 4 and is gated on a subsystem's seam work being
   finished; moving early buys packaging nobody is using yet.
2. **A relocation PR, when it comes, changes zero bytes of content.** Git replays concurrent edits
   onto a pure rename; a move mixed with edits conflicts, and this subsystem sees ~2.6 commits/day.
3. **New TPU code goes in `hardware_backend/tpu/`,** and reaches core through a registration or a
   platform method, never a new branch in core.
4. **Hooks are debt.** `sglang_torchtpu/hooks/` is empty on purpose. Anything added there needs the
   four-line header in that module's docstring and an upstream issue link.
5. **Do not publish this distribution to PyPI yet.** `_legacy.py` resolves against the fork's
   `python/` tree; installed next to stock sglang it would hand users an ImportError.
