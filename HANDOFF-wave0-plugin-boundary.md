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

### 5.1 Contract test (CPU, no TPU needed) — do this first
`hardware_backend/tpu/tests/test_platform_contract.py`, registered with `register_cpu_ci`.

Assert, for `TpuOOTPlatform`:
- `is_out_of_tree()` is True **and** `is_tpu()` is True;
- for every platform method called within an `is_out_of_tree()` gate in `python/sglang/srt`, the
  resolved implementation is not the base class's raising stub, i.e.
  `getattr(type(p), name) is not getattr(SRTPlatform, name)` for the raising ones;
- `get_dsa_kv_pool_cls()` returns the same class as `get_nsa_kv_pool_cls()`;
- the speculative capability table matches `arg_groups/speculative_hook.py` (topk=1 only, no
  rejection sampling) and `speculative/draft_utils.py` (the two `pallas_tpu_rpa_*` backends).

This is the gate for "the platform satisfies every method an OOT gate calls" and it is the thing
that catches the next sync breaking us.

### 5.2 Gate audit tool — `tools/plugin/gate_audit.py`
Scan `python/sglang/srt` for `is_out_of_tree()`, classify each gate as pure-OOT or
`or is_tpu()`-guarded, extract the `current_platform.<method>(` calls in the following window, and
emit the inventory the contract test consumes. Run it after every upstream sync; it is how §3's
divergence list stays current.

### 5.3 Lint — `tools/plugin/check_tpu_branches.py`
Reject, in CI: a new `is_tpu()` / `_is_tpu` branch in `python/sglang/srt` outside the TPU
directories, and a new TPU file created anywhere but `hardware_backend/tpu/`. Grandfather everything
that exists today via a baseline count or an allowlist; the number must only go down.

### 5.4 CI wiring
Mode A is what runs today. Add **mode B**: `pip install -e python/ -e hardware_backend/tpu/` with
`SGLANG_PLATFORM=tpu_oot`, running the existing TPU suites. **This is the step that needs your
hardware.** Mode B green on the full TPU suite is the definition of done for wave 0.

### 5.5 Close the divergences in §3
Then, and only then, make `tpu_oot` the default and rename the in-tree entry point to `tpu_intree`.

---

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

**Nothing in mode B has been run.** The only verification performed is that every module compiles
and that `activate()`'s advertised class path matches the class that exists.

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
