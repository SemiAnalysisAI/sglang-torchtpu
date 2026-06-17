# EAGLE3 TPU spec-decode — branch map

Snapshot of the branches produced for the EAGLE3 V1 speculative-decode + torch.compile
work on TorchTPU. `origin/main` = `993552028a` at the time of writing.

## Deliverable branches

### `pyc96/eagle3-compile-perf-squash`  (single-commit squash)
Base `ea3fdcb38c` + **one squashed commit**.

| commit | subject |
|---|---|
| `0e819d7e12` | [TPU][spec] torch.compile support + perf for EAGLE3 speculative decoding |
| `ea3fdcb38c` | [TPU][spec] Vectorize spec-decode utility kernels *(base)* |

The squash collapses 4 contiguous commits — `2f8472c269` (add torch.compile support),
`9626f157d7` (bucketing / persistent compile-cache / platform plumbing), `5bc2754d69`
(fused draft loop), `9ec288c970` (verify-path host reductions + correctness) — into one.
Its tree is byte-identical to `9ec288c970`.

### `pyc96/eagle3-m0-spec-decode-perf`  (full topic-split history, rebased on main)
7 commits on top of `origin/main`:

| commit | subject |
|---|---|
| `4e1d66f6d4` | Debug/profiling harness for EAGLE3 V1 bring-up |
| `0ef92af460` | Tests for EAGLE3 TPU compile, verify reductions, and guards |
| `9ec288c970` | EAGLE3 verify-path host reductions and spec-decode correctness |
| `5bc2754d69` | Fused EAGLE3 draft loop via torch.compile |
| `9626f157d7` | torch.compile bucketing, persistent compile-cache, and platform plumbing |
| `2f8472c269` | Add torch.compile support for EAGLE3 speculative decoding on TPU |
| `ea3fdcb38c` | Vectorize spec-decode utility kernels |

Rebased onto `origin/main` from the old base `3d35047c83`; the pre-rebase "format +
remove flash attn backend" commit was auto-dropped as already upstream. Two conflicts
were resolved during the rebase (an `is_tpu` import in `linear.py`; the `draft_kv_lens`
override reconciled with main's single-return Pallas op in `attention_pallas_rpa_v3.py`).

## Backup branches (safety nets, do not delete until merged)

| branch | commit | what it is |
|---|---|---|
| `backup/pre-rebase-squashtip` | `4b04b75f93` | branch tip *before* the rebase onto main (5-commit squash on old base `3d35047c83`) |
| `backup/eagle3-m0-pre-squash`  | `1434451c67` | the original ~65-commit history *before* the topic squash |

## Relationships

```
origin/main (993552028a)
        └── ea3fdcb38c  Vectorize spec-decode utility kernels
              ├── pyc96/eagle3-compile-perf-squash : ea3fdcb38c + 0e819d7e12 (1 squashed commit)
              └── 2f8472c269 → 9626f157d7 → 5bc2754d69 → 9ec288c970 → 0ef92af460 → 4e1d66f6d4
                    = pyc96/eagle3-m0-spec-decode-perf (7 commits, topic-split)
```

Both work branches contain the same functional code; they differ only in commit grouping
(`-compile-perf-squash` = one commit through the verify-reductions stage, no tests/debug;
`-m0-spec-decode-perf` = full topic-split incl. tests + debug/profiling harness).

Neither work branch has been pushed; updating a previously-pushed branch needs
`git push --force-with-lease` (history was rewritten).
