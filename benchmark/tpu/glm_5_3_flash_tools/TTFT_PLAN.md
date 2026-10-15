# GLM-5.3-Flash DSA: 8k1k TTFT plan (target 370–390 ms)

For review. Owner: this session. Slices: **46p2** (qinghan-3131) and **5rcp** (qinghan-4836).
psdr (de1c) and x5kk (4d14) belong to the agent testing KV offload, PD and prefix cache.

## Where we are

- Working commit: **a63aea39f4** (branch `glm-dsa-tpu-rebase`, local). DSA with all three
  SparseCore users (indexer top-k, sparse-MLA gather, MoE gathers) and 256-token sparse-attention
  gather chunks. No SparseCore halt.
- DFlash + DSA, 8k1k c1, simulated accept 3.5: **median TTFT 527 ms, median TPOT 2.56 ms.**
- Dense MTP reference (older commit): 330 ms TTFT, 2.04 ms TPOT. Dense DFlash on today's tree has
  not been measured yet.

## Benchmark and rules

- One metric set per change: 8k1k c1 cookbook bench (8 prompts, 2 warmups), simulated accept 3.5,
  median TTFT, P99 TTFT and median TPOT. Every change is also checked for TPOT.
- One change per run, against a63aea39f4 on the same slice. Same-slice reruns vary by ~10 ms, so a
  change has to beat that to count.
- Profiles come from the full 8192-token prefill (no shortened prompts). The trace buffer only holds
  part of one prefill, so each profile run captures two windows (start of prefill, and the
  profiler started later into the same request) to cover all layers.

## Where the 527 ms goes (8192-token prefill, from the partial profile)

| Item | Per layer | Per prefill (est.) | Tuned? |
|---|---|---|---|
| Elementwise fusions / glue | ~1.3 ms | ~55 ms | partly |
| MoE gmm_v2 | ~1.2 ms | ~55 ms | yes |
| MoE SparseCore gathers + XLA un-permute | ~0.8 ms | ~35 ms | gathers yes, un-permute no |
| KDA | ~0.7 ms | ~23 ms | yes |
| SparseCore top-k + sort | ~1.3 ms | ~14 ms | no |
| Sparse attention kernel (batch 16, DSV4 defaults) | ~1.2 ms | ~14 ms | no |
| SparseCore KV gather | ~1.2 ms | ~13 ms | no |
| gather_page_ids | ~0.56 ms | ~6 ms | no |
| StreamIndex scoring (bq 128, bkvp 2, 4 row chunks) | ~0.5 ms | ~6 ms | no |
| Idle gaps inside the graph | ~30% of module time | ~100 ms? | no |
| Host / scheduling outside the graph | ? | ? | unmeasured |

The DSA kernels together are ~50–60 ms. Getting from 527 to 370–390 also needs the idle gaps and
host time, which the next profile has to size.

## Plan

### Step 0 (running now, started 22:32)
- 46p2: a63aea39f4 as is (reference on this slice).
- 5rcp: sparse-attention batch 32 (default 16).

### Step 1: baseline and attribution (next ~1 h, both slices)
1. **46p2: dense DFlash, same tree** (`dense_dflash_acc35`). Tells us how much of the 527 ms is DSA
   at all. If dense DFlash is ~350 ms, the target means removing nearly all DSA overhead; if it is
   ~450 ms, part of the work is outside DSA.
2. **5rcp: two-window profile of the 8192-token prefill on a63aea39f4.** Per-op time for all 45
   layers, plus idle gaps and the host time between device programs.

### Step 2: DSA kernel tuning (in order of expected gain)
| # | Change | Expected | Risk |
|---|---|---|---|
| 2a | Sparse-attention batch size and gather chunk sweep (batch 16/32/64 x chunk 256/512) | 5–10 ms | low; VMEM limits at large batch |
| 2b | Indexer: no 2048-row chunking (one StreamIndex pass + one top-k per layer) | 5–10 ms | low; score buffer 8192 x 2816 is small |
| 2c | StreamIndex block sizes (queries per block 128 -> 256, KV pages per block 2 -> 4) | 2–5 ms | low |
| 2d | Top-k: cheaper selection at prefill sizes; check XLA vs SparseCore per size (also the +0.1 ms TPOT) | 5–10 ms TTFT, ~0.1 ms TPOT | medium |

### Step 3: non-DSA prefill work (if Step 1 shows the gap is there)
| # | Change | Expected | Risk |
|---|---|---|---|
| 3a | MoE un-permute gather on the SparseCore (or fused into the combine) | ~20 ms | medium; shared MoE code, needs a bit-exactness test |
| 3b | Idle gaps: find what the TensorCore waits on (collectives, SparseCore call-done waits, small-op chains) and fuse or reorder | 30–80 ms | depends on the profile |
| 3c | Host time before/after the prefill graph (metadata build, sampling, DFlash drafter prefill) | 10–40 ms | low |

### Rough path to the target
527 -> ~495 after Step 2 -> ~420 after 3a/3c -> 370–390 needs most of the idle gaps (3b). Whether
3b is reachable depends on Step 1's profile; I will report that before starting it.

## Not in scope here
- KV offload, PD disaggregation, prefix cache: other agent, psdr / x5kk.
- MTP + DSA TTFT: same kernels, so the gains carry over; measured once at the end.
- Long-context (AIME) correctness on DSA: after the 8k1k work.

## Commit policy
Each kept change is its own local commit on `glm-dsa-tpu-rebase` with its before/after numbers in the
message. a63aea39f4 stays the fallback until a later commit beats it on both TTFT and TPOT.
