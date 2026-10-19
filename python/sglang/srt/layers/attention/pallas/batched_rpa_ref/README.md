# batched_rpa_ref — reference (merge-base) batched RPA kernel, A/B only

Verbatim copy of `layers/attention/pallas/batched_rpa/` at this branch's merge
base with main (`49959c5cfa`, the kernel PR #662's PCP runs on), with the
package imports repointed. Used only by the replicated-pool prefill path when
`SGLANG_TPU_PCP_REF_KERNEL=1`, to measure this branch's kernel against it:

| 16K prefill (2K q rows/rank, 32 heads), v7x | this branch's kernel | reference kernel |
|---|---|---|
| attention per step | 46.5 ms (`RPAm-snh-…-q256-k1024`) | 35.3 ms (`RPAm-p256-b1-q512-k512`) |
| 32K second chunk | 132 ms | 102 ms |

Not meant to be shipped: the regression it exposes should be fixed in
`batched_rpa/` (bisect kernel.py / schedule.py / bref_override.py changes),
after which this directory goes away.
