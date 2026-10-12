# Combined TPU run (user-approved 2026-10-12 for the next slot): one 2x2x4 slice, three phases.
#  sglang: main + the 5 open Kimi-K3 PRs (snapshot p1-heads), all opt-in flags; TPOT 8k/1k c1 and
#          1k/1k c4 on this slice, then mid-decode xprof (bs1 8k, bs4 1k) to /mnt/kimik3-ssd/profiles.
#  spike:  Inferact megakernel decode step inside torch_tpu workers via SGLang's bridge (synthetic
#          weights, SGLang stack): does libtpu 0.0.47 compile and run it, and how fast.
#  mk:     Inferact megakernel standalone (jax 0.11.1): bs1 at 1k and 8k context, bs4 at 1k.
PHASES="sglang spike mk"
SG_PLAN=/mnt/workspace/dagent/e2e/plans/plan-combo1-sglang.sh
SPIKE_PLAN=/mnt/workspace/dagent/e2e/plans/plan-mkspike1.sh
MK_PLAN=/mnt/workspace/dagent/e2e/plans/plan-mk1.sh
SG_TIMEOUT_S=6600
SPIKE_TIMEOUT_S=4800
MK_TIMEOUT_S=6000
