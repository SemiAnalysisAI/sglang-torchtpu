# SPDX-License-Identifier: Apache-2.0
"""Two-level Pallas all-reduce for single-token TP decode tensors (opt-in).

all_reduce
  Why: XLA's all-reduce of a [1, H] bf16 decode activation over a multi-host
       TP group is latency bound (tpu7x 2x2x4, TP32: 21 us at H=7168); a
       host-then-cross-host one-shot Pallas exchange takes 13 us.
  How: AROUND GroupCoordinator.all_reduce; the TP group's eligible tensors go
       to sglang_tpu.twolevel_allreduce when SGLANG_TPU_ENABLE_TWOLEVEL_ALLREDUCE
       registered the op, everything else calls the original.
  Upstream PR or issue: not filed.
  Remove when: GroupCoordinator exposes a platform all-reduce backend seam.

merged_moe_all_reduce
  Why: Kimi-K3's latent MoE reduces the routed latent and the shared-expert
       output; with SGLANG_K3_MERGE_MOE_ALLREDUCE core calls one function for
       both, which the TPU op serves as one packed exchange (no concat/split).
  How: AROUND sglang.srt.models.kimi_k3.merged_moe_all_reduce.
  Upstream PR or issue: not filed.
  Remove when: the GroupCoordinator seam above takes several tensors.
"""


def all_reduce(original, *args, **kwargs):
    from sglang_tpu.twolevel_allreduce import all_reduce_around

    return all_reduce_around(original, *args, **kwargs)


def merged_moe_all_reduce(original, *args, **kwargs):
    from sglang_tpu.twolevel_allreduce import merged_moe_all_reduce_around

    return merged_moe_all_reduce_around(original, *args, **kwargs)
