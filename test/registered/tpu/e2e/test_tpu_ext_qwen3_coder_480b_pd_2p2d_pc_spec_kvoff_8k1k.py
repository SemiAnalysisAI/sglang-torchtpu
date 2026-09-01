"""Qwen3-Coder-480B-A35B-Instruct-FP8 external-cluster serving benchmark: 4-host P/D disaggregation: 2-host prefill + 2-host decode engines
(tp=16 each) over raiden, prefix caching + EAGLE3 spec + KV offload.

Workload: single-turn 8k-input / 1k-output serving; 'long requests' are
single-turn 64k-input requests mixed into the batch. Prefix-cache hit
rate is recorded but not gated (a single-turn session cannot re-see a
prefix).

Sweep: concurrency {1,16,64} x long requests {0,1,4} (legs with more long
requests than slots are skipped). One server serves every leg; the client's
concurrency semaphore is what varies. The 2x2x4 slice is split into two 2x2x2 sub-slices by overriding the
GKE-injected TPU_WORKER_HOSTNAMES / TPU_TOPOLOGY per engine; rank 0
fronts both engines with the mini-LB on :30000 (the benched endpoint).

The two workload files of a (model, recipe) pair share one launch group (same
launch_script + weights), so the JobSet brings the server up once and rank 0
runs both files against it. All gates live in external_bench_base.py.
"""

import unittest

from external_bench_base import build_matrix_test

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=7200, suite="external-nightly-2x2x4", nightly=True)

MULTIHOST_LAUNCH = {
    "launch_script": "recipe/external_bench/qwen3_coder_480b_pd_2p2d_pc_spec_kvoff.sh",
    "weights": "Qwen3-Coder-480B-A35B-Instruct-FP8",
    "timeout_per_file": 5400,
}

TestExtQwen3Coder480bPd2p2dPcSpecKvoff8k1k = build_matrix_test(
    bench_model="qwen3_coder_480b",
    bench_recipe="pd_2p2d_pc_spec_kvoff",
    workload="8k1k",
)

if __name__ == "__main__":
    unittest.main()
