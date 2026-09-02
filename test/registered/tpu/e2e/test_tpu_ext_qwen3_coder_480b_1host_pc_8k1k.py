"""Qwen3-Coder-480B-A35B-Instruct-FP8 external-cluster serving benchmark: 1 host tp=8/ep=8, prefix caching on.

Workload: single-turn 8k-input / 1k-output serving; 'long requests' are
single-turn LONG_TARGET_CONTEXT-input requests (~120k gpt-oss / 64k qwen) mixed into the batch. Prefix-cache hit
rate is recorded but not gated (a single-turn session cannot re-see a
prefix).

Sweep: concurrency {1,16,64} x long requests {0,1,4} (legs with more long
requests than slots are skipped). One server serves every leg; the client's
concurrency semaphore is what varies. Rank 0 of the 2-host slice serves single-host; the second host idles
(2x2x2 is the smallest external slice).

The two workload files of a (model, recipe) pair share one launch group (same
launch_script + weights), so the JobSet brings the server up once and rank 0
runs both files against it. All gates live in external_bench_base.py.
"""

import unittest

from external_bench_base import build_matrix_test

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=5400, suite="external-nightly-2x2x2", nightly=True)

MULTIHOST_LAUNCH = {
    "launch_script": "recipe/external_bench/qwen3_coder_480b_1host_pc.sh",
    "weights": "Qwen3-Coder-480B-A35B-Instruct-FP8",
    "timeout_per_file": 8100,
}

TestExtQwen3Coder480b1hostPc8k1k = build_matrix_test(
    bench_model="qwen3_coder_480b",
    bench_recipe="1host_pc",
    workload="8k1k",
)

if __name__ == "__main__":
    unittest.main()
