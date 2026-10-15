# GLM-5.3-Flash TPU working scripts (temporary, PR #1369 only)

Everything used to bring up, benchmark and validate GLM-5.3-Flash on TPU v7x
(tpuq 2x2x1 slices, TP8), copied as-is from the working directory. Not meant
to merge: pod names, paths and slice names are hard-coded.

- Eval / bench server configs: `benchmark/tpu/recipe/configs/glm_5_3_flash_eval/`
  (deployed on the pods under that same package path).
- Launch and orchestration: `run-*.sh`, `tpu-*.sh` (boot, 8k1k bench, profile,
  AIME, feature smoke tests), `bootstrap-port-pod.sh` / `deploy-port.sh`
  (set up a fresh slice from a local commit), `tpu_env.sh`, `memguard.sh`.
- Benchmarks and checks: `bench_*.py`, `aime26.py`, `prefix_test.py`,
  `dsa_sample.py`, `accept_check.py`.
- Profiling: `prof_*.py`, `round_ops.py`, `ops_by_source.py`, `inner_gaps.py`.
- Notes: `FEATURE_READINESS.md` (prefix cache / KV offload / PD status).
