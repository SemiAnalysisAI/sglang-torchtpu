# Kimi-K3 megakernel: spike, CPU dry runs and TPU harness

Tools for the experimental megakernel port (see `sglang_tpu/kimi_k3_megakernel/README.md`).

- `spike/mk_bridge_spike.py` — runs Inferact's full Kimi-K3 decode step as a torch custom op
  through SGLang's `cpu_export_jax_op` bridge inside 32 torch_tpu workers (torchrun, 4 hosts x 8),
  with synthetic megakernel-layout weights; prints per-step time. Env: `MK_SRC`, `MK_CKPT`,
  `MK_BATCH`, `MK_CTX`, `MK_POS`, `MK_STEPS`.
- `dryrun/` — CPU-only checks that must pass before any TPU run:
  `export_flat_cpu.py` / `export_check_cpu.py` (jax.export of the step to one TPU custom call),
  `dryrun_bridge.py` (bridge registration + FakeTensor shapes), `dryrun_hostfill.py`
  (synthetic weight blocks), `fake_mk_server.py` / `fake_sgl_server.py` (servers for client and
  harness dry runs).
- `harness/` — 4-host GKE JobSet drivers. Each pod runs the driver named by the recipe with
  `PLAN`, `OUT`, `NHOSTS`, `JOBSET_JOB_INDEX`, `JS_NAME`, `POD_NAMESPACE`:
  - `combo_driver.sh` runs phases (`sglang`, `spike`, `mk`) in order with a timeout, a process
    cleanup and a cross-host barrier each;
  - `run_variants_v5.sh` + `client_v5.sh`: SGLang server per variant, eval, TPOT sweep, and
    xprof captures started mid-decode (`PROFILE_SPECS`, written under `PROFILE_ROOT`);
  - `mk_spike_driver.sh`: the bridge spike per config under torchrun;
  - `mk_driver.sh` + `mk_client.py`: the megakernel's own OpenAI server per config
    (installs jax 0.11.1, so it must run last), driven with equal-length batches; decode
    timing comes from the server's `target_batch:` lines.
  - `plans/`: the prepared combined run.
- `k3_xprof.py`, `planes.py` — summarize one rank's `xplane.pb` (modules, decode launch period,
  op categories per decode step).

Budget rule for this work: at most 2 TPU runs per 24 h, and only once the port is complete.
