# MiMo-V2.5 DP-attention on TPU — single-host (tp8) → 2-host (tp16) port

Scripts to reproduce the validated single-host run and scale it to **TP=16 across
2 hosts**. All configs mirror what was validated on one `tpu7x-8` host.

## Files
| file | what |
|---|---|
| `common_env.sh`            | shared env (paths, offline HF, libtpu/SparseCore flags). Sourced by the others. |
| `launch_single_host.sh`    | **validated** baseline: 1 host, tp8/ep8/dp2. |
| `launch_tp16_multihost.sh` | tp16/ep16/dp4 across 2 hosts. Run on both, `NODE_RANK` differs. |
| `bench.sh`                 | 1k/1k, concurrency 4, detailed metrics (TTFT/ITL/TPOT/E2E). Run on rank-0. |

## Version lockstep (both/all hosts must match)
- sglang-torchtpu: branch containing DP-attention fix `312cb2bf5`
  (validated on `main @ 11059a1d3`). Check:
  `git -C <repo> merge-base --is-ancestor 312cb2bf5 HEAD && echo OK`
- torch_tpu wheel `0.1.1.dev20260707090224`, torch `2.11.0+cpu`, libtpu `0.0.40`.
  Check: `python -c "import torch_tpu,torch;print(torch_tpu.__version__,torch.__version__)"`
- `MODEL` staged to a **local** path (`/cache/...`), never a `/gcs` GCS-FUSE mount
  (mmap SIGBUS). Same path on every host.

## DP-attention sizing rules (why the numbers are what they are)
- **`attention_tp = tp_size / dp_size` must equal `num_key_value_heads` (=4 for MiMo)**
  — fused-QKV constraint. tp8→dp2, **tp16→dp4** (both give attention_tp=4).
- **`torch_compile_max_bs ≥ max_running × dp_size`** — DP decode padding requests a
  batch bucket of that size. tp8/dp2 c=4 → 8; **tp16/dp4 c=4 → 16**.
- `ep_size = tp_size` (experts sharded across all devices): 8 → **16**.
- Weights are ~half the per-device size at tp16 (~19 GB vs ~37 GB), so `mem_fraction`
  has more headroom; 0.5 is safe.

## Single host (baseline)
```bash
./launch_single_host.sh            # wait for "The server is fired up"
./bench.sh                         # in another shell
```

## TP=16 on 2 hosts
Manual (2 GCE VMs / SSH). Open the rendezvous port (`DIST_PORT`, default 10011)
between the hosts first.
```bash
# Host 0 (rank 0 — hosts the HTTP endpoint):
./launch_tp16_multihost.sh 0

# Host 1 (rank 1 — point it at host 0):
RANK0_HOST=<host0-internal-ip> ./launch_tp16_multihost.sh 1
```
Then, on host 0 (or any machine that can reach it):
```bash
./bench.sh                                  # if run on host 0
TARGET_HOST=<host0-internal-ip> ./bench.sh  # if run remotely
```

GKE TPU pod slice (`tpu7x-16`): `TPU_WORKER_ID` / `TPU_WORKER_HOSTNAMES` are set
automatically — run `./launch_tp16_multihost.sh` with **no arg** on every worker;
rank, peer list and tp=16 auto-derive.

## Notes
- `--load-format dummy` (random weights) is in the launch scripts for a fast
  plumbing/perf smoke test. **Remove it** from `EXTRA_ARGS` to load a real
  checkpoint (outputs are garbage with dummy weights).
- Bench targets **rank 0 only** — in sglang multi-node the tokenizer/HTTP server
  runs on node-rank 0; other nodes run schedulers/workers.
- ⚠️ The tp16/dp4 config is **extrapolated** from the hardware-validated tp8/dp2
  run (only 1 host was available here). The scaling rules above are applied, but
  confirm on the 2-host slice — watch first launch for the fused-qkv assert
  (`attention_tp == num_key_value_heads`) and the `BatchSizeBucketer` size error
  (bump `TORCH_COMPILE_MAX_BS` if it fires).
