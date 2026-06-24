"""Standalone TPU op-isolation reproducer for the SpecV2 AC-2 near-tie-trace blocker.

Context: capturing per-emitted-token top-2 logits from the LIVE V2 overlap verify hangs
the scheduler (10 in-server approaches all hang; see work_dirs/specv2/results/
AC2_TRACE_BLOCKED.md). This script proves the OPS themselves are fine in isolation
(argmax/topk/scatter/where + D2H + Event all complete in <10s on torch_tpu), isolating
the hang to the overlap-scheduler result/lifetime interaction, NOT the computation.

Run: SGLANG_PLATFORM=tpu .venv/bin/python3 benchmark/tpu/spec/tpu_overlap_trace_repro.py
"""

import signal
import time

import torch
import torch_tpu  # noqa

N, V = 8, 151936  # bs*draft_token_num, Qwen3 vocab
dev = "tpu"
dm = torch.get_device_module(dev)


def stage(name, fn, timeout=40):
    # run fn() under an alarm; if it blocks past timeout, report HANG
    def _to(*a):
        raise TimeoutError()

    old = signal.signal(signal.SIGALRM, _to)
    signal.alarm(timeout)
    t0 = time.time()
    try:
        fn()
        signal.alarm(0)
        print(f"  OK  {name}  ({time.time()-t0:.2f}s)")
    except TimeoutError:
        print(f"  HANG {name}  (> {timeout}s)")
    except Exception as e:
        signal.alarm(0)
        print(f"  ERR {name}: {type(e).__name__}: {e}")
    finally:
        signal.signal(signal.SIGALRM, old)


torch.manual_seed(0)
logits = torch.randn(N, V, dtype=torch.bfloat16, device=dev)


def s_argmax():
    t1 = torch.argmax(logits, dim=-1, keepdim=True)
    c = t1.to("cpu")
    print("   argmax->", c.flatten().tolist()[:3], end=" ")


def s_topk():
    tv, ti = torch.topk(logits, 2, dim=-1)
    ev = dm.Event()
    cpu_i = ti.to("cpu", non_blocking=True)
    cpu_v = tv.to("cpu", non_blocking=True)
    ev.record()
    ev.synchronize()
    print("   topk->", cpu_i[0].tolist(), end=" ")


def s_scatter():
    t1 = torch.argmax(logits, dim=-1, keepdim=True)
    masked = logits.scatter(-1, t1, float("-inf"))
    t2 = torch.argmax(masked, dim=-1, keepdim=True)
    ids = torch.cat([t1, t2], dim=-1)
    ev = dm.Event()
    cpu = ids.to("cpu", non_blocking=True)
    ev.record()
    ev.synchronize()
    print("   scatter->", cpu[0].tolist(), end=" ")


def s_where():
    t1 = torch.argmax(logits, dim=-1, keepdim=True)
    t1v = logits.gather(-1, t1)
    masked = torch.where(logits < t1v, logits, torch.full_like(logits, float("-inf")))
    t2 = torch.argmax(masked, dim=-1, keepdim=True)
    ids = torch.cat([t1, t2], dim=-1)
    ev = dm.Event()
    cpu = ids.to("cpu", non_blocking=True)
    ev.record()
    ev.synchronize()
    print("   where->", cpu[0].tolist(), end=" ")


print("=== isolating the hanging op on torch_tpu (N=%d, V=%d) ===" % (N, V))
stage("argmax + D2H (control)", s_argmax)
stage("topk(k=2) + D2H + Event.sync", s_topk)
stage("scatter(-inf)+argmax + D2H + Event.sync", s_scatter)
stage("where(-inf)+argmax + D2H + Event.sync", s_where)
print("DONE")
