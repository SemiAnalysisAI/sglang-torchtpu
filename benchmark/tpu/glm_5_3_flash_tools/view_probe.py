"""Which tensor views make torch_tpu launch an execute_as_strided program?

Runs each case between `arange` marker programs under an xprof capture; count
the as_strided modules between markers in the trace (see view_probe_count.py).
usage (one chip): python view_probe.py <out_dir>
"""
import sys, time
from pathlib import Path

import torch
import torch_tpu  # noqa: F401

from sglang.srt.hardware_backend.tpu.profiler import start_tpu_profile, stop_tpu_profile

dev = "tpu"
g2 = torch.compile(lambda t: t + 1, dynamic=False)  # [1, 8]
g1 = torch.compile(lambda t: t + 1, dynamic=False)  # [8]
buf = torch.zeros(1, 8, dtype=torch.int64, device=dev)
flat = torch.zeros(8, dtype=torch.int64, device=dev)
src = torch.ones(1, dtype=torch.int64, device=dev)
a = torch.ones(1, 8, dtype=torch.int64, device=dev)

mask2d = torch.zeros(1, 8, dtype=torch.bool, device=dev)
mask2d[0, 0] = True
CASES = {
    "out_into_flat_viewed_2d_then_flat": lambda: (torch.add(a, a, out=flat.view(1, 8)), g1(flat)),
    "copy_into_flat_viewed_2d_then_flat": lambda: (flat.view(1, 8).copy_(a), g1(flat)),
    "where_out_into_flat_viewed_2d": lambda: (torch.where(mask2d, a, a + 1, out=flat.view(1, 8)), g1(flat)),
    "d2h_base": lambda: flat.cpu(),
    "d2h_reshape_view": lambda: buf.reshape(-1).cpu(),
    "base_flat_control": lambda: g1(flat),
}


def marker():
    torch.arange(3, device=dev).cpu()


for fn in CASES.values():  # compile everything before tracing
    fn(); marker()
torch.tpu.synchronize()

trace = start_tpu_profile(output_dir=Path(sys.argv[1]), with_cpu=False)
for name, fn in CASES.items():
    marker()
    for _ in range(4):
        out = fn()
    o = out[-1] if isinstance(out, tuple) else out
    if o.device.type != "cpu":
        o.cpu()
    torch.tpu.synchronize()
    time.sleep(0.01)
marker()
torch.tpu.synchronize()
stop_tpu_profile(trace)
print("cases:", list(CASES))
