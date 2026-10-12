"""Local CPU dry run of the bridge spike up to kernel launch: op registration (signature
checks), FakeTensor shape propagation through the bridge's fake impl (jax.export of the
shard_map'd megakernel step), dtype mapping of every input. Nothing executes on a TPU."""

import os
import sys

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=32"
os.environ.setdefault("MK_CKPT", "/mnt/kimik3-ssd/kimik3")
import numpy as np
import torch
import torch_tpu  # noqa: F401

try:
    torch.utils.rename_privateuse1_backend("tpu")
except RuntimeError as e:
    print("rename skipped:", e)
import types

if not hasattr(torch, "tpu"):
    torch.tpu = types.SimpleNamespace(
        is_initialized=lambda: True,
        current_device=lambda: 0,
        device_count=lambda: 1,
        is_available=lambda: True,
    )
from jax.sharding import Mesh

import sglang_tpu.pallas_export as pe

pe.get_torch_tpu_device_kind = lambda: "TPU7x"
import mk_bridge_spike as spike
from torch_tpu._internal.pallas.pallas import _PallasDevice

batch, context = int(sys.argv[1]), int(sys.argv[2])
mesh = Mesh(
    np.array(
        [_PallasDevice(id=i, process_index=i, device_kind="TPU7x") for i in range(32)]
    ),
    ("tp",),
)
op, names, weights_abs, state_specs = spike.build_decode_op(mesh, batch, context)
print("registered", op, "inputs:", len(names) + 6)
from torch._subclasses.fake_tensor import FakeTensorMode

tdt = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "uint32": torch.uint32,
    "uint8": torch.uint8,
    "int32": torch.int32,
}
with FakeTensorMode(allow_non_fake_inputs=True):
    args = [
        torch.empty(
            (1,) + tuple(weights_abs[n].shape[1:]),
            dtype=tdt[str(weights_abs[n].dtype)],
            device="tpu",
        )
        for n in names
    ]
    args += [torch.empty((1,) + s, dtype=d, device="tpu") for s, d in state_specs]
    args += [
        torch.zeros((batch,), dtype=torch.int32, device="tpu"),
        torch.zeros((batch,), dtype=torch.int32, device="tpu"),
    ]
    outs = op(*args)
print("outputs:", [(tuple(o.shape), o.dtype, str(o.device)) for o in outs])
GB = sum(a.numel() * a.element_size() for a in args) / 1e9
print(f"per-rank input bytes: {GB:.1f} GB")
print("DRYRUN_OK")
