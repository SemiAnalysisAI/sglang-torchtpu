"""Exercise the spike's host-side random block generation for every weight family (real
per-layer shapes, CPU only) to catch dtype/view errors before a TPU run."""

import os
import sys
import time

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=32"
import jax
import numpy as np
import torch
from jax._src.pallas.mosaic import tpu_info
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh

chip = tpu_info.chip_version_from_device_kind("TPU7x")
info = tpu_info.get_tpu_info_for_chip(chip, 1)
pltpu.get_tpu_info = lambda: info
tpu_info.get_tpu_info = lambda: info
sys.path.insert(0, "/home/chandrasekhard/tpu-megakernels")
from kimi import decode_megakernel as model
from kimi import load as kimi_load

mesh = Mesh(np.array(jax.devices()), ("tp",))
wa = kimi_load.abstract_weights(mesh, "/mnt/kimik3-ssd/kimik3", layers=93)
tdt = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "uint32": torch.uint32,
    "uint8": torch.uint8,
    "int32": torch.int32,
}
gen = torch.Generator().manual_seed(1234)
for n in sorted(wa):
    shape = (1,) + tuple(wa[n].shape[1:])
    dt = tdt[str(wa[n].dtype)]
    per_layer = n in model.NAMES
    sub = shape[2:] if per_layer else shape[1:]
    t0 = time.time()
    if dt in (torch.bfloat16, torch.float32):
        if n.endswith("norm") or n in (
            "attn_res",
            "mlp_res",
            "output_res",
            "final_norm",
        ):
            host = torch.ones(sub, dtype=dt)
        elif n == "router_bias":
            host = torch.zeros(sub, dtype=dt)
        else:
            host = (torch.randn(sub, generator=gen) * 0.02).to(dt)
    elif dt == torch.float8_e4m3fn:
        host = (
            (torch.randint(0, 256, sub, generator=gen, dtype=torch.int32) & 0xBF)
            .to(torch.uint8)
            .view(torch.float8_e4m3fn)
        )
        assert not torch.isnan(host.float()).any(), n
    elif dt == torch.uint8:
        host = torch.full(sub, 120, dtype=torch.uint8)
    elif dt == torch.uint32:
        host = torch.randint(
            -(2**31), 2**31 - 1, sub, generator=gen, dtype=torch.int32
        ).view(torch.uint32)
    elif dt == torch.int32:
        host = torch.zeros(sub, dtype=torch.int32)
    else:
        raise TypeError(dt)
    assert tuple(host.shape) == tuple(sub) and host.dtype == dt, (
        n,
        host.shape,
        host.dtype,
    )
    print(
        f"{n:14s} per_layer={per_layer!s:5s} block {tuple(sub)} {dt} {host.numel() * host.element_size() / 1e6:8.1f} MB  {time.time() - t0:.1f}s",
        flush=True,
    )
    del host
print("HOSTFILL_OK")
