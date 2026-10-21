# SPDX-License-Identifier: Apache-2.0
"""Profiler ABI smoke only. This does not check numerical accuracy."""
import argparse
import importlib.metadata
import json
from pathlib import Path

import jax
import jax.numpy as jnp

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
fn = jax.jit(lambda x: x + 1)
x = jnp.ones((128, 128), jnp.float32)
for _ in range(5):
    x = fn(x)
    x.block_until_ready()
print("PROFILER_ABI_SMOKE", json.dumps({name: importlib.metadata.version(name) for name in ("jax", "jaxlib", "libtpu")}), flush=True)
with jax.profiler.trace(str(args.output)):
    for _ in range(3):
        x = fn(x)
        x.block_until_ready()
files = list(args.output.rglob("*.xplane.pb"))
if not files or any(p.stat().st_size == 0 for p in files):
    raise RuntimeError("Profiler wrote no nonempty XPlane")
print("PROFILER_ABI_SMOKE_CAPTURED", flush=True)
