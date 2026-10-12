"""Cross-platform (CPU host -> TPU target) export of the Kimi-K3 decode megakernel.

Checks only that the installed JAX/jaxlib can trace the megakernel and lower it to
StableHLO + Mosaic MLIR for TPU; nothing is compiled by libtpu or executed.

Usage: python -B export_check.py <batch> <recipe: plain|tuned> <context> [checkpoint_dir|-]
"""

import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=32"
sys.dont_write_bytecode = True
sys.path.insert(0, "/home/chandrasekhard/tpu-megakernels")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax._src.pallas.mosaic import tpu_info  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P  # noqa: E402

chip = tpu_info.chip_version_from_device_kind("TPU7x")
info = tpu_info.get_tpu_info_for_chip(chip, 1)
pltpu.get_tpu_info = lambda: info
tpu_info.get_tpu_info = lambda: info
tpu_info.get_device_kind = lambda: "TPU7x"
tpu_info.is_tpu_device = lambda: True

from kimi import decode_megakernel as model  # noqa: E402
from kimi import load as kimi_load  # noqa: E402

batch = int(sys.argv[1])
recipe = sys.argv[2]
context = int(sys.argv[3])
checkpoint = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != "-" else None

print(
    "jax",
    jax.__version__,
    "devices",
    len(jax.devices()),
    "tpu info",
    info.chip_version,
    flush=True,
)

import math  # noqa: E402

KERNEL_DEFAULTS = ("mla_cache_write_in_attend", "mla_pv_mxu_hilo")
options = {
    "gate_first_kda_projection": True,
    "kernel_options": tuple(sorted(KERNEL_DEFAULTS)),
}
if recipe == "tuned":
    eight = batch == 8
    options.update(
        residual_reduction_order="relaxed",
        mla_cache_batch_tile_size=math.gcd(4, batch),
        alias_expert_scales=batch >= 6,
        attention_token_scatter=eight,
        moe_token_scatter_output=eight,
        moe_routed_reduce_scatter=batch in (6, 8),
        sequence_parallel_pre_attention=eight,
    )

mesh = Mesh(np.array(jax.devices()), ("tp",))
sharded = NamedSharding(mesh, P("tp"))
replicated = NamedSharding(mesh, P())
started = time.perf_counter()
weights_abs = kimi_load.abstract_weights(mesh, checkpoint, layers=93)
print(f"abstract weights in {time.perf_counter() - started:.1f} s", flush=True)
for name in (
    "k_conv",
    "k_norm",
    "k_a_log",
    "k_dt",
    "router",
    "latent_norm",
    "attn_res",
    "expert_gu",
    "expert_gus",
    "expert_down",
    "expert_ds",
):
    print(f"  {name}: {weights_abs[name].shape} {weights_abs[name].dtype}")

kda_layers, mla_layers = 69, 24
states_abs = (
    jax.ShapeDtypeStruct(
        (32, batch, kda_layers, 3, 3, 3, 128), jnp.bfloat16, sharding=sharded
    ),
    jax.ShapeDtypeStruct(
        (32, batch, kda_layers, 3, 128, 128), jnp.float32, sharding=sharded
    ),
    jax.ShapeDtypeStruct(
        (32, batch, mla_layers, 3, context, 256), jnp.bfloat16, sharding=sharded
    ),
    jax.ShapeDtypeStruct(
        (32, batch, mla_layers, 3, context, 128), jnp.bfloat16, sharding=sharded
    ),
)
rows_abs = jax.ShapeDtypeStruct((batch,), jnp.int32, sharding=replicated)

decode = model.make_decode(
    mesh, batch_size=batch, layers=93, donate=True, sequence_rows=1, **options
)
started = time.perf_counter()
exported = jax.export.export(decode, platforms=["tpu"])(
    weights_abs, states_abs, rows_abs, rows_abs
)
elapsed = time.perf_counter() - started
text = exported.mlir_module()
print(
    f"exported in {elapsed:.1f} s; StableHLO text {len(text) / 1e6:.2f} MB", flush=True
)
print("tpu_custom_call count:", text.count("tpu_custom_call"))
print(
    "all_reduce ops:",
    text.count("stablehlo.all_reduce"),
    "all_gather ops:",
    text.count("stablehlo.all_gather"),
)
print("in/out avals:", len(exported.in_avals), len(exported.out_avals))
for aval in exported.out_avals:
    print("  out", aval)

import json
import re

match = re.search(
    r"stablehlo.custom_call @tpu_custom_call\((.*?)\) \{(.*?)\} :", text, re.S
)
if match:
    attrs = match.group(2)
    cfg = re.search(r'backend_config = "(.*?)"(?:,|\s)', attrs, re.S)
    if cfg:
        raw = cfg.group(1).replace("\\22", '"').replace("\\0A", "\n")
        try:
            config = json.loads(raw)
            body = config.get("custom_call_config", {}).get("body", "")
            meta = {
                k: v
                for k, v in config.get("custom_call_config", {}).items()
                if k != "body"
            }
            print(
                "custom_call_config keys/values (body omitted):",
                json.dumps(meta)[:2000],
            )
            print("Mosaic body (base64) bytes:", len(body))
            print(
                "top-level keys:",
                {
                    k: (v if k != "custom_call_config" else "...")
                    for k, v in config.items()
                },
            )
        except Exception as error:  # noqa: BLE001
            print("could not parse backend_config:", type(error).__name__, raw[:400])
    print("custom_call attrs (truncated):", attrs[:600].replace("\n", " "))
