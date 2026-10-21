# SPDX-License-Identifier: Apache-2.0
"""CPU-host TPU compilation and LLO dumps, with each case in its own process.

Uses the topology/sharding compilation pattern from pallas-tools run_aot.py.
This does not execute TPU code or establish numerical accuracy.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path

CASES = (
    "hadamard128",
    "reshape_pool",
    "reshape_pack",
    "reshape_mask",
    "reshape_mask_bool",
    "reshape_output",
    "original_small",
    "original_mask_fix",
    "reference_decode",
    "candidate_decode",
    "candidate_decode32",
    "candidate_decode_b32",
    "reference_prefill",
    "candidate_prefill",
)


def build(case, root, out):
    import jax
    import jax.numpy as jnp
    import reference
    from jax.experimental import pallas as pl

    if case == "hadamard128":
        spec = importlib.util.spec_from_file_location(
            "kpool_candidate", root / "sglang_tpu/attention/pallas/kpool_compress.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        def kernel(x, y):
            y[...] = module._hadamard128(x[...])

        shape = jax.ShapeDtypeStruct((32, 128), jnp.float32)
        return pl.pallas_call(kernel, out_shape=shape, name=case), (shape,), ()

    if case.startswith("reshape_"):
        if case == "reshape_pool":
            shape, output, dtype = (128, 128), (32, 128), jnp.float32

            def kernel(x, y):
                y[...] = jnp.sum(x[...].reshape(32, 4, 128), axis=1)

        elif case == "reshape_pack":
            shape, output, dtype = (32, 256), (8, 256), jnp.uint32

            def kernel(x, y):
                records = x[...].reshape(8, 4, 256)
                value = records[:, 0, :]
                for lane in range(1, 4):
                    value |= records[:, lane, :] << (lane * 8)
                y[...] = value

        elif case == "reshape_mask_bool":
            shape, output, dtype = (32,), (8, 1), jnp.uint32

            def kernel(x, y):
                live = (x[...] > 0).reshape(8, 4, 1)
                mask = jnp.where(live[:, 0, :], jnp.uint32(255), jnp.uint32(0))
                for lane in range(1, 4):
                    mask |= jnp.where(
                        live[:, lane, :], jnp.uint32(255), jnp.uint32(0)
                    ) << (lane * 8)
                y[...] = mask

        elif case == "reshape_output":
            shape, output, dtype = (8, 256), (8, 1, 256), jnp.uint32

            def kernel(x, y):
                y[:, 0, :] = x[...]

        else:
            shape, output, dtype = (32,), (8, 1), jnp.int32

            def kernel(x, y):
                y[...] = jnp.sum(x[...].reshape(8, 4, 1), axis=1)

        call = pl.pallas_call(
            kernel, out_shape=jax.ShapeDtypeStruct(output, dtype), name=case
        )
        return call, (jax.ShapeDtypeStruct(shape, dtype),), ()

    pages = 17 if case.startswith("original_") else 14849
    tokens = 256 if case.startswith("original_") else 32 if "decode" in case else 8192
    batch = 32 if case.endswith("_b32") else 1
    specs = tuple(
        jax.ShapeDtypeStruct(shape, dtype)
        for shape, dtype in (
            ((pages, 128, 256), jnp.bfloat16),
            ((pages, 32, 4, 256), jnp.uint8),
            ((4, 128), jnp.float32),
            ((batch,), jnp.int32),
            (((pages - 1) // batch * batch,), jnp.int32),
            ((batch + 1,), jnp.int32),
            ((3,), jnp.int32),
        )
    )
    if case.startswith("reference_"):
        fn = reference.computation
    else:
        path = root / "sglang_tpu/attention/pallas/kpool_compress.py"
        if case.startswith("original_"):
            path = out / "original.py"
            path.write_bytes(
                subprocess.check_output(
                    [
                        "git",
                        "-C",
                        str(root),
                        "show",
                        (
                            "75a03373e46643a9373cf9bcd1ed7fdfd794385d:"
                            "sglang_tpu/attention/pallas/kpool_compress.py"
                        ),
                    ]
                )
            )
            if case == "original_mask_fix":
                source = path.read_text()
                start = source.index("        ends = jnp.arange(32,")
                end = source.index("        # Serialize", start)
                source = (
                    source[:start]
                    + """        mask = jnp.zeros((8, 1), jnp.uint32)
        for lane in range(4):
            ends = jnp.arange(8, dtype=jnp.int32)[:, None] * 16 + lane * 4 + 3
            live = (ends >= desc_ref[3, i]) & (ends < desc_ref[4, i])
            mask |= jnp.where(live, jnp.uint32(255), jnp.uint32(0)) << (lane * 8)

"""
                    + source[end:]
                )
                path.write_text(source)
        spec = importlib.util.spec_from_file_location("kpool_candidate", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fn = module.compress_and_store
    extra = {}
    if case.startswith("candidate_decode"):
        extra = {
            "decode": True,
            "decode_tile_rows": 32 if case.endswith("32") and batch == 1 else 16,
        }
    return lambda *args: fn(*args, tokens, 4, **extra), specs, (1,)


def worker(opts):
    out = opts.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["LIBTPU_INIT_ARGS"] = (
        f"--xla_jf_dump_to={out}/llo --xla_mosaic_dump_to={out}/mosaic "
        "--xla_jf_dump_hlo_text=true --xla_jf_dump_llo_text=true "
        "--xla_mosaic_enable_llo_source_annotations=true "
        "--xla_mosaic_enable_dump_debug_info=true "
        "--xla_jf_emit_annotations=true --xla_jf_collect_llo_stack_trace=true"
    )
    import jax
    import numpy as np
    from jax.experimental.topologies import get_topology_desc

    root = Path(__file__).resolve().parents[3]
    result = {
        "case": opts.worker,
        "target": opts.target,
        "executed": False,
        "accuracy": "not_checked",
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("jax", "jaxlib", "libtpu")
        },
        "source_revision": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "candidate_sha256": hashlib.sha256(
            (root / "sglang_tpu/attention/pallas/kpool_compress.py").read_bytes()
        ).hexdigest(),
    }
    (out / "manifest.json").write_text(json.dumps(result, indent=2))
    try:
        fn, inputs, donated = build(opts.worker, root, out)
        device = get_topology_desc(opts.target, platform="tpu").devices[0]
        sharding = jax.sharding.SingleDeviceSharding(device)
        mesh = jax.sharding.Mesh(np.array([device]), ("aot",))
        start = time.perf_counter()
        with jax.set_mesh(mesh):
            lowered = jax.jit(
                fn,
                in_shardings=(sharding,) * len(inputs),
                out_shardings=sharding,
                donate_argnums=donated,
            ).lower(*inputs)
            (out / "stablehlo.mlir").write_text(lowered.as_text())
            compiled = lowered.compile()
        result["compile_seconds"] = time.perf_counter() - start
        result["compile_complete"] = True
        try:
            (out / "optimized_hlo.txt").write_text(compiled.as_text())
        except jax.errors.JaxRuntimeError:
            # Some libtpu executables fail HLO text deserialization after
            # successful code generation. Preserve the independent LLO result.
            result["optimized_hlo_export_error"] = traceback.format_exc()
        result["memory"] = str(compiled.memory_analysis())
        result["final_llo"] = [
            str(p.relative_to(out))
            for p in out.rglob("*final_bundles.txt")
            if "schedule-analysis_" not in p.name
        ]
        result["status"] = "compiled" if result["final_llo"] else "missing_llo"
    except Exception:
        result["status"] = "failed"
        result["error"] = traceback.format_exc()
        raise
    finally:
        (out / "result.json").write_text(json.dumps(result, indent=2))
        print("AOT_RESULT", json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", default="tpu7x:2x2x1")
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--worker", choices=CASES)
    parser.add_argument("--archive", type=Path)
    opts = parser.parse_args()
    if opts.worker:
        worker(opts)
        return
    opts.output.mkdir(parents=True, exist_ok=True)
    summary = []
    for case in opts.cases:
        out = opts.output / case
        out.mkdir(exist_ok=True)
        print("AOT_START", case, flush=True)
        command = [
            sys.executable,
            __file__,
            "--worker",
            case,
            "--target",
            opts.target,
            "--output",
            str(out),
        ]
        with (out / "compile.log").open("w") as log:
            try:
                proc = subprocess.run(
                    command, stdout=log, stderr=log, timeout=600, check=False
                )
                returncode = proc.returncode
            except subprocess.TimeoutExpired:
                returncode = 124
        path = out / "result.json"
        result = (
            json.loads(path.read_text()) if path.exists() else {"status": "crashed"}
        )
        result.update(case=case, returncode=returncode)
        if returncode:
            result["log_tail"] = (out / "compile.log").read_text()[-12000:]
        summary.append(result)
        (opts.output / "summary.json").write_text(json.dumps(summary, indent=2))
        print("AOT_CASE_RESULT", json.dumps(result), flush=True)
    print("AOT_SUMMARY", json.dumps(summary), flush=True)
    if opts.archive:
        # Keep complete final LLO and the matching pre-delay companion. The
        # full set of compiler intermediate dumps remains in the task artifact.
        names = {
            "summary.json",
            "result.json",
            "manifest.json",
            "compile.log",
            "stablehlo.mlir",
            "optimized_hlo.txt",
        }
        files = [
            path
            for path in opts.output.rglob("*")
            if path.is_file()
            and (
                path.name in names
                or path.name.endswith(
                    ("-final_bundles.txt", "-packed-bundles-post-ra.txt")
                )
            )
        ]
        temporary = opts.archive.with_suffix(opts.archive.suffix + ".tmp")
        with tarfile.open(temporary, "w:gz") as archive:
            for path in sorted(files):
                archive.add(path, arcname=path.relative_to(opts.output))
        temporary.replace(opts.archive)
        print("AOT_ARTIFACT_READY", str(opts.archive), flush=True)


if __name__ == "__main__":
    main()
