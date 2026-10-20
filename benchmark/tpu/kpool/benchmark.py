# SPDX-License-Identifier: Apache-2.0
"""Same-device pooling latency comparison; numerical validation is a later gate."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import subprocess
import time
import traceback
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import reference

PIPELINE_BASELINE_REVISION = "985b4f8c9f2151cfae7c110156f7433892e1ee60"


def load_candidate(root, baseline_output=None):
    relative = "sglang_tpu/attention/pallas/kpool_compress.py"
    path = root / relative
    if baseline_output is not None:
        path = baseline_output / "single_prefetch.py"
        path.write_bytes(
            subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(root),
                    "show",
                    f"{PIPELINE_BASELINE_REVISION}:{relative}",
                ]
            )
        )
    spec = importlib.util.spec_from_file_location("kpool_candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compress_and_store, hashlib.sha256(path.read_bytes()).hexdigest()


def inputs(pages, tokens, decode):
    """One request, fragmented pages, and an already updated raw cache."""
    if pages < max(17, (tokens + 127) // 128 + 1):
        raise ValueError("Insufficient pages for the requested token bucket")
    raw = jax.jit(lambda key: jax.random.normal(key, (pages, 128, 256), jnp.bfloat16))(
        jax.random.key(42)
    )
    page_map = np.random.default_rng(42).permutation(np.arange(1, pages))
    tail = (
        jnp.zeros((4, 128), jnp.float32),
        jnp.asarray([132 if decode else tokens], jnp.int32),
        jnp.asarray(page_map, jnp.int32),
        jnp.asarray([0, 1 if decode else tokens], jnp.int32),
        jnp.asarray([1, 1, 1] if decode else [0, 0, 1], jnp.int32),
    )
    jax.block_until_ready((raw, tail))
    return raw, tail


def benchmark_case(candidate, baseline, pages, tokens, decode, opts, output):
    name = f"{'decode' if decode else 'prefill'}_p{pages}_t{tokens}"
    print("CASE_START", name, flush=True)
    raw, tail = inputs(pages, tokens, decode)
    variants = {
        "reference": reference.computation,
        "single_prefetch": baseline,
        "candidate": candidate,
    }
    states = {}
    try:
        for label, function in variants.items():

            def entry(*args, _fn=function):
                return _fn(*args)

            entry.__name__ = f"kpool_{label}"
            cache = jnp.zeros((pages, 32, 4, 256), jnp.uint8)
            cache.block_until_ready()
            start = time.perf_counter()
            compiled = (
                jax.jit(entry, static_argnums=(7, 8), donate_argnums=(1,))
                .lower(raw, cache, *tail, tokens, 4)
                .compile()
            )
            compile_s = time.perf_counter() - start
            hlo_error = None
            try:
                (output / f"{name}-{label}.hlo.txt").write_text(compiled.as_text())
            except jax.errors.JaxRuntimeError:
                hlo_error = traceback.format_exc()
            memory = compiled.memory_analysis()
            stats = {
                field: getattr(memory, field, None)
                for field in (
                    "argument_size_in_bytes",
                    "output_size_in_bytes",
                    "alias_size_in_bytes",
                    "temp_size_in_bytes",
                )
            }
            pointer = cache.unsafe_buffer_pointer()
            for _ in range(opts.warmup):
                cache = compiled(raw, cache, *tail)
                cache.block_until_ready()
            aliases = cache.unsafe_buffer_pointer() == pointer
            if not aliases:
                raise RuntimeError(f"{label}: donated pooled cache was not reused")
            states[label] = {
                "call": compiled,
                "cache": cache,
                "pointer": pointer,
                "samples": [],
                "compile_seconds": compile_s,
                "memory": stats,
                "hlo_export_error": hlo_error,
            }
            print("COMPILED", name, label, json.dumps(stats), flush=True)
        # Alternate order each round to reduce systematic drift on one allocation.
        for iteration in range(opts.repeats):
            order = list(states) if iteration % 2 == 0 else list(reversed(states))
            for label in order:
                state = states[label]
                start = time.perf_counter_ns()
                cache = state["call"](raw, state["cache"], *tail)
                cache.block_until_ready()
                state["samples"].append((time.perf_counter_ns() - start) / 1e6)
                if cache.unsafe_buffer_pointer() != state["pointer"]:
                    raise RuntimeError(f"{label}: pooled cache allocation changed")
                state["cache"] = cache
        rows = {}
        for label, state in states.items():
            samples = state["samples"]
            rows[label] = {
                "median_ms": float(np.median(samples)),
                "p10_ms": float(np.percentile(samples, 10)),
                "p90_ms": float(np.percentile(samples, 90)),
                "samples_ms": samples,
                "compile_seconds": state["compile_seconds"],
                "memory": state["memory"],
                "pooled_cache_buffer_reused": True,
                "hlo_export_error": state["hlo_export_error"],
            }
        # Profile separate invocations so profiler collection does not enter
        # the host-blocking latency samples. Device timing is analyzed later.
        with jax.profiler.trace(str(output / name / "xprof")):
            for _ in range(5):
                for state in states.values():
                    state["cache"] = state["call"](raw, state["cache"], *tail)
                    state["cache"].block_until_ready()
        result = {
            "name": name,
            "pages": pages,
            "token_bucket": tokens,
            "actual_new_tokens": 1 if decode else tokens,
            "closed_pools": 1 if decode else tokens // 4,
            "variants": rows,
            "speedup_pending_accuracy": rows["reference"]["median_ms"]
            / rows["candidate"]["median_ms"],
            "pipeline_speedup_pending_accuracy": rows["single_prefetch"]["median_ms"]
            / rows["candidate"]["median_ms"],
            "device_profile": str(output / name / "xprof"),
        }
        print("CASE_RESULT", json.dumps(result), flush=True)
        return result
    finally:
        states.clear()
        del raw, tail
        jax.clear_caches()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    opts = parser.parse_args()
    if opts.warmup < 1 or opts.repeats < 2:
        parser.error("Require warmup >= 1 and repeats >= 2")
    root = Path(__file__).resolve().parents[3]
    opts.output.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "running",
        "accuracy": "not_checked",
        "scope": "module latency, one device, dispatch plus synchronized execution",
        "timing": "compile and input creation excluded; warm cache; donated output",
        "warmup": opts.warmup,
        "repeats": opts.repeats,
        "cases": [],
    }
    try:
        if jax.default_backend() != "tpu":
            raise RuntimeError("This benchmark requires a TPU")
        candidate, digest = load_candidate(root)
        baseline, baseline_digest = load_candidate(root, opts.output)
        result.update(
            source_revision=subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
            ).strip(),
            candidate_sha256=digest,
            pipeline_baseline_revision=PIPELINE_BASELINE_REVISION,
            pipeline_baseline_sha256=baseline_digest,
            devices=[str(d) for d in jax.devices()],
            device_kind=jax.devices()[0].device_kind,
            runtime={
                name: importlib.metadata.version(name)
                for name in ("jax", "jaxlib", "libtpu", "numpy")
            },
        )
        # Two fixed representative points; AOT covers compilation separately.
        for pages, tokens, decode in (
            (14849, 32, True),
            (14849, 8192, False),
        ):
            result["cases"].append(
                benchmark_case(
                    candidate, baseline, pages, tokens, decode, opts, opts.output
                )
            )
            (opts.output / "metrics.json").write_text(json.dumps(result, indent=2))
        result["status"] = "completed_pending_accuracy"
    except Exception:
        result["status"] = "failed"
        result["error"] = traceback.format_exc()
        raise
    finally:
        (opts.output / "metrics.json").write_text(json.dumps(result, indent=2))
        print("BENCHMARK_RESULT", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
