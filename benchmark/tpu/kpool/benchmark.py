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
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import reference

PIPELINE_BASELINE_REVISION = "985b4f8c9f2151cfae7c110156f7433892e1ee60"
DOUBLE_PREFETCH_REVISION = "61edd0e421a877158f7bfdfdfd2ae6133173cbfc"
PREVIOUS_DECODE_REVISION = "de72e465f066024c2e678b63de7d4cd73212a404"
PREVIOUS_CANDIDATE_REVISION = "880e8ace2f2f3f06cc66eb9a1235b6e6d7bf2335"


def load_candidate(root, baseline_output=None, revision=PIPELINE_BASELINE_REVISION):
    relative = "sglang_tpu/attention/pallas/kpool_compress.py"
    path = root / relative
    if baseline_output is not None:
        path = baseline_output / f"baseline-{revision[:12]}.py"
        path.write_bytes(
            subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(root),
                    "show",
                    f"{revision}:{relative}",
                ]
            )
        )
    spec = importlib.util.spec_from_file_location("kpool_candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compress_and_store, hashlib.sha256(path.read_bytes()).hexdigest()


def inputs(pages, tokens, decode, batch=1, query_len=1, raw=None):
    """Fragmented request pages and an already updated raw cache."""
    if pages < max(17, (tokens + 127) // 128 + 1):
        raise ValueError("Insufficient pages for the requested token bucket")
    if raw is None:
        raw = jax.jit(
            lambda key: jax.random.normal(key, (pages, 128, 256), jnp.bfloat16)
        )(jax.random.key(42))
    page_count = (pages - 1) // batch * batch
    page_map = np.random.default_rng(42).permutation(np.arange(1, page_count + 1))
    starts = (
        np.full(batch, 131, np.int32)
        if query_len == 1
        else np.resize(np.array([131, 127, 511, 130], np.int32), batch)
    )
    tail = (
        jnp.zeros((4, 128), jnp.float32),
        jnp.asarray(starts + query_len, jnp.int32)
        if decode
        else jnp.full((batch,), tokens, jnp.int32),
        jnp.asarray(page_map, jnp.int32),
        jnp.arange(batch + 1, dtype=jnp.int32) * query_len
        if decode
        else jnp.asarray([0, tokens], jnp.int32),
        jnp.asarray([batch, batch, batch] if decode else [0, 0, 1], jnp.int32),
    )
    jax.block_until_ready((raw, tail))
    return raw, tail


def benchmark_case(
    candidate,
    baseline,
    double_prefetch,
    previous,
    pages,
    tokens,
    decode,
    opts,
    output,
    batch=1,
    query_len=1,
    raw=None,
):
    name = f"{'decode' if decode else 'prefill'}_p{pages}_t{tokens}"
    if batch != 1:
        name += f"_b{batch}"
    if query_len != 1:
        name += f"_q{query_len}"
    print("CASE_START", name, flush=True)
    raw, tail = inputs(pages, tokens, decode, batch, query_len, raw)
    variants = {
        "reference": reference.computation,
        "single_prefetch": baseline,
        "double_prefetch": double_prefetch,
        "candidate": candidate,
    }
    if decode:
        variants = {
            "reference": reference.computation,
            "previous_candidate": previous,
            "decode_fixed32": partial(candidate, decode=True, decode_tile_rows=32),
            "candidate": partial(candidate, decode=True, decode_tile_rows=16),
        }
    if decode and opts.decode_pipeline:
        variants = {
            "reference": reference.computation,
            "previous_candidate": partial(
                previous, decode=True, decode_tile_rows=16, decode_max_q=query_len
            ),
            "decode_serial": partial(
                candidate, decode=True, decode_max_q=query_len, decode_pipeline=False
            ),
            "candidate": partial(
                candidate, decode=True, decode_max_q=query_len, decode_pipeline=True
            ),
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
            if opts.hlo_text:
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
        result = {
            "name": name,
            "pages": pages,
            "token_bucket": tokens,
            "batch": batch,
            "query_length": query_len if decode else tokens,
            "actual_new_tokens": batch * query_len if decode else tokens,
            "closed_pools": (
                int(
                    np.sum(
                        np.asarray(tail[1]) // 4
                        - (np.asarray(tail[1]) - query_len) // 4
                    )
                )
                if decode
                else tokens // 4
            ),
            "previous_decode_branch": bool(decode and opts.decode_pipeline),
            "variants": rows,
            "speedup_pending_accuracy": rows["reference"]["median_ms"]
            / rows["candidate"]["median_ms"],
            "pipeline_speedup_pending_accuracy": rows[
                "previous_candidate" if decode else "single_prefetch"
            ]["median_ms"]
            / rows["candidate"]["median_ms"],
            (
                "serial_to_pipeline_speedup_pending_accuracy"
                if decode and opts.decode_pipeline
                else "pooling_tile_speedup_pending_accuracy"
            ): rows[
                ("decode_serial" if opts.decode_pipeline else "decode_fixed32")
                if decode
                else "double_prefetch"
            ]["median_ms"]
            / rows["candidate"]["median_ms"],
            "device_profile": None,
            "profile_status": "requested" if opts.profile else "not_requested",
        }
        # Persist timings before optional profiler initialization. A native
        # profiler crash cannot be caught by Python's exception handlers.
        (output / f"{name}-latency.json").write_text(json.dumps(result, indent=2))
        print("LATENCY_RESULT", json.dumps(result), flush=True)
        if opts.profile:
            # Profile separate invocations; exclude collection from timings.
            with jax.profiler.trace(str(output / name / "xprof")):
                for _ in range(5):
                    for state in states.values():
                        state["cache"] = state["call"](raw, state["cache"], *tail)
                        state["cache"].block_until_ready()
            result["device_profile"] = str(output / name / "xprof")
            result["profile_status"] = "collected"
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
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=("decode", "prefill"),
        default=["decode", "prefill"],
    )
    parser.add_argument("--hlo-text", action="store_true")
    parser.add_argument("--decode-batches", nargs="+", type=int, default=[1, 32])
    parser.add_argument("--decode-query-lengths", nargs="+", type=int, default=[1])
    parser.add_argument("--decode-pipeline", action="store_true")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Collect XProf after saving timings; requires a compatible profiler ABI",
    )
    opts = parser.parse_args()
    if opts.warmup < 1 or opts.repeats < 2:
        parser.error("Require warmup >= 1 and repeats >= 2")
    if any(b < 1 or b > 32 for b in opts.decode_batches):
        parser.error("decode batches must be between 1 and 32")
    if any(q < 1 or q > 32 for q in opts.decode_query_lengths):
        parser.error("decode query lengths must be between 1 and 32")
    if not opts.decode_pipeline and opts.decode_query_lengths != [1]:
        parser.error("multi-token decode requires --decode-pipeline")
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
        double_prefetch, double_digest = load_candidate(
            root, opts.output, DOUBLE_PREFETCH_REVISION
        )
        previous_revision = (
            PREVIOUS_DECODE_REVISION
            if opts.decode_pipeline
            else PREVIOUS_CANDIDATE_REVISION
        )
        previous, previous_digest = load_candidate(root, opts.output, previous_revision)
        result.update(
            source_revision=subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
            ).strip(),
            candidate_sha256=digest,
            pipeline_baseline_revision=PIPELINE_BASELINE_REVISION,
            pipeline_baseline_sha256=baseline_digest,
            double_prefetch_revision=DOUBLE_PREFETCH_REVISION,
            double_prefetch_sha256=double_digest,
            previous_candidate_revision=previous_revision,
            previous_candidate_sha256=previous_digest,
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
            if ("decode" if decode else "prefill") not in opts.cases:
                continue
            shared_raw = None
            if decode and opts.decode_pipeline:
                print("INPUT_START", pages, flush=True)
                shared_raw, _ = inputs(pages, 32, True)
                print("INPUT_READY", pages, flush=True)
            for query_len in opts.decode_query_lengths if decode else [1]:
                for batch in opts.decode_batches if decode else [1]:
                    bucket = ((batch * query_len + 31) // 32) * 32 if decode else tokens
                    result["cases"].append(
                        benchmark_case(
                            candidate,
                            baseline,
                            double_prefetch,
                            previous,
                            pages,
                            bucket,
                            decode,
                            opts,
                            opts.output,
                            batch,
                            query_len,
                            shared_raw,
                        )
                    )
            del shared_raw
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
