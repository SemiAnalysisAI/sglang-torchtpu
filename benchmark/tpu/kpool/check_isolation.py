# SPDX-License-Identifier: Apache-2.0
"""Compare traced graphs against a Git baseline without running TPU numerics."""

import argparse
import hashlib
import json
import subprocess
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def load(source, filename):
    module = types.ModuleType("kpool_isolation")
    # The reviewed local/Git sources use the same filename for comparable traces.
    exec(compile(source, filename, "exec"), module.__dict__)  # noqa: S102
    return module.compress_and_store


def graph(fn, specs, tokens, options):
    traced = jax.make_jaxpr(lambda *args: fn(*args, tokens, **options))(*specs)
    # Include constants as well as the nested Pallas jaxpr. Source locations
    # are deliberately excluded; moving unchanged code must not fail this gate.
    digest = hashlib.sha256(str(traced).encode())
    for value in traced.consts:
        value = np.asarray(value)
        digest.update(str((value.shape, value.dtype)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--q1-batch", action="store_true")
    parser.add_argument("--q1-default", action="store_true")
    parser.add_argument(
        "--baseline-unrolled",
        action="store_true",
        help="Baseline uses unrolling by default; verify opt-in graph equivalence",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    relative = "sglang_tpu/attention/pallas/kpool_compress.py"
    before = subprocess.check_output(
        ["git", "-C", str(root), "show", f"{args.baseline}:{relative}"]
    )
    after = (root / relative).read_bytes()
    old, new = load(before, relative), load(after, relative)
    cases = []
    for batch in (1, 2, 3, 32):
        specs = tuple(
            jax.ShapeDtypeStruct(shape, dtype)
            for shape, dtype in (
                ((129, 128, 256), jnp.bfloat16),
                ((129, 32, 4, 256), jnp.uint8),
                ((4, 128), jnp.float32),
                ((batch,), jnp.int32),
                ((128 // batch * batch,), jnp.int32),
                ((batch + 1,), jnp.int32),
                ((3,), jnp.int32),
            )
        )
        configurations = [(t, {}) for t in (32, 128, 8192)]
        configurations += [
            (
                max(32, batch * q),
                {
                    "decode": True,
                    "decode_tile_rows": rows,
                    "decode_max_q": q,
                    "decode_pipeline": pipeline,
                },
            )
            for q in (1, 2, 6, 9, 16, 32)
            for rows in (16, 32)
            for pipeline in (False, True)
        ]
        configurations.append((32, {"decode": True}))
        configurations = [
            (tokens, {**options, "decode_q1_unroll": unroll})
            for tokens, options in configurations
            for unroll in ((True,) if args.baseline_unrolled else (False, True))
        ]
        for tokens, options in configurations:
            if args.q1_batch:
                options = {**options, "decode_q1_batch": True}
            changed = (
                options.get("decode_max_q") == 1
                and options.get("decode_pipeline")
                and batch > 1
                and options["decode_q1_unroll"]
                and not args.baseline_unrolled
            )
            baseline_options = {
                key: value
                for key, value in options.items()
                if key != "decode_q1_unroll"
            }
            if args.q1_batch:
                changed = options.get("decode_max_q") == 1
                baseline_options = {
                    key: value
                    for key, value in options.items()
                    if key != "decode_q1_batch"
                }
            if args.q1_default:
                changed = (
                    options.get("decode_max_q") == 1
                    and options.get("decode_pipeline")
                    and options.get("decode_tile_rows") == 16
                    and batch == 32
                    and not options["decode_q1_unroll"]
                )
                baseline_options = options
            left = graph(old, specs, tokens, baseline_options)
            right = graph(new, specs, tokens, options)
            passed = (left != right) if changed else (left == right)
            if args.q1_default and changed:
                passed = passed and right == graph(
                    new, specs, tokens, {**options, "decode_q1_batch": True}
                )
            cases.append(
                {
                    "batch": batch,
                    "tokens": tokens,
                    "options": options,
                    "expected_changed": bool(changed),
                    "baseline_graph_sha256": left,
                    "candidate_graph_sha256": right,
                    "passed": passed,
                }
            )
    result = {
        "jax_version": jax.__version__,
        "baseline": args.baseline,
        "baseline_sha256": hashlib.sha256(before).hexdigest(),
        "candidate_sha256": hashlib.sha256(after).hexdigest(),
        "scope": "trace identity; no TPU compilation or numerical validation",
        "cases": cases,
        "passed": all(case["passed"] for case in cases),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(
        "ISOLATION_RESULT",
        json.dumps(
            {
                "passed": result["passed"],
                "cases": len(cases),
                "expected_changed": sum(case["expected_changed"] for case in cases),
                "failures": [case for case in cases if not case["passed"]],
            }
        ),
        flush=True,
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
