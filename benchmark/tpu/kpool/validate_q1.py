# SPDX-License-Identifier: Apache-2.0
"""Performance-checkpoint gate: exact cache bytes, matching the existing unit rule."""

import argparse
import json
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import reference

from benchmark import load_candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    opts = parser.parse_args()
    opts.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[3]
    candidate, digest = load_candidate(root)
    baseline = jax.jit(reference.computation, static_argnums=(7, 8))
    batched = jax.jit(
        partial(candidate, decode=True, decode_max_q=1, decode_q1_batch=True),
        static_argnums=(7, 8),
        donate_argnums=(1,),
    )
    previous = jax.jit(
        partial(candidate, decode=True, decode_max_q=1, decode_q1_batch=False),
        static_argnums=(7, 8),
        donate_argnums=(1,),
    )
    automatic = jax.jit(
        partial(candidate, decode=True, decode_max_q=1),
        static_argnums=(7, 8),
        donate_argnums=(1,),
    )
    results = []
    cases = [(b, "random") for b in (1, 2, 3, 8, 16, 32)]
    cases += [
        (32, mode)
        for mode in (
            "mixed",
            "no_close",
            "padding",
            "shared",
            "shared_nonadjacent",
            "zero",
            "tiny",
            "nonfinite",
        )
    ]
    for index, (batch, mode) in enumerate(cases):
        rng = np.random.default_rng(2048 + index)
        pages_per_request = 8
        pages = batch * pages_per_request + 1
        raw = rng.normal(size=(pages, 128, 256)).astype(np.float32)
        old = rng.integers(0, 256, (pages, 32, 4, 256), dtype=np.uint8)
        mapping = rng.permutation(np.arange(1, pages)).astype(np.int32)
        starts = np.resize(
            np.array([3, 7, 11, 15, 127, 131, 255, 511], np.int32), batch
        )
        lengths = np.ones(batch, np.int32)
        if mode == "mixed":
            lengths[1::3] = 0
            starts[2::3] += 1
        elif mode == "no_close":
            starts += 1
        elif mode == "padding":
            lengths[:] = 0
        elif mode == "shared":
            starts = np.resize(np.array([3, 7, 11, 15], np.int32), batch)
            for request in range(batch):
                mapping[request * pages_per_request] = mapping[
                    (request // 4) * 4 * pages_per_request
                ]
        elif mode == "shared_nonadjacent":
            starts[0], starts[-1] = 3, 7
            mapping[(batch - 1) * pages_per_request] = mapping[0]
        elif mode == "zero":
            raw[..., :128] = 0
        elif mode == "tiny":
            raw[..., :128] *= 1e-7
        elif mode == "nonfinite":
            raw[mapping[0], 0, 0] = np.nan
            raw[mapping[pages_per_request], 4, 0] = np.inf
            raw[mapping[2 * pages_per_request], 8, 0] = -np.inf
        cu = np.concatenate(([0], np.cumsum(lengths))).astype(np.int32)
        tokens = max(32, batch)
        args = (
            jnp.asarray(raw, jnp.bfloat16),
            jnp.asarray(rng.normal(size=(4, 128)), jnp.float32),
            jnp.asarray(starts + lengths),
            jnp.asarray(mapping),
            jnp.asarray(cu),
            jnp.asarray([batch, batch, batch], jnp.int32),
        )

        def call(fn, args=args, old=old, tokens=tokens):
            return np.asarray(
                fn(args[0], jnp.asarray(old.copy()), *args[1:], tokens, 4)
            )

        print("ACCURACY_CASE_START", batch, mode, flush=True)
        expected = call(baseline)
        row = {"batch": batch, "mode": mode, "rule": "exact cache byte equality"}
        functions = [("previous", previous), ("batched", batched)]
        if batch == 32:
            functions.append(("automatic", automatic))
        for label, fn in functions:
            try:
                actual = call(fn)
                mismatch = np.argwhere(actual != expected)
                row[label] = {
                    "passed": not len(mismatch),
                    "mismatched_bytes": len(mismatch),
                    "first_indices": mismatch[:8].tolist(),
                }
            except Exception as exc:  # noqa: BLE001 -- persist every failed case and fail the gate
                row[label] = {"passed": False, "error": str(exc)}
        results.append(row)
        print("ACCURACY_CASE_RESULT", json.dumps(row), flush=True)
        (opts.output / "accuracy.json").write_text(
            json.dumps(
                {
                    "candidate_sha256": digest,
                    "jax": jax.__version__,
                    "cases": results,
                    "passed": all(
                        r["batched"]["passed"]
                        and r.get("automatic", {"passed": True})["passed"]
                        for r in results
                    ),
                    "complete": len(results) == len(cases),
                },
                indent=2,
            )
            + "\n"
        )
    if not all(
        r["batched"]["passed"] and r.get("automatic", {"passed": True})["passed"]
        for r in results
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
