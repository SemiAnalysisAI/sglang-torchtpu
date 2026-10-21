# SPDX-License-Identifier: Apache-2.0
"""Expose intermediate values for the exact-byte checkpoint, not for timing."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import reference
from jax.experimental import pallas as pl

from benchmark import load_candidate

STAGES = (
    "pooled_f32",
    "pooled_bf16",
    "rotation_f32",
    "rotated_bf16",
    "exponent",
    "scale",
    "scaled",
    "quantized",
    "exact_inverse_quantized",
)


def stages(raw, ape, rotate):
    raw = raw.astype(jnp.float32)
    scores = raw[:, :, 128:] + ape[None, :, :]
    scores -= jnp.max(scores, axis=1, keepdims=True)
    weight = jnp.exp(scores)
    weight /= jnp.sum(weight, axis=1, keepdims=True)
    pooled = jnp.sum(weight * raw[:, :, :128], axis=1)
    rounded = pooled.astype(jnp.bfloat16).astype(jnp.float32)
    rotation = rotate(rounded)
    rotated = rotation.astype(jnp.bfloat16).astype(jnp.float32)
    amax = jnp.maximum(jnp.max(jnp.abs(rotated), axis=1, keepdims=True), 1e-4)
    exponent = jnp.ceil(jnp.log2(amax / 448.0))
    scale = jnp.exp2(exponent)
    scaled = rotated / scale
    quantized = (
        jnp.clip(scaled, -448.0, 448.0).astype(jnp.float8_e4m3fn).astype(jnp.float32)
    )
    inverse = jax.lax.bitcast_convert_type(
        ((127 - exponent.astype(jnp.int32)) << 23).astype(jnp.uint32), jnp.float32
    )
    exact_q = (
        jnp.clip(rotated * inverse, -448.0, 448.0)
        .astype(jnp.float8_e4m3fn)
        .astype(jnp.float32)
    )
    return jnp.stack(
        [
            jnp.broadcast_to(x, pooled.shape)
            for x in (
                pooled,
                rounded,
                rotation,
                rotated,
                exponent,
                scale,
                scaled,
                quantized,
                exact_q,
            )
        ]
    ), weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    opts = parser.parse_args()
    opts.output.mkdir(parents=True, exist_ok=True)
    candidate, digest = load_candidate(Path(__file__).resolve().parents[3])
    butterfly = candidate.__globals__["_hadamard128"]

    def kernel(raw, ape, out, weights):
        result, weight = stages(
            raw[...], ape[...], lambda x: butterfly(x) * (128**-0.5)
        )
        out[...] = result
        weights[...] = weight

    probe = jax.jit(
        pl.pallas_call(
            kernel,
            out_shape=(
                jax.ShapeDtypeStruct((9, 32, 128), jnp.float32),
                jax.ShapeDtypeStruct((32, 4, 128), jnp.float32),
            ),
            grid=(1,),
        )
    )
    oracle = jax.jit(lambda raw, ape: stages(raw, ape, reference.hadamard_rotate))
    records = []
    for index, batch in enumerate((1, 2, 3, 8, 16, 32)):
        # Reproduce the random validation inputs before selecting their raw rows.
        rng = np.random.default_rng(2048 + index)
        pages = batch * 8 + 1
        raw = rng.normal(size=(pages, 128, 256)).astype(np.float32)
        rng.integers(0, 256, (pages, 32, 4, 256), dtype=np.uint8)
        mapping = rng.permutation(np.arange(1, pages)).astype(np.int32)
        starts = np.resize(
            np.array([3, 7, 11, 15, 127, 131, 255, 511], np.int32), batch
        )
        ape = jnp.asarray(rng.normal(size=(4, 128)), jnp.float32)
        selected = np.stack(
            [
                raw[mapping[i * 8 + pos // 128], pos % 128 - 3 : pos % 128 + 1]
                for i, pos in enumerate(starts)
            ]
        )
        rows = jnp.asarray(selected[np.arange(32) % batch], jnp.bfloat16)
        expected, expected_weights = map(np.asarray, oracle(rows, ape))
        actual, actual_weights = map(np.asarray, probe(rows, ape))
        row = {"batch": batch, "stages": {}}
        for stage, name in enumerate(STAGES):
            differences = np.argwhere(actual[stage] != expected[stage])
            row["stages"][name] = {
                "count": len(differences),
                "max_abs_error": float(np.max(np.abs(actual[stage] - expected[stage]))),
                "first": [
                    {
                        "index": idx.tolist(),
                        "reference": float(expected[stage][tuple(idx)]),
                        "pallas": float(actual[stage][tuple(idx)]),
                    }
                    for idx in differences[:4]
                ],
            }
        qdiff = np.argwhere(actual[7] != expected[7])
        row["quantization_mismatches"] = [
            {
                "index": idx.tolist(),
                "reference": expected[:, idx[0], idx[1]].tolist(),
                "pallas": actual[:, idx[0], idx[1]].tolist(),
            }
            for idx in qdiff[:8]
        ]
        row["exact_inverse_vs_reference_count"] = int(
            np.count_nonzero(actual[8] != expected[7])
        )
        row["weights_max_abs_error"] = float(
            np.max(np.abs(actual_weights - expected_weights))
        )
        records.append(row)
        print("PRECISION_DIAGNOSTIC", json.dumps(row), flush=True)
    (opts.output / "precision.json").write_text(
        json.dumps(
            {"candidate_sha256": digest, "stages": STAGES, "cases": records}, indent=2
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
