"""The vendored Kimi-K3 decode megakernel under SGLang's JAX pin.

Export: SGLang's decode program (program.make_decode_program) exports on the
host, at TP32 over 32 CPU devices, to one TPU custom call with no
stablehlo.partition_id, for the 1- and 8-row tiles of 8-row state arrays with
SGLang's uint8 experts. Components: upstream's own CPU checks
(Inferact/tpu-megakernels tests/test_decode_megakernel_cpu.py at the vendored
commit) run the kernel's arithmetic with ``interpret=True``.
"""

import importlib.util
import json
import os
import subprocess
import sys
import unittest

_WORKER_ARG = "--export-worker"
if _WORKER_ARG in sys.argv:
    # TP32 needs 32 devices; the export never touches a TPU.
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=32"
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_tpu_ci(est_time=90, suite="stage-a-test-tpu-unit", chips=1)

_HAS_JAX = importlib.util.find_spec("jax") is not None
_ROWS = 8
_CONTEXT = 384
_LAYERS = 93


def _export(batch_size: int) -> dict:
    """Export one decode tile; runs in the 32-device worker process."""
    from unittest import mock

    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    from sglang_tpu.kimi_k3_megakernel import program
    from sglang_tpu.kimi_k3_megakernel._vendor.kimi import load
    from sglang_tpu.pallas_export import tpu_info_for_cpu_export

    mesh = Mesh(np.array(jax.devices()), ("tp",))
    sharded = NamedSharding(mesh, P("tp"))
    replicated = NamedSharding(mesh, P())
    shapes = load.selected_local_shapes(_LAYERS)
    with (
        mock.patch(
            "sglang_tpu.pallas_export.get_torch_tpu_device_kind",
            return_value="TPU7x",
        ),
        tpu_info_for_cpu_export(),
    ):
        weights = load.abstract_weights(mesh, None, layers=_LAYERS)
        # SGLang's experts: uint8 bytes (4x the uint32 sublanes), uint8 scales.
        for name in ("expert_gu", "expert_down"):
            layers, experts, words, width = shapes[name]
            weights[name] = jax.ShapeDtypeStruct(
                (32, layers, experts, words * 4, width), jnp.uint8, sharding=sharded
            )
        for name in ("expert_gus", "expert_ds"):
            weights[name] = jax.ShapeDtypeStruct(
                (32, *shapes[name]), jnp.uint8, sharding=sharded
            )
        step, _, output_specs, donated, _ = program.make_decode_program(
            mesh, tuple(shapes), batch_size=batch_size, layers=_LAYERS, tuned=True
        )
        states = [
            jax.ShapeDtypeStruct(
                (32 * shape[0], *shape[1:]), jnp.dtype(dtype), sharding=sharded
            )
            for shape, dtype in program.state_shapes(_ROWS, _CONTEXT, _LAYERS)
        ]
        rank = jax.ShapeDtypeStruct((32,), jnp.int32, sharding=sharded)
        rows = jax.ShapeDtypeStruct((batch_size,), jnp.int32, sharding=replicated)
        jitted = jax.jit(
            step,
            donate_argnums=donated,
            keep_unused=True,
            out_shardings=tuple(NamedSharding(mesh, spec) for spec in output_specs),
        )
        exported = jax.export.export(jitted, platforms=["tpu"])(
            *(weights[name] for name in shapes), *states, rank, rows, rows
        )
    module = exported.mlir_module()
    return {
        "custom_calls": module.count("tpu_custom_call"),
        "partition_ids": module.count("partition_id"),
        "aliased_outputs": module.count("tf.aliasing_output"),
        "outputs": [[list(a.shape), str(a.dtype)] for a in exported.out_avals],
    }


@unittest.skipUnless(_HAS_JAX, "requires JAX")
class TestDecodeProgramExport(CustomTestCase):
    def test_exports_one_custom_call_without_partition_id(self):
        """A partition_id outside the custom call crashes torch_tpu's second
        SPMD partitioning, so the rank must come in as an operand."""
        for batch_size in (1, 8):
            with self.subTest(batch_size=batch_size):
                completed = subprocess.run(
                    [sys.executable, __file__, _WORKER_ARG, str(batch_size)],
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr[-4000:])
                result = json.loads(completed.stdout.strip().splitlines()[-1])
                self.assertEqual(result["custom_calls"], 1)
                self.assertEqual(result["partition_ids"], 0)
                # The four donated states come back in place.
                self.assertEqual(result["aliased_outputs"], 4)
                self.assertEqual(
                    result["outputs"],
                    [
                        [[batch_size, 163840], "float32"],
                        [[256, 69, 3, 3, 3, 128], "bfloat16"],
                        [[256, 69, 3, 128, 128], "float32"],
                        [[256, 24, 3, _CONTEXT, 256], "bfloat16"],
                        [[256, 24, 3, _CONTEXT, 128], "bfloat16"],
                    ],
                )


@unittest.skipUnless(_HAS_JAX, "requires JAX")
class TestVendoredKernelComponents(CustomTestCase):
    """Upstream's CPU component checks, ported from pytest."""

    def setUp(self):
        import jax
        import jax.numpy as jnp
        import numpy as np

        from sglang_tpu.kimi_k3_megakernel._vendor import kimi
        from sglang_tpu.kimi_k3_megakernel._vendor.kimi import (
            decode_megakernel as megakernel,
        )
        from sglang_tpu.kimi_k3_megakernel._vendor.kimi import load as kimi_load

        self.jax, self.jnp, self.np = jax, jnp, np
        self.kimi, self.megakernel, self.kimi_load = kimi, megakernel, kimi_load

    def _dot(self, left, right, contract=(1, 0)):
        return self.jax.lax.dot_general(
            left,
            right,
            (((contract[0],), (contract[1],)), ((), ())),
            preferred_element_type=self.jnp.float32,
        )

    def _bf16(self, rng, shape):
        return self.jnp.asarray(rng.normal(size=shape), self.jnp.float32).astype(
            self.jnp.bfloat16
        )

    def test_mxu_dot_padding_matches_unpadded_matmul(self):
        for rows in (1, 4, 8):
            with self.subTest(rows=rows):
                rng = self.np.random.default_rng(rows)
                left, right = self._bf16(rng, (rows, 256)), self._bf16(rng, (256, 128))
                actual = self.megakernel._dot(left, right)
                self.assertEqual(actual.shape, (rows, 128))
                self.np.testing.assert_allclose(
                    actual, self._dot(left, right), rtol=1e-5, atol=1e-4
                )

    def test_probability_value_mxu_matches_split_dot(self):
        jnp = self.jnp
        for rows in (1, 8):
            for mode in ("bf16", "hilo"):
                with self.subTest(rows=rows, mode=mode):
                    rng = self.np.random.default_rng(rows)
                    probabilities = jnp.asarray(
                        rng.uniform(size=(rows, 128)), jnp.float32
                    )
                    values = self._bf16(rng, (128, 128))
                    actual = self.megakernel._probability_value_mxu(
                        probabilities, values, mode
                    )
                    high = probabilities.astype(jnp.bfloat16)
                    expected = self._dot(high, values)
                    if mode == "hilo":
                        low = (probabilities - high.astype(jnp.float32)).astype(
                            jnp.bfloat16
                        )
                        expected += self._dot(low, values)
                    self.assertEqual(actual.shape, (rows, 128))
                    self.np.testing.assert_allclose(
                        actual, expected, rtol=1e-5, atol=1e-4
                    )

    def test_attention_scores_pad_partial_mxu_row_tile(self):
        rng = self.np.random.default_rng(4)
        queries, keys = self._bf16(rng, (5, 256)), self._bf16(rng, (128, 256))
        actual = self.megakernel._attention_scores(queries, keys)
        self.assertEqual(actual.shape, (5, 128))
        self.np.testing.assert_allclose(
            actual, self._dot(queries, keys, contract=(1, 1)), rtol=1e-5, atol=1e-4
        )

    def test_residual_mixture_matches_direct_formula(self):
        jax, jnp, np = self.jax, self.jnp, self.np
        for relaxed in (False, True):
            with self.subTest(relaxed=relaxed):
                rng = np.random.default_rng(7)
                blocks = self._bf16(rng, (8, 256))
                prefix = self._bf16(rng, (1, 256))
                folded = self._bf16(rng, (1, 256))
                actual = self.megakernel._residual_mixture_row(
                    blocks,
                    prefix,
                    folded,
                    5,
                    epsilon=1e-5,
                    relaxed_reduction_order=relaxed,
                )
                values = jnp.concatenate((blocks[:5], prefix), axis=0).astype(
                    jnp.float32
                )
                scores = jnp.sum(values * folded.astype(jnp.float32), axis=1)
                scores *= jax.lax.rsqrt(jnp.mean(values * values, axis=1) + 1e-5)
                expected = jnp.sum(
                    jax.nn.softmax(scores)[:, None] * values, axis=0, keepdims=True
                )
                np.testing.assert_allclose(
                    np.asarray(actual, dtype=np.float32),
                    np.asarray(expected, dtype=np.float32),
                    rtol=5e-3,
                    atol=5e-3,
                )

    def test_interpreted_quantized_expert_dot_matches_dense_dequant(self):
        jax, jnp, np = self.jax, self.jnp, self.np
        from jax.experimental import pallas as pl

        rng = np.random.default_rng(11)
        source = self._bf16(rng, (2, 256))
        packed = jnp.asarray(
            rng.integers(0, np.iinfo(np.uint32).max, size=(32, 128), dtype=np.uint32)
        )
        scales = jnp.asarray(rng.integers(124, 130, size=(8, 128), dtype=np.uint8))
        expected = self._dot(source, self.kimi.unpack_mxfp4(packed, scales))

        def kernel(source_ref, weights_ref, scales_ref, output_ref):
            self.megakernel._quantized_expert_dot(
                source_ref, weights_ref, scales_ref, output_ref, 256, 128
            )

        for storage in ("mxfp4", "reblocked_fp8"):
            with self.subTest(storage=storage):
                if storage == "reblocked_fp8":
                    weights, dot_scales, lossless = self.kimi_load._reblock_one(
                        packed, scales, 256
                    )
                    self.assertTrue(bool(lossless))
                else:
                    weights, dot_scales = packed, scales
                actual = pl.pallas_call(
                    kernel,
                    out_shape=jax.ShapeDtypeStruct((2, 128), jnp.float32),
                    interpret=True,
                )(source, weights, dot_scales)
                np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


if __name__ == "__main__":
    if _WORKER_ARG in sys.argv:
        print(json.dumps(_export(int(sys.argv[-1]))))
    else:
        unittest.main()
