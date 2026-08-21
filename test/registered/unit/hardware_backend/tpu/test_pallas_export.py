# SPDX-License-Identifier: Apache-2.0

"""Unit tests for TorchTPU abstract-device JAX export topology."""

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.hardware_backend.tpu import pallas_export
from sglang.srt.runtime_context import get_resources
from sglang.test.ci.ci_register import register_cpu_ci, register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_tpu_ci(est_time=20, suite="stage-a-test-tpu-unit")


class TestPallasExportEnvironment(CustomTestCase):
    def tearDown(self):
        get_resources().tpu_group_routes.clear()

    def test_tp8_device_metadata_reproduces_validated_routes(self):
        routes = pallas_export.device_attributes_to_group_routes(
            tuple(
                {"id": device_id, "slice_index": 0}
                for device_id in (0, 1, 4, 5, 6, 7, 2, 3)
            )
        )

        self.assertEqual(routes, (0, 1, 4, 5, 6, 7, 2, 3))

    def test_two_host_tp16_device_metadata_produces_complete_routes(self):
        expected = (
            8,
            9,
            12,
            13,
            14,
            15,
            10,
            11,
            0,
            1,
            4,
            5,
            6,
            7,
            2,
            3,
        )
        routes = pallas_export.device_attributes_to_group_routes(
            tuple(
                {"id": device_id, "slice_index": 0} for device_id in expected
            )
        )

        self.assertEqual(routes, expected)
        self.assertEqual(sorted(routes), list(range(16)))

    def test_rejects_invalid_route_ids_or_multiple_slices(self):
        with self.assertRaisesRegex(ValueError, "permutation"):
            pallas_export.device_attributes_to_group_routes(
                ({"id": 0, "slice_index": 0}, {"id": 0, "slice_index": 0})
            )
        with self.assertRaisesRegex(ValueError, "single TPU slice"):
            pallas_export.device_attributes_to_group_routes(
                ({"id": 0, "slice_index": 0}, {"id": 1, "slice_index": 1})
            )

    def test_gathers_topology_over_cpu_process_group(self):
        device_ids = (0, 1, 4, 5, 6, 7, 2, 3)
        attributes = tuple(
            {
                "id": device_id,
                "coords": (rank, 0, 0),
                "core_on_chip": 0,
                "device_kind": "TPU v7",
                "process_index": 0,
                "slice_index": 0,
            }
            for rank, device_id in enumerate(device_ids)
        )
        device_module = SimpleNamespace(
            get_local_device_attributes=Mock(return_value=attributes[0])
        )
        cpu_group = object()

        def gather(output, _local, *, group):
            self.assertIs(group, cpu_group)
            output[:] = attributes

        with (
            patch.object(torch, "get_device_module", return_value=device_module),
            patch(
                "torch.distributed.all_gather_object", side_effect=gather
            ) as gather_op,
        ):
            routes = pallas_export.get_tpu_group_rank_to_device_id(
                tuple(range(8)),
                cpu_group,
            )
            cached_routes = pallas_export.get_tpu_group_rank_to_device_id(
                tuple(range(8)),
                cpu_group,
            )

        self.assertEqual(routes, device_ids)
        self.assertEqual(cached_routes, device_ids)
        gather_op.assert_called_once()

    def test_abstract_export_devices_scale_with_runtime_topology(self):
        routes = (8, 9, 12, 13, 14, 15, 10, 11, 0, 1, 4, 5, 6, 7, 2, 3)
        with patch.object(
            pallas_export,
            "get_pallas_tpu_device_kind",
            return_value="TPU7x",
        ):
            devices = pallas_export.get_torchtpu_export_devices(routes)

        self.assertEqual(tuple(device.id for device in devices), routes)
        self.assertEqual(devices[0].client.device_count(), 16)
        self.assertTrue(all(device.client is devices[0].client for device in devices))

    def test_configure_export_environment_does_not_force_cpu(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("JAX_PLATFORMS", None)
            os.environ.pop("XLA_FLAGS", None)

            pallas_export.configure_jax_export_platform()
            pallas_export.configure_jax_export_env(16)

            self.assertNotIn("JAX_PLATFORMS", os.environ)
            self.assertNotIn("XLA_FLAGS", os.environ)

    def test_maps_torchtpu_device_kind_to_pallas_spelling(self):
        cases = (
            ("TPU v6e", "TPU v6 lite"),
            ("TPU v7", "TPU7x"),
            ("TPU v5e", "TPU v5e"),
        )
        for torch_device_kind, pallas_device_kind in cases:
            with self.subTest(torch_device_kind=torch_device_kind), patch.object(
                pallas_export,
                "get_torch_tpu_device_kind",
                return_value=torch_device_kind,
            ):
                self.assertEqual(
                    pallas_export.get_pallas_tpu_device_kind(), pallas_device_kind
                )

    @unittest.skipUnless(importlib.util.find_spec("jax"), "requires JAX")
    def test_abstract_export_context_reports_and_restores_tpu_target(self):
        from jax._src.pallas.mosaic import tpu_info
        from jax.experimental.pallas import tpu as pltpu

        original_public = pltpu.get_tpu_info
        original_private = tpu_info.get_tpu_info
        original_device_kind = tpu_info.get_device_kind
        original_is_tpu_device = tpu_info.is_tpu_device

        with patch.object(
            pallas_export, "get_pallas_tpu_device_kind", return_value="TPU7x"
        ):
            with pallas_export.tpu_info_for_cpu_export():
                self.assertEqual(tpu_info.get_device_kind(), "TPU7x")
                self.assertTrue(tpu_info.is_tpu_device())
                self.assertEqual(pltpu.get_tpu_info().chip_version.value, "7x")
                self.assertIs(pltpu.get_tpu_info(), tpu_info.get_tpu_info())

        self.assertIs(pltpu.get_tpu_info, original_public)
        self.assertIs(tpu_info.get_tpu_info, original_private)
        self.assertIs(tpu_info.get_device_kind, original_device_kind)
        self.assertIs(tpu_info.is_tpu_device, original_is_tpu_device)

    @unittest.skipUnless(
        importlib.util.find_spec("jax") and importlib.util.find_spec("torch_tpu"),
        "requires JAX and TorchTPU",
    )
    def test_jax_op_preserves_unused_input_without_forcing_cpu(self):
        script = textwrap.dedent("""
            import json
            import os

            import jax
            import torch
            import torch_tpu  # noqa: F401

            from sglang.srt.hardware_backend.tpu.pallas_jax_op import (
                cpu_export_jax_op,
            )

            def add_one(x: jax.Array, unused: jax.Array) -> jax.Array:
                return x + 1

            op = cpu_export_jax_op(
                "sglang_tpu_test::abstract_export_add_one",
                add_one,
            )
            x = torch.arange(8, dtype=torch.float32, device="tpu")
            unused = torch.zeros(8, dtype=torch.float32, device="tpu")
            y = op(x, unused)
            torch.tpu.synchronize()
            print(
                json.dumps(
                    {
                        "jax_platforms": os.environ.get("JAX_PLATFORMS"),
                        "result": y.cpu().tolist(),
                    }
                )
            )
            """)
        env = os.environ.copy()
        env.pop("JAX_PLATFORMS", None)
        env.pop("XLA_FLAGS", None)
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
        )
        self.assertEqual(
            completed.returncode,
            0,
            f"subprocess stderr:\n{completed.stderr}",
        )
        result = json.loads(completed.stdout.strip().splitlines()[-1])
        self.assertIsNone(result["jax_platforms"])
        self.assertEqual(result["result"], [float(value) for value in range(1, 9)])


if __name__ == "__main__":
    unittest.main()
