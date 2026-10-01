"""CPU coverage of coordinate discovery and rank-consistent launch plans."""

import itertools
import os
import random
import socket
import subprocess
import sys
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from unittest.mock import patch

from sglang.srt.hardware_backend.tpu import topology_launcher as launcher
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestTopologyLauncher(unittest.TestCase):
    def setUp(self):
        self.config = launcher.LaunchConfig(("source", "destination"), (2, 2, 2, 2))

    def test_all_four_host_permutations_preserve_rank_endpoint_identity(self):
        coords = [(0, 0, 0), (2, 0, 0), (0, 0, 1), (2, 0, 1)]
        for permutation in itertools.permutations(range(4)):
            with self.subTest(permutation=permutation):
                config = replace(
                    self.config,
                    hosts=tuple(f"host-{i}" for i in permutation),
                    topology=(4, 2, 2, 2),
                )
                order = launcher.canonical_host_order(
                    config, [list(coords[i]) for i in permutation]
                )
                for original_index, physical_rank in enumerate(permutation):
                    env = launcher.launch_environment(config, order, original_index)
                    self.assertEqual(env["NODE_RANK"], str(physical_rank))
                    self.assertEqual(env["MASTER_ADDR"], "host-0")
                    self.assertEqual(
                        env["TPU_WORKER_HOSTNAMES"], "host-0,host-1,host-2,host-3"
                    )
                    self.assertEqual(
                        env["TORCH_TPU_SLICEBUILDER_ADDRESSES"].split(",")[
                            physical_rank * 8
                        ],
                        f"host-{physical_rank}:8070",
                    )

    def test_random_host_permutations_and_representative_chips(self):
        rng = random.Random(2948)
        coords = sorted(
            itertools.product(range(0, 8, 2), range(0, 4, 2), range(4)),
            key=lambda c: c[::-1],
        )
        for _ in range(100):
            permutation = rng.sample(range(len(coords)), len(coords))
            config = replace(
                self.config,
                hosts=tuple(f"h{i}" for i in permutation),
                topology=(8, 4, 4, 2),
            )
            samples = [
                [
                    coords[i][0] + rng.randrange(2),
                    coords[i][1] + rng.randrange(2),
                    coords[i][2],
                ]
                for i in permutation
            ]
            order = launcher.canonical_host_order(config, samples)
            self.assertEqual([permutation[i] for i in order], list(range(len(coords))))

    def test_invalid_topology_placement_and_ports_fail_closed(self):
        for changes in (
            {"hosts": ()},
            {"hosts": ("a", "a")},
            {"topology": (2, 2, 3, 2)},
            {"topology": (2, 0, 2, 2)},
            {"topology": (3, 2, 2, 2)},
            {"chips_per_host": (0, 2, 1)},
            {"devices_per_host": 4},
            {"bootstrap_port": 8070},
            {"master_port": 65536},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.config, **changes).validate()
        for coords in (
            [],
            [[0, 0, 0]],
            [[0, 0, 0], [1, 1, 0]],
            [[0, 0, 0], [0, 0, 2]],
            [[0, 0, 0], [-1, 0, 1]],
            [[0, 0, 0], [True, 0, 1]],
            [[0, 0, 0], [0, 1]],
        ):
            with self.subTest(coords=coords), self.assertRaises(ValueError):
                launcher.canonical_host_order(self.config, coords)

    def test_probe_isolated_from_worker_world_and_never_falls_back(self):
        result = subprocess.CompletedProcess(
            [], 0, "runtime noise\nSGLANG_TPU_COORDS=[0, 0, 1]\n", ""
        )
        with (
            patch.dict(os.environ, {key: "old" for key in launcher._WORKER_ENV}),
            patch.object(launcher, "_run_child", return_value=result) as run,
        ):
            self.assertEqual(
                launcher.query_local_coordinates(self.config, 1, 10), [0, 0, 1]
            )
            environment = run.call_args.args[1]
            self.assertNotIn("RANK", environment)
            self.assertNotIn("LOCAL_RANK", environment)
            self.assertNotIn("WORLD_SIZE", environment)
            self.assertEqual(environment["NODE_RANK"], "1")
            self.assertEqual(environment["MASTER_ADDR"], "source")
            self.assertEqual(environment["TORCH_TPU_TOPOLOGY"], "2,2,2,2")
            self.assertIn("--nproc-per-node=8", run.call_args.args[0])
            self.assertEqual(run.call_args.kwargs["timeout"], 10)
            self.assertEqual(os.environ["RANK"], "old")
        for output in ("noise", "SGLANG_TPU_COORDS=[]", "SGLANG_TPU_COORDS=null"):
            with (
                self.subTest(output=output),
                patch.object(
                    launcher,
                    "_run_child",
                    return_value=subprocess.CompletedProcess([], 0, output, ""),
                ),
                self.assertRaises((RuntimeError, ValueError)),
            ):
                launcher.query_local_coordinates(self.config, 1, 10)
        with (
            patch.object(
                launcher,
                "_run_child",
                side_effect=subprocess.TimeoutExpired("probe", 1),
            ),
            self.assertRaises(subprocess.TimeoutExpired),
        ):
            launcher.query_local_coordinates(self.config, 1, 1)

    def test_commands_use_canonical_node_rank(self):
        env = launcher.launch_environment(self.config, [1, 0], 0)
        for mode, args in (
            ("server", ["--model-path=model", "--tp=16", "--ep=16"]),
            ("torchrun", ["probe.py", "--action=restore"]),
        ):
            command = launcher.launch_command(mode, self.config, env, args)
            self.assertIn("--node-rank=1", command)
            self.assertEqual(command[-len(args) :], args)
        for flag in (
            "--node-rank=0",
            "--node-r=0",
            "--nnodes=4",
            "--use-ray",
            "--dist-init-addr=elsewhere:20",
            "--device=cpu",
        ):
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                launcher.launch_command("server", self.config, env, [flag])
        with self.assertRaises(ValueError):
            launcher.launch_command(
                "torchrun", self.config, env, ["--node-rank=0", "worker.py"]
            )

    def test_hosts_agree_before_launch_using_real_cpu_store(self):

        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        config = replace(
            self.config, hosts=("localhost", "127.0.0.1"), bootstrap_port=port
        )
        local = threading.local()

        def host(index):
            local.coords = [0, 0, 1 - index]
            return launcher.resolve_launch(config, index, 10, "cpu-store-test")

        with patch.object(
            launcher,
            "query_local_coordinates",
            side_effect=lambda config, index, timeout: local.coords,
        ):
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(host, (0, 1)))
        for order, coords, _store in results:
            self.assertEqual(order, [1, 0])
            self.assertEqual(coords, [[0, 0, 1], [0, 0, 0]])

    def test_child_completion_and_timeout_cleanup(self):
        result = launcher._run_child(
            [sys.executable, "-c", "print('complete')"],
            os.environ.copy(),
            timeout=10,
            capture_output=True,
        )
        self.assertEqual(result.stdout.strip(), "complete")
        self.assertEqual(result.returncode, 0)
        with self.assertRaises(subprocess.TimeoutExpired):
            launcher._run_child(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                os.environ.copy(),
                timeout=0.1,
                capture_output=True,
            )


if __name__ == "__main__":
    unittest.main()
