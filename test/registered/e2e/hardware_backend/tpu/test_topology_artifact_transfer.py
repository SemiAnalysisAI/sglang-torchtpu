"""Fresh-process TPU artifact restore must survive reversed input host order.

Run this unittest on every host of a dedicated multi-host slice. The worker
mode also supports transferring artifacts between independent allocations.
"""

import argparse
import faulthandler
import hashlib
import importlib.metadata
import json
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu.eval_support import RecipeLauncher

# Exec mode runs the unittest on every host; no model or server is needed.
register_tpu_ci(
    est_time=300,
    suite="tpu-integration",
    nightly=True,
    labels=["multihost", "collectives", "compile-cache"],
)


def coordinate_worker_with_traceback(*args):
    from sglang.srt.hardware_backend.tpu.node_normalization import _coordinate_worker

    faulthandler.dump_traceback_later(60, repeat=True)
    print(f"coordinate probe {args[2]}:{args[0]} starting", flush=True)
    _coordinate_worker(*args)
    print(
        f"coordinate probe {args[2]}:{args[0]} entering interpreter teardown",
        flush=True,
    )


def worker():
    import torch
    import torch.distributed as dist
    from torch_tpu._internal.compile import tpu_torch_compile as native
    from torch_tpu._internal.distributed.device_mesh import get_device_assignment

    from sglang.srt.hardware_backend.tpu.compilation.direct_compile import (
        direct_compile,
    )
    from sglang.srt.hardware_backend.tpu.compile_cache import (
        add_tpu_rank_assignment_to_compile_cache_key,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action", choices=("populate", "restore", "expect-reject"), required=True
    )
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dist.init_process_group("tpu_dist", timeout=timedelta(seconds=120))
    control = dist.new_group(backend="gloo", timeout=timedelta(seconds=120))
    rank, world = dist.get_rank(), dist.get_world_size()
    add_tpu_rank_assignment_to_compile_cache_key()
    contract = {
        "tag": torch.compiler.config.cache_key_tag,
        "assignment": list(get_device_assignment()),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "torch-tpu", "jax", "libtpu")
        },
    }
    # Materialize the input before counting compiles in the tested region.
    x = torch.full((1, 128), rank + 7, dtype=torch.int32, device="tpu")
    x.cpu()
    artifact = args.artifacts / f"rank-{rank}.bin"
    metadata = args.artifacts / f"rank-{rank}.json"
    result = {
        "rank": rank,
        "contract": contract,
        "attributes": torch.tpu.get_local_device_attributes(),
        "action": args.action,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    before = torch.tpu._get_cache_requests()

    def region(value):
        local = value * 3 + 1
        gathered = torch.empty((world, 128), dtype=value.dtype, device=value.device)
        dist.all_gather_into_tensor(gathered, local)
        reduced = local.clone()
        dist.all_reduce(reduced)
        return local, gathered, reduced

    with torch.no_grad():
        if args.action == "populate":
            compiled = direct_compile(
                region, specialization_key=lambda value: tuple(value.shape)
            )
            outputs = compiled(x)
            executable = next(iter(compiled.cache.values())).executable
            data = native.serialize_executable(executable)
            args.artifacts.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes(data)
            metadata.write_text(json.dumps(contract, sort_keys=True))
        else:
            error = None
            try:
                saved = json.loads(metadata.read_text())
                if saved != contract:
                    error = "incompatible rank assignment or runtime"
                data = artifact.read_bytes()
            except (OSError, ValueError) as exc:
                error = str(exc)
            errors = [None] * world
            dist.all_gather_object(errors, error, group=control)
            if any(errors):
                result.update(status="REJECTED", errors=errors)
                (args.output / f"rank-{rank}.json").write_text(
                    json.dumps(result, default=str)
                )
                assert args.action == "expect-reject", errors
                dist.barrier(group=control)
                dist.destroy_process_group()
                return
            assert args.action == "restore", "Noncanonical control unexpectedly matched"
            with patch.object(
                native,
                "traverse_and_compile",
                side_effect=AssertionError("Restore must not compile"),
            ):
                executable = native.load_serialized_executable(data)
                outputs = native.execute(executable, [x])
        actual = [value.cpu() for value in outputs]
    expected_gather = (torch.arange(world, dtype=torch.int32) * 3 + 22)[:, None].expand(
        world, 128
    )
    expected = (
        torch.full((1, 128), rank * 3 + 22, dtype=torch.int32),
        expected_gather,
        torch.full((1, 128), int(expected_gather[:, 0].sum()), dtype=torch.int32),
    )
    result.update(
        status=(
            "PASS"
            if all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
            else "WRONG_OUTPUT"
        ),
        sha256=hashlib.sha256(data).hexdigest(),
        cache_requests=torch.tpu._get_cache_requests() - before,
        samples=[value[:, 0].tolist() for value in actual],
    )
    (args.output / f"rank-{rank}.json").write_text(
        json.dumps(result, default=str, indent=2)
    )
    print(json.dumps(result, default=str), flush=True)
    assert result["status"] == "PASS", result
    if args.action == "restore":
        assert result["cache_requests"] == 0, result
    dist.barrier(group=control)
    dist.destroy_process_group()


class TestTopologyArtifactTransfer(unittest.TestCase):
    def test_restore_after_reversing_host_order(self):
        import torch.distributed as dist

        from sglang.srt.hardware_backend.tpu import node_normalization as normalization
        from sglang.srt.server_args import ServerArgs

        hosts = tuple(
            filter(None, os.environ.get("TPU_WORKER_HOSTNAMES", "").split(","))
        )
        # Missing infrastructure must not silently turn this TPU gate green.
        self.assertGreaterEqual(
            len(hosts), 2, "requires a dedicated multi-host TPU slice"
        )
        host_index = int(os.environ["TPU_WORKER_ID"])
        self.assertIn(host_index, range(len(hosts)))
        config = normalization.TopologyConfig(
            hosts,
            normalization.dimensions(os.environ["TORCH_TPU_TOPOLOGY"]),
            devices_per_host=int(os.environ.get("TORCH_TPU_DEVICES_PER_HOST", "8")),
        )
        store = dist.TCPStore(
            hosts[0],
            29591,
            len(hosts),
            host_index == 0,
            timeout=timedelta(seconds=360),
        )
        world = len(hosts) * config.devices_per_host

        with tempfile.TemporaryDirectory(prefix="tpu-topology-test-") as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            environment = {
                **os.environ,
                "OMP_NUM_THREADS": "1",
                "TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS": "false",
                "TORCH_TPU_TIER2_COMPILATION_CACHE": "false",
                "SGLANG_DIRECT_COMPILE_ASYNC": "0",
                "SGLANG_TPU_RANK_LOCAL_OPS": "0",
            }

            def phase(name, phase_config, action, *, normalize=True):
                index = phase_config.hosts.index(hosts[host_index])
                output = root / name
                args = [
                    __file__,
                    "--worker",
                    "--action",
                    action,
                    "--artifacts",
                    str(artifacts),
                    "--output",
                    str(output),
                ]
                env = {
                    **environment,
                    "TORCHINDUCTOR_CACHE_DIR": str(root / f"cache-{name}"),
                }
                error = None
                try:
                    env.update(phase_config.environment(index))
                    with patch.dict(os.environ, env):
                        # The same preparation step the existing Engine invokes.
                        # Disabling it is the negative control, not another launcher.
                        server_args = ServerArgs(
                            model_path="dummy",
                            device="tpu",
                            nnodes=len(hosts),
                            node_rank=index,
                            tp_size=world,
                            dist_init_addr=f"{phase_config.hosts[0]}:{phase_config.master_port}",
                            tpu_node_normalization=normalize,
                        )
                        with patch.object(
                            normalization,
                            "_coordinate_worker",
                            coordinate_worker_with_traceback,
                        ):
                            normalization.prepare_tpu_environment(server_args)
                        ordered_hosts = os.environ["TPU_WORKER_HOSTNAMES"].split(",")
                        order = [
                            phase_config.hosts.index(host) for host in ordered_hosts
                        ]
                        # torchrun is test-only: exercise serialized collectives
                        # without loading a model or replacing the server launcher.
                        command = [
                            sys.executable,
                            "-m",
                            "torch.distributed.run",
                            f"--nnodes={len(hosts)}",
                            f"--nproc-per-node={config.devices_per_host}",
                            f"--node-rank={os.environ['NODE_RANK']}",
                            f"--master-addr={os.environ['MASTER_ADDR']}",
                            f"--master-port={phase_config.master_port}",
                            *args,
                        ]
                        processes = RecipeLauncher(
                            command=command,
                            start_new_session=True,
                            terminate_timeout=10,
                            tag=name,
                        )
                        try:
                            processes.start()
                            result = processes.wait(timeout=300)
                        finally:
                            processes.shutdown()
                    self.assertEqual(result, 0, f"{name} workers failed")
                    rows = [
                        json.loads(path.read_text())
                        for path in output.glob("rank-*.json")
                    ]
                    self.assertEqual(len(rows), config.devices_per_host)
                    self.assertTrue(
                        all(
                            row["status"]
                            == ("REJECTED" if action == "expect-reject" else "PASS")
                            for row in rows
                        )
                    )
                    for row in rows:
                        key = f"sha256/{row['rank']}"
                        if action == "populate":
                            store.set(key, row["sha256"])
                        elif action == "restore":
                            self.assertEqual(row["sha256"], store.get(key).decode())
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                store.set(f"{name}/{host_index}", json.dumps(error))
                errors = [
                    json.loads(store.get(f"{name}/{i}")) for i in range(len(hosts))
                ]
                self.assertEqual(errors, [None] * len(hosts), f"{name}: {errors}")
                print(f"{name}: all {world} ranks passed", flush=True)
                return {"host_order": order}

            plan = phase("populate", config, "populate")
            # Ship the actual serialized bytes through the CPU store, so restores
            # require neither a shared filesystem nor the producing process.
            for path in artifacts.iterdir():
                store.set(f"artifact/{path.name}", path.read_bytes())
            for rank in range(world):
                for suffix in ("bin", "json"):
                    name = f"rank-{rank}.{suffix}"
                    data = store.get(f"artifact/{name}")
                    (artifacts / name).write_bytes(data)

            reversed_config = replace(config, hosts=hosts[::-1])
            restored = phase("restore-reversed", reversed_config, "restore")
            original_hosts = [hosts[i] for i in plan["host_order"]]
            restored_hosts = [reversed_config.hosts[i] for i in restored["host_order"]]
            self.assertEqual(restored_hosts, original_hosts)
            # Deliberately bypass normalization in reverse *physical* host order.
            # The compatibility check must reject before native deserialization.
            phase(
                "reject-mismatch",
                replace(config, hosts=tuple(original_hosts[::-1])),
                "expect-reject",
                normalize=False,
            )
            store.set(f"done/{host_index}", "1")
            if host_index == 0:
                store.wait([f"done/{i}" for i in range(len(hosts))])


if __name__ == "__main__":
    if "--worker" in sys.argv:
        sys.argv.remove("--worker")
        worker()
    else:
        unittest.main()
