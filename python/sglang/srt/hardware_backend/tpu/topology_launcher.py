"""Launch SGLang with host ranks derived from TPU coordinates, not pod order."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import signal
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path

_COORDS_PREFIX = "SGLANG_TPU_COORDS="
_COORDS_PROGRAM = """
import json
import os
import sys
from datetime import timedelta
import torch.distributed as dist
from torch_tpu._internal.distributed import tpu_distributed
dist.init_process_group('gloo', timeout=timedelta(seconds=int(sys.argv[1])))
coords = tpu_distributed.local_device_coords()
if int(os.environ['LOCAL_RANK']) == 0:
    print('SGLANG_TPU_COORDS=' + json.dumps(coords), flush=True)
dist.barrier()
dist.destroy_process_group()
"""
_WORKER_ENV = (
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "NODE_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
    "TORCH_TPU_SLICEBUILDER_ADDRESSES",
)


def dimensions(value: str) -> tuple[int, ...]:
    result = tuple(int(part) for part in value.lower().replace("x", ",").split(","))
    if not result or any(part <= 0 for part in result):
        raise ValueError(f"Expected positive topology dimensions, got {value!r}")
    return result


@dataclass(frozen=True)
class LaunchConfig:
    hosts: tuple[str, ...]
    topology: tuple[int, ...]
    devices_per_host: int = 8
    chips_per_host: tuple[int, ...] = (2, 2, 1)
    bootstrap_port: int = 29590
    master_port: int = 29500
    slicebuilder_port: int = 8070

    def validate(self) -> None:
        if not self.hosts or len(set(self.hosts)) != len(self.hosts):
            raise ValueError("Provide a nonempty list of distinct worker hosts")
        if any(
            not host or ":" in host or any(c.isspace() for c in host)
            for host in self.hosts
        ):
            raise ValueError("Use worker DNS names or IPv4 addresses without ports")
        if len(self.topology) not in (3, 4) or len(self.chips_per_host) != 3:
            raise ValueError("Topology must have 3 or 4 dimensions; chip bounds need 3")
        if any(
            type(n) is not int or n <= 0
            for n in (*self.topology, *self.chips_per_host, self.devices_per_host)
        ):
            raise ValueError(
                "Topology, chip bounds, and device counts must be positive integers"
            )
        if any(
            size % tile for size, tile in zip(self.topology[:3], self.chips_per_host)
        ):
            raise ValueError("Only complete, uniformly tiled hosts are supported")
        host_count = math.prod(
            size // tile for size, tile in zip(self.topology[:3], self.chips_per_host)
        )
        if (
            host_count != len(self.hosts)
            or math.prod(self.topology) != len(self.hosts) * self.devices_per_host
        ):
            raise ValueError("Topology does not match the host and device counts")
        ports = (
            self.bootstrap_port,
            self.master_port,
            *range(
                self.slicebuilder_port, self.slicebuilder_port + self.devices_per_host
            ),
        )
        if len(set(ports)) != len(ports) or any(
            not 1 <= port <= 65535 for port in ports
        ):
            raise ValueError(
                "Bootstrap, master, and SliceBuilder ports must be distinct valid ports"
            )


def canonical_host_order(
    config: LaunchConfig, coordinates: list[list[int]]
) -> list[int]:
    """Return input host indices in TorchTPU's Z-major, X-minor host order."""
    config.validate()
    if len(coordinates) != len(config.hosts):
        raise ValueError("Missing host coordinates")
    host_coords = []
    for coords in coordinates:
        if len(coords) != 3 or any(
            type(c) is not int or not 0 <= c < bound
            for c, bound in zip(coords, config.topology)
        ):
            raise ValueError(f"Invalid slice-relative TPU coordinates: {coords}")
        host_coords.append(
            tuple(c // tile for c, tile in zip(coords, config.chips_per_host))
        )
    expected = set(
        itertools.product(
            *(
                range(size // tile)
                for size, tile in zip(config.topology[:3], config.chips_per_host)
            )
        )
    )
    if set(host_coords) != expected or len(set(host_coords)) != len(host_coords):
        raise ValueError(
            f"Duplicate or incomplete physical host placement: {host_coords}"
        )
    return sorted(range(len(host_coords)), key=lambda index: host_coords[index][::-1])


def _run_child(command, environment, *, timeout=None, capture_output=False):
    """Forward termination and reap the whole child process group on failure."""
    process = subprocess.Popen(
        command,
        env=environment,
        start_new_session=True,
        text=True,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.PIPE if capture_output else None,
    )

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    previous = {
        signum: signal.signal(signum, interrupted)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    finally:
        try:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.communicate()
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)


def query_local_coordinates(
    config: LaunchConfig, host_index: int, timeout: int
) -> list[int]:
    # GKE requires SliceBuilder initialization before PJRT can report coords.
    # Bootstrap a temporary full world in input-host order, without loading a
    # model. All probe processes must exit before assigning model-worker ranks.
    child_env = {
        key: value for key, value in os.environ.items() if key not in _WORKER_ENV
    }
    child_env.update(
        launch_environment(config, list(range(len(config.hosts))), host_index)
    )
    child_env["PYTHONPATH"] = os.pathsep.join(path for path in sys.path if path)
    result = _run_child(
        [
            *_torchrun_command(config, child_env),
            "--no-python",
            sys.executable,
            "-c",
            _COORDS_PROGRAM,
            str(timeout),
        ],
        child_env,
        capture_output=True,
        timeout=timeout,
    )
    if result.returncode:
        raise RuntimeError(
            f"Coordinate probe failed ({result.returncode}): {result.stderr}"
        )
    for line in reversed(result.stdout.splitlines()):
        if line.startswith(_COORDS_PREFIX):
            coords = json.loads(line[len(_COORDS_PREFIX) :])
            if not isinstance(coords, list) or not coords:
                raise ValueError("TorchTPU returned no physical coordinates")
            return coords
    raise RuntimeError(
        f"Coordinate probe returned no result: {result.stdout}\n{result.stderr}"
    )


def resolve_launch(config: LaunchConfig, host_index: int, timeout: int, run_id: str):
    """Exchange probe results on a CPU-only store before assigning model ranks."""
    import torch.distributed as dist

    config.validate()
    if not 0 <= host_index < len(config.hosts) or timeout <= 0 or not run_id:
        raise ValueError("Invalid host index, timeout, or rendezvous id")
    store = dist.TCPStore(
        config.hosts[0],
        config.bootstrap_port,
        len(config.hosts),
        host_index == 0,
        timeout=timedelta(seconds=timeout),
    )
    scoped = dist.PrefixStore(run_id, store)
    record = {"config": asdict(config), "coords": None, "error": None}
    try:
        record["coords"] = query_local_coordinates(config, host_index, timeout)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        record["error"] = str(error)
    scoped.set(f"host/{host_index}", json.dumps(record))
    records = [
        json.loads(scoped.get(f"host/{index}")) for index in range(len(config.hosts))
    ]
    errors = [entry["error"] for entry in records]
    if any(errors):
        raise RuntimeError(
            f"TPU coordinate discovery failed; refusing pod-index fallback: {errors}"
        )
    if any(entry["config"] != records[0]["config"] for entry in records):
        raise ValueError("Hosts disagree on the TPU launch configuration")
    coordinates = [entry["coords"] for entry in records]
    order = canonical_host_order(config, coordinates)
    scoped.set(f"ready/{host_index}", "1")
    scoped.wait([f"ready/{index}" for index in range(len(config.hosts))])
    # Keep the server-side store alive until the launched workers exit.
    return order, coordinates, store


def launch_environment(
    config: LaunchConfig, order: list[int], host_index: int
) -> dict[str, str]:
    config.validate()
    if sorted(order) != list(range(len(config.hosts))) or host_index not in order:
        raise ValueError("Invalid canonical host permutation")
    hosts = [config.hosts[index] for index in order]
    return {
        "NODE_RANK": str(order.index(host_index)),
        "NNODES": str(len(hosts)),
        "MASTER_ADDR": hosts[0],
        "MASTER_PORT": str(config.master_port),
        "TPU_WORKER_HOSTNAMES": ",".join(hosts),
        "TPU_TOPOLOGY": "x".join(map(str, config.topology[:3])),
        "TORCH_TPU_TOPOLOGY": ",".join(map(str, config.topology)),
        "TORCH_TPU_DEVICES_PER_HOST": str(config.devices_per_host),
        "TORCH_TPU_BASE_PORT": str(config.slicebuilder_port),
        "TORCH_TPU_SLICEBUILDER_ADDRESSES": ",".join(
            f"{host}:{port}"
            for host in hosts
            for port in range(
                config.slicebuilder_port,
                config.slicebuilder_port + config.devices_per_host,
            )
        ),
    }


def _torchrun_command(config: LaunchConfig, env: dict[str, str]) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nnodes={len(config.hosts)}",
        f"--nproc-per-node={config.devices_per_host}",
        f"--node-rank={env['NODE_RANK']}",
        f"--master-addr={env['MASTER_ADDR']}",
        f"--master-port={config.master_port}",
    ]


def launch_command(
    mode: str, config: LaunchConfig, env: dict[str, str], args: list[str]
) -> list[str]:
    if not args:
        raise ValueError(
            "Provide server arguments or a torchrun worker script after --"
        )
    if mode == "torchrun":
        if args[0].startswith("-"):
            raise ValueError("Provide a worker script path, not torchrun options")
        return [*_torchrun_command(config, env), *args]
    forbidden = {
        "--nnodes",
        "--node-rank",
        "--dist-init-addr",
        "--nccl-init-addr",
        "--device",
        "--tpu-worker-hostnames",
        "--torch-tpu-topology",
        "--torch-tpu-slicebuilder-addresses",
        "--use-ray",
    }
    # SGLang's argument parser accepts unambiguous option abbreviations.
    if any(
        arg.startswith("--")
        and arg.split("=", 1)[0] != "--tp"
        and any(flag.startswith(arg.split("=", 1)[0]) for flag in forbidden)
        for arg in args
    ):
        raise ValueError(
            "The topology launcher owns distributed arguments; Ray is not supported"
        )
    return [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--device=tpu",
        f"--nnodes={len(config.hosts)}",
        f"--node-rank={env['NODE_RANK']}",
        f"--dist-init-addr={env['MASTER_ADDR']}:{config.master_port}",
        *args,
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--hosts",
        default=os.environ.get("TPU_WORKER_HOSTNAMES"),
        required="TPU_WORKER_HOSTNAMES" not in os.environ,
    )
    parser.add_argument(
        "--host-index",
        type=int,
        required=True,
        help="This host's index in --hosts, not its physical rank",
    )
    parser.add_argument(
        "--topology",
        type=dimensions,
        required=True,
        help="Full physical device topology, e.g. 2,2,2,2 on TPU7x",
    )
    parser.add_argument("--chips-per-host", type=dimensions, default=(2, 2, 1))
    parser.add_argument("--devices-per-host", type=int, default=8)
    parser.add_argument("--bootstrap-port", type=int, default=29590)
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--slicebuilder-port", type=int, default=8070)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--rendezvous-id", required=True, help="Same fresh identifier on every host"
    )
    parser.add_argument("--mode", choices=("server", "torchrun"), default="server")
    parser.add_argument(
        "--manifest", type=Path, help="Write this host's launch plan to a new JSON file"
    )
    parser.add_argument("args", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    config = LaunchConfig(
        tuple(options.hosts.split(",")),
        options.topology,
        options.devices_per_host,
        options.chips_per_host,
        options.bootstrap_port,
        options.master_port,
        options.slicebuilder_port,
    )
    config.validate()
    args = options.args[1:] if options.args[:1] == ["--"] else options.args
    # Reject malformed commands before opening any rendezvous or PJRT client.
    launch_command(
        options.mode, config, {"NODE_RANK": "0", "MASTER_ADDR": config.hosts[0]}, args
    )
    order, coordinates, store = resolve_launch(
        config, options.host_index, options.timeout, options.rendezvous_id
    )
    updates = launch_environment(config, order, options.host_index)
    command = launch_command(options.mode, config, updates, args)
    plan = {
        "host_index": options.host_index,
        "coordinates": coordinates,
        "host_order": order,
        "environment": updates,
        "command": command,
    }
    if options.manifest:
        with options.manifest.open("x") as output:
            json.dump(plan, output, indent=2)
    print(json.dumps(plan), flush=True)
    environment = {
        key: value for key, value in os.environ.items() if key not in _WORKER_ENV
    }
    environment.update(updates)
    result = _run_child(command, environment).returncode
    del store
    raise SystemExit(result if result >= 0 else 128 - result)


if __name__ == "__main__":
    main()
