"""Run baseline/candidate GLM kpool serving checks on one TPU allocation.

Both source trees must use the same installed dependencies and local model.
The process owns only the server groups it launches. Source identities are
recorded by the caller in SOURCE_REVISION files when using archived trees.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import bench_kpool_serving as client


def stop(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def ready(process, url, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Server exited with code {process.returncode}")
        try:
            with urllib.request.urlopen(url + "/health", timeout=5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(5)
    raise TimeoutError("Server did not become healthy within the startup budget")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--startup-timeout", type=int, default=3600)
    parser.add_argument("--port", type=int, default=30000)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not (args.model / "config.json").is_file():
        parser.error("--model must be a local checkpoint directory")
    cases = args.output_dir / "cases.json"
    client.prepare(SimpleNamespace(model=str(args.model.resolve()), output=cases))
    base_url = f"http://127.0.0.1:{args.port}"
    for label, root in (("baseline", args.baseline), ("candidate", args.candidate)):
        # Do not mistake an unrelated server for the process launched below.
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", args.port))
        root = root.resolve()
        revision = (root / "SOURCE_REVISION").read_text().strip()
        if not revision:
            raise ValueError(f"Missing source identity for {label}")
        overrides = {
            "MODEL": str(args.model.resolve()),
            "HOST": "127.0.0.1",
            "PORT": str(args.port),
            "TORCH_TPU_TIER2_NAME": f"kpool-pair-{label}-{os.getpid()}",
            "TORCH_TPU_TIER3_ROOT": str(args.output_dir / f"compile-{label}"),
        }
        command = [
            sys.executable,
            "-u",
            "-m",
            "benchmark.tpu.launch_server",
            "--config",
            "recipe/configs/glm_5_3_flash/server.py",
            "--runtime-envs",
            json.dumps(overrides),
        ]
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join((str(root / "python"), str(root)))
        env["PYTHONUNBUFFERED"] = "1"
        identity = {
            "source_revision": revision,
            "command": command,
            "pythonpath": env["PYTHONPATH"],
            "model": overrides["MODEL"],
        }
        (args.output_dir / f"{label}-launch.json").write_text(
            json.dumps(identity, indent=2)
        )
        with (args.output_dir / f"{label}-server.log").open("w") as log:
            process = subprocess.Popen(
                command,
                cwd=root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                ready(process, base_url, args.startup_timeout)
                client.run(
                    SimpleNamespace(
                        cases=cases,
                        base_url=base_url,
                        label=label,
                        source_revision=revision,
                        output_tokens=64,
                        warmups=1,
                        repeats=3,
                        output=args.output_dir / f"{label}.json",
                    )
                )
                if process.poll() is not None:
                    raise RuntimeError("Server exited during measured requests")
            finally:
                stop(process)
    client.compare(
        SimpleNamespace(
            baseline=args.output_dir / "baseline.json",
            candidate=args.output_dir / "candidate.json",
            output=args.output_dir / "comparison.json",
        )
    )


if __name__ == "__main__":
    main()
