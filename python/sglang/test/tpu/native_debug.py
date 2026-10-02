"""Capture native stacks when a short TPU worker phase stalls."""

import contextlib
import json
import os
import subprocess
import threading
from urllib.parse import urlsplit

import psutil


def runtime_diagnostic_environment():
    """Opt-in native lifecycle logging; do not change runtime behavior."""
    modules = (
        "tpunetd_client=3,session_manager=3,vbar_control=3,tpu_hal=2,"
        "tpu_system_state=2,pjrt_state=1,pjrt_c_api_client=1"
    )
    return {
        "TPU_MIN_LOG_LEVEL": "0",
        "TPU_STDERR_LOG_LEVEL": "0",
        "TPU_VMODULE": modules,
        "TF_CPP_MIN_LOG_LEVEL": "0",
        "TF_CPP_VMODULE": modules,
        "GRPC_VERBOSITY": "DEBUG",
        # Do not trace RPC headers/payloads or channel args (proxy credentials).
        "GRPC_TRACE": "connectivity_state",
    }


def log_runtime_environment(phase):
    # Never dump the whole environment: CI workers may carry storage credentials.
    keys = (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "NODE_RANK",
        "TPU_WORKER_ID",
        "TPU_WORKER_HOSTNAMES",
        "TORCH_TPU_TOPOLOGY",
        "TORCH_TPU_SLICEBUILDER_ADDRESSES",
        "TPU_PROCESS_ADDRESSES",
        "TPU_PROCESS_BOUNDS",
        "TPU_CHIPS_PER_PROCESS_BOUNDS",
        "TPU_HOST_BOUNDS",
        "TPU_CHIPS_PER_HOST_BOUNDS",
        "TPU_VISIBLE_CHIPS",
        "VBAR_NODE_IP",
        "TPU_PREMAPPED_BUFFER_SIZE",
        "TORCH_TPU_TIER2_COMPILATION_CACHE",
        "NO_PROXY",
        "no_proxy",
        "no_grpc_proxy",
        *runtime_diagnostic_environment(),
    )
    environment = {key: os.environ.get(key) for key in keys}
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "grpc_proxy",
        "VBAR_CONTROL_SERVICE_URL",
    ):
        value = os.environ.get(key)
        if value:
            try:
                parsed = urlsplit(value if "://" in value else "//" + value)
                value = {"host": parsed.hostname, "port": parsed.port}
            except ValueError:
                value = "<set; invalid URL>"
        environment[key] = value
    print(
        json.dumps({"phase": phase, "pid": os.getpid(), "environment": environment}),
        flush=True,
    )


@contextlib.contextmanager
def trace_slow_workers():
    parent = psutil.Process(os.getpid())
    stopped = threading.Event()

    def dump():
        try:
            workers = [
                process
                for process in parent.children(recursive=True)
                if any(
                    "spawn_main" in arg or arg == "--worker"
                    for arg in process.cmdline()
                )
            ]
            # torchrun's own argv also includes --worker. Sample the actual leaf
            # workers, not the launcher waiting for them.
            parent_pids = {process.ppid() for process in workers}
            workers = [process for process in workers if process.pid not in parent_pids]
            for process in workers:
                try:
                    connections = [
                        {
                            "local": tuple(c.laddr),
                            "remote": tuple(c.raddr),
                            "status": c.status,
                        }
                        for c in process.net_connections(kind="tcp")
                    ]
                    print(
                        json.dumps(
                            {
                                "pid": process.pid,
                                "status": process.status(),
                                "connections": connections,
                            }
                        ),
                        flush=True,
                    )
                except psutil.Error as error:
                    print(f"Worker socket capture failed: {error}", flush=True)
            for process in workers[:: max(1, len(workers) - 1)][:2]:
                if stopped.is_set():
                    break
                print(f"Native stack for slow TPU worker pid={process.pid}", flush=True)
                subprocess.run(
                    ["py-spy", "dump", "--native", "--pid", str(process.pid)],
                    timeout=10,
                    check=False,
                )
        except (OSError, psutil.Error, subprocess.TimeoutExpired) as error:
            print(f"Native stack capture failed: {error}", flush=True)

    def monitor():
        while not stopped.wait(30):
            dump()

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join(timeout=15)
