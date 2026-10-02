"""Capture native stacks when a short TPU worker phase stalls."""

import contextlib
import os
import subprocess
import threading

import psutil


@contextlib.contextmanager
def trace_slow_workers():
    parent = psutil.Process(os.getpid())

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
            for process in workers[:2]:
                print(f"Native stack for slow TPU worker pid={process.pid}", flush=True)
                subprocess.run(
                    ["py-spy", "dump", "--native", "--pid", str(process.pid)],
                    timeout=20,
                    check=False,
                )
        except (OSError, psutil.Error, subprocess.TimeoutExpired) as error:
            print(f"Native stack capture failed: {error}", flush=True)

    timer = threading.Timer(45, dump)
    timer.daemon = True
    timer.start()
    try:
        yield
    finally:
        timer.cancel()
        timer.join(timeout=45)
