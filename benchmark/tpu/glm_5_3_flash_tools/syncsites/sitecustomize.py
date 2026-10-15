"""Debug only: count the Python call sites of TPU -> CPU copies (eager code, not traced graphs).

Loaded via PYTHONPATH for one diagnostic boot. Every 20 s rank processes print their top sites
with the tag SYNCSITES.
"""
import collections
import os
import threading
import time
import traceback

import torch

_COUNTS = collections.Counter()
_LOCK = threading.Lock()


def _site():
    frames = [f for f in traceback.extract_stack()[:-2] if "/sglang/" in f.filename]
    return " <- ".join(
        f"{f.filename.split('/sglang/')[-1]}:{f.lineno}" for f in reversed(frames[-4:])
    )


def _wrap(name):
    orig = getattr(torch.Tensor, name)

    def wrapper(self, *args, **kwargs):
        if not torch.compiler.is_compiling() and self.device.type == "tpu":
            if name != "to" or _to_cpu(args, kwargs):
                with _LOCK:
                    _COUNTS[(name, _site())] += 1
        return orig(self, *args, **kwargs)

    setattr(torch.Tensor, name, wrapper)


def _to_cpu(args, kwargs):
    dev = kwargs.get("device", args[0] if args else None)
    return (isinstance(dev, str) and dev.startswith("cpu")) or (
        isinstance(dev, torch.device) and dev.type == "cpu"
    )


_FLAG = "/dev/shm/home/qinghanc/work/syncsites/ARM"


def _reporter():
    # Wrapping Tensor methods breaks Dynamo tracing, so arm only after warmup.
    while not os.path.exists(_FLAG):
        time.sleep(2)
    for _n in ("cpu", "item", "tolist", "numpy", "to"):
        _wrap(_n)
    while True:
        time.sleep(20)
        with _LOCK:
            top = _COUNTS.most_common(15)
        if top:
            print(f"SYNCSITES pid={os.getpid()}", flush=True)
            for (name, site), n in top:
                print(f"SYNCSITES {n:7d} {name:6s} {site}", flush=True)


if os.environ.get("SYNCSITES") == "1":
    threading.Thread(target=_reporter, daemon=True).start()
