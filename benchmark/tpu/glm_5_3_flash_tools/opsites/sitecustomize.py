"""Debug only: count Python call sites of small eager TPU ops after warmup (arm via ARM file).

Wraps the op families the MTP round still launches eagerly (to/add/sub/getitem/arange/
index_copy_/repeat_interleave/reshape-like views) and prints the top sites every 20 s with the
tag OPSITES. Loaded via PYTHONPATH for one diagnostic boot; never committed.
"""
import collections
import os
import threading
import time
import traceback

import torch

_FLAG = "/dev/shm/home/qinghanc/work/opsites/ARM"
_COUNTS = collections.Counter()
_LOCK = threading.Lock()


def _site():
    frames = [f for f in traceback.extract_stack()[:-2] if "/sglang/" in f.filename]
    return " <- ".join(
        f"{f.filename.split('/sglang/')[-1]}:{f.lineno}" for f in reversed(frames[-3:])
    )


def _is_tpu(x):
    return isinstance(x, torch.Tensor) and x.device.type == "tpu"


def _wrap_method(name):
    orig = getattr(torch.Tensor, name)

    def wrapper(self, *args, **kwargs):
        if not torch.compiler.is_compiling() and _is_tpu(self):
            with _LOCK:
                _COUNTS[(name, _site())] += 1
        return orig(self, *args, **kwargs)

    setattr(torch.Tensor, name, wrapper)


def _wrap_fn(mod, name):
    orig = getattr(mod, name)

    def wrapper(*args, **kwargs):
        dev = kwargs.get("device")
        if not torch.compiler.is_compiling() and (
            (isinstance(dev, torch.device) and dev.type == "tpu")
            or (isinstance(dev, str) and dev.startswith("tpu"))
            or any(_is_tpu(a) for a in args)
        ):
            with _LOCK:
                _COUNTS[(name, _site())] += 1
        return orig(*args, **kwargs)

    setattr(mod, name, wrapper)


def _reporter():
    while not os.path.exists(_FLAG):
        time.sleep(2)
    # Only methods compiled-forward guards do not check; wrapping __getitem__,
    # view or torch.arange triggers a forbidden recompile.
    for n in ("to", "index_copy_", "repeat_interleave"):
        _wrap_method(n)
    while True:
        time.sleep(20)
        with _LOCK:
            top = _COUNTS.most_common(40)
        if top:
            print(f"OPSITES pid={os.getpid()}", flush=True)
            for (name, site), n in top:
                print(f"OPSITES {n:7d} {name:18s} {site}", flush=True)


if os.environ.get("OPSITES") == "1":
    threading.Thread(target=_reporter, daemon=True).start()
