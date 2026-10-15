"""Debug only: wrap spec-decode helpers in record_function scopes after warmup (arm via ARM file).

Lets a TPU+CPU profile attribute every eager launch (host `Execute (tt_jit_...)` events) to the
enclosing helper. Loaded via PYTHONPATH for one diagnostic boot; never committed.
"""
import functools
import importlib
import os
import sys
import threading
import time

_FLAG = "/dev/shm/home/qinghanc/work/scopes/ARM"

_FUNCS = {
    "sglang.srt.speculative.eagle_utils": [
        "eagle_prepare_for_decode",
        "eagle_sample",
        "verify_bookkeeping_func",
        "build_eagle_verify_input",
    ],
    "sglang.srt.speculative.eagle_worker_common": [
        "prepare_for_draft",
        "prepare_for_draft_extend",
        "build_eagle_verify_input",
        "run_eagle_verify",
        "eagle_prepare_for_verify",
    ],
    "sglang.srt.speculative.spec_utils": [
        "commit_mamba_states_after_verify",
        "spec_prepare_for_decode",
    ],
}
_METHODS = {
    "sglang.srt.speculative.eagle_worker_v2": {
        "EAGLEWorker": [
            "draft",
            "_prepare_draft_forward_batch",
            "_fused_draft_forward",
            "_draft_extend_for_decode",
        ],
        "EAGLEWorkerV2": ["forward_batch_generation", "verify"],
    },
    "sglang.srt.managers.overlap_utils": {
        "FutureMap": ["resolve_seq_lens_cpu", "publish", "stash"],
    },
}


def _wrap(fn, label):
    # torch record_function does not reach xprof here; JAX's TraceMe does.
    from jax.profiler import TraceAnnotation

    @functools.wraps(fn)
    def inner(*args, **kwargs):
        with TraceAnnotation("SCOPE:" + label):
            return fn(*args, **kwargs)

    return inner


def _arm():
    while not os.path.exists(_FLAG):
        time.sleep(2)
    wrapped = {}
    for mod_name, names in _FUNCS.items():
        mod = importlib.import_module(mod_name)
        for n in names:
            if hasattr(mod, n):
                wrapped[(mod_name, n)] = _wrap(getattr(mod, n), n)
                setattr(mod, n, wrapped[(mod_name, n)])
    # Rebind names other modules imported directly.
    for m in list(sys.modules.values()):
        name = getattr(m, "__name__", "") or ""
        if not name.startswith("sglang.srt"):
            continue
        for (_, n), w in wrapped.items():
            if getattr(m, n, None) is not None and getattr(getattr(m, n), "__wrapped__", None) is None:
                if getattr(getattr(m, n), "__name__", None) == n:
                    setattr(m, n, w)
    for mod_name, classes in _METHODS.items():
        mod = importlib.import_module(mod_name)
        for cls_name, names in classes.items():
            cls = getattr(mod, cls_name, None)
            for n in names if cls is not None else []:
                if hasattr(cls, n):
                    setattr(cls, n, _wrap(getattr(cls, n), f"{cls_name}.{n}"))
    print("SCOPES armed", len(wrapped), flush=True)


if os.environ.get("SCOPES") == "1":
    threading.Thread(target=_arm, daemon=True).start()
