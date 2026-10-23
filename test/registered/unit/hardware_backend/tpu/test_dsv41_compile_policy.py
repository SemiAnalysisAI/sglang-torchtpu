"""The segmented coordinator must satisfy the decoder policy callable contract."""

import ast
import functools
import inspect
from pathlib import Path
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_bound_coordinator_accepts_install_signature_and_preserves_call():
    compiled_graphs = []

    class CompileRecorder:
        def _compile(self, fn):
            compiled_graphs.append(fn)
            return fn

    # Only the base compiler is replaced. The production subclass is executed
    # unchanged, without importing a device runtime for this Python contract.
    root = Path(__file__).resolve().parents[5]
    path = root / "sglang_tpu/dsv41/compile_policy.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    namespace = {"DecoderTpuCompilePolicy": CompileRecorder, "wraps": functools.wraps}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    policy = namespace["DSV41CompilePolicy"]()

    calls = []

    class Model:
        def forward(self, input_ids, positions, *, forward_batch):
            calls.append((input_ids, positions, forward_batch))
            return forward_batch

    model = Model()
    segments = (lambda value: value, lambda value: value)
    finish = lambda value: value
    logits = lambda value: value
    model.model = SimpleNamespace(segments=segments, finish=finish)
    model.logits_processor = SimpleNamespace(forward=logits)
    policy.owner = model
    original = model.forward
    compiled = policy._compile(original)

    # DecoderTpuCompilePolicy.install attaches this metadata before installing
    # the forward shim. A bound method raises AttributeError here.
    compiled.__signature__ = inspect.signature(original)
    assert inspect.signature(compiled) == inspect.signature(original)
    model.forward = compiled
    batch = object()
    assert model.forward([1], [0], forward_batch=batch) is batch
    assert calls == [([1], [0], batch)]
    assert compiled_graphs == [*segments, logits]
    assert model.model.finish is finish


def test_final_segment_finishes_once_without_extra_host_boundary():
    """Exercise the real coordinator/segment methods with opaque tensor tokens."""
    import sys
    import types

    root = Path(__file__).resolve().parents[5]
    path = root / "sglang_tpu/models/deepseek_v41.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DSV41Model"
    )
    methods = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name in ("run_segment", "forward")
    ]
    namespace = {}
    for node in methods:
        exec(
            compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"),
            namespace,
        )

    events = []
    inputs = SimpleNamespace(shape=(8,))

    class Layer:
        engram = SimpleNamespace(layer_hash_index=4)

        def __call__(self, residual, carry, *args):
            assert residual is None
            events.append(("layer", carry))
            return ("carry",)

    owner = SimpleNamespace(end_layer=2, layers=[None, Layer()], segment_starts=(1,))

    def finish(carry):
        events.append(("finish", carry))
        return ("norm", "aux")

    owner.finish = finish
    owner.host_engram = SimpleNamespace(
        stage_layer=lambda *args: events.append(("stage", args))
    )
    owner.segments = (functools.partial(namespace["run_segment"], owner, 1, 2),)
    sync_module = types.ModuleType("torch_tpu._internal.sync.sync")
    sync_module.synchronize = lambda values, wait: events.append(("sync", values, wait))
    # Restore sys.modules even when this test is run standalone without pytest.
    key = "torch_tpu._internal.sync.sync"
    previous = sys.modules.get(key)
    sys.modules[key] = sync_module
    try:
        out = namespace["forward"](owner, inputs, object(), object())
    finally:
        if previous is None:
            del sys.modules[key]
        else:
            sys.modules[key] = previous
    assert out == ("norm", "aux")
    assert events == [
        ("stage", (4, 8)),
        ("layer", None),
        ("finish", ("carry",)),
        ("sync", ["norm", "aux"], False),
    ]

    events.clear()
    owner.end_layer = 3
    assert namespace["run_segment"](owner, 1, 2, None, inputs, None, None, None) == (
        "carry",
    )
    assert events == [("layer", None)]
