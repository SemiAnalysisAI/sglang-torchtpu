import importlib.util
from pathlib import Path
import unittest

import torch


_NONSTRICT_PATH = (
    Path(__file__).resolve().parents[4]
    / "python"
    / "sglang"
    / "srt"
    / "hardware_backend"
    / "tpu"
    / "nonstrict.py"
)
_SPEC = importlib.util.spec_from_file_location("tpu_nonstrict", _NONSTRICT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
nonstrict = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(nonstrict)


def _identity_backend(gm, example_inputs):
    return gm


class TestTpuNonstrict(unittest.TestCase):
    def test_make_fx_static_argnames_pytree_inputs(self):
        def fn(xs, *, scale):
            return {"y": xs["x"] * scale}

        gm = nonstrict.make_fx(fn, static_argnames="scale")(
            {"x": torch.ones(2)}, scale=2
        )

        self.assertTrue(hasattr(gm, "_nonstrict_jit_in_spec"))
        self.assertTrue(hasattr(gm, "_nonstrict_jit_out_spec"))
        torch.testing.assert_close(gm(torch.ones(2))[0], torch.tensor([2.0, 2.0]))

    def test_traceable_collective_remap_context(self):
        def fn(x):
            return x + 1

        with nonstrict.traceable_collective_remap():
            gm = nonstrict.make_fx(fn)(torch.ones(2))

        torch.testing.assert_close(gm(torch.ones(2))[0], torch.tensor([2.0, 2.0]))

    def test_jit_static_argnames_pytree_io_and_cache(self):
        traces = 0

        def fn(xs, *, scale):
            nonlocal traces
            traces += 1
            return {"y": xs["x"] * scale}

        compiled = nonstrict.jit(
            fn, static_argnames="scale", backend=_identity_backend
        )

        torch.testing.assert_close(
            compiled({"x": torch.ones(2)}, scale=2)["y"],
            torch.tensor([2.0, 2.0]),
        )
        self.assertEqual(traces, 1)
        self.assertEqual(compiled.cache_size(), 1)

        compiled({"x": torch.ones(2) + 1}, scale=2)
        self.assertEqual(traces, 1)
        self.assertEqual(compiled.cache_size(), 1)

        compiled({"x": torch.ones(3)}, scale=2)
        self.assertEqual(traces, 2)
        self.assertEqual(compiled.cache_size(), 2)

        compiled({"x": torch.ones(3)}, scale=3)
        self.assertEqual(traces, 3)
        self.assertEqual(compiled.cache_size(), 3)

    def test_jit_rejects_non_tensor_dynamic_args(self):
        with self.assertRaisesRegex(TypeError, "dynamic arguments must be Tensors"):
            nonstrict.jit(lambda x, label: x, backend=_identity_backend)(
                torch.ones(2), "label"
            )

    def test_jit_requires_hashable_static_args(self):
        with self.assertRaisesRegex(TypeError, "must be hashable"):
            nonstrict.jit(
                lambda x, scale: x, static_argnames="scale", backend=_identity_backend
            )(torch.ones(2), [])


if __name__ == "__main__":
    unittest.main()
