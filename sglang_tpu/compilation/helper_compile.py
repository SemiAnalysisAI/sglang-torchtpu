# SPDX-License-Identifier: Apache-2.0
"""TPU helper compilation: no dynamic-shape helpers, static helpers via compiled_op."""

from __future__ import annotations

from sglang.srt.compilation.helper_compile import HelperCompilePolicy


class TpuHelperCompilePolicy(HelperCompilePolicy):
    def compile_helper(self, fn, *, dynamic, backend, options):
        # torch_tpu has no dynamic-shape compile; inside a compiled model graph
        # the helper is traced inline anyway.
        return fn

    def compile_static_helper(self, fn, *, warmup, model, uses_rng):
        from sglang.srt.compilation.compiled_op import compiled_op

        return compiled_op(warmup=warmup, model=model, uses_rng=uses_rng)(fn)


TPU_HELPER_COMPILE_POLICY = TpuHelperCompilePolicy()
