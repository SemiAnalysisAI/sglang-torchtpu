# SPDX-License-Identifier: Apache-2.0
"""Temporary, coarse R1 accuracy controls. Unset means production behavior.

Read before compilation. These controls are for isolated R1 experiments only.
They are not public serving options or defaults for other models.
"""

import os


def _choice(name: str, allowed: tuple[str, ...]) -> str:
    value = os.environ.get(name, allowed[0])
    if value not in allowed:
        raise ValueError(f"{name}={value!r}; expected one of {allowed}")
    return value


WEIGHTS = _choice("SGLANG_R1_TEST_WEIGHTS", ("production", "mla_bf16", "native"))
MATH = _choice("SGLANG_R1_TEST_MATH", ("production", "fp32"))
KV = _choice("SGLANG_R1_TEST_KV", ("production", "bf16"))
NATIVE_WEIGHTS = WEIGHTS == "native"
MLA_BF16 = WEIGHTS in ("mla_bf16", "native")
PRECISE_MATH = MATH == "fp32"
MOE_SCALES = _choice("SGLANG_R1_TEST_MOE_SCALES", ("auto", "production", "fp32"))
FP32_MOE_SCALES = MOE_SCALES == "fp32" or (MOE_SCALES == "auto" and PRECISE_MATH)
ACTIVATIONS = _choice("SGLANG_R1_TEST_ACTIVATIONS", ("auto", "production", "block128"))
# Preserve earlier experiments when unset, but allow activation grouping to be
# held fixed while testing weights without the FP32 arithmetic bundle.
DENSE_ACT128 = ACTIVATIONS == "block128" or (ACTIVATIONS == "auto" and PRECISE_MATH)
ACTIVATION_BLOCK = 128 if DENSE_ACT128 else 512


def settings() -> dict:
    return dict(
        weights=WEIGHTS,
        math=MATH,
        moe_scales=MOE_SCALES,
        fp32_moe_scales=FP32_MOE_SCALES,
        activations=ACTIVATIONS,
        kv=KV,
        mla_bf16=MLA_BF16,
        native_weights=NATIVE_WEIGHTS,
        moe_activation_block=ACTIVATION_BLOCK,
    )
