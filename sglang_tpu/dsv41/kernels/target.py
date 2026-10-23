# SPDX-License-Identifier: Apache-2.0
"""Distinguish CPU reference execution from TorchTPU's CPU-hosted export."""


def is_tpu_kernel_target():
    from jax.experimental.pallas import tpu as pltpu

    try:
        return pltpu.get_tpu_info().generation is not None
    except (ValueError, NotImplementedError, AssertionError):
        return False
