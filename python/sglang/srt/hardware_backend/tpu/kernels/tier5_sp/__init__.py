# SPDX-License-Identifier: Apache-2.0
"""Pallas primitives used by the Tier-5 narrow sequence-parallel path."""

from sglang.srt.hardware_backend.tpu.kernels.tier5_sp.reduce_scatter import (
    reduce_scatter,
)

__all__ = ["reduce_scatter"]
