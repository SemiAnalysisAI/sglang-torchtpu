# SPDX-License-Identifier: Apache-2.0
"""Intra-chip tensor-parallel matmul helpers for 2-way pooler/unpooler weights.

Public API:
  * ``matmul_intra_tp_all_gather`` -- the entry point (in ``api.py``); dispatches to the routed-gather
    combine (``_matmul_tp_intra_topo.py``, the fastest -- the topology-aware bridge-last routed gather,
    schedule in ``_routing.py``, routing by physical coord so any 16-core 2x2x2 order works) or the
    flat-broadcast fallback (``_matmul_tp_intra_bcast.py``).
  * ``matmul_intra_tp_output_gather`` -- the SP pooler helper: keep tokens sharded, compute with
    half-W, and gather only the 2 output-feature halves over the on-chip pair. Large TP16 inputs use
    a bridge-overlapped Pallas kernel; other shapes use the XLA fallback.

Each core reads only ``[N/2, K]`` of W; a pre-matmul core-bridge token exchange assembles the chip's
``[T/8, K]`` block so FLOPs stay non-redundant. ``core_axis`` (size 2) must be the physical on-chip
pair.
"""

from .api import (
    matmul_intra_tp_all_gather,
    matmul_intra_tp_all_gather_post_moe_epilogue,
    matmul_intra_tp_output_gather,
)

__all__ = [
    "matmul_intra_tp_all_gather",
    "matmul_intra_tp_all_gather_post_moe_epilogue",
    "matmul_intra_tp_output_gather",
]
