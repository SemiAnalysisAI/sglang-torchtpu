# SPDX-License-Identifier: Apache-2.0
"""matmul_replicated_weights_all_gather: best-performing ``all_gather(x, axis=0) @ W`` for a REPLICATED
weight on tpu7x (the emitter's compute-then-communicate reorder).

Public API:
  * ``matmul_all_gather`` -- size-dispatching entry point (the only public API); lives in
    ``api.py`` and routes to the Pallas kernels in ``_matmul_broadcast.py``
    (small-T broadcast) and ``_matmul_topo.py`` (mid/upper-T topology-aware hypercube gather).

It dispatches internally (by shape) to the small-T fused broadcast kernel, the mid/upper-T
topology-aware hypercube gather (any 16-core 2x2x2 slice), the large-T pure-XLA reorder
``lax.all_gather(x @ W)``, or plain naive -- none of which is a separate public entry point.
"""

from .api import matmul_all_gather

__all__ = [
    "matmul_all_gather",
]
