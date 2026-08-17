# SPDX-License-Identifier: Apache-2.0
"""This forward's real (unpadded) token count, as a device tensor.

The compiled-graph token bucketer pads a batch up to the next bucket, and the
padded rows carry the embedding of token id 0 through the whole decoder. Layers
that want to skip work for them need the real count, but they run *inside* the
compiled region while the count is only known outside it.

Passing it as a Python int would bake it into a Dynamo guard, and prefill chunk
lengths vary per request (3328, 3072, 2816, ...), so every new length would
recompile -- which this repo turns into a hard error after warmup. Handing out a
freshly allocated tensor each step changes object identity and recompiles for the
same reason. So keep ONE buffer per device and update it in place: identity stays
stable across steps while the value flows at runtime.

Published by the TPU compile policy before it enters the compiled decoder; read
by ``hardware_backend.tpu.fp8_moe``. Kept in its own module so the runtime op
does not import the compilation package.
"""

from __future__ import annotations

from typing import Optional

import torch

# Identity-stable device buffer holding the count. Never reallocated for a given
# device, so the compiled graph sees the same tensor object on every step.
_BUFFER: Optional[torch.Tensor] = None

# Whether consumers should emit their masking ops at all. Must be constant for a
# given compiled graph, so callers drive it from forward mode (extend and decode
# compile separately).
_PUBLISHED: bool = False


def publish_real_token_count(
    *, device: torch.device, real_q: int, enabled: bool
) -> None:
    """Publish this forward's real token count. Call OUTSIDE the compiled region.

    ``enabled=False`` makes :func:`real_token_count` return ``None`` so consumers
    trace no masking ops at all, rather than tracing a mask that is a no-op.
    """
    global _BUFFER, _PUBLISHED
    _PUBLISHED = enabled
    if not enabled:
        return
    if _BUFFER is None or _BUFFER.device != device:
        _BUFFER = torch.tensor([real_q], dtype=torch.int32, device=device)
        return
    _BUFFER.fill_(real_q)


def real_token_count() -> Optional[torch.Tensor]:
    """The count published for this forward, or ``None`` if there is none.

    ``None`` means "no padding information available", and callers must then
    treat every row as real -- never as "no row is real", which would silently
    drop the batch.
    """
    if not _PUBLISHED:
        return None
    return _BUFFER


def reset_real_token_count() -> None:
    """Drop the published count and its buffer (teardown / tests)."""
    global _BUFFER, _PUBLISHED
    _BUFFER = None
    _PUBLISHED = False
