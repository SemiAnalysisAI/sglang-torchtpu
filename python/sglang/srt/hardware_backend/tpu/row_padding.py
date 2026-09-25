"""Pad per-request tensors to a compiled batch width in one TPU program."""

from __future__ import annotations

import functools
from collections.abc import Hashable, Sequence

import torch
import torch.nn.functional as F
from torch._dynamo import eval_frame

# (owner, attribute name, target rows, fill)
PadTarget = tuple[object, str, int, float]


def _pad_rows_eager(
    tensors: Sequence[torch.Tensor], rows: tuple[int, ...], fills: tuple[float, ...]
) -> list[torch.Tensor]:
    return [
        F.pad(t, [0, 0] * (t.dim() - 1) + [0, r - t.shape[0]], value=fill)
        for t, r, fill in zip(tensors, rows, fills)
    ]


def _pad_rows_key(
    tensors: Sequence[torch.Tensor], rows: tuple[int, ...], fills: tuple[float, ...]
) -> Hashable:
    return tuple((tuple(t.shape), t.dtype) for t in tensors), rows, fills


@functools.cache
def _compiled_pad_rows():
    from sglang.srt.hardware_backend.tpu.compilation.direct_compile import (
        direct_compile,
    )

    # Eagerly, each pad is an allocation plus a cat: two programs and about
    # 90 us of host dispatch per tensor.
    return direct_compile(_pad_rows_eager, specialization_key=_pad_rows_key)


def _compilation_frozen() -> bool:
    # set_stance has no public getter; direct_compile reads the same field.
    return eval_frame._stance.stance == "fail_on_recompile"


class RowPadder:
    """Pads per-request tensors; on TPU, each call is one compiled program.

    A program is specialized on the logical batch size, and a new one costs
    about 60-80 ms to trace and compile. So programs are compiled only when a
    tensor layout is first seen before compilation is frozen, which normally
    happens in the server's warmup request: that call compiles the layout for
    every ``(logical, physical)`` pair in ``warm_batch_sizes``. Any other
    shape is padded eagerly, one allocation and cat per tensor.
    """

    def __init__(self, warm_batch_sizes: Sequence[tuple[int, int]] = ()) -> None:
        self._warm_batch_sizes = tuple(warm_batch_sizes)
        self._warmed_layouts: set[Hashable] = set()

    def pad(
        self,
        *,
        tensors: Sequence[torch.Tensor | None],
        rows: Sequence[int],
        fills: Sequence[float] | None = None,
        batch_size: tuple[int, int] | None = None,
    ) -> list[torch.Tensor | None]:
        """Extend each tensor's first dim to its target row count.

        New rows hold the tensor's fill (default 0). None, non-tensor values
        and tensors that already have enough rows are returned unchanged.
        ``batch_size`` is this call's ``(logical, physical)`` request count;
        it enables warming the layout.
        """
        if fills is None:
            fills = (0,) * len(tensors)
        out = list(tensors)
        tpu, other = [], []
        for i, t in enumerate(tensors):
            if not isinstance(t, torch.Tensor) or t.shape[0] >= rows[i]:
                continue
            # A zero-row input would be a zero-size executable argument.
            (tpu if t.device.type == "tpu" and t.shape[0] > 0 else other).append(i)
        if tpu:
            args = dict(
                tensors=[tensors[i] for i in tpu],
                rows=tuple(rows[i] for i in tpu),
                fills=tuple(fills[i] for i in tpu),
            )
            if batch_size is not None:
                self._warm(batch_size=batch_size, **args)
            compiled = _compiled_pad_rows()
            if _pad_rows_key(**args) in compiled.cache:
                for i, value in zip(tpu, compiled(**args)):
                    out[i] = value
            else:
                other += tpu
        if other:
            padded = _pad_rows_eager(
                tensors=[tensors[i] for i in other],
                rows=tuple(rows[i] for i in other),
                fills=tuple(fills[i] for i in other),
            )
            for i, value in zip(other, padded):
                out[i] = value
        return out

    def pad_attrs(
        self,
        targets: Sequence[PadTarget],
        *,
        batch_size: tuple[int, int] | None = None,
    ) -> None:
        """Rebind each ``owner.name`` tensor to its padded copy, as one group."""
        values = [getattr(owner, name) for owner, name, _, _ in targets]
        padded = self.pad(
            tensors=values,
            rows=[rows for _, _, rows, _ in targets],
            fills=[fill for _, _, _, fill in targets],
            batch_size=batch_size,
        )
        for (owner, name, _, _), value, old in zip(targets, padded, values):
            if value is not old:
                setattr(owner, name, value)

    def _warm(
        self,
        *,
        tensors: list[torch.Tensor],
        rows: tuple[int, ...],
        fills: tuple[float, ...],
        batch_size: tuple[int, int],
    ) -> None:
        real_b, pad_b = batch_size
        if real_b <= 0 or _compilation_frozen():
            return
        # Rows per request of each input and output; warming needs both whole.
        in_per_req = [t.shape[0] // real_b for t in tensors]
        out_per_req = [r // pad_b for r in rows]
        if any(t.shape[0] != n * real_b for t, n in zip(tensors, in_per_req)) or any(
            r != n * pad_b for r, n in zip(rows, out_per_req)
        ):
            return
        layout = (
            tuple(
                (t.dtype, tuple(t.shape[1:]), n, m)
                for t, n, m in zip(tensors, in_per_req, out_per_req)
            ),
            fills,
        )
        if layout in self._warmed_layouts:
            return
        self._warmed_layouts.add(layout)
        compiled = _compiled_pad_rows()
        for warm_real_b, warm_pad_b in self._warm_batch_sizes:
            compiled(
                # Only the shapes matter; the padded result is discarded.
                tensors=[
                    t.new_empty((n * warm_real_b, *t.shape[1:]))
                    for t, n in zip(tensors, in_per_req)
                ],
                rows=tuple(m * warm_pad_b for m in out_per_req),
                fills=fills,
            )
