# SPDX-License-Identifier: Apache-2.0
"""Stage pipeline-local expert parameters on the host during checkpoint loading."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

import torch

_SHARDS_PER_EXPERT = {"w1": 2, "w3": 2, "w2": 1}
_UPLOAD_CHUNK_BYTES = 256 << 20
_MAX_STAGED = 32
_active_stager: ContextVar[ExpertParamStager | None] = ContextVar(
    "tpu_expert_stager", default=None
)


class ExpertWriteTracker:
    """Track distinct expert shards until every expert of a parameter is loaded."""

    def __init__(self):
        self._seen: dict[int, set[tuple[int, str]]] = {}

    def record(self, param, expert_id: int | None, shard_id: str | None) -> bool:
        shards = _SHARDS_PER_EXPERT.get(shard_id)
        if shards is None or expert_id is None:
            return False
        seen = self._seen.setdefault(id(param), set())
        seen.add((expert_id, shard_id))
        if len(seen) != param.shape[0] * shards:
            return False
        del self._seen[id(param)]
        return True


class ExpertParamStager:
    """Reuse host buffers and upload a parameter when its last expert arrives."""

    def __init__(self):
        self._staged: dict[int, tuple[torch.nn.Parameter, torch.Tensor]] = {}
        self._pool: dict[tuple, list[torch.Tensor]] = {}
        self._tracker = ExpertWriteTracker()
        self._written: set[int] = set()

    def staging(self, param) -> bool:
        return id(param) not in self._written

    def host_param(self, param: torch.nn.Parameter) -> torch.Tensor:
        entry = self._staged.get(id(param))
        if entry is None:
            if len(self._staged) >= _MAX_STAGED:
                oldest = next(iter(self._staged.values()))[0]
                self.flush(oldest)
            key = (tuple(param.shape), param.dtype)
            free = self._pool.get(key)
            buffer = (
                free.pop()
                if free
                else torch.empty(param.shape, dtype=param.dtype, device="cpu")
            )
            host = torch.Tensor._make_subclass(type(param), buffer, False)
            host.__dict__.update(param.__dict__)
            entry = (param, host)
            self._staged[id(param)] = entry
        return entry[1]

    def written(self, param, expert_id: int, shard_id: str) -> None:
        if self._tracker.record(param, expert_id, shard_id):
            self.flush(param)

    def flush(self, param) -> None:
        entry = self._staged.pop(id(param), None)
        if entry is None:
            return
        param, host = entry
        device = param.device
        per_expert = max(1, host[0].numel() * host.element_size())
        step = max(1, _UPLOAD_CHUNK_BYTES // per_expert)
        chunks = [
            host[start : start + step].to(device, copy=True)
            for start in range(0, host.shape[0], step)
        ]
        param.data = torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]
        if device.type == "tpu":
            from torch_tpu._internal.sync.sync import synchronize

            synchronize([param.data], wait=True)
        self._written.add(id(param))
        buffer = torch.Tensor._make_subclass(torch.Tensor, host.data, False)
        self._pool.setdefault((tuple(host.shape), host.dtype), []).append(buffer)

    def flush_all(self) -> None:
        for param, _ in list(self._staged.values()):
            self.flush(param)


def get_expert_stager() -> ExpertParamStager | None:
    return _active_stager.get()


@contextmanager
def expert_loading_context():
    """Upload remaining staged parameters before weight post-processing."""
    stager = ExpertParamStager()
    token = _active_stager.set(stager)
    try:
        yield stager
        stager.flush_all()
    finally:
        _active_stager.reset(token)
