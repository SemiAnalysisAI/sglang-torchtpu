# SPDX-License-Identifier: Apache-2.0
"""TPU sampling-batch metadata adaptations.

Why: TorchTPU specializes device-side ``cat`` by both input shapes, so decode
     batch merges otherwise compile for every arrival/completion shape pair.
How: REPLACE the three small ``adjusted_*`` seams on SamplingBatchInfo. Keep an
     immutable host mirror, update it on filter, and upload merged tensors.
Upstream PR or issue: tracked by the sglang-torchtpu PR carrying this hook.
Remove when: core sampling batches can merge immutable request metadata on the
             host without a TPU plugin hook.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.runtime_context import get_exec
from sglang.srt.utils.common import is_pin_memory_available

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo


_HOST_PARAMS_ATTR = "_tpu_basic_sampling_params"


@dataclasses.dataclass(frozen=True)
class BasicSamplingParams:
    temperatures: tuple[float, ...]
    top_ps: tuple[float, ...]
    top_ks: tuple[int, ...]
    min_ps: tuple[float, ...]
    sampling_seed: Optional[tuple[int, ...]]

    @classmethod
    def from_batch(cls, batch: ScheduleBatch) -> BasicSamplingParams:
        deterministic = get_exec().deterministic.enable_deterministic_inference
        reqs = batch.reqs
        return cls(
            temperatures=tuple(r.sampling_params.temperature for r in reqs),
            top_ps=tuple(r.sampling_params.top_p for r in reqs),
            top_ks=tuple(r.sampling_params.top_k for r in reqs),
            min_ps=tuple(r.sampling_params.min_p for r in reqs),
            sampling_seed=(
                tuple(
                    (
                        r.sampling_params.sampling_seed
                        if r.sampling_params.sampling_seed is not None
                        else 42
                    )
                    for r in reqs
                )
                if deterministic
                else None
            ),
        )

    def filter(self, keep_indices: list[int]) -> BasicSamplingParams:
        def select(values):
            return tuple(values[i] for i in keep_indices)

        return BasicSamplingParams(
            temperatures=select(self.temperatures),
            top_ps=select(self.top_ps),
            top_ks=select(self.top_ks),
            min_ps=select(self.min_ps),
            sampling_seed=(
                select(self.sampling_seed) if self.sampling_seed is not None else None
            ),
        )

    def merge(self, other: BasicSamplingParams) -> BasicSamplingParams:
        sampling_seed = None
        if self.sampling_seed is not None and other.sampling_seed is not None:
            sampling_seed = self.sampling_seed + other.sampling_seed
        return BasicSamplingParams(
            temperatures=self.temperatures + other.temperatures,
            top_ps=self.top_ps + other.top_ps,
            top_ks=self.top_ks + other.top_ks,
            min_ps=self.min_ps + other.min_ps,
            sampling_seed=sampling_seed,
        )

    def to_device(
        self, device: str
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        pin = is_pin_memory_available(device)
        temperatures = (
            torch.tensor(self.temperatures, dtype=torch.float, pin_memory=pin)
            .view(-1, 1)
            .to(device, non_blocking=True)
        )
        top_ps = torch.tensor(self.top_ps, dtype=torch.float, pin_memory=pin).to(
            device, non_blocking=True
        )
        top_ks = torch.tensor(self.top_ks, dtype=torch.int32, pin_memory=pin).to(
            device, non_blocking=True
        )
        min_ps = torch.tensor(self.min_ps, dtype=torch.float, pin_memory=pin).to(
            device, non_blocking=True
        )
        sampling_seed = (
            torch.tensor(self.sampling_seed, dtype=torch.int64, pin_memory=pin).to(
                device, non_blocking=True
            )
            if self.sampling_seed is not None
            else None
        )
        return temperatures, top_ps, top_ks, min_ps, sampling_seed


def initialize_basic_sampling_params(
    sampling_info: SamplingBatchInfo, batch: ScheduleBatch, _vocab_size: int
) -> None:
    setattr(sampling_info, _HOST_PARAMS_ATTR, BasicSamplingParams.from_batch(batch))


def filter_basic_sampling_params(
    sampling_info: SamplingBatchInfo,
    keep_indices: list[int],
    _keep_indices_device: torch.Tensor,
) -> None:
    params = getattr(sampling_info, _HOST_PARAMS_ATTR, None)
    if params is not None:
        setattr(sampling_info, _HOST_PARAMS_ATTR, params.filter(keep_indices))


def merge_basic_sampling_params(
    sampling_info: SamplingBatchInfo, other: SamplingBatchInfo
) -> bool:
    params = getattr(sampling_info, _HOST_PARAMS_ATTR, None)
    other_params = getattr(other, _HOST_PARAMS_ATTR, None)
    compatible_seeds = (
        params is not None
        and other_params is not None
        and ((params.sampling_seed is None) == (other_params.sampling_seed is None))
    )
    if not compatible_seeds:
        setattr(sampling_info, _HOST_PARAMS_ATTR, None)
        return False

    params = params.merge(other_params)
    (
        sampling_info.temperatures,
        sampling_info.top_ps,
        sampling_info.top_ks,
        sampling_info.min_ps,
        sampling_info.sampling_seed,
    ) = params.to_device(sampling_info.device)
    setattr(sampling_info, _HOST_PARAMS_ATTR, params)
    return True


def copy_basic_sampling_params(
    result: SamplingBatchInfo, sampling_info: SamplingBatchInfo
) -> SamplingBatchInfo:
    params = getattr(sampling_info, _HOST_PARAMS_ATTR, None)
    if params is not None:
        setattr(result, _HOST_PARAMS_ATTR, params)
    return result
