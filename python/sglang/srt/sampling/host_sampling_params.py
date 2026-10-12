"""Host mirror of per-request sampling params for host-staged batches.

Filter and merge run on the host copy, then whole tensors are re-published to
the sampler device, so the device never runs a gather or cat per batch-size pair.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple

import torch

HOST_SAMPLING_FIELDS = ("temperatures", "top_ps", "top_ks", "min_ps", "sampling_seed")

SamplingParamTensors = Dict[str, Optional[torch.Tensor]]


def _version(tensor: torch.Tensor) -> Optional[int]:
    try:
        return tensor._version
    except RuntimeError:
        # Inference tensors cannot report in-place changes.
        return None


@dataclasses.dataclass(frozen=True)
class HostSamplingParams:
    """Host copies plus the device tensors last published from them.

    Attributes:
      host: CPU tensors keyed by field name.
      published: Device tensor and its version at publish time, keyed by field name.
    """

    host: SamplingParamTensors
    published: Dict[str, Tuple[Optional[torch.Tensor], Optional[int]]]

    @classmethod
    def publish(cls, host: SamplingParamTensors, device: str) -> HostSamplingParams:
        assert all(v is None or v.device.type == "cpu" for v in host.values())
        published = {}
        for name, value in host.items():
            resident = None if value is None else value.to(device, non_blocking=True)
            published[name] = (
                resident,
                None if resident is None else _version(resident),
            )
        return cls(host=host, published=published)

    def device_values(self) -> SamplingParamTensors:
        return {name: tensor for name, (tensor, _) in self.published.items()}

    def matches(self, current: SamplingParamTensors) -> bool:
        """False once an override replaced or mutated a published tensor."""
        for name, (tensor, version) in self.published.items():
            if current[name] is not tensor:
                return False
            if tensor is not None and (version is None or _version(tensor) != version):
                return False
        return True

    def filtered(self, keep_indices: List[int]) -> SamplingParamTensors:
        return {
            name: None if value is None else value[keep_indices]
            for name, value in self.host.items()
        }

    def merged(self, other: HostSamplingParams) -> SamplingParamTensors:
        return {
            name: (
                torch.cat((value, other.host[name]))
                if value is not None and other.host[name] is not None
                else value
            )
            for name, value in self.host.items()
        }
