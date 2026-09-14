"""Bounded value-checked host-to-device cache for read-only attention metadata."""

import torch


class HostMetadataCache:
    def __init__(self):
        self._entries = {}

    def get(self, name: str, value: torch.Tensor, device) -> torch.Tensor:
        assert value.device.type == "cpu"
        cached = self._entries.get(name)
        key = str(device)
        if cached is not None:
            previous, device_key, resident = cached
            if (
                device_key == key
                and previous.dtype == value.dtype
                and previous.shape == value.shape
                and torch.equal(previous, value)
            ):
                return resident
        # Keep our own CPU snapshot: callers may subsequently mutate their input.
        snapshot = value.clone()
        resident = snapshot.to(device, non_blocking=True)
        self._entries[name] = snapshot, key, resident
        return resident
