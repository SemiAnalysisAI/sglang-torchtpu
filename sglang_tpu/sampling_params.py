# SPDX-License-Identifier: Apache-2.0
"""Retain host-born sampling parameters for scheduler filter/merge operations.

Public tensors stay on the sampler device. Only batch changes upload new values;
forward copies share those immutable tensors without a per-token transfer.
"""

import torch

_FIELDS = ("temperatures", "top_ps", "top_ks", "min_ps", "sampling_seed")


def _version(tensor):
    try:
        return tensor._version
    except RuntimeError:
        # Inference tensors cannot detect in-place changes; retain the base path.
        return None


def _publish(info, host):
    device_values = {}
    for name, value in host.items():
        if value is not None:
            value = value.to(info.device, non_blocking=True)
        setattr(info, name, value)
        device_values[name] = (value, None if value is None else _version(value))
    info._tpu_sampling_params = (host, device_values)


def _host_params(info):
    state = getattr(info, "_tpu_sampling_params", None)
    if state is None:
        return None
    host, published = state
    for name, (tensor, version) in published.items():
        # Overrides may replace or mutate public parameters. Never reuse a stale
        # host mirror in that case; preserve the normal tensor-based behavior.
        if getattr(info, name) is not tensor or (
            tensor is not None and (version is None or _version(tensor) != version)
        ):
            return None
    return host


def initialize_sampling_params(info):
    host = {name: getattr(info, name) for name in _FIELDS}
    assert all(value is None or value.device.type == "cpu" for value in host.values())
    _publish(info, host)


def filter_sampling_params(info, keep_indices):
    host = _host_params(info)
    if host is None:
        return False
    _publish(
        info,
        {
            name: None if value is None else value[keep_indices]
            for name, value in host.items()
        },
    )
    return True


def merge_sampling_params(info, other):
    left, right = _host_params(info), _host_params(other)
    if left is None or right is None:
        return False
    _publish(
        info,
        {
            name: (
                torch.cat((value, right[name]))
                if value is not None and right[name] is not None
                else value
            )
            for name, value in left.items()
        },
    )
    return True
