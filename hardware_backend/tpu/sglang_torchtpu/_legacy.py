# SPDX-License-Identifier: Apache-2.0
"""Stable import path for TPU code that still lives in the fork's ``python/`` tree.

New code should import TPU symbols through this module rather than reaching into
``sglang.srt.hardware_backend.tpu`` directly, so that the eventual relocation
(wave 4) is a change to this file instead of a change to every caller.

The direction reverses at relocation: this module becomes the real implementation
and the in-tree path becomes the shim. Nothing here is part of the public API of
the plugin -- it is transitional scaffolding with a known end date.
"""

from __future__ import annotations

import importlib
from types import ModuleType

_IN_TREE_ROOT = "sglang.srt.hardware_backend.tpu"


def legacy_module(name: str) -> ModuleType:
    """Import ``<in-tree TPU package>.<name>``.

    Raises a message that names the migration state rather than a bare
    ModuleNotFoundError, because the common cause is this package being
    installed next to *stock* sglang, which ships no TPU code at all.
    """
    try:
        return importlib.import_module(f"{_IN_TREE_ROOT}.{name}")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"{_IN_TREE_ROOT}.{name} is not importable. During the plugin "
            "migration this package depends on the fork's python/ tree; it does "
            "not work against stock sglang yet. Install the fork with "
            "`pip install -e python/`."
        ) from exc
