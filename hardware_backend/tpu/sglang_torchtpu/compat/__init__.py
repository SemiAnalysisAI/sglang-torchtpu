# SPDX-License-Identifier: Apache-2.0
"""Version-keyed shims across sglang releases.

Each entry resolves one API difference so a single plugin release can span a
small range of sglang versions. Keep every shim short, name the upstream change
it tracks, and delete it once the supported range no longer includes the old
spelling.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

__all__ = ["first_available"]


def first_available(obj: Any, *names: str) -> Optional[Callable]:
    """Return the first bound attribute of ``obj`` among ``names``, else None.

    For use where upstream renamed a method and the plugin must work either side
    of the rename.

    **No current users.** ``get_dsa_kv_pool_cls`` / ``get_nsa_kv_pool_cls`` looked
    like such a rename and is not one: both spellings are raising stubs, because
    TPU has no DSA pool at all, so ``TpuOOTPlatform`` refuses outright instead of
    resolving between them. The helper is kept because the first real rename is a
    question of when, not whether -- delete it if a wave passes without one.
    """
    for name in names:
        attr = getattr(obj, name, None)
        if attr is not None:
            return attr
    return None
