"""Split a request's cached prefix into full-KV hits, SWA misses and SWA restores.

Only meaningful when SWA is the cache's only match validator besides full KV;
otherwise a cut in the match cannot be attributed to SWA, and every count is zero.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec

if TYPE_CHECKING:
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, MatchResult


class SwaPrefixMatch(msgspec.Struct, frozen=True):
    """Match-time prefix lengths, in tokens."""

    full_kv_hit_length: int = 0
    full_kv_device_hit_length: int = 0
    # Prefix every validator accepted on device.
    device_hit_length: int = 0
    # Deepest prefix every validator accepted, including host-restorable windows and
    # backed-up FULL; a load-back declined past it is a capacity loss, not an SWA miss.
    reusable_length: int = 0


class SwaPrefixAdmission(msgspec.Struct, frozen=True):
    """Per-request token counts at admission."""

    full_kv_device_hit: int = 0
    full_kv_host_hit: int = 0
    swa_miss: int = 0
    swa_restored: int = 0


def swa_prefix_match(
    *, tree_cache: BasePrefixCache, match_result: MatchResult
) -> SwaPrefixMatch:
    if not tree_cache.swa_is_only_match_cut():
        return SwaPrefixMatch()
    device_hit = len(match_result.device_indices)
    reusable = max(
        match_result.best_match_length, device_hit + match_result.host_hit_length
    )
    # Producers that skip the tree walk (streaming sessions) leave the full-KV
    # lengths at 0; whatever the match reuses has full KV too.
    return SwaPrefixMatch(
        full_kv_hit_length=max(match_result.full_kv_hit_length, reusable),
        full_kv_device_hit_length=max(
            match_result.full_kv_device_hit_length, device_hit
        ),
        device_hit_length=device_hit,
        reusable_length=reusable,
    )


def swa_prefix_admission(
    *, match: SwaPrefixMatch, prefix_len: int
) -> SwaPrefixAdmission:
    """`prefix_len` is the admitted prefix after load-back. Tokens whose FULL is on
    device but which lie past the device hit lack only their SWA window; admitting
    them means load-back restored it (from host, or storage in buffer mode)."""
    full_kv = match.full_kv_hit_length
    full_kv_device = match.full_kv_device_hit_length
    return SwaPrefixAdmission(
        full_kv_device_hit=full_kv_device,
        full_kv_host_hit=full_kv - full_kv_device,
        swa_miss=max(0, full_kv - max(prefix_len, match.reusable_length)),
        swa_restored=max(0, min(prefix_len, full_kv_device) - match.device_hit_length),
    )
