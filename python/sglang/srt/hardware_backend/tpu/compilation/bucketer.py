"""TPU shape bucketing utilities for dynamic input compilation."""

from __future__ import annotations

import abc
import bisect
from typing import Any

from sglang.srt.hardware_backend.tpu._logging import get_logger

logger = get_logger(__name__)


class BucketScheme(abc.ABC):
    """Base class defining how a progression of bucket sizes is generated."""

    def __init__(self, buckets: list[int]):
        if not buckets:
            raise ValueError("Bucket list cannot be empty.")
        self._buckets = sorted(list(set(buckets)))
        self.min_val = self._buckets[0]
        self.max_val = self._buckets[-1]

    @property
    def buckets(self) -> list[int]:
        """Sorted list of bucket sizes including min_val and max_val."""
        return self._buckets

    def next_bucket(self, n: int) -> int:
        """Finds the smallest compiled bucket size >= n in O(log N) time."""
        if n > self.max_val:
            raise ValueError(
                f"Requested size {n} exceeds the maximum bucket size {self.max_val}."
            )
        idx = bisect.bisect_left(self.buckets, n)
        return self.buckets[idx]

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, BucketScheme):
            return False
        return self._buckets == other._buckets


class NoneBucketScheme(BucketScheme):
    """Capped at max_size only (no bucketing)."""

    def __init__(self, min_val: int, max_val: int):
        super().__init__([max_val])

    def __repr__(self) -> str:
        return "none"


class FixedBucketScheme(BucketScheme):
    """Linear progression with a fixed interval step."""

    def __init__(self, min_val: int, max_val: int, step: int):
        self.step = step
        buckets = set()
        n = min_val
        while n < max_val:
            buckets.add(n)
            n += step
        buckets.add(max_val)
        super().__init__(list(buckets))

    def __repr__(self) -> str:
        return f"fixed_{self.step}"


class PowerBucketScheme(BucketScheme):
    """Geometric progression with a power multiplier."""

    def __init__(
        self,
        min_val: int,
        max_val: int,
        multiplier: int = 8,
    ):
        self.multiplier = multiplier
        buckets = set()
        buckets.add(min_val)
        buckets.add(max_val)

        n = max(1, min_val)
        while n < max_val:
            buckets.add(n)
            n *= multiplier
        super().__init__(list(buckets))

    def __repr__(self) -> str:
        return f"power_{self.multiplier}"


class CustomBucketScheme(BucketScheme):
    """A custom list of pre-defined bucket boundaries."""

    def __init__(self, custom_buckets: list[int]):
        super().__init__(custom_buckets)

    def __repr__(self) -> str:
        return f"custom={self.buckets}"


class Bucketer:
    """Maps dynamic runtime dimension shapes to discrete compile-time buckets."""

    def __init__(
        self,
        name: str,
        scheme: BucketScheme,
    ):
        self.name = name
        self._scheme = scheme

    @property
    def max_val(self) -> int:
        return self._scheme.max_val

    @property
    def min_val(self) -> int:
        return self._scheme.min_val

    @property
    def buckets(self) -> list[int]:
        return self._scheme.buckets

    def next_bucket(self, n: int) -> int:
        """Rounds runtime shape size up to next compiled bucket in O(log N) time."""
        return self._scheme.next_bucket(n)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"scheme={self._scheme}, "
            f"min_val={self.min_val}, "
            f"max_val={self.max_val}, "
            f"buckets={self.buckets}"
            f")"
        )
