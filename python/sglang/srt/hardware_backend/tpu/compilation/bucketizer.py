"""TPU sequence-length bucketing for compilation"""

from __future__ import annotations

from enum import Enum

from sglang.srt.hardware_backend.tpu._logging import get_logger

logger = get_logger(__name__)


class BucketScheme(Enum):
    """Bucketizer for prompt length.

    NONE:        Single bucket at context_length
    POWER_TWO:   e.g. (16, 32, 64, ...)
    POWER_FOUR:  e.g. (16, 64, 256, ...)
    POWER_EIGHT: e.g. (16, 128, 1024, ...)
    """

    NONE = "none"
    POWER_TWO = "power_two"
    POWER_FOUR = "power_four"
    POWER_EIGHT = "power_eight"

    @property
    def step(self) -> int | None:
        return {
            BucketScheme.NONE: None,
            BucketScheme.POWER_TWO: 2,
            BucketScheme.POWER_FOUR: 4,
            BucketScheme.POWER_EIGHT: 8,
        }[self]


class PromptBucketizer:
    """Prompt bucketizer for TPU compilation."""

    # Process-singleton instance.
    _instance: "PromptBucketizer | None" = None

    # Smallest bucket.
    MIN_PROMPT_LENGTH_BUCKET: int = 64

    def __init__(
        self,
        context_length: int,
        min_bucket: int = MIN_PROMPT_LENGTH_BUCKET,
        scheme: BucketScheme = BucketScheme.POWER_EIGHT,
    ):
        self.context_length = context_length
        self.min_bucket = min_bucket
        self.scheme = scheme
        self.buckets = self._compute()

    def _compute(self) -> list[int]:
        """Computes the list of bucket boundaries based on the scheme."""
        if self.scheme is BucketScheme.NONE:
            return [self.context_length]

        buckets: list[int] = []
        n = self.min_bucket
        step = self.scheme.step
        while n < self.context_length:
            buckets.append(n)
            n *= step

        if not buckets:
            buckets = [min(self.context_length, self.min_bucket)]
        return buckets

    def next_compile_bucket(self, n: int) -> int:
        """Rounds an input length up to the next compiled bucket boundary."""
        for b in self.buckets:
            if b >= n:
                return b
        return n

    def prompt_len_for_bucket(self, bucket: int) -> int:
        return min(bucket, self.context_length - 1)

    @classmethod
    def initialize(
        cls,
        context_length: int,
        scheme: BucketScheme = BucketScheme.POWER_EIGHT,
    ) -> "PromptBucketizer":
        """Construct or return the bucketizer singleton.

        Idempotent for matching `(context_length, scheme)`. Raises if
        already initialized with a different config.
        """
        if cls._instance is not None:
            if (
                cls._instance.context_length == context_length
                and cls._instance.scheme is scheme
            ):
                return cls._instance
            raise RuntimeError(
                "PromptBucketizer already initialized with a different "
                f"config: existing=(ctx={cls._instance.context_length}, "
                f"scheme={cls._instance.scheme.value}) vs requested="
                f"(ctx={context_length}, scheme={scheme.value})."
            )
        cls._instance = cls(context_length, scheme=scheme)
        logger.info(
            "PromptBucketizer initialized: context_length=%d, scheme=%s, "
            "buckets=%s",
            cls._instance.context_length, cls._instance.scheme.value,
            cls._instance.buckets,
        )
        return cls._instance

    @classmethod
    def get_instance(cls) -> "PromptBucketizer":
        """Returns the active singleton. Raises if uninitialized."""
        if cls._instance is None:
            raise RuntimeError(
                "PromptBucketizer not initialized. Call "
                "PromptBucketizer.initialize(context_length) first."
            )
        return cls._instance
