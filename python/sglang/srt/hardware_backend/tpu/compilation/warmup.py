"""TPU compilation mode warmup."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from sglang.srt.entrypoints.warmup import warmup
from sglang.srt.hardware_backend.tpu._logging import get_logger
from sglang.srt.hardware_backend.tpu.compilation.bucketizer import (
    PromptBucketizer,
)
from sglang.srt.managers.io_struct import GenerateReqInput

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = get_logger(__name__)


@warmup("tpu_compilation_warmup")
async def tpu_compilation_warmup(
    disaggregation_mode: str, tokenizer_manager: "TokenizerManager"
):
    bucketizer = PromptBucketizer.initialize(tokenizer_manager.context_len)
    logger.info(
        "[compilation warmup] starting %d-bucket prewarm "
        "(context_length=%d): %s",
        len(bucketizer.buckets), bucketizer.context_length, bucketizer.buckets,
    )
    start = time.perf_counter()
    for token_bucket in bucketizer.buckets:
        bucket_start = time.perf_counter()
        prompt_len = bucketizer.prompt_len_for_bucket(token_bucket)
        req = GenerateReqInput(
            input_ids=[0] * prompt_len,
            sampling_params={"max_new_tokens": 1, "temperature": 0},
        )
        try:
            async for _ in tokenizer_manager.generate_request(req, None):
                pass
        except Exception as e:
            logger.warning(
                "[compilation warmup] bucket %d failed: %s", token_bucket, e,
            )
            continue
        logger.info(
            "[compilation warmup] bucket %d done in %.1fs",
            token_bucket, time.perf_counter() - bucket_start,
        )
    logger.info(
        "[compilation warmup] %d buckets all warmed up in %.1fs",
        len(bucketizer.buckets), time.perf_counter() - start,
    )
    logger.info(
        "[compilation warmup] finished %d-bucket prewarm "
        "(context_length=%d): %s",
        len(bucketizer.buckets), bucketizer.context_length, bucketizer.buckets,
    )
