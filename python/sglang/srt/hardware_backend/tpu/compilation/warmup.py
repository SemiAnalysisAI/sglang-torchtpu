"""TPU compilation warmup to precompile with static shapes."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING

from sglang.srt.entrypoints.warmup import warmup
from sglang.srt.hardware_backend.tpu._logging import get_logger
from sglang.srt.managers.io_struct import GenerateReqInput

if TYPE_CHECKING:
    from sglang.srt.hardware_backend.tpu.compilation.bucketer import Bucketer
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = get_logger(__name__)


class TpuWarmupManager:
    """Trigger warmup to precompile for all static shapes."""

    def __init__(
        self,
        tokenizer_manager: TokenizerManager,
        t_bucketer: Bucketer,
        b_bucketer: Bucketer,
    ):
        """Initializes the warmup manager with configured shape bucketers."""
        self.tokenizer_manager = tokenizer_manager
        self.t_bucketer = t_bucketer
        self.b_bucketer = b_bucketer

    async def run_warmup(self) -> None:
        """Warms up all valid shape bucket combinations."""
        t_buckets = self.t_bucketer.buckets
        b_buckets = self.b_bucketer.buckets

        # Identify valid shape cells where T_bucket >= B_bucket
        valid_cells = [
            (t, b)
            for b in b_buckets
            for t in t_buckets
            if t >= b
        ]
        n_cells = len(valid_cells)

        logger.info(
            "[compilation warmup] starting (T, B) prewarm: T-buckets=%s, "
            "B-buckets=%s, %d valid cells × (EXTEND + DECODE) = up to "
            "%d compile entries",
            t_buckets,
            b_buckets,
            n_cells,
            2 * n_cells,
        )

        overall = time.perf_counter()
        for b in b_buckets:
            for t in t_buckets:
                if t < b:
                    # Mathematically unreachable: prefill token count (T)
                    # must be at least equal to the batch size (B).
                    continue

                await self._warmup_cell(t, b)

        logger.info(
            "[compilation warmup] all %d cells (× 2 modes) warmed in %.1fs",
            n_cells,
            time.perf_counter() - overall,
        )

    async def _warmup_cell(self, t: int, b: int) -> None:
        """Runs concurrent requests to warm up the (T, B) shape."""
        cell_start = time.perf_counter()
        # Distribute total token budget T across B requests.
        per_req_len = max(1, t // b)
        tasks = []
        for _ in range(b):
            req = GenerateReqInput(
                input_ids=[0] * per_req_len,
                sampling_params={
                    "max_new_tokens": 2,
                    "temperature": 0,
                },
            )
            tasks.append(asyncio.create_task(self._drain(req)))
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.warning(
                "[compilation warmup] cell (T=%d, B=%d) failed: %s",
                t,
                b,
                e,
            )
            return

        logger.info(
            "[compilation warmup] cell (T=%d, B=%d) done in %.1fs "
            "(per_req_len=%d)",
            t,
            b,
            time.perf_counter() - cell_start,
            per_req_len,
        )

    async def _drain(self, req: GenerateReqInput) -> None:
        """Awaits streaming completion for a single request."""
        async for _ in self.tokenizer_manager.generate_request(req, None):
            pass


@warmup("tpu_compilation_warmup")
async def tpu_compilation_warmup(
    disaggregation_mode: str, tokenizer_manager: TokenizerManager
):
    """TPU server warmup hook entrypoint to pre-warm compile caches."""
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.hardware_backend.tpu import envs
    from sglang.srt.hardware_backend.tpu.compilation.static_shape_shim import (
        _MODEL_SHIM_RULES,
        ShimRules,
    )

    if not envs.TPU_PRECOMPILATION_ENABLED:
        logger.info(
            "[compilation warmup] TPU_PRECOMPILATION_ENABLED is False. "
            "Skipping static shape warmup."
        )
        return

    # Verify model architecture has registered static shape rules
    model_config = ModelConfig.from_server_args(tokenizer_manager.server_args)
    architectures = getattr(model_config.hf_config, "architectures", [])

    active_hook = None
    for arch in architectures:
        if arch in _MODEL_SHIM_RULES:
            active_hook = _MODEL_SHIM_RULES[arch]
            break

    if active_hook is None:
        logger.warning(
            "[compilation warmup] Active model architectures %s have no "
            "registered static shape shim rules. Skipping static shape "
            "warmup.",
            architectures,
        )
        return

    # Execute rules hook using a runner proxy to resolve bucketer schemes
    rules = ShimRules()
    proxy = SimpleNamespace(
        model_config=SimpleNamespace(context_len=tokenizer_manager.context_len),
        max_running_requests=int(
            getattr(tokenizer_manager.server_args, "max_running_requests", None)
            or 8
        ),
    )
    active_hook(rules, proxy)

    if rules.t_bucketer is None or rules.b_bucketer is None:
        logger.warning(
            "[compilation warmup] Registered rules hook did not populate "
            "t_bucketer and b_bucketer. Skipping static shape warmup."
        )
        return

    # Run warmup using the resolved bucketers
    manager = TpuWarmupManager(
        tokenizer_manager, rules.t_bucketer, rules.b_bucketer
    )
    await manager.run_warmup()
