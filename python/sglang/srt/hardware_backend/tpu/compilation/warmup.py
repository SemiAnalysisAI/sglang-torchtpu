"""TPU compilation-mode warmup — prewarms the (T, B, mode) compile-cache product.

The TPU backend uses ``torch.compile(..., dynamic=False)``, so every
distinct ``(T_bucket, B_bucket, mode)`` tuple triggers its own ~4500-fx-
node, ~140-second backbone compile. This warmup fires synthetic requests
that exercise each ``(T_bucket, B_bucket)`` cell of the product before
the server starts serving real traffic, so first-request latency stays
fast.

For each ``(T_bucket, B_bucket)`` pair, sends ``B_bucket`` concurrent
requests with a prompt length of ``≈ T_bucket / B_bucket`` and
``max_new_tokens=2`` so each request runs:

  * One **prefill** forward → covers ``(T=T_bucket, B=B_bucket, EXTEND)``
    once the scheduler batches the concurrent requests.
  * One **decode** forward  → covers ``(T=B_bucket, B=B_bucket, DECODE)``
    (note: in DECODE, T == B since each running seq contributes one
    query token per step).

Without ``max_new_tokens >= 2`` the decode forward never runs (the
prefill samples the first token directly and the request returns),
leaving the DECODE entries uncovered — the first real request then
stalls on the decode compile.

Uses the active model's ShimRules to dynamically resolve bucketers, and
skips mathematically unreachable prefill shape cells (T_bucket < B_bucket)
to minimize startup overhead.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING

from sglang.srt.entrypoints.warmup import warmup
from sglang.srt.hardware_backend.tpu._logging import get_logger
from sglang.srt.managers.io_struct import GenerateReqInput

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager
    from sglang.srt.hardware_backend.tpu.compilation.bucketer import Bucketer

logger = get_logger(__name__)


class TpuWarmupManager:
    """Manages TPU compilation pre-warmup cell generation.

    Coordinates prompt and batch size bucketers, filters out unreachable prefill
    shapes, generates concurrent requests targeting each compile-time bucket
    coordinate (T, B), and drives them to completion to compile both EXTEND
    and DECODE paths on startup.
    """

    def __init__(self, tokenizer_manager: TokenizerManager, t_bucketer: Bucketer, b_bucketer: Bucketer):
        """Initializes the warmup manager with bucketers configured by the shim rules.

        Args:
            tokenizer_manager: TokenizerManager instance providing request endpoints.
            t_bucketer: Active Bucketer regulating the token dimension (T).
            b_bucketer: Active Bucketer regulating the batch dimension (B).
        """
        self.tokenizer_manager = tokenizer_manager
        self.t_bucketer = t_bucketer
        self.b_bucketer = b_bucketer
        self.max_running = b_bucketer.scheme.max_val

    async def run_warmup(self) -> None:
        """Runs the entire warmup loop for all valid, reachable compilation shape cells."""
        t_buckets = self.t_bucketer.buckets
        b_buckets = self.b_bucketer.buckets

        # Calculate total valid cells where T_bucket >= B_bucket
        valid_cells = [
            (t, b) for b in b_buckets for t in t_buckets
            if b <= self.max_running and t >= b
        ]
        n_cells = len(valid_cells)

        logger.info(
            "[compilation warmup] starting (T, B) prewarm: T-buckets=%s, "
            "B-buckets=%s, %d valid cells × (EXTEND + DECODE) = up to %d compile entries",
            t_buckets, b_buckets, n_cells, 2 * n_cells,
        )

        overall = time.perf_counter()
        for b in b_buckets:
            # The largest bucket exists only to absorb the scheduler's
            # over-subscription on top of ``max_running_requests``; we
            # cannot actually fire ``B > max_running`` concurrent requests
            # here because the scheduler will queue the excess and the
            # warmup batch shape collapses to ``max_running``. Leave that
            # bucket to lazy-compile at runtime if it ever fires.
            if b > self.max_running:
                logger.info(
                    "[compilation warmup] skipping over-sub bucket B=%d "
                    "(> max_running=%d); will lazy-compile if hit.",
                    b, self.max_running,
                )
                continue

            for t in t_buckets:
                if t < b:
                    # A batch of size B contains B active sequences. During the prefill stage,
                    # each sequence has at least 1 token. Thus, the total batch sequence length T
                    # is mathematically guaranteed to satisfy T >= B. Any cell where T_bucket < B_bucket
                    # is unreachable and can be safely skipped.
                    continue

                await self._warmup_cell(t, b)

        logger.info(
            "[compilation warmup] all %d cells (× 2 modes) warmed in %.1fs",
            n_cells, time.perf_counter() - overall,
        )

    async def _warmup_cell(self, t: int, b: int) -> None:
        """Warms up a single shape bucket cell (T, B) by launching concurrent requests.

        Args:
            t: Total batch token size (T_bucket).
            b: Active batch size (B_bucket).
        """
        cell_start = time.perf_counter()
        # Per-request prompt length so the *batched* prefill totals
        # ≈ T_bucket. Floor at 1 to keep the request well-formed.
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
            tasks.append(
                asyncio.create_task(self._drain(req))
            )
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.warning(
                "[compilation warmup] cell (T=%d, B=%d) failed: %s",
                t, b, e,
            )
            return

        logger.info(
            "[compilation warmup] cell (T=%d, B=%d) done in %.1fs "
            "(per_req_len=%d)",
            t, b, time.perf_counter() - cell_start, per_req_len,
        )

    async def _drain(self, req: GenerateReqInput) -> None:
        """Sends one request to SGLang engine and awaits its streaming completion.

        Args:
            req: GenerateReqInput request structure containing inputs and parameters.
        """
        async for _ in self.tokenizer_manager.generate_request(req, None):
            pass


@warmup("tpu_compilation_warmup")
async def tpu_compilation_warmup(
    disaggregation_mode: str, tokenizer_manager: TokenizerManager
):
    """TPU entrypoint invoked by the server's warmup hook manager.

    Validates if padding is enabled globally and if the active model architecture
    has custom compilation rules registered in static_shape_shim.py. Dynamically
    resolves and extracts the bucketer schemes to drive the pre-compilation warmup.
    """
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.hardware_backend.tpu import envs
    from sglang.srt.hardware_backend.tpu.compilation.static_shape_shim import (
        _MODEL_SHIM_RULES,
        ShimRules,
    )

    if not envs.TPU_INPUT_PADDING_ENABLED:
        logger.info(
            "[compilation warmup] TPU_INPUT_PADDING_ENABLED is False. "
            "Skipping static shape warmup."
        )
        return

    # Skip warmup if the active model has no registered static shape rules
    model_config = ModelConfig.from_server_args(tokenizer_manager.server_args)
    architectures = getattr(model_config.hf_config, "architectures", [])
    
    active_hook = None
    for arch in architectures:
        if arch in _MODEL_SHIM_RULES:
            active_hook = _MODEL_SHIM_RULES[arch]
            break

    if active_hook is None:
        logger.info(
            "[compilation warmup] Active model architectures %s have no "
            "registered static shape shim rules. Skipping static shape warmup.",
            architectures,
        )
        return

    # Execute the rules hook using a lightweight runner proxy to dynamically populate the bucketers
    rules = ShimRules()
    proxy = SimpleNamespace(
        model_config=SimpleNamespace(context_len=tokenizer_manager.context_len),
        max_running_requests=int(getattr(tokenizer_manager.server_args, "max_running_requests", None) or 8),
    )
    active_hook(rules, proxy)

    if rules.t_bucketer is None or rules.b_bucketer is None:
        logger.info(
            "[compilation warmup] Registered rules hook did not populate t_bucketer and b_bucketer. "
            "Skipping static shape warmup."
        )
        return

    # Run the warmup generically using the dynamically retrieved bucketers
    manager = TpuWarmupManager(tokenizer_manager, rules.t_bucketer, rules.b_bucketer)
    await manager.run_warmup()
