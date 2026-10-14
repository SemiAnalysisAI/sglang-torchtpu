# SPDX-License-Identifier: Apache-2.0
"""Server-argument contract of the Kimi-K3 decode megakernel.

The megakernel keeps every decoding request's KDA and MLA state in its own
row-major buffers (``KimiK3MegakernelRunner.states``). It reads SGLang's
paged ``MLATokenToKVPool`` / ``MambaPool`` once, at the prefill->decode
handoff, and never writes decoded tokens back. Anything that reads those pools
after the handoff would see the prefill-time state, so every such reader is
refused or turned off here, before the model loads:

* a decode batch larger than the megakernel's rows would fall back to SGLang's
  decode over stale caches -> ``--max-running-requests`` <= max rows;
* the radix cache would publish finished requests' decoded tokens with stale
  KV/Mamba state for later prefix hits -> radix cache off;
* positions past the state buffers' context are out of range -> the context
  must fit the megakernel context;
* speculative decoding, PP, DP attention, P/D disaggregation and the
  hierarchical cache read or move pool state the megakernel does not keep.

This module imports nothing heavy so ``apply_server_args_defaults`` can call it
in the orchestrator before any TPU runtime starts.
"""

from __future__ import annotations

import logging

from .slots import ROW_BUCKETS

logger = logging.getLogger(__name__)

_TP_SIZE = 32
_EP_SIZE = 8
_MIN_CONTEXT = 384
_CONTEXT_ALIGN = 128
_RESOLVER = "tpu.kimi_k3_megakernel"


def is_megakernel_enabled() -> bool:
    """Return True when the Kimi-K3 TPU decode megakernel is enabled."""
    from sglang_tpu.envs import envs

    return bool(envs.SGLANG_TPU_K3_MEGAKERNEL.get())


def megakernel_max_rows() -> int:
    """Decode rows the megakernel state holds, validated against its tiles."""
    from sglang_tpu.envs import envs

    max_rows = int(envs.SGLANG_TPU_K3_MEGAKERNEL_MAX_ROWS.get())
    if not 1 <= max_rows <= ROW_BUCKETS[-1]:
        raise ValueError(
            f"SGLANG_TPU_K3_MEGAKERNEL_MAX_ROWS must be in [1, {ROW_BUCKETS[-1]}], "
            f"got {max_rows}: the megakernel compiles row tiles {ROW_BUCKETS}."
        )
    return max_rows


def megakernel_context(context_length: int) -> int:
    """State-buffer tokens per row: ``context_length`` rounded up to a cache tile."""
    aligned = -(-int(context_length) // _CONTEXT_ALIGN) * _CONTEXT_ALIGN
    return max(_MIN_CONTEXT, aligned)


def validate_megakernel_server_args(server_args) -> None:
    """Refuse or resolve server args the megakernel's state ownership forbids.

    No-op unless ``SGLANG_TPU_K3_MEGAKERNEL`` is set. Unset limits are declared
    (``max_running_requests``, ``context_length``); the radix cache is turned
    off as a declared resolution; explicit values that conflict raise.
    """
    if not is_megakernel_enabled():
        return

    from sglang.srt.arg_groups.overrides import declare_resolution, resolving_view
    from sglang_tpu.envs import envs

    cfg = resolving_view(server_args)
    prefix = "SGLANG_TPU_K3_MEGAKERNEL=1"

    if cfg.tp_size != _TP_SIZE or cfg.ep_size != _EP_SIZE:
        raise ValueError(
            f"{prefix} requires --tp-size {_TP_SIZE} --ep-size {_EP_SIZE} "
            f"(got --tp-size {cfg.tp_size} --ep-size {cfg.ep_size}): the kernel's "
            "collectives and expert layout are built for that topology."
        )
    unsupported = {
        "--speculative-algorithm": cfg.speculative_algorithm is not None,
        "--pipeline-parallel-size > 1": cfg.pp_size > 1,
        "--dp-size > 1": cfg.dp_size > 1,
        "--enable-dp-attention": bool(cfg.enable_dp_attention),
        "--disaggregation-mode": cfg.disaggregation_mode != "null",
        "--enable-hierarchical-cache": bool(cfg.enable_hierarchical_cache),
    }
    refused = [flag for flag, on in unsupported.items() if on]
    if refused:
        raise ValueError(
            f"{prefix} does not support {', '.join(refused)}: decode state lives "
            "only in the megakernel's buffers, which these features do not read."
        )

    max_rows = megakernel_max_rows()
    if cfg.max_running_requests is None:
        declare_resolution(server_args, _RESOLVER, max_running_requests=max_rows)
    elif cfg.max_running_requests > max_rows:
        raise ValueError(
            f"{prefix} requires --max-running-requests <= {max_rows} "
            f"(SGLANG_TPU_K3_MEGAKERNEL_MAX_ROWS), got {cfg.max_running_requests}: "
            "a larger decode batch would fall back to SGLang's decode, whose KV "
            "and Mamba caches the megakernel does not update."
        )

    if not cfg.disable_radix_cache:
        logger.warning(
            "%s: disabling the radix cache; finished requests' decoded tokens "
            "have no valid KV/Mamba state in SGLang's pools to share.",
            prefix,
        )
        declare_resolution(server_args, _RESOLVER, disable_radix_cache=True)

    mk_context = envs.SGLANG_TPU_K3_MEGAKERNEL_CONTEXT.get()
    if cfg.context_length is None:
        if mk_context is None:
            raise ValueError(
                f"{prefix} requires --context-length or "
                "SGLANG_TPU_K3_MEGAKERNEL_CONTEXT: the megakernel allocates "
                "decode state for the full context of every row."
            )
        declare_resolution(server_args, _RESOLVER, context_length=int(mk_context))
    elif mk_context is not None and cfg.context_length > megakernel_context(mk_context):
        raise ValueError(
            f"{prefix}: --context-length {cfg.context_length} exceeds the "
            f"megakernel context {megakernel_context(mk_context)} "
            f"(SGLANG_TPU_K3_MEGAKERNEL_CONTEXT={mk_context}); positions past it "
            "have no decode state."
        )
