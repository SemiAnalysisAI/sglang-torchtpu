# SPDX-License-Identifier: Apache-2.0
"""Startup contract of ``SGLANG_TPU_ENABLE_K3_MEGAKERNEL``.

The megakernel keeps each decoding request's KDA and MLA state in its own
per-row arrays. It reads SGLang's paged KV and Mamba pools once, when the
request enters decode, and never writes decoded tokens back. Every feature that
would read those pools after that point, run decode outside the megakernel, or
lay the model out differently is refused here, before the weights load.

Imports nothing heavy: TpuSRTPlatform.apply_server_args_defaults calls it in
the orchestrator, before any TPU runtime starts.
"""

from __future__ import annotations

from sglang.srt.arg_groups.overrides import declare_resolution, resolving_view

# The compiled batch tiles; the state arrays hold max(rows) rows.
ROW_BUCKETS = (1, 2, 4, 8)
# The kernel's collectives and expert layout are built for exactly this mesh.
TP_SIZE = 32
EP_SIZE = 8
# The MLA caches are written in 128-token tiles; the kernel needs 3 of them.
CONTEXT_ALIGN = 128
MIN_CONTEXT = 384
ARCHITECTURES = ("KimiK3ForConditionalGeneration", "KimiK3LinearForCausalLM")

_ATTENTION_BACKEND = "pallas_tpu_mla"
_FLAG = "SGLANG_TPU_ENABLE_K3_MEGAKERNEL"
_SOURCE = "tpu.kimi_k3_megakernel"


def megakernel_context(context_length: int) -> int:
    """Tokens of state per row: ``context_length`` rounded up to a cache tile."""
    aligned = -(-context_length // CONTEXT_ALIGN) * CONTEXT_ALIGN
    return max(MIN_CONTEXT, aligned)


def row_buckets(max_rows: int) -> list[int]:
    """The tiles a megakernel with ``max_rows`` state rows compiles."""
    return [bucket for bucket in ROW_BUCKETS if bucket <= max_rows]


def check_ranks_agree(local: tuple, gathered: list[tuple]) -> None:
    """Refuse a launch whose TP ranks would build different megakernels.

    Every rank reads its own environment and server args, and the program the
    tuned recipe, the row count or the context selects carries the TP
    collectives, so ranks that disagree would wait on each other's
    collectives forever or decode garbage.
    """
    if any(other != local for other in gathered):
        by_rank = ", ".join(f"rank {r}: {g}" for r, g in enumerate(gathered))
        raise RuntimeError(
            f"{_FLAG}: TP ranks disagree on the megakernel's (rows, context, "
            f"tuned recipe, layers, vocab); set the same server args and "
            f"SGLANG_TPU_OPT_K3_MEGAKERNEL_TUNED_RECIPE on every host. {by_rank}"
        )


def _model_architectures(cfg) -> list[str]:
    from sglang.srt.utils.hf_transformers.config import get_config

    hf_config = get_config(
        cfg.model_path, trust_remote_code=cfg.trust_remote_code, revision=cfg.revision
    )
    return list(hf_config.architectures or ())


def _refused_flags(cfg) -> list[str]:
    """Set flags the megakernel cannot serve correctly, by their CLI names."""
    checks = {
        # A row would be decoded once by SGLang over its stale pools.
        "--enable-mixed-chunk": cfg.enable_mixed_chunk,
        "--speculative-algorithm": cfg.speculative_algorithm is not None,
        # These shard or move the KV / Mamba pools the handoff reads.
        "--decode-context-parallel-size > 1": cfg.dcp_size > 1,
        "--attention-context-parallel-size > 1": cfg.attn_cp_size > 1,
        "--pipeline-parallel-size > 1": cfg.pp_size > 1,
        "--dp-size > 1": cfg.dp_size > 1,
        # --enable-dp-attention has resolved into attn_dp_size by now.
        "--attention-data-parallel-size > 1": cfg.attn_dp_size > 1,
        "--disaggregation-mode": cfg.disaggregation_mode != "null",
        "--enable-hierarchical-cache": cfg.enable_hierarchical_cache,
        "--enable-pdmux": cfg.enable_pdmux,
        # The handoff reads the latent KV as bf16 values.
        f"--kv-cache-dtype {cfg.kv_cache_dtype}": cfg.kv_cache_dtype
        not in ("auto", "bf16", "bfloat16"),
        # Decode would silently ignore the adapters / return nothing extra.
        "--enable-lora / --lora-paths": bool(cfg.enable_lora) or bool(cfg.lora_paths),
        "--enable-return-hidden-states": cfg.enable_return_hidden_states,
        "--enable-return-routed-experts": cfg.enable_return_routed_experts,
        # The tuned recipe reduces in a batch-dependent order.
        "--enable-deterministic-inference": cfg.enable_deterministic_inference,
        # The kernel's expert layout is 112 experts per rank, never rebalanced.
        "--enable-eplb": cfg.enable_eplb,
        "--ep-num-redundant-experts": cfg.ep_num_redundant_experts != 0,
        # The dense weight families are read from the checkpoint's safetensors.
        f"--load-format {cfg.load_format}": cfg.load_format
        not in ("auto", "safetensors"),
        # Decode is dispatched from the dynamo-compiled forward only.
        "--tpu-compile-tracer direct": cfg.tpu_compile_tracer != "dynamo",
        # The handoff reads the paged latent pool this backend allocates.
        f"--attention-backend {cfg.attention_backend}": cfg.attention_backend
        not in (None, _ATTENTION_BACKEND),
    }
    return [flag for flag, refused in checks.items() if refused]


def _resolve_rows(server_args, cfg) -> None:
    rows = cfg.max_running_requests
    if rows is None:
        rows = ROW_BUCKETS[-1]
        declare_resolution(server_args, _SOURCE, max_running_requests=rows)
    elif rows not in ROW_BUCKETS:
        raise ValueError(
            f"{_FLAG} requires --max-running-requests in {ROW_BUCKETS} (the "
            f"megakernel's state rows and compiled tiles), got {rows}."
        )
    buckets = row_buckets(rows)
    if cfg.tpu_compile_bs is None:
        # Pad every decode batch to a tile the megakernel compiled.
        declare_resolution(server_args, _SOURCE, tpu_compile_bs=buckets)
    elif sorted(cfg.tpu_compile_bs) != buckets:
        raise ValueError(
            f"{_FLAG} with --max-running-requests {rows} requires --tpu-compile-bs "
            f"{' '.join(map(str, buckets))} (the megakernel's tiles), got "
            f"{cfg.tpu_compile_bs}."
        )


def validate_server_args(server_args) -> None:
    """Refuse, or resolve, the server args ``SGLANG_TPU_ENABLE_K3_MEGAKERNEL`` forbids."""
    cfg = resolving_view(server_args)
    architectures = _model_architectures(cfg)
    if not set(architectures) & set(ARCHITECTURES):
        raise ValueError(
            f"{_FLAG} serves Kimi-K3 ({', '.join(ARCHITECTURES)}) only; "
            f"--model-path {cfg.model_path} is {architectures}."
        )
    if cfg.tp_size != TP_SIZE or cfg.ep_size != EP_SIZE:
        raise ValueError(
            f"{_FLAG} requires --tp-size {TP_SIZE} --ep-size {EP_SIZE}, got "
            f"--tp-size {cfg.tp_size} --ep-size {cfg.ep_size}: the kernel's "
            "collectives and expert layout are built for that mesh."
        )
    refused = _refused_flags(cfg)
    if refused:
        raise ValueError(
            f"{_FLAG} does not support {', '.join(refused)}: decode state lives in "
            "the megakernel's own arrays and decode runs only through it."
        )
    if not cfg.enable_torch_compile:
        raise ValueError(
            f"{_FLAG} requires --enable-torch-compile: the megakernel decode step "
            "is dispatched from the compiled forward."
        )
    if not cfg.disable_radix_cache:
        raise ValueError(
            f"{_FLAG} requires --disable-radix-cache: decoded tokens have no KV or "
            "Mamba state in SGLang's pools to share."
        )
    if cfg.context_length is None:
        raise ValueError(
            f"{_FLAG} requires --context-length: the megakernel holds state for the "
            "full context of every row (55 KB per token per rank)."
        )
    _resolve_rows(server_args, cfg)
    if cfg.attention_backend is None:
        declare_resolution(server_args, _SOURCE, attention_backend=_ATTENTION_BACKEND)
    if cfg.kv_cache_dtype == "auto":
        # Before the platform's fp8 default for the Pallas RPA backends.
        declare_resolution(server_args, _SOURCE, kv_cache_dtype="bfloat16")
