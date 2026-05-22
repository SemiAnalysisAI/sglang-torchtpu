"""TPU graph runner using the nonstrict JIT helper."""

from __future__ import annotations

import bisect
import contextlib
import logging
from typing import TYPE_CHECKING, NamedTuple

import torch

from sglang.srt.hardware_backend.tpu import nonstrict
from sglang.srt.hardware_backend.tpu.attention_pallas import (
    _PAGE_SIZE,
    _bucket_kv_len,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import (
    log_info_on_rank0,
    require_gathered_buffer,
    require_mlp_sync,
    require_mlp_tp_gather,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class DecodeKey(NamedTuple):
    batch_size: int
    kv_len_bucket: int


def _tuple_from_cpu_values(values) -> tuple[int, ...] | None:
    if values is None:
        return None
    if isinstance(values, torch.Tensor):
        values = values.tolist()
    return tuple(int(v) for v in values)


def get_batch_sizes_to_capture(model_runner: ModelRunner) -> list[int]:
    server_args = model_runner.server_args
    max_bs = min(
        server_args.cuda_graph_max_bs or server_args.torch_compile_max_bs,
        model_runner.req_to_token_pool.size,
    )
    # Decode schedules exactly one token per request, so the static compile
    # shape is the request batch size.
    capture_bs = [
        bs
        for bs in (server_args.cuda_graph_bs or [])
        if bs > 0 and bs <= max_bs
    ]
    if max_bs > 0 and max_bs not in capture_bs:
        capture_bs.append(max_bs)
    capture_bs = list(sorted(set(capture_bs)))
    assert capture_bs, f"{capture_bs=}, {max_bs=}"
    return capture_bs


class TpuGraphRunner:
    """Runs eligible SGLang batches through nonstrict.jit(..., backend="tpu")."""

    def __init__(self, model_runner: ModelRunner):
        self.model_runner = model_runner
        self.device = model_runner.device
        self.decode = nonstrict.jit(
            self._run_decode,
            static_argnames=("decode_key",),
            backend="tpu",
            tracing_mode="real",
        )
        self.decode_buffers: dict[str, torch.Tensor] = {}
        self._compiled_decode_keys: set[DecodeKey] = set()
        self._logged_decode_replay_buckets: set[tuple[int, int, int]] = set()

        self.is_encoder_decoder = model_runner.model_config.is_encoder_decoder
        self.require_gathered_buffer = require_gathered_buffer(model_runner.server_args)
        self.require_mlp_tp_gather = require_mlp_tp_gather(model_runner.server_args)
        self.require_mlp_sync = require_mlp_sync(model_runner.server_args)
        self.enable_two_batch_overlap = (
            model_runner.server_args.enable_two_batch_overlap
        )
        self.dp_size = model_runner.server_args.dp_size
        self.pp_size = model_runner.server_args.pp_size
        self.disable_padding = model_runner.server_args.disable_cuda_graph_padding

        assert (
            not self.model_runner.server_args.enable_lora
        ), "TpuGraphRunner does not support LoRA yet."
        assert (
            not self.enable_two_batch_overlap
        ), "TpuGraphRunner does not support two batch overlap yet."
        assert (
            not self.require_mlp_tp_gather
        ), "TpuGraphRunner does not support MLP TP gather yet."
        assert (
            not self.require_mlp_sync
        ), "TpuGraphRunner does not support MLP sync yet."
        assert (
            not self.require_gathered_buffer
        ), "TpuGraphRunner does not support gathered buffer yet."
        assert (
            model_runner.spec_algorithm == SpeculativeAlgorithm.NONE
        ), "TpuGraphRunner does not support speculative inference yet."
        assert (
            not self.is_encoder_decoder
        ), "TpuGraphRunner does not support encoder-decoder models yet."
        assert self.dp_size == 1, "TpuGraphRunner does not support DP yet."
        assert self.pp_size == 1, "TpuGraphRunner does not support PP yet."

        self.capture_bs = get_batch_sizes_to_capture(model_runner)
        self.max_bs = max(self.capture_bs)
        self.decode_buffers = self._create_decode_buffers(self.max_bs)
        log_info_on_rank0(logger, f"Capture TPU nonstrict graph bs {self.capture_bs}")

        self.capture_forward_mode = ForwardMode.DECODE
        self.capture_hidden_mode = CaptureHiddenMode.NULL

        self.capture()

    def can_run(self, forward_batch: ForwardBatch) -> bool:
        capture_hidden_mode = forward_batch.capture_hidden_mode
        if capture_hidden_mode is None:
            capture_hidden_mode = CaptureHiddenMode.NULL
        return (
            # Normal decode enters through ModelRunner.forward_decode.
            forward_batch.forward_mode.is_decode()
            and self._select_decode_batch_size(forward_batch.batch_size) is not None
            and self._decode_key(forward_batch) is not None
            and capture_hidden_mode == self.capture_hidden_mode
        )

    def capture(self) -> None:
        for bs in self.capture_bs:
            self.warmup_decode_batch_size(bs)

    def _select_decode_batch_size(self, raw_bs: int) -> int | None:
        if self.disable_padding:
            return raw_bs if raw_bs in self.capture_bs else None
        index = bisect.bisect_left(self.capture_bs, raw_bs)
        if index == len(self.capture_bs):
            return None
        return self.capture_bs[index]

    def _create_decode_buffers(self, max_bs: int) -> dict[str, torch.Tensor]:
        with torch.device(self.device):
            return {
                "input_ids": torch.zeros((max_bs,), dtype=torch.int64),
                "positions": torch.zeros((max_bs,), dtype=torch.int64),
                "req_pool_indices": torch.zeros((max_bs,), dtype=torch.int64),
                "seq_lens": torch.ones((max_bs,), dtype=torch.int64),
                "out_cache_loc": torch.zeros((max_bs,), dtype=torch.int64),
            }

    def _prepare_decode_inputs(
        self, forward_batch: ForwardBatch, bs: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_bs = forward_batch.batch_size
        if raw_bs == bs:
            return (
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.out_cache_loc,
            )

        buffers = self.decode_buffers
        input_ids = buffers["input_ids"][:bs]
        positions = buffers["positions"][:bs]
        req_pool_indices = buffers["req_pool_indices"][:bs]
        seq_lens = buffers["seq_lens"][:bs]
        out_cache_loc = buffers["out_cache_loc"][:bs]

        input_ids[:raw_bs].copy_(forward_batch.input_ids)
        positions[:raw_bs].copy_(forward_batch.positions)
        req_pool_indices[:raw_bs].copy_(forward_batch.req_pool_indices)
        seq_lens[:raw_bs].copy_(forward_batch.seq_lens)
        out_cache_loc[:raw_bs].copy_(forward_batch.out_cache_loc)

        # Match SGLang CUDA graph replay: padded requests use reserved request
        # and KV slot 0, so their dummy work does not mutate real request state.
        input_ids[raw_bs:bs].zero_()
        positions[raw_bs:bs].zero_()
        req_pool_indices[raw_bs:bs].zero_()
        seq_lens[raw_bs:bs].fill_(1)
        out_cache_loc[raw_bs:bs].zero_()
        return input_ids, positions, req_pool_indices, seq_lens, out_cache_loc

    def warmup_decode_batch_size(self, bs: int) -> None:
        """Compile one fixed-shape decode batch.

        Unlike CUDA graph replay, TorchTPU compiled functions do not require
        stable input tensor addresses. The dummy tensors below are only used to
        produce the first trace for this static shape. Real decode steps pass
        their live scheduler tensors directly to the shared compiled callable.
        """
        with torch.device(self.device):
            # Decode has one input token per request, so each dummy tensor has
            # shape `(bs,)`. Prefill/extend would need token-count buckets.
            input_ids = torch.zeros((bs,), dtype=torch.int64)
            req_pool_indices = torch.arange(bs, dtype=torch.int64)
            seq_lens = torch.ones((bs,), dtype=torch.int64)
            out_cache_loc = torch.arange(bs, dtype=torch.int64)
            positions = torch.zeros((bs,), dtype=torch.int64)

        with torch.no_grad():
            # First call traces and compiles the bs-specific executable through
            # nonstrict.jit/backend="tpu". The collective remap is only needed
            # while make_fx is tracing tensor-parallel collectives.
            decode_key = DecodeKey(batch_size=bs, kv_len_bucket=_PAGE_SIZE)
            with nonstrict.traceable_collective_remap():
                self.decode(
                    decode_key,
                    input_ids,
                    positions,
                    req_pool_indices,
                    seq_lens,
                    out_cache_loc,
                    self.model_runner.req_to_token_pool.req_to_token,
                    *self._kv_buffers(),
                )
            self._compiled_decode_keys.add(decode_key)

    def _decode_key(self, forward_batch: ForwardBatch) -> DecodeKey | None:
        bs = self._select_decode_batch_size(forward_batch.batch_size)
        seq_lens = _tuple_from_cpu_values(forward_batch.seq_lens_cpu)
        if bs is None or seq_lens is None:
            return None
        max_len = max(seq_lens) if seq_lens else _PAGE_SIZE
        kv_len_bucket = max(_PAGE_SIZE, _bucket_kv_len(max_len))
        if kv_len_bucket > self.model_runner.req_to_token_pool.req_to_token.shape[1]:
            return None
        return DecodeKey(batch_size=bs, kv_len_bucket=kv_len_bucket)

    def _kv_buffers(self) -> tuple[torch.Tensor, ...]:
        token_to_kv_pool = self.model_runner.token_to_kv_pool
        # KV cache tensors are dynamic graph inputs. Passing them explicitly
        # prevents the compiled function from closing over stale buffer objects.
        return tuple(token_to_kv_pool.k_buffer) + tuple(token_to_kv_pool.v_buffer)

    def _run_decode(
        self,
        decode_key: DecodeKey,
        input_ids,
        positions,
        req_pool_indices,
        seq_lens,
        out_cache_loc,
        req_to_token,
        *kv_buffers,
    ) -> torch.Tensor:
        bs = decode_key.batch_size
        model = self.model_runner.model
        attn_backend = self.model_runner.attn_backend
        req_to_token_pool = self.model_runner.req_to_token_pool
        token_to_kv_pool = self.model_runner.token_to_kv_pool
        num_k_buffers = len(token_to_kv_pool.k_buffer)
        old_req_to_token = req_to_token_pool.req_to_token
        old_k_buffers = token_to_kv_pool.k_buffer
        old_v_buffers = token_to_kv_pool.v_buffer
        old_decode_kv_len_bucket = getattr(
            attn_backend, "decode_kv_len_bucket", None
        )
        try:
            # SGLang stores these tensors behind pool objects that model
            # layers read from forward_batch. Temporarily point the pools at
            # the dynamic tensor inputs so the tracer sees them as graph
            # arguments instead of Python-closed globals.
            req_to_token_pool.req_to_token = req_to_token
            token_to_kv_pool.k_buffer = list(kv_buffers[:num_k_buffers])
            token_to_kv_pool.v_buffer = list(kv_buffers[num_k_buffers:])
            if hasattr(attn_backend, "decode_kv_len_bucket"):
                attn_backend.decode_kv_len_bucket = decode_key.kv_len_bucket
            forward_batch = ForwardBatch(
                forward_mode=self.capture_forward_mode,
                batch_size=bs,
                input_ids=input_ids,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool=token_to_kv_pool,
                attn_backend=self.model_runner.attn_backend,
                out_cache_loc=out_cache_loc,
                seq_lens_sum=bs,
                return_logprob=False,
                positions=positions,
                spec_algorithm=self.model_runner.spec_algorithm,
                spec_info=None,
                capture_hidden_mode=self.capture_hidden_mode,
                num_token_non_padded=input_ids.new_full((1,), bs),
                global_forward_mode=self.capture_forward_mode,
            )
            output = model.forward(input_ids, positions, forward_batch)
            # Return only logits so replay matches SGLang's decode contract.
            return output.next_token_logits
        finally:
            req_to_token_pool.req_to_token = old_req_to_token
            token_to_kv_pool.k_buffer = old_k_buffers
            token_to_kv_pool.v_buffer = old_v_buffers
            if hasattr(attn_backend, "decode_kv_len_bucket"):
                attn_backend.decode_kv_len_bucket = old_decode_kv_len_bucket

    def replay(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool,
        pp_proxy_tensors=None,
    ) -> LogitsProcessorOutput:
        del skip_attn_backend_init, pp_proxy_tensors
        raw_bs = forward_batch.batch_size
        decode_key = self._decode_key(forward_batch)
        assert decode_key is not None
        bs = decode_key.batch_size
        replay_bucket = (raw_bs, bs, decode_key.kv_len_bucket)
        if replay_bucket not in self._logged_decode_replay_buckets:
            log_info_on_rank0(
                logger,
                "Replay TPU nonstrict graph "
                f"bs={raw_bs} bucket_bs={bs} kv_bucket={decode_key.kv_len_bucket}",
            )
            self._logged_decode_replay_buckets.add(replay_bucket)
        (
            input_ids,
            positions,
            req_pool_indices,
            seq_lens,
            out_cache_loc,
        ) = self._prepare_decode_inputs(forward_batch, bs)
        # TorchTPU compile is shape-specialized, not address-specialized like
        # CUDA graphs. Exact bucket hits pass live SGLang tensors; padded
        # bucket hits pass reusable static buffers.
        if decode_key in self._compiled_decode_keys:
            trace_context = contextlib.nullcontext()
        else:
            trace_context = nonstrict.traceable_collective_remap()
        with trace_context:
            logits = self.decode(
                decode_key,
                input_ids,
                positions,
                req_pool_indices,
                seq_lens,
                out_cache_loc,
                self.model_runner.req_to_token_pool.req_to_token,
                *self._kv_buffers(),
            )
        self._compiled_decode_keys.add(decode_key)
        return LogitsProcessorOutput(
            next_token_logits=logits[:raw_bs],
            hidden_states=None,
        )
