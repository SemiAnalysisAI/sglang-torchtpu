# SPDX-License-Identifier: Apache-2.0
"""Runtime runner and TorchTPU custom-op registration for the Kimi-K3 decode megakernel."""

from __future__ import annotations

import inspect
import logging
import os
from collections.abc import Callable, Sequence
from threading import Lock
from types import SimpleNamespace
from typing import Any

import torch

from .handoff import (
    apply_row_move,
    build_padded_token_locs_and_mask,
    extract_mamba_buffers,
    extract_mla_kv_buffers,
    handoff_request_to_megakernel_row,
)
from .program import make_decode_program, state_shapes
from .slots import ROW_BUCKETS, RowMove, SlotTable, row_bucket
from .weights import (
    _own_storage,
    load_dense_families,
    stack_sglang_moe_experts_in_place,
)

logger = logging.getLogger(__name__)

_STATE_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

_DECODE_OPS: dict[tuple[int, int, str, int, tuple[str, ...]], Callable] = {}
_DECODE_OPS_LOCK = Lock()


def is_megakernel_enabled() -> bool:
    """Return True when the Kimi-K3 TPU decode megakernel is enabled."""
    return os.environ.get("SGLANG_TPU_K3_MEGAKERNEL") == "1"


def allocate_decode_states(
    rows: int,
    context: int,
    layers: int = 93,
    device: torch.device | str = "tpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate the 4 megakernel-owned decode state base tensors on ``device``."""
    specs = state_shapes(rows, context, layers=layers)
    states = tuple(
        _own_storage(torch.zeros(shape, dtype=_STATE_DTYPES[dtype_name], device=device))
        for shape, dtype_name in specs
    )
    return states  # type: ignore[return-value]


def register_megakernel_decode_op(
    mesh: Any,
    weight_names: Sequence[str],
    *,
    batch_size: int,
    layers: int = 93,
    recipe: str = "tuned",
    vocab_size: int = 163840,
) -> Callable:
    """Register and return the TorchTPU custom op for one decode batch bucket."""
    weight_names_tuple = tuple(weight_names)
    cache_key = (batch_size, layers, recipe, vocab_size, weight_names_tuple)
    op = _DECODE_OPS.get(cache_key)
    if op is not None:
        return op

    with _DECODE_OPS_LOCK:
        op = _DECODE_OPS.get(cache_key)
        if op is not None:
            return op

        import jax
        from jax.sharding import NamedSharding
        from torch_tpu._internal.pallas import pallas as pallas_impl

        from sglang_tpu.pallas_export import tpu_info_for_cpu_export
        from sglang_tpu.pallas_jax_op import _CpuExportJaxCallable

        with tpu_info_for_cpu_export():
            decode_step, input_specs, output_specs, state_operands, aliases = (
                make_decode_program(
                    mesh,
                    weight_names_tuple,
                    batch_size=batch_size,
                    layers=layers,
                    recipe=recipe,
                )
            )
            output_shardings = tuple(
                NamedSharding(mesh, spec) for spec in output_specs
            )
            jit_fn = jax.jit(
                decode_step,
                donate_argnums=state_operands,
                keep_unused=True,
                out_shardings=output_shardings,
            )
            op_name = (
                f"sglang_tpu_pallas::kimi_k3_megakernel_decode_b{batch_size}_l{layers}_{recipe}"
            )
            trace_key = pallas_impl._get_kernel_invocation_key(
                op_name,
                [],
                {
                    "batch_size": batch_size,
                    "layers": layers,
                    "recipe": recipe,
                    "donate_argnums": state_operands,
                    "output_partition_specs": tuple(map(str, output_specs)),
                },
            )
            callable_op = _CpuExportJaxCallable(
                name=f"jax_{op_name}",
                jit_fn=jit_fn,
                trace_key=trace_key,
                mesh=mesh,
                input_partition_specs=input_specs,
                output_partition_specs=output_specs,
                static_argnums=(),
                donate_argnums=list(state_operands),
                input_output_aliases=aliases,
            )
            callable_op.sglang_output_shardings = output_shardings

        params = [
            inspect.Parameter(
                f"operand{index}",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=torch.Tensor,
            )
            for index in range(len(input_specs))
        ]
        sig = inspect.Signature(params, return_annotation=torch.Tensor)
        mutates_args = tuple(f"operand{index}" for index in state_operands)

        def step(*operands: torch.Tensor) -> torch.Tensor:
            out = callable_op(*operands)
            return out[0]

        def fake_step(*operands: torch.Tensor) -> torch.Tensor:
            tokens = operands[-2]
            return tokens.new_empty((batch_size, vocab_size), dtype=torch.float32)

        step.__signature__ = sig  # type: ignore[attr-defined]
        fake_step.__signature__ = sig  # type: ignore[attr-defined]

        custom_op = torch.library.custom_op(
            op_name,
            step,
            mutates_args=mutates_args,
            device_types=["tpu"],
        )
        custom_op.register_fake(fake_step)
        _DECODE_OPS[cache_key] = custom_op
        return custom_op


class KimiK3MegakernelRunner:
    """Manages megakernel weights, decode state slots, handoff, and decode dispatch."""

    def __init__(
        self,
        *,
        weights: dict[str, torch.Tensor],
        weight_names: Sequence[str] | None = None,
        states: Sequence[torch.Tensor] | None = None,
        decode_ops: dict[int, Callable] | None = None,
        max_rows: int = 8,
        context: int = 4096,
        layers: int = 93,
        vocab_size: int = 163840,
        recipe: str = "tuned",
        tp_rank: int = 0,
        device: torch.device | str = "tpu",
        language_model: Any = None,
    ) -> None:
        if max_rows not in ROW_BUCKETS:
            raise ValueError(f"max_rows must be one of {ROW_BUCKETS}, got {max_rows}")
        self.max_rows = max_rows
        self.context = context
        self.layers = layers
        self.vocab_size = vocab_size
        self.recipe = recipe
        self.tp_rank = int(tp_rank)
        self.device = torch.device(device)
        self._language_model = language_model

        if weight_names is None:
            from ._vendor.kimi.load import selected_local_shapes

            expected = tuple(
                selected_local_shapes(layers, vocab=vocab_size).keys()
            )
            if set(expected).issubset(weights.keys()):
                weight_names = expected
            else:
                weight_names = tuple(weights.keys())
        self.weight_names = tuple(weight_names)
        self.weights = {
            name: _own_storage(weights[name].to(self.device))
            for name in self.weight_names
        }
        self._weight_tensors = tuple(self.weights[name] for name in self.weight_names)

        if states is None:
            self.states = allocate_decode_states(
                max_rows, context, layers=layers, device=self.device
            )
        else:
            if len(states) != 4:
                raise ValueError(f"Expected 4 decode state tensors, got {len(states)}")
            self.states = tuple(_own_storage(s.to(self.device)) for s in states)  # type: ignore[assignment]

        self.slot_table = SlotTable(max_rows=max_rows)
        self._active_gen: dict[str, int] = {}
        self._handoff_warmed = False

        self.buckets = tuple(b for b in ROW_BUCKETS if b <= max_rows)
        if decode_ops is not None:
            self._decode_ops = dict(decode_ops)
        else:
            from torch_tpu._internal.pallas import get_pallas_mesh

            mesh = get_pallas_mesh(("tp",), (32,))
            self._decode_ops = {
                b: register_megakernel_decode_op(
                    mesh,
                    self.weight_names,
                    batch_size=b,
                    layers=layers,
                    recipe=recipe,
                    vocab_size=vocab_size,
                )
                for b in self.buckets
            }

        self._input_to_row: dict[int, torch.Tensor] = {
            b: _own_storage(torch.arange(b, dtype=torch.long, device=self.device))
            for b in self.buckets
        }
        self._mk_positions: dict[int, torch.Tensor] = {
            b: _own_storage(torch.zeros((b,), dtype=torch.int32, device=self.device))
            for b in self.buckets
        }
        self._row_to_output: dict[int, torch.Tensor] = {
            b: _own_storage(torch.arange(b, dtype=torch.long, device=self.device))
            for b in self.buckets
        }
        self._row_index_devs: tuple[torch.Tensor, ...] = tuple(
            _own_storage(torch.tensor([r], dtype=torch.long, device=self.device))
            for r in range(max_rows)
        )
        self._vocab_offset_dev = _own_storage(
            torch.tensor(
                [self.tp_rank * (vocab_size // 32)],
                dtype=torch.int32,
                device=self.device,
            )
        )
        try:
            from sglang.srt.layers.logits_processor import LogitsProcessorOutput

            self._logits_output_cls: Any = LogitsProcessorOutput
        except Exception:
            if self.device.type == "tpu":
                raise
            self._logits_output_cls = SimpleNamespace

    @classmethod
    def from_model(cls, language_model: Any) -> KimiK3MegakernelRunner:
        """Build the megakernel runner from a loaded ``KimiK3LinearForCausalLM``."""
        from sglang.srt.platforms import current_platform
        from sglang.srt.runtime_context import get_model, get_parallel

        parallel = get_parallel()
        if parallel.tp_size != 32 or parallel.moe_ep_size != 8:
            raise ValueError(
                "Kimi-K3 TPU decode megakernel requires tp_size=32 and ep_size=8, "
                f"got tp_size={parallel.tp_size}, moe_ep_size={parallel.moe_ep_size}"
            )
        tp_rank = parallel.tp_rank

        checkpoint_path = os.environ.get("SGLANG_TPU_K3_MEGAKERNEL_CHECKPOINT")
        if not checkpoint_path:
            checkpoint_path = getattr(language_model.config, "_name_or_path", None)
        if not checkpoint_path:
            try:
                checkpoint_path = get_model().model_path
            except Exception:
                checkpoint_path = None
        if not checkpoint_path:
            raise ValueError(
                "Could not determine checkpoint path for Kimi-K3 megakernel; "
                "set SGLANG_TPU_K3_MEGAKERNEL_CHECKPOINT."
            )

        raw_context = os.environ.get("SGLANG_TPU_K3_MEGAKERNEL_CONTEXT")
        if raw_context is not None:
            context = int(raw_context)
        else:
            model_ctx = None
            try:
                model_ctx = get_model().context_length
            except Exception:
                pass
            context = int(model_ctx) if model_ctx else 4096
        context = max(384, ((context + 127) // 128) * 128)

        max_rows = int(os.environ.get("SGLANG_TPU_K3_MEGAKERNEL_MAX_ROWS", "8"))
        recipe = os.environ.get("SGLANG_TPU_K3_MEGAKERNEL_RECIPE", "tuned")
        layers = int(language_model.config.num_hidden_layers)
        vocab_size = int(language_model.config.vocab_size)
        device = next(language_model.parameters()).device

        moe_layers = [
            layer.mlp.experts
            for layer in language_model.model.layers[1:]
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts")
        ]
        logger.info(
            "[KimiK3Megakernel] Stacking %d MoE layers in-place on rank %d...",
            len(moe_layers),
            tp_rank,
        )
        expert_weights = stack_sglang_moe_experts_in_place(moe_layers, device=device)
        if device.type == "tpu":
            current_platform.synchronize()

        logger.info(
            "[KimiK3Megakernel] Loading dense families from %s on rank %d...",
            checkpoint_path,
            tp_rank,
        )
        dense_weights = load_dense_families(
            checkpoint_path, rank=tp_rank, device=device, layers=layers
        )
        if device.type == "tpu":
            current_platform.synchronize()

        from ._vendor.kimi.load import selected_local_shapes

        weight_names = tuple(
            selected_local_shapes(layers, vocab=vocab_size).keys()
        )
        weights = {**dense_weights, **expert_weights}
        runner = cls(
            weights=weights,
            weight_names=weight_names,
            max_rows=max_rows,
            context=context,
            layers=layers,
            vocab_size=vocab_size,
            recipe=recipe,
            tp_rank=tp_rank,
            device=device,
            language_model=language_model,
        )
        if device.type == "tpu":
            current_platform.synchronize()
        logger.info(
            "[KimiK3Megakernel] Initialized runner on rank %d (max_rows=%d, context=%d, recipe=%s)",
            tp_rank,
            max_rows,
            context,
            recipe,
        )
        return runner

    def _resolve_pools(self, forward_batch: Any) -> tuple[Any, Any]:
        req_pool = getattr(forward_batch, "req_to_token_pool", None)
        kv_pool = getattr(forward_batch, "token_to_kv_pool", None)
        if req_pool is not None and kv_pool is not None:
            return req_pool, kv_pool

        try:
            from sglang.srt.model_executor.forward_context import (
                get_req_to_token_pool,
                get_token_to_kv_pool,
                has_forward_context,
            )

            if has_forward_context():
                req_pool = req_pool or get_req_to_token_pool()
                kv_pool = kv_pool or get_token_to_kv_pool()
                if req_pool is not None and kv_pool is not None:
                    return req_pool, kv_pool
        except Exception:
            pass

        if self._language_model is not None:
            shim = getattr(self._language_model, "forward", None)
            policy = getattr(shim, "policy", None)
            model_runner = getattr(policy, "model_runner", None)
            if model_runner is not None:
                req_pool = req_pool or getattr(
                    model_runner, "req_to_token_pool", None
                )
                kv_pool = kv_pool or getattr(model_runner, "token_to_kv_pool", None)
        return req_pool, kv_pool

    @staticmethod
    def _is_dummy_warmup_batch(forward_batch: Any, req_indices: list[int]) -> bool:
        if getattr(forward_batch, "reqs", None) is not None:
            return False
        if bool(getattr(forward_batch, "_is_dummy_warmup", False)):
            return True
        return all(idx == 0 for idx in req_indices)

    @staticmethod
    def _extract_live_rids_and_gens(
        forward_batch: Any,
        req_indices: list[int],
        req_to_token_pool: Any,
    ) -> tuple[list[str], list[int]]:
        reqs = getattr(forward_batch, "reqs", None)
        gen_tensor = (
            getattr(req_to_token_pool, "req_generation", None)
            if req_to_token_pool is not None
            else None
        )
        rids: list[str] = []
        gens: list[int] = []
        for i, pool_idx in enumerate(req_indices):
            gen = int(gen_tensor[pool_idx].item()) if gen_tensor is not None else 0
            gens.append(gen)
            if (
                reqs is not None
                and i < len(reqs)
                and getattr(reqs[i], "rid", None) is not None
            ):
                rids.append(str(reqs[i].rid))
            else:
                rids.append(f"__pool_{pool_idx}_gen_{gen}")
        return rids, gens

    def _mamba_slot_dev(self, req_to_token_pool: Any, req_pool_idx: int) -> torch.Tensor:
        mapping = getattr(
            req_to_token_pool, "req_index_to_mamba_index_mapping", None
        )
        if mapping is None:
            return torch.tensor([req_pool_idx], dtype=torch.long, device=self.device)
        if mapping.device.type == "cpu":
            slot = int(mapping[req_pool_idx].item())
            return torch.tensor([slot], dtype=torch.long, device=self.device)
        req_idx_dev = torch.tensor(
            [req_pool_idx], dtype=torch.long, device=mapping.device
        )
        return mapping.index_select(0, req_idx_dev).to(torch.long)

    def _handoff_one_request(
        self,
        *,
        row: int,
        req_pool_idx: int,
        cached_len: int,
        req_to_token_pool: Any,
        token_to_kv_pool: Any,
    ) -> None:
        conv_buffers, recurrent_buffers = extract_mamba_buffers(req_to_token_pool)
        mla_kv_buffers = extract_mla_kv_buffers(token_to_kv_pool)
        token_locs_dev, valid_mask_dev = build_padded_token_locs_and_mask(
            req_to_token_pool.req_to_token,
            req_pool_idx=req_pool_idx,
            cached_len=cached_len,
            context=self.context,
            device=self.device,
        )
        mamba_slot_dev = self._mamba_slot_dev(req_to_token_pool, req_pool_idx)
        handoff_request_to_megakernel_row(
            self.states,
            row_dev=self._row_index_devs[row],
            mamba_slot_dev=mamba_slot_dev,
            token_locs_dev=token_locs_dev,
            valid_mask_dev=valid_mask_dev,
            conv_buffers=conv_buffers,
            recurrent_buffers=recurrent_buffers,
            mla_kv_buffers=mla_kv_buffers,
            m_kb=self.weights["m_kb"],
        )

    def _resolve_pad_b(self, real_b: int) -> int:
        if self._language_model is not None:
            shim = getattr(self._language_model, "forward", None)
            policy = getattr(shim, "policy", None)
            b_bucketer = getattr(policy, "b_bucketer", None)
            if b_bucketer is not None:
                pad_b = int(b_bucketer.next_bucket(real_b))
                if pad_b in self._decode_ops:
                    return pad_b
        return row_bucket(real_b)

    def prepare_batch(self, forward_batch: Any) -> None:
        """Synchronize ``SlotTable``, hand off new decode requests, and stage step metadata."""
        if not forward_batch.forward_mode.is_decode():
            return

        real_b = getattr(forward_batch, "_tpu_real_b", None)
        if real_b is None:
            real_b = int(forward_batch.seq_lens.shape[0])
        else:
            real_b = int(real_b)

        if real_b <= 0 or real_b > self.max_rows:
            return

        pad_b = self._resolve_pad_b(real_b)
        seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
        if seq_lens_cpu is None:
            seq_lens_cpu = forward_batch.seq_lens.cpu()
        req_indices_cpu = getattr(forward_batch, "req_pool_indices_cpu", None)
        if req_indices_cpu is None:
            req_indices_cpu = forward_batch.req_pool_indices.cpu()

        req_indices = [int(x) for x in req_indices_cpu[:real_b].tolist()]
        seq_lens = [int(x) for x in seq_lens_cpu[:real_b].tolist()]

        if self._is_dummy_warmup_batch(forward_batch, req_indices):
            if not self._handoff_warmed:
                req_pool, kv_pool = self._resolve_pools(forward_batch)
                if req_pool is not None and kv_pool is not None:
                    self._handoff_one_request(
                        row=0,
                        req_pool_idx=0,
                        cached_len=0,
                        req_to_token_pool=req_pool,
                        token_to_kv_pool=kv_pool,
                    )
                    apply_row_move(
                        self.states,
                        RowMove(source=0, destination=0),
                        src_dev=self._row_index_devs[0],
                        dst_dev=self._row_index_devs[0],
                    )
                    self._handoff_warmed = True

            input_to_row_cpu = torch.arange(pad_b, dtype=torch.long)
            positions_cpu = torch.full((pad_b,), -1, dtype=torch.int32)
            positions_cpu[:real_b] = 0
            row_to_output_cpu = torch.arange(pad_b, dtype=torch.long)
            self._input_to_row[pad_b].copy_(input_to_row_cpu)
            self._mk_positions[pad_b].copy_(positions_cpu)
            self._row_to_output[pad_b].copy_(row_to_output_cpu)
            return

        req_pool, kv_pool = self._resolve_pools(forward_batch)
        if req_pool is None or kv_pool is None:
            raise RuntimeError(
                "KimiK3MegakernelRunner could not resolve req_to_token_pool / token_to_kv_pool"
            )

        live_rids, live_gens = self._extract_live_rids_and_gens(
            forward_batch, req_indices, req_pool
        )
        live_gen_by_rid = dict(zip(live_rids, live_gens))

        # 1. Release finished or generation-changed requests and compact state rows.
        current_rids = [self.slot_table.owner(r) for r in range(len(self.slot_table))]
        for rid in current_rids:
            if (
                rid not in live_gen_by_rid
                or self._active_gen.get(rid) != live_gen_by_rid[rid]
            ):
                move = self.slot_table.release(rid)
                self._active_gen.pop(rid, None)
                if move is not None:
                    apply_row_move(
                        self.states,
                        move,
                        src_dev=self._row_index_devs[move.source],
                        dst_dev=self._row_index_devs[move.destination],
                    )

        # 2. Assign rows and hand off newly-entering decode requests.
        for rid, gen, req_pool_idx, seq_len in zip(
            live_rids, live_gens, req_indices, seq_lens
        ):
            if rid not in self.slot_table:
                row = self.slot_table.assign(rid)
                self._active_gen[rid] = gen
                cached_len = max(seq_len - 1, 0)
                self._handoff_one_request(
                    row=row,
                    req_pool_idx=req_pool_idx,
                    cached_len=cached_len,
                    req_to_token_pool=req_pool,
                    token_to_kv_pool=kv_pool,
                )

        # 3. Stage per-bucket input permutation, positions, and output inverse permutation.
        rows = self.slot_table.rows_for(live_rids)
        input_to_row_cpu = torch.arange(pad_b, dtype=torch.long)
        positions_cpu = torch.full((pad_b,), -1, dtype=torch.int32)
        row_to_output_cpu = torch.arange(pad_b, dtype=torch.long)
        for i, r in enumerate(rows):
            input_to_row_cpu[r] = i
            positions_cpu[r] = max(seq_lens[i] - 1, 0)
            row_to_output_cpu[i] = r

        self._input_to_row[pad_b].copy_(input_to_row_cpu)
        self._mk_positions[pad_b].copy_(positions_cpu)
        self._row_to_output[pad_b].copy_(row_to_output_cpu)

    def forward_decode(
        self, input_ids: torch.Tensor, forward_batch: Any
    ) -> Any:
        """Execute one compiled megakernel decode step for padded batch ``input_ids``."""
        del forward_batch
        b = int(input_ids.shape[0])
        mk_tokens = (
            input_ids.to(torch.int32).index_select(0, self._input_to_row[b])
            - self._vocab_offset_dev
        )
        mk_logits = self._decode_ops[b](
            *self._weight_tensors,
            *self.states,
            mk_tokens,
            self._mk_positions[b],
        )
        logits = mk_logits.index_select(0, self._row_to_output[b])
        return self._logits_output_cls(next_token_logits=logits)
