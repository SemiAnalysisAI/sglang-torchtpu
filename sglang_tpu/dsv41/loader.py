# SPDX-License-Identifier: Apache-2.0
"""RunAI transport with EP-local routed expert reads for V4.1 Flash."""

from __future__ import annotations

import inspect
import json
import logging
from pathlib import Path
from time import perf_counter

from sglang.srt.model_loader.loader import RunaiModelStreamerLoader
from sglang_tpu.dsv41.checkpoint import plan_expert_reads

logger = logging.getLogger(__name__)


class DSV41RunaiModelStreamerLoader(RunaiModelStreamerLoader):
    """Reuse construction/postprocessing; specialize only primary weight I/O."""

    def __init__(self, load_config, model_config):
        super().__init__(load_config)
        self._primary_model_config = model_config

    def _get_weights_iterator(self, source):
        config = getattr(source.model_config, "hf_config", None)
        if (
            source.model_config is not self._primary_model_config
            or getattr(config, "model_type", None) != "deepseek_v41"
            or getattr(source.model_config, "is_draft_model", False)
            or source.model_or_path
            != getattr(
                self._primary_model_config,
                "model_weights",
                self._primary_model_config.model_path,
            )
            or source.revision != self._primary_model_config.revision
            or source.prefix
            or self.load_config.draft_model_idx is not None
        ):
            yield from super()._get_weights_iterator(source)
            return

        from runai_model_streamer import SafetensorsStreamer

        from sglang.srt.model_loader.weight_utils import RUNAI_STREAMER_TENSOR_ATTR
        from sglang.srt.runtime_context import get_parallel
        from sglang.srt.utils.runai_utils import ObjectStorageModel, is_runai_obj_uri

        if (
            "tensor_names"
            not in inspect.signature(SafetensorsStreamer.stream_files).parameters
        ):
            raise RuntimeError(
                "V4.1 selective loading requires RunAI tensor_names support (tested with 0.17.0)"
            )
        folder, files = self._prepare_weights(source.model_or_path, source.revision)
        metadata_dir = (
            ObjectStorageModel.get_path(folder) if is_runai_obj_uri(folder) else folder
        )
        index = json.loads(
            (Path(metadata_dir) / "model.safetensors.index.json").read_text()
        )
        parallel = get_parallel()
        plan = plan_expert_reads(
            index["weight_map"],
            num_layers=config.num_hidden_layers,
            num_experts=config.n_routed_experts,
            ep_size=parallel.moe_ep_size,
            ep_rank=parallel.moe_ep_rank,
        )
        files = [path for path in files if path.rsplit("/", 1)[-1] in plan.shard_names]
        if len(files) != len(plan.shard_names):
            raise ValueError(
                "RunAI shard list does not cover the selected checkpoint index"
            )
        logger.info(
            "V4.1 EP rank %d/%d: reading %d/%d tensors (%d local expert tensors) from %d shards",
            parallel.moe_ep_rank,
            parallel.moe_ep_size,
            len(plan.tensor_names),
            plan.total_tensors,
            plan.local_expert_tensors,
            len(files),
        )
        # RunAI distributed streaming broadcasts one common request sequence.
        # Its contract requires identical tensor_names on every rank. Our EP
        # sets differ, so each rank reads its own ranges independently. All ranks
        # still enter stream_files: even False can perform group setup in 0.17.
        # Keep owned=True (the default): loaders may retain CPU expert tensors.
        start, size, count = perf_counter(), 0, 0
        with SafetensorsStreamer() as streamer:
            streamer.stream_files(
                files,
                device="cpu",
                is_distributed=False,
                tensor_names=plan.tensor_names,
            )
            for name, tensor in streamer.get_tensors():
                size += tensor.numel() * tensor.element_size()
                count += 1
                setattr(tensor, RUNAI_STREAMER_TENSOR_ATTR, True)
                yield name, tensor
        logger.info(
            "V4.1 EP rank %d: consumed %d selected tensors, %.3f GB in %.3f s (includes parameter loading)",
            parallel.moe_ep_rank,
            count,
            size / 1e9,
            perf_counter() - start,
        )
