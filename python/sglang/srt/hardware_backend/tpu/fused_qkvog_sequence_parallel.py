# SPDX-License-Identifier: Apache-2.0
"""Compatibility imports for the sequence-parallel QKVOG TorchTPU entrypoint."""

from sglang.srt.hardware_backend.tpu.kernels.sequence_parallel.fused_rmsnorm_qkvog_mm_a2a.fused_rmsnorm_qkvog_mm_a2a_entrypoint import (
    can_use_tpu_sp_fused_qkvog,
    make_local_rank_metadata,
    pack_local_qkvog_weights_by_physical_destination,
    prepare_tpu_sp_qkvog_weight,
    register_tpu_sp_fused_qkvog_op,
    supports_tpu_sp_qkvog_device,
    tpu_sp_fused_qkvog,
    tpu_sp_qkvog_config_unsupported_reason,
    tpu_sp_qkvog_device_unsupported_reason,
    validate_physical_to_logical,
)

__all__ = [
    "can_use_tpu_sp_fused_qkvog",
    "make_local_rank_metadata",
    "pack_local_qkvog_weights_by_physical_destination",
    "prepare_tpu_sp_qkvog_weight",
    "register_tpu_sp_fused_qkvog_op",
    "supports_tpu_sp_qkvog_device",
    "tpu_sp_fused_qkvog",
    "tpu_sp_qkvog_config_unsupported_reason",
    "tpu_sp_qkvog_device_unsupported_reason",
    "validate_physical_to_logical",
]
