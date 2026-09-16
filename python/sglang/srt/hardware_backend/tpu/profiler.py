"""TPU xprof profiler."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import msgspec

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.tpu._logging import get_logger

logger = get_logger(__name__)


def start_tpu_profile(*, output_dir: Path, with_cpu: bool = True) -> _TpuTrace:
    """Start an xprof capture writing to ``output_dir``.

    Returns a handle. Pass it to :func:`stop_tpu_profile` to end the session.

    Uses torch_tpu's start_trace, which drives libtpu's profiler directly, and
    not torch.profiler. The torch.profiler route reaches TPU device activity
    only through kineto, which needs libtpu registered via PyTorch's
    PrivateUse1ProfilerRegistry -- absent before torch 2.12, and the TPU stack
    pins torch 2.11 (pyproject_tpu.toml). On that route libtpu logs

        PyTorch PrivateUse1ProfilerRegistry not found. Native TPU profiling
        will be disabled (this is expected on PyTorch < 2.12).

    and the capture still succeeds, holding host activity only: a 43 s run gave
    905,956 /host:CPU events against /device:TPU:0 at 0.0 ms busy. start_trace
    has no such dependency: it calls libtpu's trace backend directly, so it
    should not care which torch is installed.

    ``with_cpu`` is accepted for call compatibility. libtpu's trace always
    carries the host plane, so there is nothing to switch off.
    """
    # start_trace/stop_trace live in _impl; the package root re-exports only
    # the torch.profiler-facing names (profile, xprof_trace_handler) and the
    # remote-capture server.
    from torch_tpu._internal.profiler import _impl as tpu_profiler

    output_dir.mkdir(parents=True, exist_ok=True)
    tpu_profiler.start_trace(str(output_dir), _tpu_profile_options())
    return _TpuTrace(output_dir=output_dir)


def stop_tpu_profile(trace: _TpuTrace) -> None:
    """Stop an xprof session started by :func:`start_tpu_profile`."""
    del trace  # the session is process-global; only one may run at a time
    from torch_tpu._internal.profiler import _impl as tpu_profiler

    tpu_profiler.stop_trace()


class _TpuTrace(msgspec.Struct, frozen=True):
    """Handle for an in-flight capture. torch_tpu keeps the session itself."""

    output_dir: Path


def _tpu_profile_options():
    """Build libtpu trace-limit options, or None to trace everything.

    start_trace takes the options directly, so the narrowing no longer needs to
    monkeypatch profiler_api._get_profile_options the way the torch.profiler
    route did.
    """
    num_chips = envs.SGLANG_TPU_PROFILE_NUM_CHIPS.get()
    num_sparse_cores = envs.SGLANG_TPU_PROFILE_NUM_SPARSE_CORES.get()
    chip_limit = num_chips if num_chips > 0 else None
    sparse_core_limit = num_sparse_cores if num_sparse_cores > 0 else None
    # When tracing is narrowed to a single SparseCore, also narrow to a
    # single tile within it to reduce the number of trace events to allow
    # tracing more steps, otherwise we may run into dropping trace.
    sparse_core_tile_limit = 1 if sparse_core_limit == 1 else None
    if chip_limit is None and sparse_core_limit is None:
        return None

    try:
        from torch_tpu._internal.profiler import _impl as _tpu_impl

        if not dataclasses.is_dataclass(_tpu_impl.ProfileOptions):
            raise TypeError("ProfileOptions is no longer a dataclass")

        @dataclasses.dataclass
        class _ProfileOptions(_tpu_impl.ProfileOptions):
            tpu_num_chips_to_profile_per_task: int | None = None
            tpu_num_sparse_cores_to_trace: int | None = None
            tpu_num_sparse_core_tiles_to_trace: int | None = None

        options = _ProfileOptions(
            tpu_num_chips_to_profile_per_task=chip_limit,
            tpu_num_sparse_cores_to_trace=sparse_core_limit,
            tpu_num_sparse_core_tiles_to_trace=sparse_core_tile_limit,
        )
    except (ImportError, AttributeError, TypeError):
        logger.warning(
            "[TPU profiler] this torch_tpu does not expose the expected "
            "profile-options hook. Tracing all chips/SparseCores "
            "(traces may be large)."
        )
        return None

    # The tile cap is only derived when tracing a single SparseCore; omit the
    # clause otherwise so the log doesn't read "(all tile(s))" for a branch
    # that was never entered.
    tile_clause = (
        f" ({sparse_core_tile_limit} tile(s))"
        if sparse_core_tile_limit is not None
        else ""
    )
    logger.info(
        "[TPU profiler] limiting trace to %s chip(s), %s SparseCore(s)%s per task",
        chip_limit or "all",
        sparse_core_limit or "all",
        tile_clause,
    )
    return options
