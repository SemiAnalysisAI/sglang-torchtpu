"""TPU xprof profiler."""

from __future__ import annotations

import dataclasses
import gzip
import json
from contextlib import AbstractContextManager
from pathlib import Path
from types import ModuleType

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.tpu._logging import get_logger

logger = get_logger(__name__)


_TPU_PROFILE_SHIM_FRAMES = (
    "/sglang/srt/hardware_backend/tpu/patches.py:",
    "/sglang/srt/hardware_backend/tpu/pallas_jax_op.py:",
)


@dataclasses.dataclass
class _TpuProfileSession(AbstractContextManager):
    profiler: AbstractContextManager
    output_dir: Path

    def __enter__(self):
        self.profiler.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        return self.profiler.__exit__(exc_type, exc, tb)


def start_tpu_profile(
    *, output_dir: Path, with_cpu: bool = True
) -> AbstractContextManager:
    """Start an xprof capture writing to ``output_dir``.

    Returns the entered profiler. Pass it to :func:`stop_tpu_profile` to end
    the session. ``with_cpu`` adds the host-CPU activity to the device trace.
    """
    from torch_tpu._internal.profiler import profiler_api

    _install_tpu_profile_options_patch(profiler_api)

    output_dir.mkdir(parents=True, exist_ok=True)
    handler = profiler_api.xprof_trace_handler(dir_name=str(output_dir))
    activities = [profiler_api.ProfilerActivity.TPU]
    if with_cpu:
        activities.append(profiler_api.ProfilerActivity.CPU)
    profiler = profiler_api.profile(activities=activities, on_trace_ready=handler)
    session = _TpuProfileSession(profiler=profiler, output_dir=output_dir)
    session.__enter__()
    return session


def stop_tpu_profile(profiler: AbstractContextManager) -> None:
    """Stop an xprof session started by :func:`start_tpu_profile`."""
    # Exiting the context stops the xprof session.
    profiler.__exit__(None, None, None)
    if isinstance(profiler, _TpuProfileSession):
        _normalize_tpu_profile_sources(profiler.output_dir)


def _normalize_tpu_profile_sources(output_dir: Path) -> None:
    """Promote xprof source attribution past SGLang's TPU shim frames.

    TorchTPU records generic signature and Pallas adapters as the HLO source.
    The full stack contains the SGLang operation that invoked the adapter, so
    promote that operation in both event metadata and Source Code tracks.
    """
    for path in _tpu_trace_paths(output_dir):
        if _normalize_tpu_trace_file(path):
            logger.info("[TPU profiler] normalized shim source frames in %s", path)


def _tpu_trace_paths(output_dir: Path) -> list[Path]:
    return [
        *output_dir.rglob("*.trace.json.gz"),
        *output_dir.rglob("*.trace.json"),
    ]


def _normalize_tpu_trace_file(path: Path) -> bool:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        trace = json.load(f)

    events = trace.get("traceEvents", []) if isinstance(trace, dict) else []
    changed = False
    source_replacements: dict[tuple[object, str], set[str]] = {}
    for event in events:
        args = event.get("args")
        if not isinstance(args, dict):
            continue
        source = args.get("source")
        if not _normalize_tpu_event_source(args):
            continue
        changed = True
        promoted_source = args.get("source")
        if isinstance(source, str) and isinstance(promoted_source, str):
            key = (args.get("group_id"), source)
            source_replacements.setdefault(key, set()).add(promoted_source)

    for event in events:
        args = event.get("args")
        if not isinstance(args, dict):
            continue
        name = event.get("name")
        if not isinstance(name, str):
            continue
        replacements = source_replacements.get((args.get("group_id"), name), set())
        if len(replacements) == 1:
            event["name"] = next(iter(replacements))
            changed = True

    if not changed:
        return False

    tmp_path = path.with_name(f"{path.name}.tmp")
    with opener(tmp_path, "wt", encoding="utf-8") as f:
        json.dump(trace, f, separators=(",", ":"))
    tmp_path.replace(path)
    return True


def _normalize_tpu_event_source(args: dict) -> bool:
    source_stack = args.get("source_stack")
    frames = _source_stack_frames(source_stack)
    if not frames:
        return False

    filtered_frames = [
        frame for frame in frames if not _is_tpu_profile_shim_frame(frame)
    ]
    if len(filtered_frames) == len(frames):
        return False

    source = args.get("source")
    promoted_frame = _first_sglang_source_frame(filtered_frames)
    if promoted_frame is not None and (
        not isinstance(source, str) or _is_tpu_profile_shim_frame(source)
    ):
        args["source"] = _source_frame_location(promoted_frame)

    args["source_stack"] = (
        "\n".join(filtered_frames) if isinstance(source_stack, str) else filtered_frames
    )
    return True


def _first_sglang_source_frame(frames: list[str]) -> str | None:
    for frame in frames:
        if "/sglang/" in frame:
            return frame
    return frames[0] if frames else None


def _source_stack_frames(source_stack) -> list[str]:
    if isinstance(source_stack, str):
        return [frame for frame in source_stack.splitlines() if frame]
    if isinstance(source_stack, list):
        return [frame for frame in source_stack if isinstance(frame, str)]
    return []


def _is_tpu_profile_shim_frame(frame: str) -> bool:
    return any(shim_frame in frame for shim_frame in _TPU_PROFILE_SHIM_FRAMES)


def _source_frame_location(frame: str) -> str:
    path_and_line, sep, column = frame.rpartition(":")
    if sep and column.isdigit():
        return path_and_line
    return frame


def _install_tpu_profile_options_patch(profiler_api: ModuleType) -> None:
    """Add libtpu trace-limit fields to torch_tpu profile options."""
    if getattr(profiler_api, "_sglang_tpu_profile_options_patch_installed", False):
        return

    num_chips = envs.SGLANG_TPU_PROFILE_NUM_CHIPS.get()
    num_sparse_cores = envs.SGLANG_TPU_PROFILE_NUM_SPARSE_CORES.get()
    chip_limit = num_chips if num_chips > 0 else None
    sparse_core_limit = num_sparse_cores if num_sparse_cores > 0 else None
    # When tracing is narrowed to a single SparseCore, also narrow to a
    # single tile within it to reduce the number of trace events to allow
    # tracing more steps, otherwise we may run into dropping trace.
    sparse_core_tile_limit = 1 if sparse_core_limit == 1 else None
    if chip_limit is None and sparse_core_limit is None:
        profiler_api._sglang_tpu_profile_options_patch_installed = True
        return

    try:
        from torch_tpu._internal.profiler import _impl as _tpu_impl

        orig_get_options = profiler_api._get_profile_options
        if not dataclasses.is_dataclass(_tpu_impl.ProfileOptions):
            raise TypeError("ProfileOptions is no longer a dataclass")

        @dataclasses.dataclass
        class _ProfileOptions(_tpu_impl.ProfileOptions):
            tpu_num_chips_to_profile_per_task: int | None = None
            tpu_num_sparse_cores_to_trace: int | None = None
            tpu_num_sparse_core_tiles_to_trace: int | None = None

    except (ImportError, AttributeError, TypeError):
        logger.warning(
            "[TPU profiler] this torch_tpu does not expose the expected "
            "profile-options hook. Tracing all chips/SparseCores "
            "(traces may be large)."
        )
        profiler_api._sglang_tpu_profile_options_patch_installed = True
        return

    def _get_options(
        activities: list[profiler_api.ProfilerActivity],
    ) -> _tpu_impl.ProfileOptions:
        base = orig_get_options(activities)
        try:
            # Merge into a dict first. A future ProfileOptions that grows one
            # of these fields would make **asdict raise on duplicate keywords.
            fields = dataclasses.asdict(base)
            fields.update(
                tpu_num_chips_to_profile_per_task=chip_limit,
                tpu_num_sparse_cores_to_trace=sparse_core_limit,
                tpu_num_sparse_core_tiles_to_trace=sparse_core_tile_limit,
            )
            return _ProfileOptions(**fields)
        except Exception:
            # Restore the original hook so the failure is logged once.
            profiler_api._get_profile_options = orig_get_options
            logger.warning(
                "[TPU profiler] failed to extend ProfileOptions. Tracing all "
                "chips/SparseCores (traces may be large).",
                exc_info=True,
            )
            return base

    profiler_api._get_profile_options = _get_options
    profiler_api._sglang_tpu_profile_options_patch_installed = True
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
