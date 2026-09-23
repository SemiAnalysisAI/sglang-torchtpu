"""CPU tests for asynchronous TPU work at profiler stage boundaries."""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.tpu import profiler as tpu_profiler
from sglang.srt.managers.scheduler_components import profiler_manager
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize("prefill_steps,profile_steps", [(2, 2), (3, 1)])
def test_stage_capture_drains_all_ranks(
    tmp_path, monkeypatch, prefill_steps, profile_steps
):
    # Complete device work only at synchronize(), reproducing a scheduler that
    # queues the next stage while the preceding stage is still executing.
    pending = [[], []]
    captures = {}
    active_stage = None
    ranks = {}
    host_barrier = threading.Barrier(2)
    group = object()

    def synchronize():
        rank = ranks[threading.get_ident()]
        if active_stage is not None:
            captures[active_stage][rank].extend(pending[rank])
        pending[rank].clear()

    def barrier(actual_group):
        assert actual_group is group
        host_barrier.wait(timeout=10)

    def start_profile(*, output_dir, with_cpu):
        nonlocal active_stage
        assert ranks[threading.get_ident()] == 0
        assert active_stage is None
        assert pending == [[], []]
        assert with_cpu
        active_stage = output_dir.name.rsplit("-", 1)[-1]
        captures[active_stage] = [[], []]
        return active_stage

    def stop_profile(profile):
        nonlocal active_stage
        assert ranks[threading.get_ident()] == 0
        assert profile == active_stage
        assert pending == [[], []]
        active_stage = None

    monkeypatch.setattr(profiler_manager, "_is_tpu", True)
    monkeypatch.setattr(
        profiler_manager,
        "current_platform",
        SimpleNamespace(synchronize=synchronize, is_out_of_tree=lambda: False),
    )
    monkeypatch.setattr(profiler_manager.torch.distributed, "barrier", barrier)
    monkeypatch.setattr(tpu_profiler, "start_tpu_profile", start_profile)
    monkeypatch.setattr(tpu_profiler, "stop_tpu_profile", stop_profile)

    def run_rank(rank):
        ranks[threading.get_ident()] = rank
        manager = profiler_manager.SchedulerProfilerManager(
            ps=SimpleNamespace(tp_rank=rank, dp_size=1, pp_size=1, moe_ep_size=1),
            dp_tp_cpu_group=group,
            get_forward_ct=lambda: 0,
        )
        assert manager._init_profile(
            output_dir=str(tmp_path),
            start_step=None,
            num_steps=profile_steps,
            activities=["CPU", "TPU"],
            with_stack=False,
            record_shapes=False,
            profile_by_stage=True,
            profile_id="test",
        ).success
        pending[rank].append("warmup")
        for step in range(prefill_steps):
            manager._profile_batch_predicate(
                SimpleNamespace(forward_mode=ForwardMode.EXTEND)
            )
            pending[rank].append(f"prefill-{step}")
        for step in range(profile_steps + 1):
            manager._profile_batch_predicate(
                SimpleNamespace(forward_mode=ForwardMode.DECODE)
            )
            pending[rank].append(f"decode-{step}")
        assert not manager.profile_in_progress

    with (
        envs.SGLANG_PROFILE_V2.override(False),
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        futures = [pool.submit(run_rank, rank) for rank in range(2)]
        for future in futures:
            future.result(timeout=30)

    expected_prefill = [f"prefill-{step}" for step in range(profile_steps)]
    expected_decode = [f"decode-{step}" for step in range(profile_steps)]
    assert captures == {
        "EXTEND": [expected_prefill, expected_prefill],
        "DECODE": [expected_decode, expected_decode],
    }
    assert pending == [[f"decode-{profile_steps}"], [f"decode-{profile_steps}"]]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
