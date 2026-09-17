"""Shared class fixture and role dispatch for one-host-per-engine P/D tests."""

import os
from pathlib import Path
from typing import Any, ClassVar

from benchmark.tpu import launch_server
from benchmark.tpu.launcher_test_case import TPULauncherTestCase

try:
    from sglang.test.ci.ci_register import register_tpu_ci
except ModuleNotFoundError:
    # Keep source-only configuration checks independent of serving dependencies.
    def register_tpu_ci(*_args, **_kwargs):
        return None


register_tpu_ci(
    est_time=0,
    suite="tpu-integration",
    nightly=True,
    disabled="helper module, no tests",
)


def peer_hostname(hostname: str, *, from_role: str, to_role: str) -> str:
    """Map this pod's DNS name onto the pod of the peer role."""
    needle, replacement = f"-{from_role}-", f"-{to_role}-"
    if needle not in hostname:
        raise ValueError(f"hostname {hostname!r} has no {needle!r} segment")
    return hostname.replace(needle, replacement)


class PDSingleHostTestCase(TPULauncherTestCase):
    """Run evaluations on prefill while the decode pod serves independently.

    Subclasses supply config_module, log paths, self_hostname, and
    build_model_config. They may also override check_server_configuration.
    Server startup and shutdown remain in the inherited class lifecycle.
    """

    base_url = f"http://127.0.0.1:{int(os.environ.get('LB_PORT', '8000'))}"
    health_timeout = 2 * int(os.environ.get("WAIT_SECS", "3600")) + 300
    health_generate_timeout = float(os.environ.get("HEALTH_TIMEOUT", "3600"))
    cwd = str(Path(__file__).absolute().parents[4] / "benchmark" / "tpu")
    keep_up = os.environ.get("KEEP_UP", "0") == "1"
    tag = "pd-singlehost"
    decode_log_path: ClassVar[str | None] = None
    decode_host_env: ClassVar[str | None] = None

    @staticmethod
    def self_hostname() -> str:
        raise NotImplementedError("Provide this test's hostname resolver")

    @classmethod
    def decode_hostname(cls) -> str:
        if cls.decode_host_env and os.environ.get(cls.decode_host_env):
            return os.environ[cls.decode_host_env]
        return peer_hostname(cls.self_hostname(), from_role="prefill", to_role="decode")

    @classmethod
    def get_runtime_envs(cls) -> dict[str, str | None]:
        return {
            **super().get_runtime_envs(),
            "SGLANG_LAUNCHER_HOSTNAMES": f"{cls.self_hostname()},{cls.decode_hostname()}",
            "SGLANG_LAUNCHER_RANK": "0",
        }

    @classmethod
    def build_model_config(cls) -> dict[str, Any]:
        raise NotImplementedError("Provide this test's evaluation configuration")

    @classmethod
    def run_prefill_leader(cls) -> int:
        # Evaluation dependencies are only needed on the integration path.
        from sglang.test.tpu import model_config_main

        model_config_main(
            cls.build_model_config(), base_url=cls.base_url, server_fixture=cls
        )
        return 0

    @classmethod
    def run_decode_worker(cls) -> int:
        proc = launch_server.start_launcher(
            cls.config_module + ":DECODE_CONFIG",
            {**cls.runtime_envs, "SGLANG_LAUNCHER_RANK": "0"},
            cwd=cls.cwd,
            log_path=cls.decode_log_path,
        )
        try:
            return proc.wait()
        finally:
            launch_server.stop_processes([proc], grace=60)

    @classmethod
    def run_role(cls, role: str) -> int:
        if role == "prefill":
            return cls.run_prefill_leader()
        if role == "decode":
            return cls.run_decode_worker()
        raise ValueError(f"Expected prefill or decode role, got {role!r}")
