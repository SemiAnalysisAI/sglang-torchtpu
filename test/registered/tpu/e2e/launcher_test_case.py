"""Class fixtures for tests using the shared TPU launcher.

Configuration tests can import this fixture without installing SGLang or TPU
libraries; the serving process loads those dependencies when it starts.
"""

import subprocess
import unittest
from collections.abc import Mapping
from typing import ClassVar

from benchmark.tpu import launch_server

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


class TPULauncherTestCase(unittest.TestCase):
    """Share one launcher across every test method in a subclass.

    Set config_module to a Python recipe and base_url to its server or router.
    Override get_runtime_envs when topology is only available at setup time.
    Set health_generate_timeout to check generation after server configuration
    validation, as P/D tests do once both engines and the router are healthy.
    """

    config_module: ClassVar[str] = ""
    runtime_envs: ClassVar[Mapping[str, str | None]] = {}
    base_url: ClassVar[str] = ""
    health_timeout: ClassVar[float] = 3600
    health_endpoint: ClassVar[str] = "/health"
    health_generate_timeout: ClassVar[float | None] = None
    cwd: ClassVar[str | None] = None
    log_path: ClassVar[str | None] = None
    keep_up: ClassVar[bool] = False
    tag: ClassVar[str] = "tpu-launcher"
    _launcher_process: ClassVar[subprocess.Popen | None] = None

    @classmethod
    def get_runtime_envs(cls) -> dict[str, str | None]:
        return dict(cls.runtime_envs)

    @classmethod
    def check_server_configuration(cls) -> None:
        """Optionally validate the ready servers before running any tests."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        if not cls.config_module or not cls.base_url:
            raise ValueError("Set config_module and base_url on the test class")
        cls._launcher_process = launch_server.start_launcher(
            cls.config_module,
            cls.get_runtime_envs(),
            cwd=cls.cwd,
            log_path=cls.log_path,
        )

        def check():
            if cls._launcher_process.poll() is not None:
                raise RuntimeError(
                    f"Launcher exited rc={cls._launcher_process.returncode}"
                )

        try:
            launch_server.wait_healthy(
                cls.base_url,
                cls.health_timeout,
                check,
                endpoint=cls.health_endpoint,
                tag=cls.tag,
            )
            cls.check_server_configuration()
            if cls.health_generate_timeout is not None:
                launch_server.wait_healthy(
                    cls.base_url,
                    cls.health_generate_timeout,
                    check,
                    interval=5,
                    endpoint="/health_generate",
                    tag=cls.tag,
                )
        except BaseException:
            # unittest does not call tearDownClass if setUpClass fails.
            cls._stop_launcher()
            raise

    @classmethod
    def _stop_launcher(cls) -> None:
        if cls._launcher_process is not None:
            # Allow the supervisor's 30-second child shutdown to finish.
            launch_server.stop_processes([cls._launcher_process], grace=60)
            cls._launcher_process = None

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            if not cls.keep_up:
                cls._stop_launcher()
        finally:
            super().tearDownClass()
