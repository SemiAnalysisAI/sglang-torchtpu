"""Class fixtures for tests using the shared TPU launcher.

Only the standard library and launcher are imported here, so configuration
tests can import their test classes without loading SGLang or TPU libraries.
"""

import subprocess
import unittest
from collections.abc import Mapping
from typing import ClassVar

from benchmark.tpu import launch_server


class TPULauncherTestCase(unittest.TestCase):
    """Share one launcher across every test method in a subclass.

    Set config_module to a Python recipe and base_url to its server or router.
    Override get_runtime_envs when topology is only available at setup time.
    """

    config_module: ClassVar[str] = ""
    runtime_envs: ClassVar[Mapping[str, str | None]] = {}
    base_url: ClassVar[str] = ""
    health_timeout: ClassVar[float] = 3600
    health_endpoint: ClassVar[str] = "/health"
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
