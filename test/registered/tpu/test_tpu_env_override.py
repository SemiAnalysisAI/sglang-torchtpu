"""Unit tests for the TPU environment overrides.

``apply_tpu_env_overrides`` must scrub jax-owned libtpu env vars: launchers
that previously ran jax leave ``TPU_ML_PLATFORM`` behind, libtpu then
initializes in JAX mode, and torch_tpu's multi-process PJRT init fails on
the libtpu lockfile. CPU-runnable.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=60, suite="stage-a-test-tpu-unit")

import os
import unittest
from unittest.mock import patch

from sglang.test.test_utils import CustomTestCase


class TestJaxEnvScrub(CustomTestCase):
    def test_apply_tpu_env_overrides_removes_jax_owned_env(self):
        from sglang.srt.hardware_backend.tpu.env_override import (
            apply_tpu_env_overrides,
        )

        with patch.dict(
            os.environ,
            {"TPU_ML_PLATFORM": "JAX", "TPU_ML_PLATFORM_VERSION": "1"},
        ):
            apply_tpu_env_overrides()
            self.assertNotIn("TPU_ML_PLATFORM", os.environ)
            self.assertNotIn("TPU_ML_PLATFORM_VERSION", os.environ)


if __name__ == "__main__":
    unittest.main()
