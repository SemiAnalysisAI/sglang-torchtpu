"""TPU CI entry for the all-gather ring attention dispatch tests (CPU logic)."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=60, suite="stage-a-test-tpu-unit")

import unittest

from sglang.multimodal_gen.test.unit.test_ring_admission import (  # noqa: F401
    TestTpuSplashRingCapability,
)
from sglang.multimodal_gen.test.unit.test_usp_allgather_ring import (  # noqa: F401
    TestAllGatherRingAttention,
)
from sglang.multimodal_gen.test.unit.test_usp_ring_tail_pad import (  # noqa: F401
    TestRingTailPadDispatch,
)

if __name__ == "__main__":
    unittest.main()
