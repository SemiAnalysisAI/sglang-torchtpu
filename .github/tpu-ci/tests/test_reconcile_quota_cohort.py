"""The 2x2x1 cohort split: pr-gate keeps its reserve unlent, nightly gets the rest."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import reconcile_quota

FLAVOR_POOL = {"tpu7x-2x2x1-pr-gate": "pr-gate", "tpu7x-2x2x1-nightly": "nightly-1h"}
COHORT = [
    (1, {"pr-gate": "tpu-prgate-1h", "nightly": "tpu-nightly-1h"}, {"pr-gate": 5})
]


def cq(name, flavors):
    return {
        "metadata": {"name": name},
        "spec": {"resourceGroups": [{"flavors": [{"name": f} for f in flavors]}]},
    }


CQS = [
    cq("tpu-prgate-1h", ["tpu7x-2x2x1-pr-gate", "tpu7x-2x2x1-nightly"]),
    cq("tpu-nightly-1h", ["tpu7x-2x2x1-nightly", "tpu7x-2x2x1-pr-gate"]),
]


def split(pr_gate_nodes, nightly_nodes):
    return reconcile_quota.cohort_quotas(
        cohorts=COHORT,
        cqs=CQS,
        flavor_pool=FLAVOR_POOL,
        chips={"pr-gate": 4 * pr_gate_nodes, "nightly-1h": 4 * nightly_nodes},
    )


class TestCohortQuotas(unittest.TestCase):
    def test_reserve_fills_own_pool_then_spills(self):
        got = split(pr_gate_nodes=4, nightly_nodes=4)
        self.assertEqual(got[("tpu-prgate-1h", "tpu7x-2x2x1-pr-gate")], (16, 0))
        self.assertEqual(got[("tpu-prgate-1h", "tpu7x-2x2x1-nightly")], (4, 0))
        self.assertEqual(got[("tpu-nightly-1h", "tpu7x-2x2x1-nightly")], (12, None))
        self.assertEqual(got[("tpu-nightly-1h", "tpu7x-2x2x1-pr-gate")], (0, None))

    def test_quota_never_exceeds_live_chips(self):
        for pr, nt in ((0, 0), (2, 3), (4, 2), (6, 1), (5, 5)):
            got = split(pr_gate_nodes=pr, nightly_nodes=nt)
            for flavor, pool in FLAVOR_POOL.items():
                total = sum(n for (_, f), (n, _) in got.items() if f == flavor)
                live = 4 * (pr if pool == "pr-gate" else nt)
                self.assertEqual(total, live, (pr, nt, flavor))

    def test_reserve_larger_than_cluster_leaves_nightly_nothing(self):
        got = split(pr_gate_nodes=2, nightly_nodes=2)
        self.assertEqual(got[("tpu-nightly-1h", "tpu7x-2x2x1-nightly")], (0, None))
        self.assertEqual(got[("tpu-prgate-1h", "tpu7x-2x2x1-nightly")], (8, 0))

    def test_needs_exactly_one_lane_without_reserve(self):
        both = [(1, COHORT[0][1], {"pr-gate": 1, "nightly": 1})]
        with self.assertRaises(SystemExit):
            reconcile_quota.cohort_quotas(
                cohorts=both, cqs=CQS, flavor_pool=FLAVOR_POOL, chips={}
            )

    def test_resources_shape(self):
        self.assertEqual(
            reconcile_quota.resources(chips=8, lending_limit=0),
            [
                {"name": "google.com/tpu", "nominalQuota": "8", "lendingLimit": "0"},
                {"name": "cpu", "nominalQuota": "8", "lendingLimit": "0"},
                {"name": "memory", "nominalQuota": "64Gi", "lendingLimit": "0"},
            ],
        )
        self.assertNotIn("lendingLimit", reconcile_quota.resources(chips=4)[0])


if __name__ == "__main__":
    unittest.main()
