"""Self-test for the V1<->V2 parity harness comparator (benchmark/tpu/spec/parity_v1_v2.py).

Proves the parity contract is enforced: token IDs are authoritative, and a missing
`output_ids` on either side is a hard FAILURE (not a silent fall-back to text
comparison that could hide a tokenization difference). No TPU / server needed.
"""

import importlib.util
import json
import os
import tempfile
import unittest

from sglang.test.test_utils import CustomTestCase

_HARNESS = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "benchmark",
    "tpu",
    "spec",
    "parity_v1_v2.py",
)


def _load_mod():
    spec = importlib.util.spec_from_file_location(
        "parity_v1_v2", os.path.abspath(_HARNESS)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_compare():
    return _load_mod().compare


def _write(records):
    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(records, f)
    f.close()
    return f.name


def _write_trace(lines):
    """lines: list of {"rid":..., "emitted":[[tok,[id1,id2],[lg1,lg2]],...]} dicts."""
    f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
    for ln in lines:
        f.write(json.dumps(ln) + "\n")
    f.close()
    return f.name


class TestParityComparator(CustomTestCase):
    def setUp(self):
        self.compare = _load_compare()

    def test_identical_passes(self):
        a = [{"output_ids": [1, 2, 3], "text": "x"}]
        b = [{"output_ids": [1, 2, 3], "text": "x"}]
        self.assertEqual(self.compare(_write(a), _write(b)), 0)

    def test_token_id_mismatch_fails(self):
        a = [{"output_ids": [1, 2, 3], "text": "x"}]
        b = [{"output_ids": [1, 9, 3], "text": "x"}]
        self.assertEqual(self.compare(_write(a), _write(b)), 1)

    def test_missing_output_ids_is_hard_failure(self):
        # Identical text but one side lacks token ids -> must FAIL (not pass).
        a = [{"output_ids": [1, 2, 3], "text": "same text"}]
        b = [{"output_ids": None, "text": "same text"}]
        self.assertEqual(self.compare(_write(a), _write(b)), 1)

    def test_both_missing_output_ids_is_hard_failure(self):
        a = [{"output_ids": None, "text": "same text"}]
        b = [{"output_ids": None, "text": "same text"}]
        self.assertEqual(self.compare(_write(a), _write(b)), 1)

    def test_request_count_mismatch_fails(self):
        a = [{"output_ids": [1], "text": "x"}, {"output_ids": [2], "text": "y"}]
        b = [{"output_ids": [1], "text": "x"}]
        self.assertEqual(self.compare(_write(a), _write(b)), 1)


class TestNearTieComparator(CustomTestCase):
    """Sensitivity fixture for the fail-closed bf16 near-tie gate.

    A divergence is excused ONLY with direct top-2 logit evidence proving the two
    emitted tokens are mutual 2nd-choices within the margin. Missing trace, a wrong
    token pair, or a too-large gap must FAIL.
    """

    def setUp(self):
        self.f = _load_mod().allow_near_tie_compare
        # V1 picks 20 at pos1; V2 picks 21; identical elsewhere (cascade after).
        self.a = [{"output_ids": [10, 20, 30]}]
        self.b = [{"output_ids": [10, 21, 31]}]

    def _trace(self, rid, p1_top2, p1_lg):
        # pos0 + pos2 are clear wins; pos1 carries the (configurable) tie.
        return [
            {
                "rid": rid,
                "emitted": [
                    [10, [10, 99], [8.0, 1.0]],
                    [p1_top2[0], list(p1_top2), list(p1_lg)],
                    [p1_top2[0] + 10, [p1_top2[0] + 10, 41], [7.0, 1.0]],
                ],
            }
        ]

    def test_within_margin_near_tie_passes(self):
        ta = self._trace("r0", (20, 21), (3.0, 2.7))  # V1: gap 0.3, 2nd=21=B
        tb = self._trace("s0", (21, 20), (3.0, 2.8))  # V2: gap 0.2, 2nd=20=A
        rc = self.f(
            _write(self.a),
            _write(self.b),
            _write_trace(ta),
            _write_trace(tb),
            margin=1.0,
        )
        self.assertEqual(rc, 0)

    def test_margin_above_tolerance_fails(self):
        ta = self._trace("r0", (20, 21), (5.0, 2.0))  # gap 3.0 > margin
        tb = self._trace("s0", (21, 20), (3.0, 2.8))
        rc = self.f(
            _write(self.a),
            _write(self.b),
            _write_trace(ta),
            _write_trace(tb),
            margin=1.0,
        )
        self.assertEqual(rc, 1)

    def test_wrong_token_pair_fails(self):
        # V1's 2nd choice is 77, NOT B=21 -> not a mutual near-tie -> FAIL.
        ta = self._trace("r0", (20, 77), (3.0, 2.9))
        tb = self._trace("s0", (21, 20), (3.0, 2.9))
        rc = self.f(
            _write(self.a),
            _write(self.b),
            _write_trace(ta),
            _write_trace(tb),
            margin=1.0,
        )
        self.assertEqual(rc, 1)

    def test_missing_trace_fails(self):
        ta = self._trace("r0", (20, 21), (3.0, 2.9))
        empty = _write_trace([])  # no trace for V2's diverging request
        rc = self.f(_write(self.a), _write(self.b), _write_trace(ta), empty, margin=1.0)
        self.assertEqual(rc, 1)

    def test_identical_outputs_pass_trivially(self):
        same = [{"output_ids": [10, 20, 30]}]
        ta = self._trace("r0", (20, 21), (3.0, 2.7))
        rc = self.f(
            _write(same), _write(same), _write_trace(ta), _write_trace(ta), margin=1.0
        )
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
