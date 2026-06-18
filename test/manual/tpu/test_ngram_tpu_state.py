"""Instrumented AC-4 / AC-5 validation for NGram on TPU.

Launches an NGram TPU server with the env-gated state recorder enabled
(SGLANG_NGRAM_TPU_STATE_GUARD=1). The recorder lives in NgramVerifyInput.verify()
and, for every verify step, asserts the cache/bookkeeping invariants in-process
(seq_lens / seq_lens_cpu advance by num_accept_tokens; accepted out_cache_loc is
wired into req_to_token[req_pool, old:new]; kv_committed_len / kv_allocated_len
advance by the accepted length) and appends a per-verify event record
(batch_size, rids, num_correct_drafts, num_accept_tokens, finished) to
SGLANG_NGRAM_TPU_STATE_FILE.

A concurrent workload then drives the scheduler to form bs>1 verify batches and to
exercise zero-match / accept / finished-request patterns. The test asserts on the
recorded events:
  - the recorder produced events (so the in-process invariants held during a real
    TPU decode; a violation would crash the scheduler and the requests would fail),
  - at least one verify event had batch_size >= 2 (real bs>1; fails, never skips),
  - accept patterns cover num_correct_drafts == 0 (zero-match) and > 0 (accept),
  - at least one event has a finished request while the batch had others (AC-4
    finished-request freeing), and
  - the concurrent non-greedy request is rejected request-scoped (HTTP 400) while
    the concurrent greedy requests succeed (AC-5).

Runs in float32 so bs>1 verify shapes compile (the bf16 bs=4/q=16 shape hits a
pallas SMEM compile limit; bs=2/q=8 in float32 compiles).

Usage:
    SGLANG_PLATFORM=tpu python test/manual/tpu/test_ngram_tpu_state.py
"""

import concurrent.futures
import json
import os
import tempfile
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

from _ngram_tpu_common import MODEL, skip_if_no_tpu

# Repetitive greedy prompts: the per-request n-gram corpus learns the repeated
# pattern and proposes it, yielding accept events; early steps yield zero-match.
_REPEAT_PROMPTS = [
    "The answer is yes. The answer is yes. The answer is yes. The answer is",
    "a b c a b c a b c a b c a b c a b c a b c a b",
    "one two three one two three one two three one two",
]


def _gen(rid, prompt, max_new_tokens, **sampling_overrides):
    sampling = {"temperature": 0, "max_new_tokens": max_new_tokens}
    sampling.update(sampling_overrides)
    try:
        resp = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={"text": prompt, "sampling_params": sampling, "rid": rid},
            timeout=300,
        )
        return rid, (resp.status_code, resp.text)
    except Exception as exc:  # pragma: no cover - surfaced via assertions
        return rid, (f"EXC:{exc!r}", "")


@unittest.skipIf(skip_if_no_tpu(), "torch_tpu not available")
class TestNgramTpuState(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["SGLANG_PLATFORM"] = "tpu"
        fd, cls.state_file = tempfile.mkstemp(prefix="ngram_tpu_state_", suffix=".jsonl")
        os.close(fd)
        os.remove(cls.state_file)  # recorder appends; start clean

        env = dict(os.environ)
        env["SGLANG_NGRAM_TPU_STATE_GUARD"] = "1"
        env["SGLANG_NGRAM_TPU_STATE_FILE"] = cls.state_file

        cls.process = popen_launch_server(
            MODEL,
            DEFAULT_URL_FOR_TEST,
            timeout=600,
            env=env,
            other_args=[
                "--device", "tpu",
                "--dtype", "float32",
                "--attention-backend", "pallas_tpu_rpa_v3",
                "--page-size", "1",
                "--grammar-backend", "none",
                "--mem-fraction-static", "0.4",
                # Cap concurrency at 2 and use draft_token_num=2 so a bs=2 verify
                # batch has combined query length q = bs * draft = 4 — the same
                # safe shape as bs=1/draft=4. Larger q (e.g. bs=2/draft=4 -> q=8 or
                # bs=4/draft=4 -> q=16) overflows the pallas attention SMEM at
                # compile time (a pre-existing backend limit, reproduces on the
                # non-spec server).
                "--max-running-requests", "2",
                "--speculative-algorithm", "NGRAM",
                "--speculative-num-draft-tokens", "2",
                "--speculative-ngram-max-bfs-breadth", "1",
                "--decode-log-interval", "8",
            ],
        )
        try:
            jobs = []
            # Long greedy requests (overlap -> bs>1, accept patterns).
            for i, prompt in enumerate(_REPEAT_PROMPTS):
                jobs.append((f"g{i}", prompt, 24, {}))
            # A short greedy request that finishes while others continue.
            jobs.append(("gshort", "Hello world", 2, {}))
            # A concurrent non-greedy request (must be rejected request-scoped).
            jobs.append(("ng", "Tell me a story", 16, {"temperature": 0.8, "top_k": 50}))

            with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as ex:
                futs = [
                    ex.submit(_gen, rid, prompt, mnt, **ov)
                    for (rid, prompt, mnt, ov) in jobs
                ]
                cls.statuses = dict(f.result() for f in futs)

            cls.events = []
            if os.path.exists(cls.state_file):
                with open(cls.state_file) as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            cls.events.append(json.loads(line))
        finally:
            kill_process_tree(cls.process.pid)

    def _status(self, rid):
        return self.statuses.get(rid, (None, ""))[0]

    def test_recorder_invariants_held(self):
        # Non-empty events => the in-process invariant asserts (seq_lens advance,
        # req_to_token wiring, kv committed/allocated, and freed-slot delta) passed
        # during a real TPU decode (a violation raises in verify() and crashes the
        # scheduler, which would fail the requests / leave the file empty).
        self.assertGreater(len(self.events), 0, "no verify events recorded")

    def test_batch_size_gt_one_formed(self):
        max_bs = max(e["batch_size"] for e in self.events)
        self.assertGreaterEqual(
            max_bs, 2, f"scheduler never formed bs>1 (max batch_size={max_bs})"
        )

    def test_unaccepted_draft_slots_freed(self):
        # The in-process recorder already asserts the exact free delta per event;
        # here require that at least one event actually freed rejected draft slots.
        self.assertTrue(
            any(e.get("freed", 0) > 0 for e in self.events),
            "no event freed unaccepted draft slots",
        )

    def test_zero_match_and_partial_accept_patterns(self):
        all_ncd = [n for e in self.events for n in e["num_correct_drafts"]]
        draft = self.events[0]["draft_token_num"]
        self.assertIn(0, all_ncd, "no zero-match (num_correct_drafts==0) event observed")
        self.assertTrue(
            any(0 < n < draft for n in all_ncd),
            f"no partial-accept (0 < num_correct_drafts < {draft}) event observed",
        )

    def test_finished_request_in_mixed_batch(self):
        # A request finished while another in the SAME bs>=2 verify batch continued.
        self.assertTrue(
            any(
                e["batch_size"] >= 2 and any(e["finished"]) and not all(e["finished"])
                for e in self.events
            ),
            "no bs>=2 event with one request finished and another continuing",
        )

    def test_same_batch_greedy_with_spec_evidence(self):
        greedy = {"g0", "g1", "g2"}
        # >=1 verify event with batch_size>=2 containing >=2 of the greedy rids.
        self.assertTrue(
            any(
                e["batch_size"] >= 2 and len(greedy.intersection(e["rids"])) >= 2
                for e in self.events
            ),
            "no bs>=2 event co-batching >=2 of {g0,g1,g2}",
        )
        # Per-request accept evidence: some greedy rid accepted a draft (ncd>0).
        self.assertTrue(
            any(
                rid in greedy and ncd > 0
                for e in self.events
                for rid, ncd in zip(e["rids"], e["num_correct_drafts"])
            ),
            "no greedy request with positive accepted-draft evidence",
        )
        # Each long greedy request participated in at least one verify event.
        seen = {rid for e in self.events for rid in e["rids"]}
        for rid in greedy:
            self.assertIn(rid, seen, f"greedy request {rid} never reached verify")

    def test_non_greedy_rejected_others_succeed(self):
        ng_status, ng_text = self.statuses.get("ng", (None, ""))
        self.assertEqual(ng_status, 400, f"statuses={self.statuses}")
        self.assertIn("greedy", ng_text.lower(), f"unexpected error body: {ng_text}")
        for rid in ("g0", "g1", "g2", "gshort"):
            self.assertEqual(self._status(rid), 200, f"statuses={self.statuses}")


if __name__ == "__main__":
    unittest.main()
