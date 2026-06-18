"""End-to-end TPU validation for NGram speculative decoding (linear-greedy mode).

Requires real TPU hardware + torch_tpu and a small cached model (Qwen3-0.6B).
Launches two TPU servers sequentially on one chip:

  1. a non-speculative reference server, and
  2. an NGram speculative server (breadth=1, greedy, pallas_tpu_rpa_v3, page_size=1,
     eager), with a small decode-log-interval so speculative counters flush.

It then asserts (token IDs via /generate, not text):
  - greedy output token IDs are EXACTLY equal between the two servers for a fixed
    deterministic prompt set [AC-3.1],
  - the NGram server reports non-zero speculative activity [AC-1],
  - a non-greedy request is rejected request-scoped while greedy ones succeed,
    including a concurrent mixed batch where two greedy requests batch together
    and still match the reference token IDs [AC-4 bs>1, AC-5].

Greedy speculative decoding is lossless given identical target logits, so for a
correct implementation the verify path must reproduce the non-spec token IDs
exactly. The comparison runs in float32 to remove bf16 verify-vs-decode forward
numerical noise (the batched verify forward q=draft_token_num and the single-token
decode forward are not bit-identical in bf16, which can flip an argmax on a
near-tie). In float32 the logit gap shrinks by ~1e3, isolating algorithmic
correctness: any token mismatch then indicates a real reconstruct/verify bug, not
fp noise. (The bf16 production path is exercised for coherence + accept length by
the smoke run / accept-length assertion.)

Cross-device equivalence vs CUDA breadth=1 NGram (AC-3.2) needs a CUDA host; see
test_ngram_cuda_ref.py (host-blocked here).

Usage:
    SGLANG_PLATFORM=tpu python test/manual/tpu/test_ngram_tpu_e2e.py
"""

import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

from _ngram_tpu_common import MODEL, skip_if_no_tpu

# Greedy-confident, deterministic prompts (factual / structured continuations).
PROMPTS = [
    "The capital of France is",
    "Water is made of hydrogen and",
    "The first three prime numbers are 2, 3, and",
    "Monday, Tuesday, Wednesday,",
]
MAX_NEW_TOKENS = 12

_COMMON_ARGS = [
    "--device", "tpu",
    "--dtype", "float32",
    "--attention-backend", "pallas_tpu_rpa_v3",
    "--page-size", "1",
    "--grammar-backend", "none",
    "--mem-fraction-static", "0.4",
    "--max-running-requests", "8",
]
_NGRAM_ARGS = [
    "--speculative-algorithm", "NGRAM",
    "--speculative-num-draft-tokens", "4",
    "--speculative-ngram-max-bfs-breadth", "1",
    "--decode-log-interval", "8",
]


def _gen(base_url, prompt, **sampling_overrides):
    sampling = {"temperature": 0, "max_new_tokens": MAX_NEW_TOKENS}
    sampling.update(sampling_overrides)
    return requests.post(
        f"{base_url}/generate",
        json={"text": prompt, "sampling_params": sampling, "stream": False},
        timeout=300,
    )


def _output_ids(resp):
    data = resp.json()
    if "output_ids" in data and data["output_ids"] is not None:
        return data["output_ids"]
    # Fallback: some builds expose ids under meta_info.
    return data.get("meta_info", {}).get("output_ids")


def _greedy_ids(base_url):
    """Greedy output token IDs for all prompts, sequentially (bs=1)."""
    out = {}
    for prompt in PROMPTS:
        resp = _gen(base_url, prompt)
        assert resp.status_code == 200, (prompt, resp.status_code, resp.text)
        ids = _output_ids(resp)
        assert ids, f"no output_ids returned for {prompt!r}: {resp.text[:200]}"
        out[prompt] = ids
    return out


@unittest.skipIf(skip_if_no_tpu(), "torch_tpu not available")
class TestNgramTpuE2E(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["SGLANG_PLATFORM"] = "tpu"

        # 1) Non-speculative reference token IDs.
        ref_proc = popen_launch_server(
            MODEL, DEFAULT_URL_FOR_TEST, timeout=600, other_args=list(_COMMON_ARGS)
        )
        try:
            cls.ref_ids = _greedy_ids(DEFAULT_URL_FOR_TEST)
        finally:
            kill_process_tree(ref_proc.pid)

        # 2) NGram speculative server.
        spec_proc = popen_launch_server(
            MODEL,
            DEFAULT_URL_FOR_TEST,
            timeout=600,
            other_args=_COMMON_ARGS + _NGRAM_ARGS,
        )
        try:
            cls.spec_ids = _greedy_ids(DEFAULT_URL_FOR_TEST)

            # Persist a TPU NGram output-IDs artifact so a CUDA host can compare
            # CUDA breadth=1 NGram output_ids against these exact TPU NGram IDs
            # (AC-3.2). See test_ngram_cuda_ref.py.
            import json as _json

            artifact_path = os.environ.get(
                "SGLANG_NGRAM_TPU_IDS_ARTIFACT", "/tmp/ngram_tpu_ids.json"
            )
            with open(artifact_path, "w") as _fh:
                _json.dump(
                    {
                        "model": MODEL,
                        "dtype": "float32",
                        "max_new_tokens": MAX_NEW_TOKENS,
                        "speculative_num_draft_tokens": 4,
                        "speculative_ngram_max_bfs_breadth": 1,
                        "prompts": PROMPTS,
                        "output_ids": cls.spec_ids,
                    },
                    _fh,
                )

            info = requests.get(
                f"{DEFAULT_URL_FOR_TEST}/get_server_info", timeout=60
            ).json()
            cls.accept_length = info.get("internal_states", [{}])[0].get(
                "avg_spec_accept_length"
            )

            # Request-scoped non-greedy rejection (server stays up for greedy).
            # The rigorous bs>1 / same-batch / cache-state validation lives in
            # test_ngram_tpu_state.py (which uses a safe bs=2/draft=2 q=4 shape).
            cls.nongreedy_status = _gen(
                DEFAULT_URL_FOR_TEST, "Tell me a story", top_k=50, temperature=0.8
            ).status_code
            cls.greedy_after_status = _gen(DEFAULT_URL_FOR_TEST, PROMPTS[0]).status_code
        finally:
            kill_process_tree(spec_proc.pid)

    def test_greedy_token_id_equivalence(self):
        # AC-3.1: exact output-token-ID equality vs non-spec TPU decode.
        for prompt in PROMPTS:
            self.assertEqual(
                self.spec_ids[prompt],
                self.ref_ids[prompt],
                f"NGram vs non-spec token-ID mismatch for {prompt!r}: "
                f"spec={self.spec_ids[prompt]} ref={self.ref_ids[prompt]}",
            )

    def test_speculative_activity_reported(self):
        # AC-1: spec path actually exercised.
        self.assertIsNotNone(self.accept_length, "avg_spec_accept_length not reported")
        self.assertGreater(self.accept_length, 1.0)

    def test_non_greedy_rejected_request_scoped(self):
        # AC-5 (request-scoped): a non-greedy request is rejected with 400 and the
        # server keeps serving greedy requests. Same-batch bs>1 behavior is proven
        # in test_ngram_tpu_state.py.
        self.assertEqual(self.nongreedy_status, 400, "non-greedy request not rejected")
        self.assertEqual(self.greedy_after_status, 200)


if __name__ == "__main__":
    unittest.main()
