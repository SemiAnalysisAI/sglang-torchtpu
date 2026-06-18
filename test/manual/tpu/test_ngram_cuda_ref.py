"""AC-3.2: TPU NGram vs CUDA breadth=1 NGram exact token-ID equivalence.

This is the CUDA-host half of AC-3. The TPU e2e (test_ngram_tpu_e2e.py) writes a
JSON artifact of exact TPU NGram greedy output_ids (model, dtype=float32, prompts,
sampling, spec args). On a CUDA host, copy that artifact here and run this test:
it launches ONLY a breadth=1 greedy CUDA NGram server with the same model /
prompts / sampling / spec args (float32) and asserts the CUDA NGram output_ids
equal the TPU NGram output_ids from the artifact, per prompt.

It skips when no CUDA device is present (e.g. on a TPU-only host) or when the
artifact is absent. Point it at the artifact with SGLANG_NGRAM_TPU_IDS_ARTIFACT
(default /tmp/ngram_tpu_ids.json).

NOTE: exact token IDs are not guaranteed bit-identical across different hardware
under floating point even at float32; if this fails purely on late near-tie
divergence, the failure message prints the per-prompt diff so the cross-device
tolerance question can be decided explicitly (it must not be silently relaxed).

Usage (on a CUDA host, after copying the artifact):
    SGLANG_NGRAM_TPU_IDS_ARTIFACT=/path/ngram_tpu_ids.json \
        python test/manual/tpu/test_ngram_cuda_ref.py
"""

import json
import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

ARTIFACT = os.environ.get("SGLANG_NGRAM_TPU_IDS_ARTIFACT", "/tmp/ngram_tpu_ids.json")


def _no_cuda():
    try:
        import torch

        return not torch.cuda.is_available()
    except Exception:
        return True


def _load_artifact():
    if not os.path.exists(ARTIFACT):
        return None
    with open(ARTIFACT) as fh:
        return json.load(fh)


def _gen_ids(prompt, max_new_tokens):
    resp = requests.post(
        f"{DEFAULT_URL_FOR_TEST}/generate",
        json={
            "text": prompt,
            "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens},
            "stream": False,
        },
        timeout=300,
    )
    assert resp.status_code == 200, (prompt, resp.status_code, resp.text)
    return resp.json()["output_ids"]


@unittest.skipIf(_no_cuda(), "CUDA device not available")
@unittest.skipUnless(_load_artifact() is not None, f"TPU IDs artifact not found at {ARTIFACT}")
class TestNgramTpuVsCudaNgram(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifact = _load_artifact()
        spec = popen_launch_server(
            cls.artifact["model"],
            DEFAULT_URL_FOR_TEST,
            timeout=600,
            other_args=[
                "--dtype", cls.artifact.get("dtype", "float32"),
                "--speculative-algorithm", "NGRAM",
                "--speculative-num-draft-tokens",
                str(cls.artifact.get("speculative_num_draft_tokens", 4)),
                "--speculative-ngram-max-bfs-breadth",
                str(cls.artifact.get("speculative_ngram_max_bfs_breadth", 1)),
            ],
        )
        try:
            mnt = cls.artifact["max_new_tokens"]
            cls.cuda_ids = {p: _gen_ids(p, mnt) for p in cls.artifact["prompts"]}
        finally:
            kill_process_tree(spec.pid)

    def test_tpu_ngram_ids_match_cuda_ngram_ids(self):
        # artifact["output_ids"] is a {prompt: output_ids} mapping.
        tpu_ids = self.artifact["output_ids"]
        for prompt in self.artifact["prompts"]:
            self.assertEqual(
                self.cuda_ids[prompt],
                tpu_ids[prompt],
                f"TPU vs CUDA NGram token-ID mismatch for {prompt!r}: "
                f"cuda={self.cuda_ids[prompt]} tpu={tpu_ids[prompt]}",
            )


if __name__ == "__main__":
    unittest.main()
