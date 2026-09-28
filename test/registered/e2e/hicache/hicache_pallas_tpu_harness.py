from sglang.test.ci.ci_register import register_tpu_ci

# Shared machinery only; the dashboard rows are the thin per-profile files
# beside this one. Registered-but-disabled because every .py under
# test/registered/ must carry a registration.
register_tpu_ci(
    est_time=0,
    suite="nightly-tpu-4-v7",
    disabled="helper module, no tests",
)

"""E2E correctness for TPU KV offloading, shared across attention backends.

A prefix served via HOST RELOAD must produce byte-identical greedy output to the
same prefix served from the DEVICE cache. Using one server, the prefix's KV is
computed exactly once (first prefill, written through to host), so the device-hit
and the host-reload read the SAME cached KV -- they match iff the host round-trip
preserves it. (Comparing reload against a fresh recompute instead would diverge on
radix-cache extend-vs-prefill float numerics, which is unrelated to offloading.)

Sequence on a small-pool offload server: prefill a target prompt, re-send it for a
DEVICE hit, send fillers to evict it to host, re-send it for a HOST reload, then
compare. We assert each cache path actually fired via cached_tokens.

The comparison is between two reads on one server, so it is content-independent:
nothing here is calibrated against a constant that a model or runtime change
could invalidate.
"""

import json
import os
import sys
import unittest
import urllib.request

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
)

MODEL = "Qwen/Qwen3-0.6B"

# Device pool 4096 tok. Target ~800 tok; 8 fillers x ~800 = ~6400 > device pool
# at either page size, so the target evicts to host (host pool = 3x holds it).
TARGET_WORDS = 800
FILLER_WORDS = 800
NUM_FILLERS = 8
MAX_NEW = 16
OFFLOAD_POOL = 4096
HICACHE_RATIO = 3

# Set by main() to the PROFILES key the test case should launch.
_PROFILE_ENV = "HICACHE_TPU_PROFILE"

# Per-profile server shape. The only axis that varies is which attention
# backend serves the KV, and therefore which physical KV layout the host
# round-trip has to preserve.
#
#   rpa_v3          token-major (HEAD_ALONG_SUBLANE), page 64
#   seq_along_lane  head-major (SEQ_ALONG_LANE), page 256 -- the batched
#                   backend's floor, since that layout tiles the page axis
#                   onto the 128 lanes
PROFILES = {
    "rpa_v3": {"attention_backend": "pallas_tpu_rpa_v3", "page_size": 64, "env": {}},
    "seq_along_lane": {
        "attention_backend": "pallas_tpu_rpa_batched",
        "page_size": 256,
        "env": {"SGLANG_TPU_USE_SEQ_ALONG_LANE_KV": "1"},
    },
}


def offload_args(profile: dict) -> list:
    return [
        "--device",
        "tpu",
        "--attention-backend",
        profile["attention_backend"],
        "--enable-torch-compile",
        "--page-size",
        str(profile["page_size"]),
        "--mem-fraction-static",
        "0.8",
        "--max-total-tokens",
        str(OFFLOAD_POOL),
        "--enable-hierarchical-cache",
        "--hicache-io-backend",
        "torch_tpu",
        # The TPU host pool mirrors the fused device pool; the io backend
        # rejects any other layout at startup.
        "--hicache-mem-layout",
        "page_first_fused",
        "--hicache-ratio",
        str(HICACHE_RATIO),
        "--enable-cache-report",
        "--trust-remote-code",
    ]


# Common English words (~1 token each) so word count ~= token count; distinct
# header per prompt so no prefix is shared across prompts.
_VOCAB = (
    "the of and to in a is that for it as was with on be by at this from but not are or "
    "his they she had which were we when your can said there use each their time will how "
    "other about many then them these so some her would make like him into has two more "
    "very after our just name good through where help much before line right too mean old"
).split()


def _make_prompt(i: int, nwords: int) -> str:
    seed = (i + 1) * 2654435761 & 0xFFFFFFFF
    words = []
    for _ in range(nwords):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        words.append(_VOCAB[seed % len(_VOCAB)])
    return f"Document {i} header alpha{i} beta{i * 7}. " + " ".join(words)


def _generate(base_url: str, prompt: str, max_new: int = MAX_NEW):
    body = json.dumps(
        {
            "text": prompt,
            "sampling_params": {"max_new_tokens": max_new, "temperature": 0.0},
        }
    ).encode()
    req = urllib.request.Request(
        base_url + "/generate", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())
    ctd = out.get("meta_info", {}).get("cached_tokens_details") or {}
    return out["text"], ctd.get("device", 0), ctd.get("host", 0)


@unittest.skipUnless(is_in_ci(), "TPU-only e2e; run on the TPU CI runner")
class TestHiCachePallasTpuOffload(CustomTestCase):
    def test_host_reload_matches_device_hit(self):
        url = DEFAULT_URL_FOR_TEST
        proc = popen_launch_server(
            MODEL,
            url,
            DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=offload_args(PROFILES[os.environ[_PROFILE_ENV]]),
            device="tpu",
        )
        try:
            target = _make_prompt(0, TARGET_WORDS)
            fillers = [_make_prompt(100 + i, FILLER_WORDS) for i in range(NUM_FILLERS)]

            _generate(url, target)  # prefill + write-through to host
            dev_text, dev_dev, dev_host = _generate(
                url, target
            )  # device hit (resident)
            for f in fillers:
                _generate(url, f, max_new=1)  # fill device -> evict target
            host_text, host_dev, host_host = _generate(url, target)  # host reload
        finally:
            kill_process_tree(proc.pid)

        self.assertGreater(
            dev_dev, 0, "expected a device cache hit on the immediate re-send"
        )
        self.assertEqual(
            dev_host, 0, "immediate re-send should hit device, not reload from host"
        )
        self.assertGreater(
            host_host, 0, "target was not reloaded from host after eviction"
        )
        # Host-reloaded KV must give byte-identical greedy output to the device hit.
        self.assertEqual(host_text, dev_text)


def _run_case(case: type) -> None:
    """Run one TestCase and exit with its verdict.

    Load the class explicitly rather than calling unittest.main(): the thin
    per-profile files are what CI executes, so ``__main__`` is the thin file,
    which holds no TestCase. Bare discovery there finds nothing and reports
    "Ran 0 tests" as a pass-shaped failure.
    """
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(case)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


def main(profile_key: str) -> None:
    """Run the offload suite for one attention-backend profile."""
    if profile_key not in PROFILES:
        raise SystemExit(f"unknown profile {profile_key!r}; have {sorted(PROFILES)}")
    os.environ[_PROFILE_ENV] = profile_key
    for k, v in PROFILES[profile_key]["env"].items():
        os.environ.setdefault(k, v)
    _run_case(TestHiCachePallasTpuOffload)


if __name__ == "__main__":
    # Not a CI row (registration is disabled); runs the rpa_v3 profile locally.
    os.environ.setdefault(_PROFILE_ENV, "rpa_v3")
    unittest.main()
