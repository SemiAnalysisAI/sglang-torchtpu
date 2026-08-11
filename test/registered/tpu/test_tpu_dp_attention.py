"""DP-attention correctness on TPU: tp_size == dp_size (attn_tp_size == 1).

The gap this closes: every existing DP-attention test on TPU checks
*self-consistency* (do repeated / cross-rank requests agree?), and the feature
combos run dp_size < tp_size. Nothing checked that a DP rank produces the
*right* tokens at dp_size == tp_size, so `Qwen3_5TextForCausalLM` shipped with a
lm_head sharded over the full TP group while the LogitsProcessor gathered over
the attention-TP group -- at attn_tp_size == 1 there is no gather at all, every
rank argmaxed its own 1/tp of the vocab row, and the server answered
'!!!!!!!!!!!!!!!!' to every prompt. Deterministic garbage passes a
self-consistency oracle.

So this test asserts CONTENT, and asserts it for enough concurrent requests that
round-robin DP dispatch has to have used every rank.

Usage:
    SGLANG_PLATFORM=tpu python test/registered/tpu/test_tpu_dp_attention.py
"""

import os
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# Runs on the TPU PR gate (single-host 4-chip / "4-v7" = 8 devices).
register_tpu_ci(est_time=420, suite="stage-b-test-tpu-4-v7")

MODEL = "Qwen/Qwen3-0.6B"
DP_SIZE = 8


def _skip_if_no_tpu():
    try:
        import importlib.util

        if importlib.util.find_spec("torch_tpu") is None:
            return True
    except Exception:
        return True
    return False


@unittest.skipIf(_skip_if_no_tpu(), "torch_tpu not available")
class TestTpuDpAttention(CustomTestCase):
    """One DP rank per device, attention data-parallel, MLP still TP-sharded."""

    @classmethod
    def setUpClass(cls):
        os.environ["SGLANG_PLATFORM"] = "tpu"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            MODEL,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--dtype",
                "bfloat16",
                "--device",
                "tpu",
                "--grammar-backend",
                "none",
                "--enable-torch-compile",
                "--tp-size",
                "8",
                "--dp-size",
                str(DP_SIZE),
                "--enable-dp-attention",
                # Required on TPU at every attn_tp_size (tpu_platform validates
                # it): it selects the vocab-parallel group for BOTH the head and
                # the logits gather, and they must agree.
                "--enable-dp-lm-head",
                "--page-size",
                "64",
                "--chunked-prefill-size",
                "512",
                "--context-length",
                "1024",
                # >= 1 running request per rank, so a concurrent wave can spread.
                "--max-running-requests",
                str(2 * DP_SIZE),
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _complete(self, prompt, max_tokens=10):
        resp = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "model": MODEL,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0,
            },
            timeout=120,
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        return resp.json()["choices"][0]["text"]

    def test_completion_coherent(self):
        self.assertIn("Paris", self._complete("The capital of France is"))

    def test_every_rank_answers_correctly(self):
        # Round-robin dispatch over DP ranks: 4x dp_size identical requests in
        # flight cannot all land on one rank. A rank whose logits are wrong (or
        # whose slot in the DP gather buffer is misaligned) shows up here even
        # when rank 0 is fine.
        prompt = "The capital of France is"
        with ThreadPoolExecutor(max_workers=4 * DP_SIZE) as pool:
            texts = list(pool.map(lambda _: self._complete(prompt), range(4 * DP_SIZE)))

        for i, text in enumerate(texts):
            self.assertIn("Paris", text, f"request {i} of {len(texts)}: {text!r}")
        # Greedy decoding: every rank must also agree token for token.
        self.assertEqual(len(set(texts)), 1, f"ranks disagreed: {set(texts)}")

    def test_prefill_on_one_rank_while_others_decode(self):
        # Under MAX_LEN padding (the only mode TPU supports) every rank must run
        # the same forward mode, so a rank holding live decode work is
        # relabelled EXTEND as soon as any other rank starts a prefill. One wave
        # of requests never produces that mix -- every rank prefills, then every
        # rank decodes -- so stagger a second wave into the first one's decode.
        # For mamba-hybrid models this path used to hit
        # "extend-idle conversion expects an empty rank" and kill the scheduler.
        first = [self._complete(p, max_tokens=64) for p in ("Count: 1, 2, 3,",)]
        with ThreadPoolExecutor(max_workers=2 * DP_SIZE) as pool:
            long_wave = pool.submit(
                lambda: [
                    self._complete("Count: 1, 2, 3,", max_tokens=64)
                    for _ in range(DP_SIZE)
                ]
            )
            time.sleep(0.5)  # let the first wave reach steady-state decode
            short_wave = list(
                pool.map(
                    lambda _: self._complete("The capital of France is"),
                    range(DP_SIZE),
                )
            )
            long_texts = long_wave.result()

        for text in short_wave:
            self.assertIn("Paris", text)
        for text in long_texts:
            self.assertEqual(text, first[0], "decode diverged while a rank prefilled")

    def test_mixed_prompt_wave(self):
        # Different prompt lengths in one wave, so ranks pad to different real
        # token counts inside the same cross-rank bucket and some ranks run the
        # idle/dummy batch while others prefill.
        prompts = [
            "The capital of France is",
            "2 + 2 =",
            "Water boils at",
            "One kilogram equals how many grams?",
        ] * DP_SIZE
        with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
            texts = list(pool.map(self._complete, prompts))

        by_prompt = {}
        for prompt, text in zip(prompts, texts):
            self.assertTrue(text.strip(), f"empty completion for {prompt!r}")
            by_prompt.setdefault(prompt, set()).add(text)
        for prompt, variants in by_prompt.items():
            self.assertEqual(
                len(variants), 1, f"{prompt!r} decoded differently: {variants}"
            )


if __name__ == "__main__":
    unittest.main()
