"""Kimi-K3's multimodal model under DP attention with TPU compilation: TP4, two
attention-DP groups of attention TP 2.

Under DP attention every rank runs the MoE layers together, so every rank must
run the same compiled program each step. When one group prefills an image, the
other groups take the input_embeds path too (``is_tpu_mm_extend_in_batch``),
and the vision tower runs collective-free on the group that owns the image.
Requests go in four at a time, an image among them, so one group prefills an
image while the other prefills or decodes text.

The reference is the same checkpoint (kimi_k3_tiny_harness.py) at TP4 without
DP attention. Greedy continuations of this random model part ways within a
few tokens at any other attention TP, so the comparison is teacher-forced:
each server continues the same prefixes (cut points through every prompt,
including a three-chunk one) by two tokens. The first token checks the
prefill, the second, where the first agrees, one decode step. Calibration on
tpu7x over these 96 prefixes:

    layout                     token 1   token 2 | token 1
    TP4 again                  100.0%    100.0%
    TP2 / EP2 (no DP)           70.8%     83.8%
    DP attention                70.8%     77.9%
    DP attention, concurrent    71.9%     76.8%
    another prompt's prefix      0.0%      --

so DP attention must stay well clear of the wrong-context floor.
"""

import tempfile
import threading
import unittest

import requests
from kimi_k3_tiny_harness import (
    PROMPTS,
    assert_no_serving_recompile,
    build_model,
    chat_with_image,
    launch,
    read_log,
)

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_tpu_ci(
    est_time=900,
    suite="nightly-tpu-4-v7",
    nightly=True,
    labels=["singlehost", "kimi-k3", "dpa", "multimodal", "compile"],
)

PORT = 21000
BASE_URL = f"http://127.0.0.1:{PORT}"
DP_ATTENTION_ARGS = ["--enable-dp-attention", "--dp-size", "2", "--enable-dp-lm-head"]
MIN_AGREEMENT = 0.55
WIDTH = 4


def _prefixes() -> list[list[int]]:
    prefixes = []
    for prompt in PROMPTS:
        n = len(prompt)
        for cut in sorted({max(1, (n * i) // 24) for i in range(1, 25)}):
            prefixes.append(prompt[:cut])
    return prefixes


PREFIXES = _prefixes()


def _next_two(prefix: list[int]) -> tuple[int, int]:
    response = requests.post(
        f"{BASE_URL}/generate",
        json={
            "input_ids": prefix,
            "sampling_params": {
                "max_new_tokens": 2,
                "temperature": 0,
                "ignore_eos": True,
            },
        },
        timeout=600,
    )
    response.raise_for_status()
    return tuple(response.json()["output_ids"])


def _serve_all(model: str, with_image: bool) -> tuple[list, object]:
    """Every prefix, ``WIDTH`` in flight at once, plus the image request."""
    results: list = [None] * len(PREFIXES)
    image: list = []
    errors: list = []
    lock = threading.Lock()
    next_index = [0]

    def worker():
        try:
            while True:
                with lock:
                    i = next_index[0]
                    next_index[0] += 1
                if i >= len(PREFIXES):
                    return
                results[i] = _next_two(PREFIXES[i])
        except Exception as e:  # surfaced on the main thread
            errors.append(e)

    def image_worker():
        try:
            image.append(chat_with_image(BASE_URL, model))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(WIDTH)]
    if with_image:
        threads.append(threading.Thread(target=image_worker))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    return results, image[0] if image else None


class TestKimiK3TinyDPAttention(CustomTestCase):
    def test_dp_attention_agrees_with_the_reference(self):
        model_dir = tempfile.TemporaryDirectory(prefix="kimi_k3_tiny_")
        self.addCleanup(model_dir.cleanup)
        build_model(model_dir.name)
        model = model_dir.name
        logs = tempfile.mkdtemp(prefix="kimi_k3_dpa_logs_")

        dpa_log = f"{logs}/dp_attention.log"
        process, log = launch(model, PORT, DP_ATTENTION_ARGS, dpa_log)
        try:
            served, image = _serve_all(model, with_image=True)
        finally:
            kill_process_tree(process.pid, wait_timeout=60)
            log.close()
        assert_no_serving_recompile(self, read_log(dpa_log), "DP attention server")

        process, log = launch(model, PORT, [], f"{logs}/reference.log")
        try:
            reference = [_next_two(prefix) for prefix in PREFIXES]
            reference_image = chat_with_image(BASE_URL, model)
        finally:
            kill_process_tree(process.pid, wait_timeout=60)
            log.close()

        first = [got[0] == want[0] for got, want in zip(served, reference)]
        second = [
            got[1] == want[1]
            for got, want in zip(served, reference)
            if got[0] == want[0]
        ]
        first_rate = sum(first) / len(first)
        second_rate = sum(second) / max(1, len(second))
        print(
            f"token 1 agreement {sum(first)}/{len(first)} = {first_rate:.3f}; "
            f"token 2 given token 1 {sum(second)}/{len(second)} = {second_rate:.3f}"
        )
        self.assertGreaterEqual(first_rate, MIN_AGREEMENT)
        self.assertGreaterEqual(second_rate, MIN_AGREEMENT)
        text, image_tokens = image
        self.assertTrue(text)
        self.assertGreater(image_tokens, 0)
        self.assertEqual(image_tokens, reference_image[1])


if __name__ == "__main__":
    unittest.main()
