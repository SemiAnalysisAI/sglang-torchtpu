"""Kimi-K3 data-parallel vision encoding on TPU (--mm-enable-dp-encoder).

Each TP rank encodes its share of a request's images on the collective-free
compiled vision runner and the projected features meet on the host (a gloo
all-gather): no device collective, which is what hung the device-gather DP
runner on TPU. Every image's compute is the same program on any rank, so a
multi-image request must give exactly the replicated server's output.
"""

import base64
import io
import os
import sys
import tempfile
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_tpu_kimi_k3_tiny_singlehost_compile as smoke  # noqa: E402

register_tpu_ci(
    est_time=420,
    suite="nightly-tpu-4-v7",
    nightly=True,
    labels=["singlehost", "kimi-k3", "multimodal", "compile"],
)

# Three grids in one request: square, wide, and one in another bucket.
IMAGE_SIZES = [(224, 224), (448, 224), (308, 196)]


def _image_url(width: int, height: int) -> str:
    from PIL import Image

    image = Image.new("RGB", (width, height))
    image.putdata(
        [
            ((x * 7 + width) % 256, (y * 5) % 256, ((x + y) * 3) % 256)
            for y in range(height)
            for x in range(width)
        ]
    )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _chat(model: str) -> tuple[str, int]:
    content = [
        {"type": "image_url", "image_url": {"url": _image_url(w, h)}}
        for w, h in IMAGE_SIZES
    ] + [{"type": "text", "text": "Compare the images."}]
    response = requests.post(
        f"{DEFAULT_URL_FOR_TEST}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": smoke.MAX_NEW_TOKENS,
            "temperature": 0,
        },
        timeout=600,
    )
    response.raise_for_status()
    body = response.json()
    return (
        body["choices"][0]["message"]["content"],
        body["usage"]["prompt_tokens_details"]["image_tokens"],
    )


def _serve(model: str, extra: list[str]) -> tuple[list, str]:
    log = tempfile.NamedTemporaryFile(mode="w+", suffix=".log")
    process = popen_launch_server(
        model,
        DEFAULT_URL_FOR_TEST,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=smoke.SERVER_ARGS + ["--enable-torch-compile"] + extra,
        env={
            "TORCH_LOGS": "recompiles",
            "SGLANG_TPU_VIT_MAX_PRECOMPILE_PATCHES": "1024",
        },
        return_stdout_stderr=(log, log),
        device="tpu",
    )
    try:
        # Twice: the second pass runs on warm programs.
        outputs = [_chat(model), _chat(model)]
        with open(log.name, errors="replace") as f:
            return outputs, f.read()
    finally:
        kill_process_tree(process.pid, wait_timeout=60)
        log.close()


class TestKimiK3TinyDPEncoder(CustomTestCase):
    def test_dp_encoder_matches_replicated_encoding(self):
        model_dir = tempfile.TemporaryDirectory(prefix="kimi_k3_tiny_")
        self.addCleanup(model_dir.cleanup)
        smoke._download_model(model_dir.name)
        replicated, _ = _serve(model_dir.name, [])
        data_parallel, log = _serve(model_dir.name, ["--mm-enable-dp-encoder"])
        served = log[log.rfind(smoke.PRECOMPILE_DONE) :]
        self.assertNotIn("Recompiling function", served)
        self.assertNotIn("Detected recompile", served)
        self.assertGreater(replicated[0][1], 0)
        self.assertEqual(data_parallel, replicated)


if __name__ == "__main__":
    unittest.main()
