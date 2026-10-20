"""Kimi-K3 Raiden P/D on a tiny random-weight model: one TP4 host per engine.

Kimi-K3 interleaves MLA layers, whose latent KV lives in the paged fused MLA
pool, with KDA layers, whose conv and recurrent state lives in a slot-indexed
Mamba pool. A disaggregated request must carry both to the decode engine. The
prefill pod first serves the requests colocated, then as the prefill engine
behind a mini-LB router; greedy output must match token for token. A decode
engine that resumed from a stale or zero KDA state, or from wrong MLA pages,
continues differently, so the comparison checks both transfers.

Every pod builds the same seeded checkpoint for the tiny architecture (at the
128-lane head dims the TPU kernels take) and loads it normally. Dummy weights
would not do: they depend on each parameter's local shard, so engines at
different TP would run different models.

The decode pod (``CI_PD_ROLE=decode``) only serves; the JobSet ends with the
prefill pod.
"""

import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    popen_launch_server,
)

register_tpu_ci(
    est_time=900,
    suite="tpu-integration",
    nightly=True,
    labels=["pd", "singlehost", "kimi-k3", "multimodal"],
)

TINY_MODEL = "inference-optimization/Kimi-K3-0.40B"
TINY_MODEL_REVISION = "d853649387ffe8f48ce0198a29ac1a44205031f7"
PROCESSOR_SOURCE = "moonshotai/Kimi-K3"
PROCESSOR_SOURCE_REVISION = "f831ab66814297da540d832a5235f8e904f29d06"
MODEL_OVERRIDE = {
    "text_config": {
        "v_head_dim": 128,
        "linear_attn_config": {
            "full_attn_layers": [4, 8],
            "head_dim": 128,
            "kda_layers": [1, 2, 3, 5, 6, 7],
            "num_heads": 8,
            "short_conv_kernel_size": 4,
            "use_full_rank_gate": True,
        },
    }
}
SERVER_ARGS = [
    "--trust-remote-code",
    "--tp-size",
    "4",
    "--ep-size",
    "4",
    "--attention-backend",
    "pallas_tpu_mla",
    "--page-size",
    "128",
    "--context-length",
    "2048",
    "--chunked-prefill-size",
    "512",
    "--max-prefill-tokens",
    "512",
    "--max-running-requests",
    "4",
    "--mem-fraction-static",
    "0.5",
    "--disable-radix-cache",
    "--grammar-backend",
    "none",
    "--enable-torch-compile",
    "--torch-compile-max-bs",
    "4",
    "--tpu-compile-tokens",
    "512",
    "--tpu-compile-bs",
    "1",
    "4",
    # The sampler every TPU P/D gate pins; greedy picks the argmax either way.
    "--sampling-backend",
    "pytorch",
]
PREFILL_PORT = 21000
DECODE_PORT = 21000
BOOTSTRAP_PORT = 21100
LB_PORT = 21200
MAX_NEW_TOKENS = 24
TOP_K = 20


def _self_hostname() -> str:
    """This pod's DNS name. JobSet pods resolve each other only by the
    subdomain-qualified name GKE puts in TPU_WORKER_HOSTNAMES."""
    hostnames = [h for h in os.environ.get("TPU_WORKER_HOSTNAMES", "").split(",") if h]
    return hostnames[0] if hostnames else socket.gethostname()


def _decode_hostname() -> str:
    """The decode pod's DNS name: this pod's with its role segment swapped.

    (pd_singlehost_base has the same helper, but importing it would bring up
    the TPU runtime in this process and leave the servers no chips.)
    """
    hostname = _self_hostname()
    if "-prefill-" not in hostname:
        raise ValueError(f"hostname {hostname!r} has no '-prefill-' segment")
    return hostname.replace("-prefill-", "-decode-")


def _prompt(length: int, seed: int) -> list[int]:
    return [(i * 7919 + seed * 104729) % 50000 + 100 for i in range(length)]


# 1300 tokens span three 512-token prefill chunks, so the KDA state the decode
# receives is the one left after chunked prefill.
PROMPTS = [_prompt(100, 1), _prompt(37, 2), _prompt(300, 3), _prompt(1300, 4)]


def _build_model(root: str) -> None:
    """Write a seeded random checkpoint of the tiny Kimi-K3 at the TPU head
    dims (``MODEL_OVERRIDE``) to ``root``: the same bytes on every pod."""
    import importlib
    import shutil
    import types
    import zlib

    import torch
    from huggingface_hub import hf_hub_download, snapshot_download
    from torch import nn

    hub = snapshot_download(
        TINY_MODEL,
        revision=TINY_MODEL_REVISION,
        allow_patterns=["*.json", "*.py", "*.model"],
        ignore_patterns=["checkpoint-*"],
    )
    package_root = tempfile.mkdtemp(prefix="kimi_k3_tiny_code_")
    package = os.path.join(package_root, "kimi_k3_tiny")
    shutil.copytree(hub, package)
    open(os.path.join(package, "__init__.py"), "w").close()

    # The reference modeling code imports fla (flash-linear-attention), which
    # is CUDA-only. Only two of its modules hold parameters; stand in for them
    # with the same parameter names and shapes, and for its kernels -- used
    # only in forward, which building the weights never runs -- with stubs.
    class ShortConvolution(nn.Conv1d):
        def __init__(self, hidden_size, kernel_size, bias=False, **kwargs):
            super().__init__(
                hidden_size,
                hidden_size,
                kernel_size,
                groups=hidden_size,
                padding=kernel_size - 1,
                bias=bias,
            )

    class FusedRMSNormGated(nn.Module):
        def __init__(self, hidden_size, eps=1e-5, elementwise_affine=True, **kwargs):
            super().__init__()
            self.weight = (
                nn.Parameter(torch.ones(hidden_size)) if elementwise_affine else None
            )

    def kernel(*args, **kwargs):
        raise RuntimeError("fla kernel stand-in called")

    stand_ins = {
        "fla": {},
        "fla.modules": dict(
            ShortConvolution=ShortConvolution, FusedRMSNormGated=FusedRMSNormGated
        ),
        "fla.ops": {},
        "fla.ops.kda": dict(chunk_kda=kernel, fused_recurrent_kda=kernel),
        "fla.ops.utils": {},
        "fla.ops.utils.index": dict(
            prepare_cu_seqlens_from_mask=kernel, prepare_lens_from_mask=kernel
        ),
        "fla.utils": dict(tensor_cache=lambda fn: fn),
    }
    saved = {name: sys.modules.get(name) for name in stand_ins}
    sys.path.insert(0, package_root)
    default_dtype = torch.get_default_dtype()
    try:
        for name, attrs in stand_ins.items():
            module = types.ModuleType(name)
            module.__dict__.update(attrs)
            sys.modules[name] = module
        configuration = importlib.import_module("kimi_k3_tiny.configuration_kimi_k3")
        modeling = importlib.import_module("kimi_k3_tiny.modeling_kimi_k3")
        config = configuration.KimiK3Config.from_pretrained(package)
        text = MODEL_OVERRIDE["text_config"]
        config.text_config.v_head_dim = text["v_head_dim"]
        config.text_config.linear_attn_config = {
            **(config.text_config.linear_attn_config or {}),
            **text["linear_attn_config"],
        }
        torch.set_default_dtype(torch.bfloat16)
        model = modeling.KimiK3ForConditionalGeneration(config)
    finally:
        torch.set_default_dtype(default_dtype)
        sys.path.remove(package_root)
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    # Some parameters are allocated uninitialized; give every one a value
    # from a generator seeded by its name. An engine at another TP sums in
    # another order, so near-ties must not decide anything: the MoE router
    # and the LM head are scaled up so each token's experts, and the greedy
    # token, win by a margin well above bf16 rounding.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "norm" in name and name.endswith("weight"):
                param.fill_(1.0)
            elif name.endswith("bias"):
                param.zero_()
            else:
                if name.endswith("block_sparse_moe.gate.weight"):
                    bound = 1.0
                elif "lm_head" in name:
                    bound = 0.5
                else:
                    bound = 0.02
                generator = torch.Generator().manual_seed(zlib.crc32(name.encode()))
                values = torch.empty(param.shape).uniform_(
                    -bound, bound, generator=generator
                )
                param.copy_(values)
    config.auto_map = json.load(open(os.path.join(hub, "config.json")))["auto_map"]
    model.save_pretrained(root, safe_serialization=True)
    for name in os.listdir(hub):
        if name.endswith((".py", ".model")) or name in (
            "tokenizer_config.json",
            "added_tokens.json",
        ):
            shutil.copy(os.path.join(hub, name), root)
    hf_hub_download(
        PROCESSOR_SOURCE,
        "preprocessor_config.json",
        revision=PROCESSOR_SOURCE_REVISION,
        local_dir=root,
    )


def _image_url() -> str:
    import base64
    import io

    from PIL import Image

    size = 224
    image = Image.new("RGB", (size, size))
    image.putdata(
        [
            ((x * 7) % 256, (y * 5) % 256, ((x + y) * 3) % 256)
            for y in range(size)
            for x in range(size)
        ]
    )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _generate(base_url: str, prompt: list[int]) -> list[int]:
    response = requests.post(
        f"{base_url}/generate",
        json={
            "input_ids": prompt,
            "sampling_params": {
                "max_new_tokens": MAX_NEW_TOKENS,
                "temperature": 0,
                "ignore_eos": True,
            },
        },
        timeout=600,
    )
    response.raise_for_status()
    return response.json()["output_ids"]


def _chat_with_image(base_url: str, model: str) -> tuple[str, int]:
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": _image_url()}},
                        {"type": "text", "text": "Describe the image."},
                    ],
                }
            ],
            "max_tokens": MAX_NEW_TOKENS,
            "temperature": 0,
        },
        timeout=600,
    )
    response.raise_for_status()
    body = response.json()
    image_tokens = body["usage"]["prompt_tokens_details"]["image_tokens"]
    return body["choices"][0]["message"]["content"], image_tokens


def _requests(base_url: str, model: str) -> dict:
    # One request at a time: both servers then decode at batch size 1, so
    # greedy output is comparable token for token.
    return {
        "text": [_generate(base_url, prompt) for prompt in PROMPTS],
        "image": _chat_with_image(base_url, model),
    }


def _token_ranks(base_url: str, prompt: list[int], tokens: list[int]) -> list[int]:
    """Rank of each of ``tokens`` among the server's top candidates when it is
    fed ``prompt + tokens`` (``TOP_K`` when it is not among them)."""
    response = requests.post(
        f"{base_url}/generate",
        json={
            "input_ids": prompt + tokens,
            "sampling_params": {"max_new_tokens": 1, "temperature": 0},
            "return_logprob": True,
            "logprob_start_len": len(prompt) - 1,
            "top_logprobs_num": TOP_K,
        },
        timeout=600,
    )
    response.raise_for_status()
    # Entry j scores the token at position logprob_start_len + j, so entry 0
    # is the prompt's last token and token k of ``tokens`` is entry k + 1.
    tops = response.json()["meta_info"]["input_top_logprobs"]
    ranks = []
    for k, token in enumerate(tokens):
        candidates = [entry[1] for entry in tops[k + 1]]
        ranks.append(candidates.index(token) if token in candidates else TOP_K)
    return ranks


def _read(path: str) -> str:
    with open(path, errors="replace") as f:
        return f.read()


def _follow(path: str, prefix: str) -> None:
    """Mirror a server log to stdout as it grows, so the pod log shows it."""

    def run():
        with open(path, errors="replace") as f:
            while True:
                line = f.readline()
                if line:
                    sys.stdout.write(f"[{prefix}] {line}")
                    sys.stdout.flush()
                else:
                    time.sleep(0.5)

    threading.Thread(target=run, daemon=True).start()


def _launch(
    model: str, port: int, args: list[str], log_path: str, env: dict | None = None
):
    log = open(log_path, "w")
    _follow(log_path, os.path.splitext(os.path.basename(log_path))[0])
    # Bind every interface: the peer engine and the router reach these servers
    # by pod name (popen_launch_server takes --host from this URL).
    process = popen_launch_server(
        model,
        f"http://0.0.0.0:{port}",
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=SERVER_ARGS + args,
        env={"TORCH_LOGS": "recompiles", **(env or {})},
        return_stdout_stderr=(log, log),
        device="tpu",
    )
    return process, log


def _disagg_args(role: str) -> list[str]:
    args = [
        "--disaggregation-mode",
        role,
        "--disaggregation-transfer-backend",
        "raiden",
        # A disaggregated engine's warmup request needs its peer and the router.
        "--skip-server-warmup",
    ]
    if role == "prefill":
        args += ["--disaggregation-bootstrap-port", str(BOOTSTRAP_PORT)]
    else:
        args += ["--disaggregation-decode-retraction-backup", "cpu_tensor"]
    return args


def _assert_no_serving_recompile(test: CustomTestCase, log: str, name: str) -> None:
    # Unfrozen recompiles log "Recompiling function"; frozen ones raise
    # "Detected recompile" and fail the request.
    test.assertNotIn("Detected recompile", log, f"{name} recompiled while serving")


def _wait_healthy(url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if requests.get(f"{url}/health", timeout=5).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(5)
    raise TimeoutError(f"{url} not healthy after {timeout}s")


class TinyPDParity:
    """The P/D-vs-colocated comparison; a variant sets the decode's extra
    server arguments (a mixin, so importing it adds no test)."""

    decode_args: list[str] = []

    def test_disaggregated_output_matches_colocated(self):
        role = os.environ.get("CI_PD_ROLE")
        if role not in ("prefill", "decode"):
            self.skipTest("needs CI_PD_ROLE=prefill or decode (a P/D JobSet)")
        model_dir = tempfile.TemporaryDirectory(prefix="kimi_k3_tiny_")
        self.addCleanup(model_dir.cleanup)
        _build_model(model_dir.name)
        model = model_dir.name
        logs = tempfile.mkdtemp(prefix="kimi_k3_pd_logs_")

        if role == "decode":
            process, log = _launch(
                model,
                DECODE_PORT,
                _disagg_args("decode") + self.decode_args,
                f"{logs}/decode.log",
            )
            try:
                # Serve until the prefill pod finishes and the JobSet ends.
                process.wait()
            finally:
                kill_process_tree(process.pid, wait_timeout=60)
                log.close()
            return

        decode_host = _decode_hostname()
        prefill_log = f"{logs}/prefill.log"
        process, log = _launch(
            model, PREFILL_PORT, _disagg_args("prefill"), prefill_log
        )
        router = None
        try:
            prefill_url = f"http://{_self_hostname()}:{PREFILL_PORT}"
            decode_url = f"http://{decode_host}:{DECODE_PORT}"
            _wait_healthy(decode_url, DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH)
            router = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "sglang_router.launch_router",
                    "--pd-disaggregation",
                    "--mini-lb",
                    "--prefill",
                    prefill_url,
                    str(BOOTSTRAP_PORT),
                    "--decode",
                    decode_url,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(LB_PORT),
                ]
            )
            lb_url = f"http://127.0.0.1:{LB_PORT}"
            _wait_healthy(lb_url, 300)
            disaggregated = _requests(lb_url, model)
        finally:
            if router is not None:
                kill_process_tree(router.pid, wait_timeout=30)
            kill_process_tree(process.pid, wait_timeout=60)
            log.close()
        _assert_no_serving_recompile(self, _read(prefill_log), "prefill")

        # The colocated reference's own greedy output.
        process, log = _launch(model, PREFILL_PORT, [], f"{logs}/colocated.log")
        try:
            expected = _requests(f"http://127.0.0.1:{PREFILL_PORT}", model)
        finally:
            kill_process_tree(process.pid, wait_timeout=60)
            log.close()
        # The rank of each disaggregated token in the reference's distribution
        # at that position, and, for scale, of another prompt's continuation.
        # (A second server: prompt logprobs change the compiled program.)
        process, log = _launch(
            model,
            PREFILL_PORT,
            [],
            f"{logs}/reference_logprobs.log",
            env={"SGLANG_TPU_ENABLE_PROMPT_LOGPROB": "1"},
        )
        try:
            reference = f"http://127.0.0.1:{PREFILL_PORT}"
            ranks = [
                _token_ranks(reference, prompt, tokens)
                for prompt, tokens in zip(PROMPTS, disaggregated["text"])
            ]
            unrelated = [
                _token_ranks(
                    reference, prompt, expected["text"][(i + 1) % len(PROMPTS)]
                )
                for i, prompt in enumerate(PROMPTS)
            ]
        finally:
            kill_process_tree(process.pid, wait_timeout=60)
            log.close()
        self.assertGreater(expected["image"][1], 0)
        for i, (got, want) in enumerate(zip(disaggregated["text"], expected["text"])):
            same = next(
                (k for k, (a, b) in enumerate(zip(got, want)) if a != b), len(want)
            )
            print(
                f"request {i}: first {same} of {len(want)} tokens match; "
                f"ranks {ranks[i]}; unrelated continuation ranks {unrelated[i]}"
            )
        self.check_outputs(disaggregated, expected, ranks)

    def check_outputs(self, disaggregated, expected, ranks) -> None:
        """Equal attention TP: the decode runs the colocated computation."""
        self.assertEqual(disaggregated, expected)


class TestKimiK3TinyPD(TinyPDParity, CustomTestCase):
    """Equal attention TP: the decode pulls the MLA pages, and the KDA state
    row on raiden's state engine."""


if __name__ == "__main__":
    unittest.main()
