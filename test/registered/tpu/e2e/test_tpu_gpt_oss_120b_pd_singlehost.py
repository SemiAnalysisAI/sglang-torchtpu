"""GPT-OSS-120B P/D disaggregation GSM8K quality gate, one host per engine."""

import argparse
import json
import os
import socket
import subprocess
import sys
import unittest
import urllib.request
from collections.abc import Mapping
from typing import Any

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu.pd_singlehost_base import PDSingleHostTestCase, peer_hostname

# Expected runtime: 45 minutes; CI uses this for scheduling and timeouts.
register_tpu_ci(
    est_time=2700,
    suite="tpu-integration",
    nightly=True,
    labels=["singlehost", "gpt-oss-120b", "pd"],
)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, *[".."] * 4))
BENCH = os.path.join(REPO, "benchmark", "tpu")

E = os.environ.get

SERVED_MODEL_NAME = "gpt-oss-120b"
DEFAULT_MODEL_PATH = "/weights/gpt-oss-120b"
MODEL_ENV = "GPT_OSS_MODEL"

PREFILL_ROLE = "prefill"
DECODE_ROLE = "decode"
ROLE_ENV = "CI_PD_ROLE"

# Each engine uses one eight-chiplet TPU host.
HOSTS_PER_ENGINE = 1
CHIPLETS_PER_HOST = 8
TP_SIZE = HOSTS_PER_ENGINE * CHIPLETS_PER_HOST
EP_SIZE = TP_SIZE

# Separate pods may reuse the same service and Raiden ports.
PREFILL_PORT = int(E("PREFILL_PORT", "30000"))
DECODE_PORT = int(E("DECODE_PORT", "30000"))
BOOTSTRAP_PORT = int(E("BOOTSTRAP_PORT", "8998"))
RAIDEN_KV_PORT = int(E("SGLANG_RAIDEN_KV_PORT", "31000"))
LOCAL_HOST = "127.0.0.1"

PYTHON = E("PYTHON", "python3")

# The runtime image excludes the JobSet renderer.
RENDER_JOBSET = os.path.join(REPO, ".github", "tpu-ci", "render_jobset.py")

# Used in generated pod names, which Kubernetes limits to 63 characters.
RUN_NAME = "gpt-oss-120b-pd-singlehost-gsm8k"

PREFILL_LOG = "/tmp/pd_singlehost_prefill.log"
DECODE_LOG = "/tmp/pd_singlehost_decode.log"
LAUNCH_CONFIG_MODULE = "recipe/configs/gpt_oss_120b/pd_singlehost.py"

PAGE_SIZE = 64
MEM_FRACTION_STATIC = "0.35"
# Use checkpoint-provided MXFP4 weights and FP8 KV cache.
QUANTIZATION = ""
KV_CACHE_DTYPE = "fp8_e4m3"
# One host has half the HBM of the two-host P/D configuration.
MAX_TOTAL_TOKENS = 393216
DISAGG_TRANSFER_BACKEND = "raiden"
# Raiden handles both P/D KV transfer and decode host-pool I/O.
HICACHE_IO_BACKEND = "raiden"
EXTRA_SERVER_ARGS: tuple[str, ...] = ("--reasoning-parser", "gpt-oss")
# GPT-OSS does not support the SparseCore fused-MoE gather path.
EXTRA_LAUNCH_ENV: dict[str, str] = {"SGLANG_TPU_FUSED_MOE_SC_GATHER": "0"}

# Defaults to a 0.77 passing floor; this topology measured 0.97.
GSM8K_BASELINE = float(E("GSM8K_THRESHOLD_GPT_OSS_120B", E("GSM8K_THRESHOLD", "0.80")))
ATOL = float(E("GSM8K_ATOL", "0.03"))
NUM_EXAMPLES = int(E("GSM8K_NUM_EXAMPLES", "100"))
NUM_THREADS = int(E("GSM8K_NUM_THREADS", "8"))
MAX_TOKENS = int(E("GSM8K_MAX_TOKENS", "1024"))


def resolve_model_path(env: Mapping[str, str] | None = None) -> str:
    """Resolve the checkpoint from the environment, else the mounted default."""
    values = os.environ if env is None else env
    return values.get(MODEL_ENV) or values.get("MODEL") or DEFAULT_MODEL_PATH


def validate_model_source(model_path: str) -> None:
    """Fail clearly when CI points at an unprovisioned local checkpoint."""
    if not os.path.isabs(model_path):
        return  # HF repo ids resolve remotely
    if not os.path.isfile(os.path.join(model_path, "config.json")):
        raise FileNotFoundError(
            f"GPT-OSS-120B checkpoint incomplete or missing at {model_path}, "
            f"provision it on the weights volume or set {MODEL_ENV}"
        )


def resolve_role(env: Mapping[str, str] | None = None) -> str:
    """Read this pod's P/D role, which the JobSet template sets per replica."""
    values = os.environ if env is None else env
    role = values.get(ROLE_ENV, "")
    if role not in (PREFILL_ROLE, DECODE_ROLE):
        raise ValueError(
            f"{ROLE_ENV} must be {PREFILL_ROLE!r} or {DECODE_ROLE!r}, got {role!r}"
        )
    return role


def self_hostname(env: Mapping[str, str] | None = None) -> str:
    """This pod's own DNS name."""
    values = os.environ if env is None else env
    hostnames = [h for h in values.get("TPU_WORKER_HOSTNAMES", "").split(",") if h]
    return hostnames[0] if hostnames else socket.gethostname()


def _common_launch_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    values = os.environ if env is None else env
    return {
        "MODEL": resolve_model_path(values),
        "TP_SIZE": str(TP_SIZE),
        "EP_SIZE": str(EP_SIZE),
        "PAGE_SIZE": str(PAGE_SIZE),
        # Raiden requires the fused KV pool provided by RPA v3.
        "ATTENTION_BACKEND": "pallas_tpu_rpa_v3",
        "QUANTIZATION": QUANTIZATION,
        "KV_CACHE_DTYPE": KV_CACHE_DTYPE,
        "DISABLE_RADIX_CACHE": "0",
        "ENABLE_TORCH_COMPILE": values.get("ENABLE_TORCH_COMPILE", "1"),
        "MEM_FRACTION_STATIC": values.get("MEM_FRACTION_STATIC", MEM_FRACTION_STATIC),
        "SGLANG_RAIDEN_KV_PORT": str(RAIDEN_KV_PORT),
        "PYTHON": PYTHON,
        **EXTRA_LAUNCH_ENV,
    }


def _shared_server_args() -> list[str]:
    # EXTRA_ARGS overrides the launchers' default TPU transfer backend.
    return [
        "--served-model-name",
        SERVED_MODEL_NAME,
        *EXTRA_SERVER_ARGS,
        "--disaggregation-transfer-backend",
        DISAGG_TRANSFER_BACKEND,
        "--max-total-tokens",
        str(MAX_TOTAL_TOKENS),
        "--skip-server-warmup",
        "--sampling-backend",
        "pytorch",
    ]


def build_prefill_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Construct the environment for the prefill engine."""
    values = os.environ if env is None else env
    extra_args = _shared_server_args()
    if values.get("EXTRA_PREFILL_ARGS"):
        extra_args.append(values["EXTRA_PREFILL_ARGS"])
    return {
        **_common_launch_env(values),
        "PORT": str(PREFILL_PORT),
        "BOOTSTRAP_PORT": str(BOOTSTRAP_PORT),
        "EXTRA_ARGS": " ".join(extra_args).strip(),
    }


def _decode_host_pool_args() -> list[str]:
    """Make the decode retraction host pool buildable on a fused TPU KV pool."""
    return [
        "--hicache-mem-layout",
        "page_first_fused",
        "--hicache-io-backend",
        HICACHE_IO_BACKEND,
    ]


def build_decode_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Construct the environment for the decode engine."""
    values = os.environ if env is None else env
    extra_args = _shared_server_args() + _decode_host_pool_args()
    if values.get("EXTRA_DECODE_ARGS"):
        extra_args.append(values["EXTRA_DECODE_ARGS"])
    return {
        **_common_launch_env(values),
        "PORT": str(DECODE_PORT),
        "EXTRA_ARGS": " ".join(extra_args).strip(),
    }


def _get_json(url: str, timeout: int = 30) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def transport_failures(*, prefill_info: dict[str, Any], decode_info: dict[str, Any]):
    """Reject a run where the two engines are not a raiden P/D pair."""
    failures = []
    for label, info, expected_mode in (
        (PREFILL_ROLE, prefill_info, PREFILL_ROLE),
        (DECODE_ROLE, decode_info, DECODE_ROLE),
    ):
        mode = info.get("disaggregation_mode")
        if mode != expected_mode:
            failures.append(
                f"{label} engine reports disaggregation_mode={mode!r}, expected "
                f"{expected_mode!r}: it is not serving a P/D role"
            )
        backend = info.get("disaggregation_transfer_backend")
        if backend != DISAGG_TRANSFER_BACKEND:
            failures.append(
                f"{label} engine reports disaggregation_transfer_backend="
                f"{backend!r}, expected {DISAGG_TRANSFER_BACKEND!r}"
            )
    return failures


def pd_transport_runner(*, prefill_url: str, decode_url: str):
    """Verify that both servers report the expected P/D roles and backend."""

    def run(base_url: str) -> dict[str, Any]:
        prefill_info = _get_json(f"{prefill_url}/server_info")
        decode_info = _get_json(f"{decode_url}/server_info")
        failures = transport_failures(
            prefill_info=prefill_info, decode_info=decode_info
        )
        print(
            f"[pd-singlehost] lb={base_url} prefill="
            f"{prefill_info.get('disaggregation_mode')}/"
            f"{prefill_info.get('disaggregation_transfer_backend')} decode="
            f"{decode_info.get('disaggregation_mode')}/"
            f"{decode_info.get('disaggregation_transfer_backend')}",
            flush=True,
        )
        return {
            "score": 0.0 if failures else 1.0,
            "_passed": not failures,
            "_failures": failures,
        }

    return run


def build_model_config(*, decode_host: str) -> dict[str, Any]:
    """Assemble the gate list, which needs the decode pod's resolved name."""
    return {
        "model": SERVED_MODEL_NAME,
        "test_name": "gpt-oss-120b-pd-singlehost-gsm8k",
        "variant": f"PD-singlehost-TP{TP_SIZE}+EP{EP_SIZE}-raiden",
        # Keep this specialized P/D result off the standard accuracy dashboard.
        "gates": [
            {
                "dataset": "pd_transport",
                "label": "pd-transport",
                "runner": pd_transport_runner(
                    prefill_url=f"http://{LOCAL_HOST}:{PREFILL_PORT}",
                    decode_url=f"http://{decode_host}:{DECODE_PORT}",
                ),
            },
            {
                "dataset": "gsm8k",
                "label": "gsm8k",
                "baseline": GSM8K_BASELINE,
                "atol": ATOL,
                "num_examples": NUM_EXAMPLES,
                "num_threads": NUM_THREADS,
                "max_tokens": MAX_TOKENS,
                "temperature": 0.0,
            },
        ],
    }


class TestPdSingleHostGptOss120bIntegration(PDSingleHostTestCase):
    """Configure the shared P/D fixture for this test's recipe and gates."""

    config_module = LAUNCH_CONFIG_MODULE
    repo_root = REPO
    cwd = BENCH
    log_path = PREFILL_LOG
    decode_log_path = DECODE_LOG
    self_hostname = staticmethod(self_hostname)

    @classmethod
    def build_model_config(cls) -> dict[str, Any]:
        return build_model_config(decode_host=cls.decode_hostname())

    @classmethod
    def setUpClass(cls) -> None:
        validate_model_source(resolve_model_path())
        super().setUpClass()

    def test_accuracy(self) -> None:
        self.run_evaluations()


def render_jobset_yaml(
    image: str = "us-central1-docker.pkg.dev/sgl-project/tpu-ci/sglang:latest",
    lane: str = "nightly",
    weights_pvc: str = "model-weights-filestore",
    run_name: str = RUN_NAME,
) -> str:
    """Render the JobSet for this lane: one host of prefill, one of decode."""
    rel_path = os.path.relpath(__file__, REPO)
    cmd = [
        PYTHON,
        RENDER_JOBSET,
        "--pd-prefill-hosts",
        str(HOSTS_PER_ENGINE),
        "--pd-decode-hosts",
        str(HOSTS_PER_ENGINE),
        "--lane",
        lane,
        "--image",
        image,
        "--weights-pvc",
        weights_pvc,
        "--name",
        run_name,
        # Names the dashboard row and the TIMINGS record.
        "--files",
        rel_path,
        "--exec",
        f"python3 {rel_path} integration",
    ]
    return subprocess.check_output(cmd, text=True)


class TestPdSingleHostGptOss120bConfiguration(unittest.TestCase):
    """Host-side configuration tests, runnable without TPU hardware."""

    def test_each_engine_takes_one_whole_host(self) -> None:
        for env in (build_prefill_env(env={}), build_decode_env(env={})):
            self.assertEqual(env["TP_SIZE"], str(CHIPLETS_PER_HOST))
            self.assertEqual(env["EP_SIZE"], str(CHIPLETS_PER_HOST))
            # Do not mask chiplets when each engine owns its host.
            for var in (
                "TPU_VISIBLE_DEVICES",
                "TPU_VISIBLE_CHIPS",
                "TPU_CHIPS_PER_HOST_BOUNDS",
                "TPU_HOST_BOUNDS",
            ):
                self.assertNotIn(var, env)

    def test_both_engines_select_the_raiden_transport(self) -> None:
        for env in (build_prefill_env(env={}), build_decode_env(env={})):
            self.assertIn(
                f"--disaggregation-transfer-backend {DISAGG_TRANSFER_BACKEND}",
                env["EXTRA_ARGS"],
            )

    def test_peer_hostname_maps_prefill_to_decode(self) -> None:
        self.assertEqual(
            peer_hostname(
                "tpu-ci-gpt-oss-120b-pd-singlehost-gsm8k-prefill-0-0",
                from_role=PREFILL_ROLE,
                to_role=DECODE_ROLE,
            ),
            "tpu-ci-gpt-oss-120b-pd-singlehost-gsm8k-decode-0-0",
        )
        with self.assertRaises(ValueError):
            peer_hostname(
                "no-role-segment", from_role=PREFILL_ROLE, to_role=DECODE_ROLE
            )

    def test_self_hostname_falls_back_to_the_pod_name(self) -> None:
        # Single-host pods may omit TPU_WORKER_HOSTNAMES.
        self.assertEqual(
            self_hostname({"TPU_WORKER_HOSTNAMES": "a-prefill-0-0"}), "a-prefill-0-0"
        )
        self.assertEqual(self_hostname({}), socket.gethostname())

    def test_role_must_be_set_by_the_jobset(self) -> None:
        self.assertEqual(resolve_role({ROLE_ENV: DECODE_ROLE}), DECODE_ROLE)
        for bad in ({}, {ROLE_ENV: ""}, {ROLE_ENV: "both"}):
            with self.assertRaisesRegex(ValueError, ROLE_ENV):
                resolve_role(bad)

    def test_transport_gate_rejects_a_non_disaggregated_pair(self) -> None:
        raiden = {
            "disaggregation_mode": "null",
            "disaggregation_transfer_backend": DISAGG_TRANSFER_BACKEND,
        }
        self.assertTrue(transport_failures(prefill_info=raiden, decode_info=raiden))
        native = {
            "disaggregation_mode": PREFILL_ROLE,
            "disaggregation_transfer_backend": "tpu",
        }
        self.assertTrue(
            transport_failures(
                prefill_info=native,
                decode_info={**native, "disaggregation_mode": DECODE_ROLE},
            )
        )
        self.assertEqual(
            transport_failures(
                prefill_info={
                    "disaggregation_mode": PREFILL_ROLE,
                    "disaggregation_transfer_backend": DISAGG_TRANSFER_BACKEND,
                },
                decode_info={
                    "disaggregation_mode": DECODE_ROLE,
                    "disaggregation_transfer_backend": DISAGG_TRANSFER_BACKEND,
                },
            ),
            [],
        )

    def test_launcher_scripts_resolve(self) -> None:
        for script in ("launch_prefill.sh", "launch_decode.sh"):
            self.assertTrue(
                os.path.isfile(os.path.join(BENCH, script)),
                f"BENCH does not point at benchmark/tpu: {BENCH}",
            )

    def test_local_model_path_validation(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, MODEL_ENV):
            validate_model_source("/weights/definitely-missing-gpt-oss-120b")
        validate_model_source("openai/gpt-oss-120b")

    def test_run_name_fits_the_generated_pod_name(self) -> None:
        # Kubernetes rejects generated pod names longer than 63 characters.
        pod_name = f"tpu-ci-{RUN_NAME}-xxxxx-prefill-0-0-xxxxx"
        self.assertLessEqual(len(pod_name), 63, pod_name)

    # Skip in runtime images, which do not contain the renderer.
    @unittest.skipUnless(
        os.path.exists(RENDER_JOBSET), "renderer is not shipped in the runtime image"
    )
    def test_dry_run_render_jobset_yaml(self) -> None:
        # Verify the workflow name, roles, and command remain aligned.
        yaml_content = render_jobset_yaml()
        self.assertIn(
            "generateName: tpu-ci-gpt-oss-120b-pd-singlehost-gsm8k-", yaml_content
        )
        self.assertIn("name: prefill", yaml_content)
        self.assertIn("name: decode", yaml_content)
        self.assertNotIn("name: router", yaml_content)
        self.assertIn("CI_PD_ROLE", yaml_content)
        self.assertIn(
            "python3 test/registered/tpu/e2e/test_tpu_gpt_oss_120b_pd_singlehost.py integration",
            yaml_content,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-host-per-engine P/D GSM8K")
    parser.add_argument(
        "mode",
        nargs="?",
        default="unittest",
        choices=["unittest", "integration", "integration_dry_run"],
        help="Execution mode (default: unittest)",
    )
    args, extra_args = parser.parse_known_args()

    if args.mode == "integration_dry_run":
        print(render_jobset_yaml())
        sys.exit(0)

    test_case = (
        TestPdSingleHostGptOss120bIntegration
        if args.mode == "integration"
        else TestPdSingleHostGptOss120bConfiguration
    )
    unittest.main(
        defaultTest=test_case.__name__,
        argv=[sys.argv[0], *extra_args],
        verbosity=2,
    )
