"""GPT-OSS-120B P/D throughput: a PP8 EAGLE3 prefill feeding a TP8/EP8 decode.

The JobSet gives each engine one full v7x host. The prefill runs eight
single-device pipeline stages and relays the Eagle3 aux hidden states to the
last stage; the decode pulls the KV and draft state from that last stage over
Raiden. The prefill pod runs the mini router and drives the 8k1k sweep through
it.
"""

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
from sglang.test.test_utils import dump_metric
from sglang.test.tpu.utils.pd_singlehost_base import PDSingleHostTestCase

register_tpu_ci(
    est_time=7200,
    suite="tpu-integration",
    nightly=True,
    labels=["singlehost", "gpt-oss-120b", "pd", "pp", "spec", "eagle3", "perf"],
)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, *[".."] * 4))
BENCH = os.path.join(REPO, "benchmark", "tpu")

LAUNCH_CONFIG_MODULE = "recipe/configs/gpt_oss_120b/pd_singlehost_pp_spec.py"

E = os.environ.get

SERVED_MODEL_NAME = "gpt-oss-120b"
DEFAULT_MODEL_PATH = "/weights/gpt-oss-120b"
MODEL_ENV = "GPT_OSS_MODEL"

PREFILL_ROLE = "prefill"
DECODE_ROLE = "decode"
SELF_HOST_ENV = "PD_SELF_HOST"
DECODE_HOST_ENV = "PD_DECODE_HOST"

# A v7x host has four physical chips and eight addressable TPU devices.
CHIPLETS_PER_HOST = 8
PREFILL_PP_SIZE = CHIPLETS_PER_HOST
DECODE_TP_SIZE = CHIPLETS_PER_HOST

PREFILL_PORT = int(E("PREFILL_PORT", "30000"))
DECODE_PORT = int(E("DECODE_PORT", "30000"))
LOCAL_HOST = "127.0.0.1"

WAIT_SECS = int(E("WAIT_SECS", "5400"))

PREFILL_LOG = "/tmp/pd_pp_spec_prefill.log"
DECODE_LOG = "/tmp/pd_pp_spec_decode.log"

WORKLOAD = "8k1k"
ISL, OSL, CONCUR = 8192, 1024, 64

# Random-ids prompts give a drafter nothing to predict; pin acceptance near the
# GSM8K measurement, as the single-host EAGLE3 perf test does.
SIMULATE_ACC_LEN = 2.5
# The simulated acceptance is random per verify; a 640-request mean sits well inside this.
ACCEPT_LENGTH_TOLERANCE = 0.3


def resolve_model_path(env: Mapping[str, str] | None = None) -> str:
    values = os.environ if env is None else env
    return values.get(MODEL_ENV) or values.get("MODEL") or DEFAULT_MODEL_PATH


def self_hostname(env: Mapping[str, str] | None = None) -> str:
    values = os.environ if env is None else env
    if values.get(SELF_HOST_ENV):
        return values[SELF_HOST_ENV]
    hostnames = [h for h in values.get("TPU_WORKER_HOSTNAMES", "").split(",") if h]
    return hostnames[0] if hostnames else socket.gethostname()


def _get_json(url: str, timeout: int = 30) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def engine_configuration_failures(
    *, prefill_info: Mapping[str, Any], decode_info: Mapping[str, Any]
) -> list[str]:
    expected = {
        PREFILL_ROLE: (prefill_info, {"tp_size": 1, "pp_size": PREFILL_PP_SIZE}),
        DECODE_ROLE: (decode_info, {"tp_size": DECODE_TP_SIZE, "pp_size": 1}),
    }
    failures = []
    for role, (info, sizes) in expected.items():
        if info.get("disaggregation_mode") != role:
            failures.append(
                f"{role} reports disaggregation_mode="
                f"{info.get('disaggregation_mode')!r}, expected {role!r}"
            )
        if str(info.get("speculative_algorithm", "")).upper() != "EAGLE3":
            failures.append(
                f"{role} reports speculative_algorithm="
                f"{info.get('speculative_algorithm')!r}, expected 'EAGLE3'"
            )
        for name, value in sizes.items():
            if info.get(name) != value:
                failures.append(
                    f"{role} reports {name}={info.get(name)!r}, expected {value}"
                )
    return failures


def read_spec_accept_length(metrics_text: str) -> float:
    """Max sglang:spec_accept_length across ranks; 0.0 when spec never ran."""
    values = [
        float(line.rsplit(" ", 1)[1])
        for line in metrics_text.splitlines()
        if line.startswith("sglang:spec_accept_length") and " " in line
    ]
    return max(values, default=0.0)


def build_model_config() -> dict[str, Any]:
    return {
        "model": SERVED_MODEL_NAME,
        "test_name": "gpt-oss-120b-pd-singlehost-pp-spec-c64",
        "id": "perf-gpt-oss-120b-pd-singlehost-pp8-eagle3-c64",
        "display_name": "GPT-OSS 120B P/D PP8 prefill + TP8 decode EAGLE3 (c64)",
        "variant": "PD-1P1D-PP8+TP8EP8+Raiden+EAGLE3-c64",
        "gates": [
            {
                "dataset": "bench_serving",
                "label": WORKLOAD,
                "sweep": [
                    {
                        "workload": WORKLOAD,
                        "isl": ISL,
                        "osl": OSL,
                        "concurrency": CONCUR,
                    }
                ],
                "bench_env": {
                    "DATASET_NAME": "random-ids",
                    "TOKENIZE_PROMPT": "1",
                    "MODEL": resolve_model_path(),
                    "SERVED_MODEL_NAME": SERVED_MODEL_NAME,
                },
            }
        ],
    }


class TestPdSingleHostPpSpecGptOss120bPerf(PDSingleHostTestCase):
    """Configure the shared P/D fixture for this test's recipe and sweep."""

    config_module = LAUNCH_CONFIG_MODULE
    repo_root = REPO
    cwd = BENCH
    log_path = PREFILL_LOG
    decode_log_path = DECODE_LOG
    self_hostname = staticmethod(self_hostname)
    decode_host_env = DECODE_HOST_ENV
    tag = "pd-singlehost-pp-spec"
    health_timeout = 2 * WAIT_SECS + 300
    health_generate_timeout = float(E("HEALTH_TIMEOUT", "5400"))
    # Class-level so the decode role, which skips get_runtime_envs, gets it too.
    runtime_envs = {"SGLANG_SIMULATE_ACC_LEN": str(SIMULATE_ACC_LEN)}

    @classmethod
    def build_model_config(cls) -> dict[str, Any]:
        return build_model_config()

    @classmethod
    def check_server_configuration(cls) -> None:
        failures = engine_configuration_failures(
            prefill_info=_get_json(f"http://{LOCAL_HOST}:{PREFILL_PORT}/server_info"),
            decode_info=_get_json(
                f"http://{cls.decode_hostname()}:{DECODE_PORT}/server_info"
            ),
        )
        if failures:
            raise RuntimeError("; ".join(failures))

    def test_throughput(self) -> None:
        self.run_evaluations()
        if self.role != PREFILL_ROLE:
            return
        url = f"http://{self.decode_hostname()}:{DECODE_PORT}/metrics"
        with urllib.request.urlopen(url, timeout=60) as response:
            accept_length = read_spec_accept_length(response.read().decode("utf-8"))
        config = build_model_config()
        dump_metric(
            "spec_accept_length",
            accept_length,
            labels={
                "kind": "info",
                "id": config["id"],
                "display_name": config["display_name"],
                "model": config["model"],
                "workload": WORKLOAD,
            },
        )
        self.assertAlmostEqual(
            accept_length,
            SIMULATE_ACC_LEN,
            delta=ACCEPT_LENGTH_TOLERANCE,
            msg=f"decode accept length {accept_length} on {url}: EAGLE3 or the "
            "simulated acceptance did not take effect",
        )


class TestPdSingleHostPpSpecGptOss120bConfiguration(unittest.TestCase):
    """Launch-contract tests that do not initialize TPU hardware."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.plans = {}
        for role in (PREFILL_ROLE, DECODE_ROLE):
            output = subprocess.check_output(
                [
                    sys.executable,
                    "-m",
                    "benchmark.tpu.launch_server",
                    "--config",
                    LAUNCH_CONFIG_MODULE + ":" + role.upper() + "_CONFIG",
                    "--runtime-envs",
                    json.dumps(
                        {"FORCE_SINGLE_HOST": "1", "SGLANG_HOST_IP": "127.0.0.1"}
                    ),
                    "--dry-run",
                ],
                cwd=REPO,
                text=True,
            )
            cls.plans[role] = json.loads(output)

    def test_prefill_is_eight_single_device_stages(self) -> None:
        args = self.plans[PREFILL_ROLE]["server_args"]
        self.assertEqual(args["pp_size"], PREFILL_PP_SIZE)
        self.assertEqual(args["tp_size"], 1)
        self.assertTrue(args["disable_overlap_schedule"])

    def test_decode_is_tp8_and_pulls_pipeline_pools(self) -> None:
        plan = self.plans[DECODE_ROLE]
        self.assertEqual(plan["server_args"]["tp_size"], DECODE_TP_SIZE)
        self.assertEqual(plan["server_args"]["ep_size"], DECODE_TP_SIZE)
        self.assertEqual(plan["server_args"].get("pp_size", 1), 1)
        self.assertIn("SGLANG_RAIDEN_PIPELINE_TRANSFER", plan["runtime_env_keys"])

    def test_both_engines_share_eagle3_and_raiden(self) -> None:
        for plan in self.plans.values():
            args = plan["server_args"]
            self.assertEqual(args["disaggregation_transfer_backend"], "raiden")
            self.assertEqual(args["speculative_algorithm"], "EAGLE3")
            self.assertEqual(args["speculative_num_steps"], 3)
            self.assertEqual(args["speculative_eagle_topk"], 1)
            self.assertEqual(args["speculative_num_draft_tokens"], 4)
            self.assertEqual(args["max_running_requests"], CONCUR)
            self.assertFalse(plan["server_args"].get("use_ray", False))

    def test_perf_config_builds_its_sweep_gate(self) -> None:
        from sglang.test.tpu.evals import model_config

        self.assertTrue(model_config.build_gates(build_model_config()))

    def test_accept_length_reads_the_max_over_ranks(self) -> None:
        text = (
            "# TYPE sglang:spec_accept_length gauge\n"
            'sglang:spec_accept_length{tp_rank="0"} 2.4\n'
            'sglang:spec_accept_length{tp_rank="1"} 2.6\n'
            "sglang:spec_accept_rate 0.5\n"
        )
        self.assertEqual(read_spec_accept_length(text), 2.6)
        self.assertEqual(read_spec_accept_length("sglang:num_running_reqs 1\n"), 0.0)

    def test_decode_exports_metrics(self) -> None:
        self.assertTrue(self.plans[DECODE_ROLE]["server_args"]["enable_metrics"])

    def test_engine_configuration_check(self) -> None:
        prefill = {
            "disaggregation_mode": PREFILL_ROLE,
            "speculative_algorithm": "EAGLE3",
            "tp_size": 1,
            "pp_size": PREFILL_PP_SIZE,
        }
        decode = {
            "disaggregation_mode": DECODE_ROLE,
            "speculative_algorithm": "EAGLE3",
            "tp_size": DECODE_TP_SIZE,
            "pp_size": 1,
        }
        self.assertEqual(
            engine_configuration_failures(prefill_info=prefill, decode_info=decode),
            [],
        )
        failures = engine_configuration_failures(
            prefill_info={**prefill, "pp_size": 1}, decode_info=decode
        )
        self.assertEqual(failures, ["prefill reports pp_size=1, expected 8"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="One-host-per-engine GPT-OSS P/D, PP8 prefill + EAGLE3, throughput"
    )
    parser.add_argument(
        "mode", nargs="?", default="unittest", choices=["unittest", "integration"]
    )
    args, extra_args = parser.parse_known_args()
    test_case = (
        TestPdSingleHostPpSpecGptOss120bPerf
        if args.mode == "integration"
        else TestPdSingleHostPpSpecGptOss120bConfiguration
    )
    unittest.main(
        defaultTest=test_case.__name__,
        argv=[sys.argv[0], *extra_args],
        verbosity=2,
    )
