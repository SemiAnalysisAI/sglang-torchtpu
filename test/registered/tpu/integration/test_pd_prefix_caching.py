"""Multihost P/D prefix-caching quality test for GPT-OSS-120B.

Topology: a multihost PREFILL engine and a multihost DECODE engine, each on its
own TPU slice, connected by the P/D mini load balancer. Device radix caching is
enabled and KV offload is disabled on both engines. The only hardware pass
criterion is that GSM8K clears the GPT-OSS-120B quality floor.

Every pod in both slices runs this file. ``CI_PD_ROLE`` selects the engine role
and the slice-local ``TPU_WORKER_*`` variables select the multihost rank. Without
``CI_PD_ROLE``, only the host-side configuration tests run.

Hardware invocation:

  CI_PD_ROLE=prefill python3 \
      test/registered/tpu/integration/test_pd_prefix_caching.py

The model path and GSM8K settings are environment-overridable through the
configuration structs below.
"""

import os
import signal
import socket
import subprocess
import sys
import time
import unittest
import urllib.request
from types import SimpleNamespace
from typing import Dict, Mapping, Optional, Tuple

import msgspec

SGLANG_IMPORT_ERROR = None
try:
    from sglang.test.ci.ci_register import register_tpu_ci
    from sglang.test.test_utils import dump_metric
except ModuleNotFoundError as exc:
    # Configuration tests remain useful in a source-only checkout. Hardware
    # cases require the full SGLang TPU environment.
    SGLANG_IMPORT_ERROR = exc

    def register_tpu_ci(*_args, **_kwargs):
        return None

    def dump_metric(*_args, **_kwargs):
        return None


register_tpu_ci(est_time=10800, suite="tpu-integration")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, *[".."] * 4))
BENCH = os.path.join(REPO, "benchmark", "tpu")

E = os.environ.get

PREFILL_PORT = int(E("PREFILL_PORT", "30000"))
DECODE_PORT = int(E("DECODE_PORT", "30000"))
LB_PORT = int(E("LB_PORT", "8000"))
BOOTSTRAP_PORT = int(E("BOOTSTRAP_PORT", "8998"))
WAIT_SECS = int(E("WAIT_SECS", "3600"))
KEEP_UP = E("KEEP_UP", "0") == "1"
PYTHON = E("PYTHON", "python3")

ROLE_PREFILL = "prefill"
ROLE_DECODE = "decode"

LOG_DIR = E("PD_TEST_LOG_DIR", "/tmp")
PREFILL_LOG = os.path.join(LOG_DIR, "pd_prefix_caching_prefill.log")
DECODE_LOG = os.path.join(LOG_DIR, "pd_prefix_caching_decode.log")
LB_LOG = os.path.join(LOG_DIR, "pd_prefix_caching_lb.log")


class ModelProfile(msgspec.Struct, frozen=True, kw_only=True):
    display_name: str
    served_name: str
    default_model_path: str
    model_env: str
    page_size: int
    mem_fraction_prefill: str
    mem_fraction_decode: str
    quantization: str
    kv_cache_dtype: str
    max_total_tokens: int
    gsm8k_threshold: float
    disagg_transfer_backend: str
    extra_server_args: Tuple[str, ...] = ()
    extra_launch_env: Dict[str, str] = {}
    reasoning_effort: Optional[str] = None


GPT_OSS_120B = ModelProfile(
    display_name="GPT-OSS-120B",
    served_name="gpt-oss-120b",
    default_model_path="/weights/gpt-oss-120b",
    model_env="GPT_OSS_MODEL",
    page_size=64,
    mem_fraction_prefill="0.35",
    mem_fraction_decode="0.35",
    # The checkpoint declares MXFP4 itself. Forcing fp8 would re-quantize.
    quantization="",
    kv_cache_dtype="fp8_e4m3",
    # Keep the validated multihost P/D pool bound from issue #376. Smaller
    # pools can hang GPT-OSS extend-graph precompilation.
    max_total_tokens=786_432,
    gsm8k_threshold=0.72,
    # The native TPU connector rejects GPT-OSS's hybrid SWA KV pool.
    disagg_transfer_backend="raiden",
    extra_server_args=("--reasoning-parser", "gpt-oss"),
    extra_launch_env={"SGLANG_TPU_FUSED_MOE_SC_GATHER": "0"},
    reasoning_effort="low",
)


class Gsm8kConfig(msgspec.Struct, frozen=True, kw_only=True):
    num_examples: int
    num_threads: int
    max_tokens: int
    threshold: float

    @classmethod
    def from_env(
        cls, profile: ModelProfile, env: Optional[Mapping[str, str]] = None
    ) -> "Gsm8kConfig":
        values = os.environ if env is None else env
        threshold = float(values.get("GSM8K_THRESHOLD", str(profile.gsm8k_threshold)))
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"GSM8K threshold must be in [0, 1], got {threshold}")
        return cls(
            num_examples=int(values.get("GSM8K_NUM_EXAMPLES", "100")),
            num_threads=int(values.get("GSM8K_NUM_THREADS", "8")),
            max_tokens=int(values.get("GSM8K_MAX_TOKENS", "2048")),
            threshold=threshold,
        )


class PodTopology(msgspec.Struct, frozen=True, kw_only=True):
    role: str
    group_rank: int
    group_hosts: int
    group_hostnames: Tuple[str, ...]

    @property
    def is_leader(self) -> bool:
        return self.role == ROLE_PREFILL and self.group_rank == 0

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "PodTopology":
        values = os.environ if env is None else env
        role = values.get("CI_PD_ROLE", "")
        if role not in (ROLE_PREFILL, ROLE_DECODE):
            raise ValueError(
                f"CI_PD_ROLE must be {ROLE_PREFILL!r} or {ROLE_DECODE!r}, "
                f"got {role!r}"
            )
        hostnames = tuple(
            host for host in values.get("TPU_WORKER_HOSTNAMES", "").split(",") if host
        )
        if not hostnames:
            hostnames = (socket.gethostname(),)
        return cls(
            role=role,
            group_rank=int(values.get("TPU_WORKER_ID", "0")),
            group_hosts=len(hostnames),
            group_hostnames=hostnames,
        )


def peer_group_hostname(hostname: str, from_role: str, to_role: str) -> str:
    """Map a JobSet pod hostname from one role to its peer role."""
    needle, replacement = f"-{from_role}-", f"-{to_role}-"
    if needle not in hostname:
        raise ValueError(f"hostname {hostname!r} has no {needle!r} segment")
    return hostname.replace(needle, replacement)


def resolve_model_path(
    profile: ModelProfile, env: Optional[Mapping[str, str]] = None
) -> str:
    values = os.environ if env is None else env
    return (
        values.get(profile.model_env)
        or values.get("MODEL")
        or profile.default_model_path
    )


def validate_model_source(display_name: str, model_path: str, env_name: str) -> None:
    """Fail clearly when a configured absolute checkpoint is incomplete."""
    if not os.path.isabs(model_path):
        return
    if not os.path.isfile(os.path.join(model_path, "config.json")):
        raise FileNotFoundError(
            f"{display_name} checkpoint incomplete or missing at {model_path}; "
            f"provision it on the weights volume or set {env_name}"
        )


def _common_launch_env(
    profile: ModelProfile,
    group_hosts: int,
    env: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    values = os.environ if env is None else env
    tp_size = int(values.get("TP_SIZE", str(8 * group_hosts)))
    return {
        "MODEL": resolve_model_path(profile, values),
        "TP_SIZE": str(tp_size),
        "EP_SIZE": values.get("EP_SIZE", str(tp_size)),
        "PAGE_SIZE": str(profile.page_size),
        "MEM_FRACTION_STATIC": values.get(
            "MEM_FRACTION_STATIC", profile.mem_fraction_prefill
        ),
        "ATTENTION_BACKEND": "pallas_tpu_rpa_v3",
        "QUANTIZATION": profile.quantization,
        "KV_CACHE_DTYPE": profile.kv_cache_dtype,
        "MAX_TOTAL_TOKENS": values.get(
            "MAX_TOTAL_TOKENS", str(profile.max_total_tokens)
        ),
        "CONTEXT_LEN": values.get("CONTEXT_LEN", "8192"),
        "CHUNKED_PREFILL_SIZE": values.get("CHUNKED_PREFILL_SIZE", "4096"),
        "MAX_PREFILL_TOKENS": values.get("MAX_PREFILL_TOKENS", "4096"),
        "MAX_RUNNING": values.get("MAX_RUNNING", "16"),
        "TPU_COMPILE_TOKENS": values.get("TPU_COMPILE_TOKENS", "512 2048 4096"),
        "TPU_COMPILE_BS": values.get("TPU_COMPILE_BS", "1 2 4 16"),
        "DISABLE_RADIX_CACHE": "0",
        "ENABLE_CACHE_REPORT": "1",
        "KV_OFFLOAD": "0",
        "ENABLE_TORCH_COMPILE": values.get("ENABLE_TORCH_COMPILE", "1"),
        "ENABLE_PROFILER_ENDPOINT": "0",
        "DUMP_META": "0",
        "PYTHON": PYTHON,
        "SGLANG_RAIDEN_KV_PORT": values.get("SGLANG_RAIDEN_KV_PORT", "31000"),
        **profile.extra_launch_env,
    }


def _base_extra_args(profile: ModelProfile) -> Tuple[str, ...]:
    return (
        "--served-model-name",
        profile.served_name,
        *profile.extra_server_args,
        "--disaggregation-transfer-backend",
        profile.disagg_transfer_backend,
        "--skip-server-warmup",
        "--sampling-backend",
        "pytorch",
    )


def build_prefill_env(
    profile: ModelProfile,
    group_hosts: int,
    env: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    values = os.environ if env is None else env
    launch_env = _common_launch_env(profile, group_hosts, values)
    launch_env["MEM_FRACTION_STATIC"] = values.get(
        "MEM_FRACTION_STATIC", profile.mem_fraction_prefill
    )
    launch_env["PORT"] = str(PREFILL_PORT)
    launch_env["BOOTSTRAP_PORT"] = str(BOOTSTRAP_PORT)
    launch_env["EXTRA_ARGS"] = " ".join(
        (*_base_extra_args(profile), values.get("EXTRA_PREFILL_ARGS", ""))
    ).strip()
    return launch_env


def build_decode_env(
    profile: ModelProfile,
    group_hosts: int,
    env: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    values = os.environ if env is None else env
    launch_env = _common_launch_env(profile, group_hosts, values)
    launch_env["MEM_FRACTION_STATIC"] = values.get(
        "MEM_FRACTION_STATIC", profile.mem_fraction_decode
    )
    launch_env["PORT"] = str(DECODE_PORT)
    launch_env["EXTRA_ARGS"] = " ".join(
        (*_base_extra_args(profile), values.get("EXTRA_DECODE_ARGS", ""))
    ).strip()
    return launch_env


def run_gsm8k(base_url: str, profile: ModelProfile, config: Gsm8kConfig) -> float:
    from sglang.test.run_eval import run_eval

    metrics = run_eval(
        SimpleNamespace(
            base_url=base_url.rstrip("/"),
            model=profile.served_name,
            eval_name="gsm8k",
            api="chat",
            num_examples=config.num_examples,
            num_threads=config.num_threads,
            num_shots=5,
            max_tokens=config.max_tokens,
            temperature=0.0,
            top_p=1.0,
            reasoning_effort=profile.reasoning_effort,
            chat_template_kwargs=None,
            gsm8k_data_path=None,
        )
    )
    return float(metrics["score"])


def _tee_command(argv: Tuple[str, ...], log_path: str) -> Tuple[str, ...]:
    quoted = " ".join(f"'{arg}'" for arg in argv)
    return ("bash", "-c", f"set -o pipefail; {quoted} 2>&1 | tee -a '{log_path}'")


def _launch(script: str, extra_env: Dict[str, str], log_path: str) -> subprocess.Popen:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    env = dict(os.environ)
    env.update(extra_env)
    return subprocess.Popen(
        _tee_command(("bash", os.path.join(BENCH, script)), log_path),
        env=env,
        preexec_fn=os.setsid,
        cwd=BENCH,
    )


def _kill(proc: Optional[subprocess.Popen]) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass


def _http_ok(url: str, timeout: int = 5) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return 200 <= response.status < 300
    except Exception:  # noqa: BLE001 - any transport error means not healthy
        return False


def _wait_http_200(url: str, secs: int, procs: Tuple = ()) -> None:
    deadline = time.time() + secs
    while time.time() < deadline:
        for proc in procs:
            if proc is not None and proc.poll() is not None:
                raise RuntimeError(f"process for {url} exited rc={proc.returncode}")
        if _http_ok(url):
            return
        time.sleep(10)
    raise TimeoutError(f"{url} not ready after {secs}s")


class TestPdPrefixCaching(unittest.TestCase):
    """Runs on the prefill leader and gates the whole two-slice run."""

    procs = []
    topo: PodTopology
    profile = GPT_OSS_120B

    @classmethod
    def setUpClass(cls) -> None:
        if SGLANG_IMPORT_ERROR is not None:
            raise SGLANG_IMPORT_ERROR
        cls.topo = PodTopology.from_env()
        model_path = resolve_model_path(cls.profile)
        validate_model_source(
            cls.profile.display_name, model_path, cls.profile.model_env
        )
        cls.gsm8k_config = Gsm8kConfig.from_env(cls.profile)
        cls.procs = []

        self_host = E("PD_SELF_HOST", cls.topo.group_hostnames[0])
        decode_host = E("PD_PEER_HOST") or peer_group_hostname(
            self_host, from_role=ROLE_PREFILL, to_role=ROLE_DECODE
        )
        cls.procs.append(
            _launch(
                "launch_prefill.sh",
                build_prefill_env(cls.profile, cls.topo.group_hosts),
                PREFILL_LOG,
            )
        )
        _wait_http_200(
            f"http://127.0.0.1:{PREFILL_PORT}/health", WAIT_SECS, tuple(cls.procs)
        )
        _wait_http_200(f"http://{decode_host}:{DECODE_PORT}/health", WAIT_SECS)

        cls.procs.append(
            subprocess.Popen(
                _tee_command(
                    (
                        PYTHON,
                        "-m",
                        "sglang_router.launch_router",
                        "--pd-disaggregation",
                        "--mini-lb",
                        "--prefill",
                        f"http://{self_host}:{PREFILL_PORT}",
                        "--decode",
                        f"http://{decode_host}:{DECODE_PORT}",
                        "--host",
                        "0.0.0.0",
                        "--port",
                        str(LB_PORT),
                    ),
                    LB_LOG,
                ),
                preexec_fn=os.setsid,
            )
        )
        _wait_http_200(f"http://127.0.0.1:{LB_PORT}/health", 300, tuple(cls.procs))

    @classmethod
    def tearDownClass(cls) -> None:
        if KEEP_UP:
            print("KEEP_UP=1: leaving prefill and load balancer running")
            return
        for proc in cls.procs:
            _kill(proc)

    def test_gsm8k_quality(self) -> None:
        score = run_gsm8k(
            base_url=f"http://127.0.0.1:{LB_PORT}",
            profile=self.profile,
            config=self.gsm8k_config,
        )
        dump_metric(
            "gsm8k_score",
            score,
            labels={"model": self.profile.served_name, "test": "pd_prefix_caching"},
        )
        print(f"gsm8k score={score:.4f} floor={self.gsm8k_config.threshold:.4f}")
        self.assertGreaterEqual(score, self.gsm8k_config.threshold)


def _serve_role(topo: PodTopology, profile: ModelProfile) -> int:
    """Serve a non-leader engine shard until the prefill leader is done."""
    if topo.role == ROLE_DECODE:
        script, env, log_path = (
            "launch_decode.sh",
            build_decode_env(profile, topo.group_hosts),
            DECODE_LOG,
        )
        leader_host = E("PD_PEER_HOST") or peer_group_hostname(
            topo.group_hostnames[0], from_role=ROLE_DECODE, to_role=ROLE_PREFILL
        )
    else:
        script, env, log_path = (
            "launch_prefill.sh",
            build_prefill_env(profile, topo.group_hosts),
            PREFILL_LOG,
        )
        leader_host = topo.group_hostnames[0]

    proc = _launch(script, env, log_path)
    leader_url = f"http://{leader_host}:{PREFILL_PORT}/health"
    leader_seen, misses, waited = False, 0, 0
    while True:
        return_code = proc.poll()
        if return_code is not None:
            if leader_seen and not _http_ok(leader_url):
                print(f"[{topo.role}:{topo.group_rank}] exited during teardown")
                return 0
            return return_code
        if _http_ok(leader_url):
            leader_seen, misses = True, 0
        elif leader_seen:
            misses += 1
            if misses >= 6:
                print(f"[{topo.role}:{topo.group_rank}] leader gone, stopping")
                _kill(proc)
                return 0
        elif waited > WAIT_SECS + 600:
            print(f"[{topo.role}:{topo.group_rank}] leader never healthy, stopping")
            _kill(proc)
            return 0
        time.sleep(10)
        waited += 10


class TestPdPrefixCachingConfiguration(unittest.TestCase):
    """Host-side contracts for the GPT-OSS P/D prefix-cache scenario."""

    def test_both_roles_enable_device_cache_without_speculation(self) -> None:
        for builder in (build_prefill_env, build_decode_env):
            with self.subTest(builder=builder.__name__):
                env = builder(GPT_OSS_120B, 2, env={})
                self.assertEqual(env["DISABLE_RADIX_CACHE"], "0")
                self.assertEqual(env["ENABLE_CACHE_REPORT"], "1")
                self.assertEqual(env["KV_OFFLOAD"], "0")
                self.assertNotIn("speculative-", env["EXTRA_ARGS"])
                self.assertNotIn("EAGLE", env["EXTRA_ARGS"])

    def test_multihost_size_and_model_override_reach_both_roles(self) -> None:
        override = {
            "TP_SIZE": "32",
            "EP_SIZE": "16",
            "MODEL": "/model",
        }
        prefill = build_prefill_env(GPT_OSS_120B, 2, env=override)
        decode = build_decode_env(GPT_OSS_120B, 2, env=override)
        self.assertEqual(prefill["TP_SIZE"], "32")
        self.assertEqual(decode["TP_SIZE"], "32")
        self.assertEqual(prefill["EP_SIZE"], "16")
        self.assertEqual(decode["EP_SIZE"], "16")
        self.assertEqual(prefill["MODEL"], "/model")
        self.assertEqual(decode["MODEL"], "/model")

    def test_gsm8k_quality_configuration(self) -> None:
        config = Gsm8kConfig.from_env(
            GPT_OSS_120B,
            {
                "GSM8K_NUM_EXAMPLES": "12",
                "GSM8K_NUM_THREADS": "3",
                "GSM8K_MAX_TOKENS": "512",
                "GSM8K_THRESHOLD": "0.5",
            },
        )
        self.assertEqual(config.num_examples, 12)
        self.assertEqual(config.num_threads, 3)
        self.assertEqual(config.max_tokens, 512)
        self.assertEqual(config.threshold, 0.5)

    def test_jobset_peer_hostname_mapping(self) -> None:
        self.assertEqual(
            peer_group_hostname("run-prefill-0-0.run", ROLE_PREFILL, ROLE_DECODE),
            "run-decode-0-0.run",
        )

    def test_launcher_directory_resolves_from_this_file(self) -> None:
        for script in ("launch_prefill.sh", "launch_decode.sh"):
            with self.subTest(script=script):
                self.assertTrue(os.path.isfile(os.path.join(BENCH, script)))


if __name__ == "__main__":
    if not E("CI_PD_ROLE"):
        sys.argv = [sys.argv[0], "TestPdPrefixCachingConfiguration"]
        unittest.main(verbosity=2)

    topology = PodTopology.from_env()
    if topology.is_leader:
        sys.argv = [sys.argv[0], "TestPdPrefixCaching"]
        unittest.main(verbosity=2)
    sys.exit(_serve_role(topology, GPT_OSS_120B))
