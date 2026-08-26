"""GPT-OSS-120B P/D disaggregation GSM8K quality gate, one host per engine.

The smallest topology that still runs real gpt-oss-120b P/D: prefill takes one
host at TP8/EP8, decode takes another, and raiden carries the KV between them.
The shipping lane (test_tpu_gpt_oss_120b_pd_multihost_gsm8k.py) gives each
engine two hosts, so its transport crosses both the slice-to-slice fabric and
the intra-engine one, and a red run cannot say which. Here each engine is a
single slice, so a failure is the P/D transport rather than the fabric beneath
it, and the run costs two 1-host slices instead of two 2-host ones.

Roles come from CI_PD_ROLE, one pod each. The prefill pod is also the leader:
it finds the decode pod by swapping the replicated-job segment of its own
hostname, brings up the mini-LB locally, and runs the eval. The LB rides the
prefill pod rather than a pod of its own because the Kueue quota covers only
the two TPU slices.

GSM8K over the LB is the quality signal, since disaggregation moves the whole
prefix KV between engines before the first decode step and a page that arrives
corrupted or out of order surfaces as a wrong answer rather than an error. The
transport gate ahead of it reads both engines' modes and backends, which is
what makes the score mean something -- a decode-mode engine cannot prefill on
its own, so every answer scored necessarily crossed raiden.
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import unittest
import urllib.error
import urllib.request
from typing import Any, Dict, List, Mapping, Optional, Tuple

SGLANG_IMPORT_ERROR = None
try:
    from sglang.test.ci.ci_register import register_tpu_ci
    from sglang.test.tpu import model_config_main
except ModuleNotFoundError as exc:
    # Source-only checkout: the host-side configuration tests still run.
    SGLANG_IMPORT_ERROR = exc

    def register_tpu_ci(*_args, **_kwargs):
        return None

    def model_config_main(*_args, **_kwargs):
        return None


# Two full gpt-oss-120b boots in parallel, each loading weights and compiling on
# its own slice, then the eval through the LB.
register_tpu_ci(est_time=7200, suite="tpu-integration", nightly=True)

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

# One host per engine, and a host is 8 chiplets. Unlike the co-tenant Qwen3 lane
# there is no chiplet pinning here: each engine owns its slice outright, so
# launch_server.sh leaves every chiplet visible and the slicebuilder lays out one
# rank per chiplet.
HOSTS_PER_ENGINE = 1
CHIPLETS_PER_HOST = 8
TP_SIZE = HOSTS_PER_ENGINE * CHIPLETS_PER_HOST
EP_SIZE = TP_SIZE

# Separate pods, so both engines can take the conventional port and raiden can
# take one base. Only the LB needs a port of its own, on the prefill pod.
PREFILL_PORT = int(E("PREFILL_PORT", "30000"))
DECODE_PORT = int(E("DECODE_PORT", "30000"))
LB_PORT = int(E("LB_PORT", "8000"))
BOOTSTRAP_PORT = int(E("BOOTSTRAP_PORT", "8998"))
RAIDEN_KV_PORT = int(E("SGLANG_RAIDEN_KV_PORT", "31000"))
RAIDEN_PORTS_PER_RANK = 4
LOCAL_HOST = "127.0.0.1"

WAIT_SECS = int(E("WAIT_SECS", "3600"))
KEEP_UP = E("KEEP_UP", "0") == "1"
PYTHON = E("PYTHON", "python3")

# The JobSet renderer, which lives under .dockerignore's .github/ entry and so
# is absent from the runtime image. Anything that shells out to it only works
# from a checkout, which is why the dry-run test below is guarded on this path.
RENDER_JOBSET = os.path.join(REPO, ".github", "tpu-ci", "render_jobset.py")

# The JobSet label for this lane, and the only variable part of the pod names
# JobSet generates from it, which k8s caps at 63 characters. A longer one gets
# the whole manifest rejected at submission, after the image build and the
# queue wait. test_run_name_fits_the_generated_pod_name pins the budget.
RUN_NAME = "gpt-oss-120b-pd-singlehost-gsm8k"

PREFILL_LOG = "/tmp/pd_singlehost_prefill.log"
DECODE_LOG = "/tmp/pd_singlehost_decode.log"
LB_LOG = "/tmp/pd_singlehost_lb.log"

PAGE_SIZE = 64
MEM_FRACTION_STATIC = "0.35"
# The shipping gpt-oss config: MXFP4 weights as checkpointed, FP8 KV. raiden is
# dtype-agnostic, it moves the fused pool's pages as bytes.
QUANTIZATION = ""
KV_CACHE_DTYPE = "fp8_e4m3"
# Half the two-host lane's budget, because one engine here has half the
# aggregate HBM. Bounded explicitly rather than left to mem_fraction so the two
# engines agree on a pool size the transport can pair up.
MAX_TOTAL_TOKENS = 393216
DISAGG_TRANSFER_BACKEND = "raiden"
# Everything this lane moves between device and host rides raiden, both the P/D
# hop above and the decode retraction host pool below.
HICACHE_IO_BACKEND = "raiden"
EXTRA_SERVER_ARGS: Tuple[str, ...] = ("--reasoning-parser", "gpt-oss")
# SGLANG_TPU_FUSED_MOE_SC_GATHER defaults on, and on gpt-oss's MoE the SparseCore
# gather fails to lower: _extract_indirect_offsets_from_indices rejects the
# fused_moe indexer shape, so the backend compile raises during the prefill
# engine's first forward and the scheduler dies before it can serve /health.
# Every other gpt-oss lane in the tree pins it off for the same reason.
EXTRA_LAUNCH_ENV: Dict[str, str] = {"SGLANG_TPU_FUSED_MOE_SC_GATHER": "0"}

# Inherited from the two-host lane so the two are directly comparable: halving
# the hosts changes where KV lives, not what the model computes.
GSM8K_BASELINE = float(E("GSM8K_THRESHOLD_GPT_OSS_120B", E("GSM8K_THRESHOLD", "0.72")))
ATOL = float(E("GSM8K_ATOL", "0.03"))
NUM_EXAMPLES = int(E("GSM8K_NUM_EXAMPLES", "100"))
NUM_THREADS = int(E("GSM8K_NUM_THREADS", "8"))
MAX_TOKENS = int(E("GSM8K_MAX_TOKENS", "1024"))
HEALTH_TIMEOUT = float(E("HEALTH_TIMEOUT", "3600"))


def resolve_model_path(env: Optional[Mapping[str, str]] = None) -> str:
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


def resolve_role(env: Optional[Mapping[str, str]] = None) -> str:
    """Read this pod's P/D role, which the JobSet template sets per replica."""
    values = os.environ if env is None else env
    role = values.get(ROLE_ENV, "")
    if role not in (PREFILL_ROLE, DECODE_ROLE):
        raise ValueError(
            f"{ROLE_ENV} must be {PREFILL_ROLE!r} or {DECODE_ROLE!r}, got {role!r}"
        )
    return role


def self_hostname(env: Optional[Mapping[str, str]] = None) -> str:
    """This pod's own DNS name.

    On a single-host slice the TPU webhook may leave TPU_WORKER_HOSTNAMES unset,
    which is why the pod name is the fallback rather than the other way round.
    """
    values = os.environ if env is None else env
    hostnames = [h for h in values.get("TPU_WORKER_HOSTNAMES", "").split(",") if h]
    return hostnames[0] if hostnames else socket.gethostname()


def peer_hostname(hostname: str, *, from_role: str, to_role: str) -> str:
    """Map this pod's DNS name onto the pod of the peer role.

    The two roles are replicated jobs of one JobSet, so their pods differ only in
    that segment and enableDNSHostnames makes the swapped name resolve. render_
    jobset.py rejects a run name containing either word, which is what keeps the
    swap unambiguous.
    """
    needle, replacement = f"-{from_role}-", f"-{to_role}-"
    if needle not in hostname:
        raise ValueError(f"hostname {hostname!r} has no {needle!r} segment")
    return hostname.replace(needle, replacement)


def _common_launch_env(env: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    values = os.environ if env is None else env
    return {
        "MODEL": resolve_model_path(values),
        "TP_SIZE": str(TP_SIZE),
        "EP_SIZE": str(EP_SIZE),
        "PAGE_SIZE": str(PAGE_SIZE),
        # raiden reads device pages through the fused KV pool, which only the RPA
        # v3 backend allocates. Pinned so a launcher default cannot take the
        # transport under test away from this gate.
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


def _shared_server_args() -> List[str]:
    # The launchers hardcode the native "tpu" transport then append EXTRA_ARGS,
    # so naming raiden here is what selects it -- argparse keeps the last one.
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


def build_prefill_env(env: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
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


def _decode_host_pool_args() -> List[str]:
    """Make the decode retraction host pool buildable on a fused TPU KV pool.

    Nothing here enables hierarchical cache. A PD decode server turns on a
    host_pool retraction backup by itself and builds it from the device pool,
    which under RPA v3 is fused, so the GPU-shaped defaults abort the decode
    boot. raiden is the only io backend that works: gpt-oss is hybrid-SWA, and
    the fused host pool mirrors the full-attention pool alone, leaving the
    sliding-window sub-pool with no host storage. raiden gives each sub-pool its
    own. The retraction path is not what this lane measures, it just has to
    survive startup, and it stays on the same transport as everything else here.
    """
    return [
        "--hicache-mem-layout",
        "page_first_fused",
        "--hicache-io-backend",
        HICACHE_IO_BACKEND,
    ]


def build_decode_env(env: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
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


def _tee_command(argv: Tuple[str, ...], log_path: str) -> Tuple[str, ...]:
    quoted = " ".join(f"'{a}'" for a in argv)
    return ("bash", "-c", f"set -o pipefail; {quoted} 2>&1 | tee -a '{log_path}'")


def _launch(
    cmd: Tuple[str, ...],
    log_path: str,
    extra_env: Optional[Dict[str, str]] = None,
    cwd: Optional[str] = None,
) -> subprocess.Popen:
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    return subprocess.Popen(
        _tee_command(cmd, log_path=log_path),
        env=env,
        start_new_session=True,
        cwd=cwd,
    )


def execute_prefill(log_path: str = PREFILL_LOG) -> subprocess.Popen:
    """Launch the prefill engine across this pod's whole slice."""
    return _launch(
        ("bash", os.path.join(BENCH, "launch_prefill.sh")),
        log_path=log_path,
        extra_env=build_prefill_env(),
        cwd=BENCH,
    )


def execute_decode(log_path: str = DECODE_LOG) -> subprocess.Popen:
    """Launch the decode engine across this pod's whole slice."""
    return _launch(
        ("bash", os.path.join(BENCH, "launch_decode.sh")),
        log_path=log_path,
        extra_env=build_decode_env(),
        cwd=BENCH,
    )


def execute_router(
    *, prefill_host: str, decode_host: str, log_path: str = LB_LOG
) -> subprocess.Popen:
    """Launch the mini-LB in front of the prefill and decode pods."""
    cmd = (
        PYTHON,
        "-m",
        "sglang_router.launch_router",
        "--pd-disaggregation",
        "--mini-lb",
        "--prefill",
        f"http://{prefill_host}:{PREFILL_PORT}",
        "--decode",
        f"http://{decode_host}:{DECODE_PORT}",
        "--host",
        "0.0.0.0",
        "--port",
        str(LB_PORT),
    )
    return _launch(cmd, log_path=log_path)


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
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return 200 <= r.status < 300
    except (urllib.error.URLError, TimeoutError, OSError, ConnectionError):
        return False


def _get_json(url: str, timeout: int = 30) -> Dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _wait_http_200(
    *, url: str, secs: int, procs: Tuple[Tuple[str, subprocess.Popen], ...] = ()
) -> None:
    """Poll url until it answers 200, failing early if a local process dies.

    Reports the process that died rather than the url being polled. Only local
    processes can be watched this way, so a decode pod that aborts is caught by
    the timeout instead.
    """
    deadline = time.time() + secs
    while time.time() < deadline:
        for name, p in procs:
            if p is not None and p.poll() is not None:
                raise RuntimeError(
                    f"{name} exited rc={p.returncode} while waiting on {url}"
                )
        if _http_ok(url):
            return
        time.sleep(10)
    raise TimeoutError(f"{url} not ready after {secs}s")


def transport_failures(*, prefill_info: Dict[str, Any], decode_info: Dict[str, Any]):
    """Reject a run where the two engines are not a raiden P/D pair.

    A GSM8K score alone cannot tell this lane apart from two ordinary servers
    behind a load balancer, so without this gate the eval could pass while the
    disaggregation path never ran. Both halves are checked, since either one
    degrading -- wrong role, or a fallback to the native "tpu" transport -- is
    enough to make the score meaningless.
    """
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
    """Build the gate that proves the score below came over the P/D transport."""

    def run(base_url: str) -> Dict[str, Any]:
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


def build_model_config(*, decode_host: str) -> Dict[str, Any]:
    """Assemble the gate list, which needs the decode pod's resolved name."""
    return {
        "model": SERVED_MODEL_NAME,
        "test_name": "gpt-oss-120b-pd-singlehost-gsm8k",
        "variant": f"PD-1host-TP{TP_SIZE}+EP{EP_SIZE}-raiden",
        "launch": {
            # This pod starts prefill and the LB before handing over, and decode
            # runs on its peer, so the shared runner attaches rather than
            # launching.
            "external": True,
            "health_timeout": HEALTH_TIMEOUT,
            "tag": "gpt-oss-120b-pd-singlehost-gsm8k-e2e",
        },
        # No dashboard key: feature validation on a config no accuracy lane
        # serves, so plotting its score beside one would compare two servers.
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


def run_prefill_leader() -> int:
    """Serve prefill, wait for the decode pod, then score GSM8K through the LB."""
    this_host = self_hostname()
    decode_host = peer_hostname(this_host, from_role=PREFILL_ROLE, to_role=DECODE_ROLE)
    procs: List[Tuple[str, subprocess.Popen]] = []
    try:
        procs.append((PREFILL_ROLE, execute_prefill()))
        _wait_http_200(
            url=f"http://{LOCAL_HOST}:{PREFILL_PORT}/health",
            secs=WAIT_SECS,
            procs=tuple(procs),
        )
        # No local process to watch for the peer, so this one rides the timeout.
        _wait_http_200(url=f"http://{decode_host}:{DECODE_PORT}/health", secs=WAIT_SECS)

        procs.append(
            ("router", execute_router(prefill_host=this_host, decode_host=decode_host))
        )
        _wait_http_200(
            url=f"http://{LOCAL_HOST}:{LB_PORT}/health", secs=300, procs=tuple(procs)
        )

        model_config_main(
            build_model_config(decode_host=decode_host),
            base_url=f"http://{LOCAL_HOST}:{LB_PORT}",
        )
        return 0
    finally:
        if not KEEP_UP:
            for _, p in procs:
                _kill(p)


def run_decode_worker() -> int:
    """Serve decode until the engine exits, which the JobSet deadline bounds."""
    proc = execute_decode()
    try:
        while True:
            rc = proc.poll()
            if rc is not None:
                return rc
            time.sleep(10)
    finally:
        _kill(proc)


def run_integration() -> int:
    """Dispatch on the role this pod was given."""
    validate_model_source(resolve_model_path())
    if resolve_role() == PREFILL_ROLE:
        return run_prefill_leader()
    return run_decode_worker()


def render_jobset_yaml(
    image: str = "us-central1-docker.pkg.dev/sgl-project/tpu-ci/sglang:latest",
    lane: str = "nightly",
    weights_pvc: str = "model-weights-rox",
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
        # Names the dashboard row and the TIMINGS record. The renderer rejects
        # P/D mode without it rather than emit a record with an empty `file`,
        # so this has to match what --exec actually runs.
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
            # Chiplet pinning belongs to the co-tenant lane. Naming any of these
            # here would mask chiplets and break the slicebuilder handshake that
            # launch_server.sh relies on above TP=1.
            for var in (
                "TPU_VISIBLE_DEVICES",
                "TPU_VISIBLE_CHIPS",
                "TPU_CHIPS_PER_HOST_BOUNDS",
                "TPU_HOST_BOUNDS",
            ):
                self.assertNotIn(var, env)

    def test_both_engines_disable_the_sparsecore_moe_gather(self) -> None:
        # Regression: this lane shipped without it and every run died the same
        # way. The env var defaults on, and on gpt-oss's MoE the SparseCore
        # gather has no lowering for the fused_moe indexer shape, so the prefill
        # engine's first forward raises BackendCompilerFailed and the scheduler
        # is gone before /health ever answers. The gate then reports a health
        # timeout, which reads like a slow boot rather than a launch flag.
        for env in (build_prefill_env(env={}), build_decode_env(env={})):
            self.assertEqual(env["SGLANG_TPU_FUSED_MOE_SC_GATHER"], "0")

    def test_both_engines_select_the_raiden_transport(self) -> None:
        for env in (build_prefill_env(env={}), build_decode_env(env={})):
            # The launchers hardcode the native transport before appending
            # EXTRA_ARGS, so this is an override, not a default.
            self.assertIn(
                f"--disaggregation-transfer-backend {DISAGG_TRANSFER_BACKEND}",
                env["EXTRA_ARGS"],
            )
            self.assertEqual(env["ATTENTION_BACKEND"], "pallas_tpu_rpa_v3")
            self.assertEqual(env["KV_CACHE_DTYPE"], KV_CACHE_DTYPE)
            self.assertEqual(env["QUANTIZATION"], "")
            self.assertIn("--reasoning-parser gpt-oss", env["EXTRA_ARGS"])

    def test_decode_can_build_its_retraction_host_pool(self) -> None:
        # A PD decode server enables the host_pool backup itself, and the
        # GPU-shaped defaults abort the boot on a fused TPU pool. torch_tpu is
        # not an option to fall back to: server_args rejects it outright for a
        # hybrid-SWA model, so naming raiden is what keeps the decode boot alive.
        decode_args = build_decode_env(env={})["EXTRA_ARGS"]
        self.assertIn("--hicache-mem-layout page_first_fused", decode_args)
        self.assertIn("--hicache-io-backend raiden", decode_args)
        self.assertNotIn("torch_tpu", decode_args)
        # Prefill has no retraction path, so it must not carry the flags either.
        self.assertNotIn("hicache", build_prefill_env(env={})["EXTRA_ARGS"])

    def test_no_speculative_or_hierarchical_cache_features(self) -> None:
        for env in (build_prefill_env(env={}), build_decode_env(env={})):
            self.assertNotIn("speculative", env["EXTRA_ARGS"])
            self.assertNotIn("hierarchical-cache", env["EXTRA_ARGS"])

    def test_raiden_port_window_avoids_the_service_ports(self) -> None:
        for env in (build_prefill_env(env={}), build_decode_env(env={})):
            base = int(env["SGLANG_RAIDEN_KV_PORT"])
            window = range(base, base + RAIDEN_PORTS_PER_RANK * TP_SIZE)
            for port in (PREFILL_PORT, DECODE_PORT, LB_PORT, BOOTSTRAP_PORT):
                self.assertNotIn(port, window)

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
        # A single-host slice may carry no TPU_WORKER_HOSTNAMES at all, which is
        # the case the two-host lane never hits.
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
        # Regression: two lanes shipped with a run name a few characters too
        # long and the JobSet webhook rejected the whole manifest, so the lane
        # died at submission with no pod and no log to read. k8s caps a pod
        # name at 63, and JobSet builds it from the run name plus a fixed
        # tail -- the JobSet's own generated suffix, the longer of the two
        # role names, the job and pod indices, and the pod suffix.
        pod_name = f"tpu-ci-{RUN_NAME}-xxxxx-prefill-0-0-xxxxx"
        self.assertLessEqual(len(pod_name), 63, pod_name)

    # Skipped in-pod: the gates run under model_config_main, whose bare
    # unittest.main() collects this class too, and the renderer this shells
    # out to is excluded from the image by .dockerignore. Left unguarded it
    # fails a lane whose engines and gates were all green.
    @unittest.skipUnless(
        os.path.exists(RENDER_JOBSET), "renderer is not shipped in the runtime image"
    )
    def test_dry_run_render_jobset_yaml(self) -> None:
        # The nightly pd-singlehost-e2e job passes this run_name and this exec
        # command, so a rename on either side has to land on both.
        yaml_content = render_jobset_yaml()
        self.assertIn(
            "generateName: tpu-ci-gpt-oss-120b-pd-singlehost-gsm8k-", yaml_content
        )
        self.assertIn("name: prefill", yaml_content)
        self.assertIn("name: decode", yaml_content)
        self.assertNotIn("name: router", yaml_content)
        self.assertIn("CI_PD_ROLE", yaml_content)
        self.assertIn(
            "python3 test/registered/tpu/e2e/test_tpu_gpt_oss_120b_pd_singlehost_gsm8k.py integration",
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

    if args.mode == "integration":
        # model_config_main ends in unittest.main(), which reads sys.argv, so
        # the mode word has to go or the loader hunts for a test named after it.
        sys.argv = [sys.argv[0], *extra_args]
        sys.exit(run_integration())
    sys.argv = [sys.argv[0], "TestPdSingleHostGptOss120bConfiguration", *extra_args]
    unittest.main(verbosity=2)
