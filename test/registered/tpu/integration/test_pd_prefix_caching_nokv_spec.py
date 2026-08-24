"""P/D disaggregation x prefix caching x EAGLE3 test without KV offload.

Topology: a multihost PREFILL engine and a multihost DECODE engine, each on its
own TPU slice, connected by the PD mini load balancer. EAGLE3 runs only on the
decode engine. Device radix caching stays enabled on prefill, while HiCache/KV
offload is explicitly disabled on both engines.

The prefill leader gates three properties through the load balancer:
  * a repeated long prompt reports a device prefix-cache hit and produces the
    same greedy output as its cold request;
  * the warm request performs speculative verification and its accept length
    exceeds the selected model profile's floor;
  * GSM8K clears the selected model profile's quality floor.

Every pod in both slices runs this file. ``CI_PD_ROLE`` selects the engine role
and the slice-local ``TPU_WORKER_*`` variables select the multihost rank. Without
``CI_PD_ROLE``, only the host-side configuration tests run.

Run one model profile per invocation:

  MODEL_PROFILE=gpt_oss_120b CI_PD_ROLE=prefill python3 \
      test/registered/tpu/integration/test_pd_prefix_caching_nokv_spec.py

Model paths, probe sizes, quality floors, and accept-length floors are
environment-overridable through the configuration structs below.
"""

import json
import os
import signal
import socket
import subprocess
import sys
import time
import unittest
import urllib.request
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional, Tuple

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
PREFILL_LOG = os.path.join(LOG_DIR, "pd_nokv_spec_prefill.log")
DECODE_LOG = os.path.join(LOG_DIR, "pd_nokv_spec_decode.log")
LB_LOG = os.path.join(LOG_DIR, "pd_nokv_spec_lb.log")


class ModelProfile(msgspec.Struct, frozen=True, kw_only=True):
    key: str
    display_name: str
    served_name: str
    default_model_path: str
    model_env: str
    default_draft_model_path: str
    draft_model_env: str
    page_size: int
    mem_fraction_prefill: str
    mem_fraction_decode: str
    quantization: str
    kv_cache_dtype: str
    max_total_tokens: int
    gsm8k_threshold: float
    accept_length_threshold: float
    disagg_transfer_backend: str
    extra_server_args: Tuple[str, ...] = ()
    extra_launch_env: Dict[str, str] = {}
    extra_decode_env: Dict[str, str] = {}
    reasoning_effort: Optional[str] = None
    chat_template_kwargs: Optional[Dict[str, Any]] = None


MODEL_PROFILES = {
    "gpt_oss_120b": ModelProfile(
        key="gpt_oss_120b",
        display_name="GPT-OSS-120B",
        served_name="gpt-oss-120b",
        default_model_path="/weights/gpt-oss-120b",
        model_env="GPT_OSS_MODEL",
        default_draft_model_path="lmsys/EAGLE3-gpt-oss-120b-bf16",
        draft_model_env="GPT_OSS_DRAFT_MODEL",
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
        # Functional floor used by the existing TPU EAGLE3 test. The remote
        # bring-up can calibrate a tighter PD-specific floor.
        accept_length_threshold=1.0,
        # The native TPU connector rejects GPT-OSS's hybrid SWA KV pool.
        disagg_transfer_backend="raiden",
        extra_server_args=("--reasoning-parser", "gpt-oss"),
        extra_launch_env={"SGLANG_TPU_FUSED_MOE_SC_GATHER": "0"},
        extra_decode_env={"SGLANG_TPU_ALLOW_SPEC_RECOMPILE": "1"},
        reasoning_effort="low",
    ),
    "qwen3_coder_480b": ModelProfile(
        key="qwen3_coder_480b",
        display_name="Qwen3-Coder-480B",
        served_name="qwen3-coder-480b",
        default_model_path="/weights/Qwen3-Coder-480B-A35B-Instruct-FP8",
        model_env="QWEN3_CODER_MODEL",
        default_draft_model_path=(
            "/weights/" "SGLang-EAGLE3-Qwen3-Coder-480B-A35B-Instruct-SpecForge-EigenAI"
        ),
        draft_model_env="QWEN3_CODER_DRAFT_MODEL",
        page_size=256,
        mem_fraction_prefill="0.6",
        mem_fraction_decode="0.6",
        quantization="fp8",
        kv_cache_dtype="fp8_e4m3",
        max_total_tokens=32_768,
        gsm8k_threshold=0.80,
        accept_length_threshold=1.0,
        disagg_transfer_backend="tpu",
        extra_server_args=("--tpu-fp8-runtime", "channelwise"),
        chat_template_kwargs={"enable_thinking": False},
    ),
}


class CacheSpecProbeConfig(msgspec.Struct, frozen=True, kw_only=True):
    page_size: int
    target_words: int
    max_new_tokens: int
    accept_length_threshold: float

    @classmethod
    def from_env(
        cls, profile: ModelProfile, env: Optional[Mapping[str, str]] = None
    ) -> "CacheSpecProbeConfig":
        values = os.environ if env is None else env
        suffix = profile.key.upper()
        threshold = float(
            values.get(
                f"ACCEPT_LENGTH_THRESHOLD_{suffix}",
                values.get(
                    "ACCEPT_LENGTH_THRESHOLD",
                    str(profile.accept_length_threshold),
                ),
            )
        )
        if threshold < 1.0:
            raise ValueError(
                "accept-length threshold must be at least 1.0, " f"got {threshold}"
            )
        return cls(
            page_size=profile.page_size,
            target_words=int(values.get("CACHE_PROBE_TARGET_WORDS", "2048")),
            max_new_tokens=int(values.get("CACHE_PROBE_MAX_NEW_TOKENS", "256")),
            accept_length_threshold=threshold,
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
        suffix = profile.key.upper()
        threshold = float(
            values.get(
                f"GSM8K_THRESHOLD_{suffix}",
                values.get("GSM8K_THRESHOLD", str(profile.gsm8k_threshold)),
            )
        )
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


def select_profile(raw: str) -> ModelProfile:
    key = raw.strip()
    if key not in MODEL_PROFILES:
        allowed = ", ".join(MODEL_PROFILES)
        raise ValueError(
            f"MODEL_PROFILE must name exactly one of: {allowed} (got {key!r})"
        )
    return MODEL_PROFILES[key]


def resolve_model_path(
    profile: ModelProfile, env: Optional[Mapping[str, str]] = None
) -> str:
    values = os.environ if env is None else env
    return (
        values.get(profile.model_env)
        or values.get("MODEL")
        or profile.default_model_path
    )


def resolve_draft_model_path(
    profile: ModelProfile, env: Optional[Mapping[str, str]] = None
) -> str:
    values = os.environ if env is None else env
    return (
        values.get(profile.draft_model_env)
        or values.get("DRAFT_MODEL")
        or profile.default_draft_model_path
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
        # Pin the negative feature contract. An inherited shell value must not
        # silently turn this test into another HiCache/offload scenario.
        "KV_OFFLOAD": "0",
        "ENABLE_TORCH_COMPILE": values.get("ENABLE_TORCH_COMPILE", "1"),
        "ENABLE_PROFILER_ENDPOINT": "0",
        "DUMP_META": "0",
        "PYTHON": PYTHON,
        "SGLANG_RAIDEN_KV_PORT": values.get("SGLANG_RAIDEN_KV_PORT", "31000"),
        **profile.extra_launch_env,
    }


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
        (
            "--served-model-name",
            profile.served_name,
            *profile.extra_server_args,
            "--disaggregation-transfer-backend",
            profile.disagg_transfer_backend,
            "--skip-server-warmup",
            "--sampling-backend",
            "pytorch",
            values.get("EXTRA_PREFILL_ARGS", ""),
        )
    ).strip()
    return launch_env


def build_decode_env(
    profile: ModelProfile,
    group_hosts: int,
    env: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    values = os.environ if env is None else env
    launch_env = _common_launch_env(profile, group_hosts, values)
    launch_env.update(profile.extra_decode_env)
    launch_env["MEM_FRACTION_STATIC"] = values.get(
        "MEM_FRACTION_STATIC", profile.mem_fraction_decode
    )
    launch_env["PORT"] = str(DECODE_PORT)
    draft_model_path = resolve_draft_model_path(profile, values)
    launch_env["DRAFT_MODEL"] = draft_model_path
    launch_env["EXTRA_ARGS"] = " ".join(
        (
            "--served-model-name",
            profile.served_name,
            *profile.extra_server_args,
            "--disaggregation-transfer-backend",
            profile.disagg_transfer_backend,
            "--skip-server-warmup",
            "--sampling-backend",
            "pytorch",
            "--speculative-algorithm",
            "EAGLE3",
            "--speculative-draft-model-path",
            draft_model_path,
            "--speculative-num-steps",
            values.get("SPECULATIVE_NUM_STEPS", "3"),
            "--speculative-eagle-topk",
            values.get("SPECULATIVE_EAGLE_TOPK", "1"),
            "--speculative-num-draft-tokens",
            values.get("SPECULATIVE_NUM_DRAFT_TOKENS", "4"),
            values.get("EXTRA_DECODE_ARGS", ""),
        )
    ).strip()
    return launch_env


_VOCAB = (
    "the of and to in a is that for it as was with on be by at this from but not "
    "are or his they she had which were we when your can said there use each "
    "their time will how other about many then them these so some her would make "
    "like him into has two more very after our just name good through where help "
    "much before line right too mean old"
).split()


def make_prompt(index: int, nwords: int) -> str:
    seed = (index + 1) * 2654435761 & 0xFFFFFFFF
    words = []
    for _ in range(nwords):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        words.append(_VOCAB[seed % len(_VOCAB)])
    return (
        f"Document {index} header alpha{index} beta{index * 7}. "
        + " ".join(words)
        + "\nContinue the document with a detailed numbered list:"
    )


class GenerationRecord(msgspec.Struct, frozen=True, kw_only=True):
    text: str
    prompt_tokens: int
    completion_tokens: int
    device_cached_tokens: int
    host_cached_tokens: int
    spec_verify_count: int
    spec_accept_length: float
    latency_seconds: float


def _generate(base_url: str, prompt: str, max_new_tokens: int) -> GenerationRecord:
    started = time.perf_counter()
    request = urllib.request.Request(
        base_url.rstrip("/") + "/generate",
        data=json.dumps(
            {
                "text": prompt,
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": max_new_tokens,
                },
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        payload = json.loads(response.read())
    meta = payload.get("meta_info") or {}
    details = meta.get("cached_tokens_details") or {}
    completion_tokens = int(meta.get("completion_tokens") or 0)
    verify_count = int(meta.get("spec_verify_ct") or 0)
    accept_length = completion_tokens / verify_count if verify_count else 0.0
    device_tokens = details.get("device")
    if device_tokens is None:
        device_tokens = meta.get("cached_tokens") or 0
    return GenerationRecord(
        text=payload["text"],
        prompt_tokens=int(meta.get("prompt_tokens") or 0),
        completion_tokens=completion_tokens,
        device_cached_tokens=int(device_tokens),
        host_cached_tokens=int(details.get("host") or 0),
        spec_verify_count=verify_count,
        spec_accept_length=accept_length,
        latency_seconds=round(time.perf_counter() - started, 3),
    )


def _flush_cache(url: str) -> None:
    with urllib.request.urlopen(url.rstrip("/") + "/flush_cache", timeout=60) as r:
        r.read()


def validate_cache_spec_probe(
    cold: GenerationRecord,
    warm: GenerationRecord,
    page_size: int,
    accept_length_threshold: float,
) -> int:
    """Validate the combined P/D, device-cache, and speculative warm path."""
    if cold.prompt_tokens < 2 * page_size:
        raise AssertionError(f"target prompt too short: {cold.prompt_tokens} tokens")
    if cold.device_cached_tokens or cold.host_cached_tokens:
        raise AssertionError(
            "cold request unexpectedly hit cache: "
            f"device={cold.device_cached_tokens} host={cold.host_cached_tokens}"
        )
    if not cold.text.strip() or cold.completion_tokens <= 0:
        raise AssertionError(
            "cold request produced no decode output, P/D KV transfer is broken"
        )
    if cold.spec_verify_count <= 0:
        raise AssertionError("cold request did not perform speculative verification")

    expected = max(page_size, (cold.prompt_tokens // page_size - 1) * page_size)
    if warm.device_cached_tokens < expected:
        raise AssertionError(
            "repeated request missed the device prefix cache: "
            f"device={warm.device_cached_tokens} expected>={expected}"
        )
    if warm.host_cached_tokens:
        raise AssertionError(
            "warm request unexpectedly loaded host KV with offload disabled: "
            f"host={warm.host_cached_tokens}"
        )
    if warm.text != cold.text:
        raise AssertionError("warm greedy output diverged from the cold baseline")
    if warm.spec_verify_count <= 0:
        raise AssertionError("warm cache-hit request did not run speculative decoding")
    if warm.spec_accept_length <= accept_length_threshold:
        raise AssertionError(
            f"warm accept_length={warm.spec_accept_length:.4f}, expected > "
            f"{accept_length_threshold:.4f}"
        )
    return expected


def run_cache_spec_probe(
    base_url: str,
    flush_url: str,
    profile: ModelProfile,
    config: CacheSpecProbeConfig,
) -> float:
    _flush_cache(flush_url)
    prompt = make_prompt(0, config.target_words)
    cold = _generate(base_url, prompt, config.max_new_tokens)
    warm = _generate(base_url, prompt, config.max_new_tokens)
    expected = validate_cache_spec_probe(
        cold=cold,
        warm=warm,
        page_size=config.page_size,
        accept_length_threshold=config.accept_length_threshold,
    )
    labels = {"model": profile.served_name, "test": "pd_prefix_caching_nokv_spec"}
    dump_metric(
        "prefix_cache_device_hit_tokens", warm.device_cached_tokens, labels=labels
    )
    dump_metric("spec_accept_length", warm.spec_accept_length, labels=labels)
    print(
        json.dumps(
            {
                "cache_spec_probe": profile.key,
                "expected_cached_tokens": expected,
                "accept_length_threshold": config.accept_length_threshold,
                "cold": msgspec.to_builtins(cold),
                "warm": msgspec.to_builtins(warm),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return warm.spec_accept_length


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
            chat_template_kwargs=profile.chat_template_kwargs,
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


class TestPdPrefixCachingNoKvSpec(unittest.TestCase):
    """Runs on the prefill leader. Its verdict gates the whole two-slice run."""

    procs = []
    topo: PodTopology
    profile: ModelProfile

    @classmethod
    def setUpClass(cls) -> None:
        if SGLANG_IMPORT_ERROR is not None:
            raise SGLANG_IMPORT_ERROR
        cls.topo = PodTopology.from_env()
        cls.profile = select_profile(E("MODEL_PROFILE", ""))
        model_path = resolve_model_path(cls.profile)
        draft_model_path = resolve_draft_model_path(cls.profile)
        validate_model_source(
            cls.profile.display_name, model_path, cls.profile.model_env
        )
        validate_model_source(
            f"{cls.profile.display_name} EAGLE3 draft",
            draft_model_path,
            cls.profile.draft_model_env,
        )
        cls.cache_spec_config = CacheSpecProbeConfig.from_env(cls.profile)
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

    def test_01_prefix_cache_and_spec_acceptance(self) -> None:
        accept_length = run_cache_spec_probe(
            base_url=f"http://127.0.0.1:{LB_PORT}",
            flush_url=f"http://127.0.0.1:{PREFILL_PORT}",
            profile=self.profile,
            config=self.cache_spec_config,
        )
        print(
            f"warm accept_length={accept_length:.4f} floor>"
            f"{self.cache_spec_config.accept_length_threshold:.4f}"
        )

    def test_02_gsm8k_quality(self) -> None:
        score = run_gsm8k(
            base_url=f"http://127.0.0.1:{LB_PORT}",
            profile=self.profile,
            config=self.gsm8k_config,
        )
        dump_metric(
            "gsm8k_score",
            score,
            labels={
                "model": self.profile.served_name,
                "test": "pd_prefix_caching_nokv_spec",
            },
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


class TestPdNoKvSpecConfiguration(unittest.TestCase):
    """Host-side contracts that guard the intended feature combination."""

    def test_profile_selection_requires_one_model(self) -> None:
        self.assertIs(select_profile("gpt_oss_120b"), MODEL_PROFILES["gpt_oss_120b"])
        for invalid in ("", "all", "gpt_oss_120b,qwen3_coder_480b", "unknown"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "MODEL_PROFILE"):
                    select_profile(invalid)

    def test_prefill_enables_device_cache_without_spec_or_offload(self) -> None:
        for profile in MODEL_PROFILES.values():
            with self.subTest(profile=profile.key):
                env = build_prefill_env(profile, 2, env={})
                self.assertEqual(env["DISABLE_RADIX_CACHE"], "0")
                self.assertEqual(env["ENABLE_CACHE_REPORT"], "1")
                self.assertEqual(env["KV_OFFLOAD"], "0")
                self.assertNotIn("hierarchical-cache", env["EXTRA_ARGS"])
                self.assertNotIn("hicache", env["EXTRA_ARGS"])
                self.assertNotIn("speculative-", env["EXTRA_ARGS"])

    def test_decode_enables_eagle3_without_offload(self) -> None:
        for profile in MODEL_PROFILES.values():
            with self.subTest(profile=profile.key):
                env = build_decode_env(profile, 2, env={})
                self.assertEqual(env["KV_OFFLOAD"], "0")
                self.assertNotIn("hierarchical-cache", env["EXTRA_ARGS"])
                self.assertNotIn("hicache", env["EXTRA_ARGS"])
                self.assertIn("--speculative-algorithm EAGLE3", env["EXTRA_ARGS"])
                self.assertIn(
                    "--speculative-draft-model-path "
                    f"{profile.default_draft_model_path}",
                    env["EXTRA_ARGS"],
                )

    def test_multihost_size_and_path_overrides_reach_both_roles(self) -> None:
        profile = MODEL_PROFILES["qwen3_coder_480b"]
        override = {
            "TP_SIZE": "32",
            "EP_SIZE": "16",
            "MODEL": "/model",
            "DRAFT_MODEL": "/draft",
        }
        prefill = build_prefill_env(profile, 2, env=override)
        decode = build_decode_env(profile, 2, env=override)
        self.assertEqual(prefill["TP_SIZE"], "32")
        self.assertEqual(decode["TP_SIZE"], "32")
        self.assertEqual(prefill["EP_SIZE"], "16")
        self.assertEqual(decode["EP_SIZE"], "16")
        self.assertEqual(prefill["MODEL"], "/model")
        self.assertEqual(decode["DRAFT_MODEL"], "/draft")

    def test_probe_requires_spec_on_the_warm_cache_hit(self) -> None:
        cold = GenerationRecord(
            text="same",
            prompt_tokens=1024,
            completion_tokens=64,
            device_cached_tokens=0,
            host_cached_tokens=0,
            spec_verify_count=32,
            spec_accept_length=2.0,
            latency_seconds=1.0,
        )
        warm = GenerationRecord(
            text="same",
            prompt_tokens=1024,
            completion_tokens=64,
            device_cached_tokens=960,
            host_cached_tokens=0,
            spec_verify_count=0,
            spec_accept_length=0.0,
            latency_seconds=0.5,
        )
        with self.assertRaisesRegex(AssertionError, "warm cache-hit request"):
            validate_cache_spec_probe(
                cold,
                warm,
                page_size=64,
                accept_length_threshold=1.0,
            )

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
        sys.argv = [sys.argv[0], "TestPdNoKvSpecConfiguration"]
        unittest.main(verbosity=2)

    topology = PodTopology.from_env()
    profile = select_profile(E("MODEL_PROFILE", ""))
    if topology.is_leader:
        sys.argv = [sys.argv[0], "TestPdPrefixCachingNoKvSpec"]
        unittest.main(verbosity=2)
    sys.exit(_serve_role(topology, profile))
