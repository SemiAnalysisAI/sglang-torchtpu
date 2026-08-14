"""P/D disaggregation with TP resharding integration test (issue #378).

Topology: a multihost PREFILL engine (TP16EP16) and a multihost DECODE engine
(DP2TP8EP16), each spanning its own TPU slice, connected by the PD mini load
balancer. Prefill engine uses attention TP=16; Decode engine uses DP=2 with
attention TP=8 per DP replica (total 16 devices).

The KV cache transfer reshards the prefill attention shards (16 ranks) to
decode attention shards (8 ranks per DP replica) via FAN_IN resharding (2:1).

The CI JobSet (.github/tpu-ci/jobset-tpu-pd.yaml) runs this file on every pod of
both slices; roles are dispatched from the CI_PD_ROLE env plus the slice-local
TPU worker env.

What each verification proves:
  * quality:          gsm8k through the LB must clear the model's accuracy floor
                      under TP resharding.
  * KV resharding:    generation requests dispatched to prefill -> decode across
                      both DP replicas produce valid, deterministic, and bit-equal
                      greedy text, certifying that KV cache was resharded and
                      transferred intact.

Roles (rank = slice-local TPU_WORKER_ID):
  prefill rank 0   launches the prefill engine, waits for decode, starts the
                   mini-LB, runs the resharding generation probe + gsm8k -- its
                   verdict gates the run.
  prefill rank >0  worker shard of the prefill engine.
  decode any rank  launches/joins the decode engine (DP2TP8EP16), serves until
                   the leader tears down.

Run one model profile per invocation on hardware:

  MODEL_PROFILE=gpt_oss_120b CI_PD_ROLE=prefill python3 \
      test/registered/tpu/integration/test_pd_tp_resharding.py

Without CI_PD_ROLE the file runs only its host-side configuration tests.
"""
import json
import os
import signal
import subprocess
import sys
import time
import unittest
import urllib.error
import urllib.request
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional, Tuple

try:
    import msgspec
except ModuleNotFoundError:
    class _MsgspecShim:
        class Struct:
            def __init_subclass__(cls, **kwargs):
                super().__init_subclass__()
                def _init(self, **kw):
                    for base in reversed(cls.__mro__):
                        for k, v in base.__dict__.items():
                            if not k.startswith("_") and not callable(v) and not isinstance(v, (property, classmethod, staticmethod)):
                                setattr(self, k, v)
                    for k, v in kw.items():
                        setattr(self, k, v)
                cls.__init__ = _init

        @staticmethod
        def to_builtins(obj):
            if hasattr(obj, "__dict__"):
                return {k: _MsgspecShim.to_builtins(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
            return obj

    msgspec = _MsgspecShim()

SGLANG_IMPORT_ERROR = None
try:
    from sglang.test.ci.ci_register import register_tpu_ci
    from sglang.test.test_utils import dump_metric
except ModuleNotFoundError as exc:
    # Keep the configuration tests runnable in a source-only checkout; the
    # integration cases need the full SGLang TPU environment anyway.
    SGLANG_IMPORT_ERROR = exc

    def register_tpu_ci(*_args, **_kwargs):
        return None

    def dump_metric(*_args, **_kwargs):
        return None


# One invocation runs ONE model profile: two engines (weight load + compile),
# the cache probe with device-pool churn, and gsm8k.
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

# Emitted by hardware_backend/tpu/raiden_transfer.py when the raiden host pool
# and DMA engine finish initializing; its absence means offload silently never
# engaged even if the probe below were to pass on the device tier alone.
RAIDEN_READY_MARKER = "Raiden KV transfer ready"

PREFILL_LOG = "/tmp/pd_prefill.log"
DECODE_LOG = "/tmp/pd_decode.log"
LB_LOG = "/tmp/pd_lb.log"


class ModelProfile(msgspec.Struct, frozen=True, kw_only=True):
    key: str
    display_name: str
    served_name: str
    default_model_path: str
    model_env: str
    page_size: int
    mem_fraction_prefill: str
    mem_fraction_decode: str
    quantization: str
    kv_cache_dtype: str
    # None sizes the KV pool from mem_fraction instead of an explicit token
    # count; device_pool_tokens_hint then feeds the churn workload sizing.
    max_total_tokens: Optional[int]
    device_pool_tokens_hint: int
    gsm8k_threshold: float
    extra_server_args: Tuple[str, ...] = ()
    extra_launch_env: Dict[str, str] = {}
    reasoning_effort: Optional[str] = None
    chat_template_kwargs: Optional[Dict[str, Any]] = None


MODEL_PROFILES = {
    "gpt_oss_120b": ModelProfile(
        key="gpt_oss_120b",
        display_name="GPT-OSS-120B",
        served_name="gpt-oss-120b",
        default_model_path="/weights/gpt-oss-120b",
        model_env="GPT_OSS_MODEL",
        page_size=64,
        mem_fraction_prefill="0.35",
        mem_fraction_decode="0.35",
        # GPT-OSS ships MXFP4 quantization_config; forcing fp8 would re-quantize.
        quantization="",
        kv_cache_dtype="fp8_e4m3",
        # An explicit small KV pool hangs the extend-graph precompile on this
        # model (issue #363), so the pool is sized from mem_fraction and the
        # churn workload from this hint (~what 0.35 yields per TP8 host).
        max_total_tokens=None,
        device_pool_tokens_hint=600_000,
        gsm8k_threshold=0.72,
        extra_server_args=("--reasoning-parser", "gpt-oss"),
        # The SparseCore ragged-gather MoE path cannot be lowered for GPT-OSS
        # (dynamic token dim in the extend graph).
        extra_launch_env={"SGLANG_TPU_FUSED_MOE_SC_GATHER": "0"},
        reasoning_effort="low",
    ),
    "qwen3_coder_480b": ModelProfile(
        key="qwen3_coder_480b",
        display_name="Qwen3-Coder-480B",
        served_name="qwen3-coder-480b",
        default_model_path="/weights/Qwen3-Coder-480B-A35B-Instruct-FP8",
        model_env="QWEN3_CODER_MODEL",
        page_size=64,
        mem_fraction_prefill="0.6",
        mem_fraction_decode="0.6",
        quantization="fp8",
        kv_cache_dtype="fp8_e4m3",
        # Small device pool so a short churn pass forces host offload.
        max_total_tokens=32768,
        device_pool_tokens_hint=32768,
        gsm8k_threshold=0.80,
        extra_server_args=("--tpu-fp8-runtime", "channelwise"),
        chat_template_kwargs={"enable_thinking": False},
    ),
}


class CacheProbeConfig(msgspec.Struct, frozen=True, kw_only=True):
    page_size: int
    device_pool_tokens: int
    target_words: int
    filler_words: int
    num_fillers: int
    max_new_tokens: int

    @classmethod
    def from_env(
        cls, profile: ModelProfile, env: Optional[Mapping[str, str]] = None
    ) -> "CacheProbeConfig":
        values = os.environ if env is None else env
        device_tokens = int(
            values.get(
                "CACHE_PROBE_DEVICE_TOKENS", str(profile.device_pool_tokens_hint)
            )
        )
        filler_words = int(values.get("CACHE_PROBE_FILLER_WORDS", "2048"))
        # ~1.3 tokens/word; sized to push ~1.2x the pool through it so the
        # churn evicts the warm target. Both are hints, not guarantees -- the
        # host-reload assertion is what actually proves eviction happened.
        filler_tokens_est = filler_words * 13 // 10
        default_fillers = (12 * device_tokens) // (10 * filler_tokens_est) + 2
        return cls(
            page_size=profile.page_size,
            device_pool_tokens=device_tokens,
            target_words=int(values.get("CACHE_PROBE_TARGET_WORDS", "2048")),
            filler_words=filler_words,
            num_fillers=int(
                values.get("CACHE_PROBE_NUM_FILLERS", str(default_fillers))
            ),
            max_new_tokens=int(values.get("CACHE_PROBE_MAX_NEW_TOKENS", "16")),
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
        threshold_raw = values.get(
            f"GSM8K_THRESHOLD_{suffix}",
            values.get("GSM8K_THRESHOLD", str(profile.gsm8k_threshold)),
        )
        threshold = float(threshold_raw)
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
                f"CI_PD_ROLE must be {ROLE_PREFILL!r} or {ROLE_DECODE!r}, got {role!r}"
            )
        hostnames = tuple(
            h for h in values.get("TPU_WORKER_HOSTNAMES", "").split(",") if h
        )
        if not hostnames:
            raise ValueError(
                "TPU_WORKER_HOSTNAMES is empty -- this test needs the "
                "slice-local GKE TPU env of a multihost node pool"
            )
        return cls(
            role=role,
            group_rank=int(values.get("TPU_WORKER_ID", "0")),
            group_hosts=len(hostnames),
            group_hostnames=hostnames,
        )


def peer_group_hostname(hostname: str, from_role: str, to_role: str) -> str:
    """Map a pod hostname of one replicated job onto its peer job's pod.

    JobSet pod hostnames are ``<jobset>-<replicatedJob>-<jobIdx>-<podIdx>`` (the
    DNS suffix, when present, is the jobset name). The replicated jobs in
    jobset-tpu-pd.yaml are named ``prefill``/``decode`` and the jobset name never
    contains ``-prefill-``/``-decode-`` (render_jobset.py rejects such run
    names), so the role segment is the unique match.
    """
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


def validate_model_source(profile: ModelProfile, model_path: str) -> None:
    """Fail clearly when CI points at an unprovisioned local checkpoint."""
    if not os.path.isabs(model_path):
        return  # HF repo ids resolve remotely
    if not os.path.isfile(os.path.join(model_path, "config.json")):
        raise FileNotFoundError(
            f"{profile.display_name} checkpoint incomplete or missing at "
            f"{model_path}; provision it on the weights volume or set "
            f"{profile.model_env}"
        )


def hicache_server_args(
    profile: ModelProfile, env: Optional[Mapping[str, str]] = None
) -> str:
    values = os.environ if env is None else env
    args = [
        "--enable-hierarchical-cache",
        "--hicache-ratio",
        values.get("HICACHE_RATIO", "2"),
        "--hicache-io-backend",
        "raiden",
        "--hicache-mem-layout",
        "page_first_fused",
        "--enable-cache-report",
    ]
    if profile.max_total_tokens is not None:
        args += ["--max-total-tokens", str(profile.max_total_tokens)]
    return " ".join(args)


def _common_launch_env(
    profile: ModelProfile, group_hosts: int, env: Optional[Mapping[str, str]] = None
) -> Dict[str, str]:
    values = os.environ if env is None else env
    tp_size = int(values.get("TP_SIZE", str(8 * group_hosts)))
    return {
        "MODEL": resolve_model_path(profile, values),
        "TP_SIZE": str(tp_size),
        "EP_SIZE": values.get("EP_SIZE", str(tp_size)),
        "PAGE_SIZE": str(profile.page_size),
        "ATTENTION_BACKEND": "pallas_tpu_rpa_v3",
        "QUANTIZATION": profile.quantization,
        "KV_CACHE_DTYPE": profile.kv_cache_dtype,
        "DISABLE_RADIX_CACHE": "0",
        "ENABLE_TORCH_COMPILE": values.get("ENABLE_TORCH_COMPILE", "1"),
        "PYTHON": PYTHON,
        **profile.extra_launch_env,
    }


def build_prefill_env(
    profile: ModelProfile, group_hosts: int, env: Optional[Mapping[str, str]] = None
) -> Dict[str, str]:
    values = os.environ if env is None else env
    served = ("--served-model-name", profile.served_name)
    tp_size = int(values.get("TP_SIZE", str(8 * group_hosts)))
    return {
        **_common_launch_env(profile, group_hosts, values),
        "TP_SIZE": str(tp_size),
        "EP_SIZE": values.get("EP_SIZE", str(tp_size)),
        "DP_SIZE": "1",
        "ENABLE_DP_ATTENTION": "0",
        "MEM_FRACTION_STATIC": values.get(
            "MEM_FRACTION_STATIC", profile.mem_fraction_prefill
        ),
        "PORT": str(PREFILL_PORT),
        "BOOTSTRAP_PORT": str(BOOTSTRAP_PORT),
        "EXTRA_ARGS": " ".join(
            (
                *served,
                *profile.extra_server_args,
                values.get("EXTRA_PREFILL_ARGS", ""),
            )
        ).strip(),
    }


def _unused_build_prefill_env(
    profile: ModelProfile, group_hosts: int, env: Optional[Mapping[str, str]] = None
) -> Dict[str, str]:
    values = os.environ if env is None else env
    served = ("--served-model-name", profile.served_name)
    return {
        **_common_launch_env(profile, group_hosts, values),
        "MEM_FRACTION_STATIC": values.get(
            "MEM_FRACTION_STATIC", profile.mem_fraction_prefill
        ),
        "PORT": str(PREFILL_PORT),
        "BOOTSTRAP_PORT": str(BOOTSTRAP_PORT),
        "EXTRA_ARGS": " ".join(
            (
                hicache_server_args(profile, values),
                *served,
                *profile.extra_server_args,
                values.get("EXTRA_PREFILL_ARGS", ""),
            )
        ).strip(),
    }


def build_decode_env(
    profile: ModelProfile, group_hosts: int, env: Optional[Mapping[str, str]] = None
) -> Dict[str, str]:
    values = os.environ if env is None else env
    tp_size = int(values.get("TP_SIZE", str(8 * group_hosts)))
    dp_size = int(values.get("DP_SIZE", "2"))
    decode_args = [
        "--served-model-name",
        profile.served_name,
        "--dp-size",
        str(dp_size),
        "--enable-dp-attention",
    ]
    if profile.max_total_tokens is not None:
        decode_args += ["--max-total-tokens", str(profile.max_total_tokens)]
    return {
        **_common_launch_env(profile, group_hosts, values),
        "TP_SIZE": str(tp_size),
        "EP_SIZE": values.get("EP_SIZE", str(tp_size)),
        "DP_SIZE": str(dp_size),
        "ENABLE_DP_ATTENTION": "1",
        "MEM_FRACTION_STATIC": values.get(
            "MEM_FRACTION_STATIC", profile.mem_fraction_decode
        ),
        "PORT": str(DECODE_PORT),
        "EXTRA_ARGS": " ".join(
            (
                *decode_args,
                *profile.extra_server_args,
                values.get("EXTRA_DECODE_ARGS", ""),
            )
        ).strip(),
    }


def _unused_build_decode_env(
    profile: ModelProfile, group_hosts: int, env: Optional[Mapping[str, str]] = None
) -> Dict[str, str]:
    values = os.environ if env is None else env
    decode_args = ["--served-model-name", profile.served_name]
    if profile.max_total_tokens is not None:
        # The decode pool must be bounded too, or its auto-derived size OOMs
        # at precompile (pool materialized unaliased as compile args+outputs).
        decode_args += ["--max-total-tokens", str(profile.max_total_tokens)]
    return {
        **_common_launch_env(profile, group_hosts, values),
        "MEM_FRACTION_STATIC": values.get(
            "MEM_FRACTION_STATIC", profile.mem_fraction_decode
        ),
        "PORT": str(DECODE_PORT),
        "EXTRA_ARGS": " ".join(
            (
                *decode_args,
                *profile.extra_server_args,
                values.get("EXTRA_DECODE_ARGS", ""),
            )
        ).strip(),
    }


def select_profile(raw: str) -> ModelProfile:
    key = raw.strip()
    if key not in MODEL_PROFILES:
        allowed = ", ".join(MODEL_PROFILES)
        raise ValueError(
            f"MODEL_PROFILE must name exactly one of: {allowed} (got {key!r}); "
            "P/D runs launch one engine pair per invocation"
        )
    return MODEL_PROFILES[key]


# ------------------------------ probe workload -------------------------------

_PROMPTS = (
    "Explain the theory of general relativity in simple terms.",
    "Write a Python function to solve the Traveling Salesperson Problem using dynamic programming.",
    "Summarize the key events that led to the Renaissance in Europe.",
    "What are the main differences between TCP and UDP protocols?",
    "Describe the process of cellular respiration step by step.",
    "How does quantum key distribution provide unconditional security in communications?",
    "Write a SQL query to find the top 3 highest spending customers in each department.",
    "Explain the concept of attention mechanism in transformer neural networks.",
)


def run_tp_resharding_probe(base_url: str, profile: ModelProfile) -> None:
    records = []
    for i, prompt in enumerate(_PROMPTS):
        rec = _generate(base_url, prompt, 32)
        if not rec.text.strip() or rec.completion_tokens <= 0:
            raise AssertionError(
                f"Request {i} produced empty decode output -- TP resharding P/D transfer failed"
            )
        records.append(rec)

    resend = _generate(base_url, _PROMPTS[0], 32)
    if resend.text != records[0].text:
        raise AssertionError(
            "Greedy generation output diverged on resend under TP resharding"
        )

    print(
        json.dumps(
            {
                "tp_resharding_probe": profile.key,
                "num_requests": len(records),
                "sample_output": records[0].text[:100],
                "latencies": [r.latency_seconds for r in records],
            },
            sort_keys=True,
        ),
        flush=True,
    )




def _launch(script: str, extra_env: Dict[str, str], log_path: str) -> subprocess.Popen:
    env = dict(os.environ)
    env.update(extra_env)
    log = open(log_path, "ab")
    return subprocess.Popen(
        ["bash", os.path.join(BENCH, script)],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,  # own pgid so teardown can killpg
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
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return 200 <= r.status < 300
    except Exception:  # noqa: BLE001 - any transport error means "not healthy"
        return False


def _wait_http_200(url: str, secs: int, procs: Tuple = ()) -> None:
    deadline = time.time() + secs
    while time.time() < deadline:
        for p in procs:
            if p is not None and p.poll() is not None:
                raise RuntimeError(f"process for {url} exited rc={p.returncode}")
        if _http_ok(url):
            return
        time.sleep(10)
    raise TimeoutError(f"{url} not ready after {secs}s")


def _assert_raiden_engaged(log_path: str) -> None:
    with open(log_path, "rb") as f:
        log = f.read().decode(errors="replace")
    if RAIDEN_READY_MARKER not in log:
        raise AssertionError(
            f"{RAIDEN_READY_MARKER!r} never appeared in {log_path} -- the raiden "
            "offload engine did not initialize on the prefill engine"
        )


# ------------------------------- leader case ---------------------------------



class TestPdTpResharding(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if SGLANG_IMPORT_ERROR is not None:
            raise unittest.SkipTest(
                f"SGLang runtime unavailable: {SGLANG_IMPORT_ERROR}"
            )
        cls.profile = select_profile(E("MODEL_PROFILE", ""))
        cls.gsm8k_config = Gsm8kConfig.from_env(cls.profile)
        cls.topo = PodTopology.from_env()
        validate_model_source(cls.profile, resolve_model_path(cls.profile))

        cls.procs: list[subprocess.Popen] = []
        try:
            prefill_env = build_prefill_env(cls.profile, cls.topo.group_hosts)
            cls.procs.append(
                _launch("launch_prefill.sh", prefill_env, PREFILL_LOG)
            )

            decode_host = peer_group_hostname(
                cls.topo.group_hostnames[0],
                from_role=ROLE_PREFILL,
                to_role=ROLE_DECODE,
            )
            _wait_healthy(
                "decode",
                f"http://{decode_host}:{DECODE_PORT}/health",
                timeout=WAIT_SECS,
            )

            _wait_healthy(
                "prefill",
                f"http://127.0.0.1:{PREFILL_PORT}/health",
                proc=cls.procs[0],
                timeout=WAIT_SECS,
            )

            lb_env = {
                "PYTHON": PYTHON,
                "PREFILL_HOST": "127.0.0.1",
                "PREFILL_PORT": str(PREFILL_PORT),
                "DECODE_HOST": decode_host,
                "DECODE_PORT": str(DECODE_PORT),
                "LB_PORT": str(LB_PORT),
            }
            cls.procs.append(
                _launch(
                    "recipe/pd_lb.py" if os.path.isfile(os.path.join(BENCH, "recipe/pd_lb.py")) else "profile/pd_lb.py",
                    lb_env,
                    LB_LOG,
                )
            )
            _wait_healthy(
                "pd_lb",
                f"http://127.0.0.1:{LB_PORT}/health",
                proc=cls.procs[-1],
                timeout=120,
            )
        except Exception:
            cls.tearDownClass()
            raise

    @classmethod
    def tearDownClass(cls) -> None:
        if KEEP_UP:
            print("[KEEP_UP=1] leaving servers running for debug")
            return
        for p in cls.procs:
            _kill(p)

    def test_01_tp_resharding_generation(self) -> None:
        run_tp_resharding_probe(
            base_url=f"http://127.0.0.1:{LB_PORT}",
            profile=self.profile,
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
                "test": "pd_tp_resharding",
            },
        )
        print(f"gsm8k score={score:.4f} floor={self.gsm8k_config.threshold:.4f}")
        self.assertGreaterEqual(score, self.gsm8k_config.threshold)


class TestPdTpReshardingConfiguration(unittest.TestCase):
    def test_profile_selection_requires_exactly_one(self) -> None:
        self.assertIs(select_profile("gpt_oss_120b"), MODEL_PROFILES["gpt_oss_120b"])
        self.assertIs(select_profile("qwen3_coder_480b"), MODEL_PROFILES["qwen3_coder_480b"])
        for bad in ("", "all", "gpt_oss_120b,qwen3_coder_480b", "unknown"):
            with self.assertRaisesRegex(ValueError, "MODEL_PROFILE"):
                select_profile(bad)

    def test_prefill_and_decode_parallel_topologies(self) -> None:
        for profile in MODEL_PROFILES.values():
            p_env = build_prefill_env(profile, 2, env={})
            self.assertEqual(p_env["TP_SIZE"], "16")
            self.assertEqual(p_env["EP_SIZE"], "16")
            self.assertEqual(p_env["DP_SIZE"], "1")
            self.assertEqual(p_env["ENABLE_DP_ATTENTION"], "0")

            d_env = build_decode_env(profile, 2, env={})
            self.assertEqual(d_env["TP_SIZE"], "16")
            self.assertEqual(d_env["EP_SIZE"], "16")
            self.assertEqual(d_env["DP_SIZE"], "2")
            self.assertEqual(d_env["ENABLE_DP_ATTENTION"], "1")
            self.assertIn("--dp-size 2", d_env["EXTRA_ARGS"])
            self.assertIn("--enable-dp-attention", d_env["EXTRA_ARGS"])

    def test_heterogeneous_attention_tp_resharding_ratio(self) -> None:
        for profile in MODEL_PROFILES.values():
            p_env = build_prefill_env(profile, 2, env={})
            d_env = build_decode_env(profile, 2, env={})
            prefill_tp = int(p_env["TP_SIZE"])
            decode_tp = int(d_env["TP_SIZE"])
            decode_dp = int(d_env["DP_SIZE"])
            prefill_attn_tp = prefill_tp
            decode_attn_tp = decode_tp // decode_dp
            self.assertEqual(prefill_attn_tp, 16)
            self.assertEqual(decode_attn_tp, 8)
            self.assertEqual(prefill_attn_tp // decode_attn_tp, 2)

    def test_gpt_oss_pool_is_not_explicitly_bounded(self) -> None:
        env = build_prefill_env(MODEL_PROFILES["gpt_oss_120b"], 2, env={})
        self.assertNotIn("--max-total-tokens", env["EXTRA_ARGS"])
        env = build_decode_env(MODEL_PROFILES["gpt_oss_120b"], 2, env={})
        self.assertNotIn("--max-total-tokens", env["EXTRA_ARGS"])

    def test_gpt_oss_does_not_force_quantization(self) -> None:
        env = build_prefill_env(MODEL_PROFILES["gpt_oss_120b"], 2, env={})
        self.assertEqual(env["QUANTIZATION"], "")

    def test_peer_hostname_maps_only_the_role_segment(self) -> None:
        self.assertEqual(
            peer_group_hostname(
                "tpu-ci-pd-tpreshard-x7f2-prefill-0-1",
                from_role=ROLE_PREFILL,
                to_role=ROLE_DECODE,
            ),
            "tpu-ci-pd-tpreshard-x7f2-decode-0-1",
        )
        with self.assertRaisesRegex(ValueError, "no '-prefill-' segment"):
            peer_group_hostname(
                "tpu-ci-pd-tpreshard-x7f2-decode-0-0",
                from_role=ROLE_PREFILL,
                to_role=ROLE_DECODE,
            )

    def test_launcher_directory_resolves_from_this_file(self) -> None:
        self.assertTrue(
            os.path.isfile(os.path.join(BENCH, "launch_prefill.sh")),
            f"BENCH does not point at benchmark/tpu: {BENCH}",
        )
        self.assertTrue(
            os.path.isfile(os.path.join(BENCH, "launch_decode.sh")),
            f"BENCH does not point at benchmark/tpu: {BENCH}",
        )

    def test_leader_is_prefill_rank_zero(self) -> None:
        env = {
            "CI_PD_ROLE": ROLE_PREFILL,
            "TPU_WORKER_ID": "0",
            "TPU_WORKER_HOSTNAMES": "a-prefill-0-0,a-prefill-0-1",
        }
        self.assertTrue(PodTopology.from_env(env).is_leader)
        self.assertFalse(PodTopology.from_env({**env, "TPU_WORKER_ID": "1"}).is_leader)
        self.assertFalse(
            PodTopology.from_env(
                {
                    "CI_PD_ROLE": ROLE_DECODE,
                    "TPU_WORKER_ID": "0",
                    "TPU_WORKER_HOSTNAMES": "a-decode-0-0,a-decode-0-1",
                }
            ).is_leader
        )

    def test_model_specific_gsm8k_threshold_wins(self) -> None:
        config = Gsm8kConfig.from_env(
            MODEL_PROFILES["qwen3_coder_480b"],
            env={"GSM8K_THRESHOLD": "0.50", "GSM8K_THRESHOLD_QWEN3_CODER_480B": "0.91"},
        )
        self.assertEqual(config.threshold, 0.91)

    def test_local_model_path_is_validated(self) -> None:
        profile = MODEL_PROFILES["qwen3_coder_480b"]
        with self.assertRaisesRegex(FileNotFoundError, profile.model_env):
            validate_model_source(profile, "/weights/definitely-missing-480b")
        validate_model_source(profile, "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8")


if __name__ == "__main__":
    if not E("CI_PD_ROLE"):
        sys.argv = [sys.argv[0], "TestPdTpReshardingConfiguration"]
        unittest.main(verbosity=2)
    topology = PodTopology.from_env()
    if not topology.is_leader:
        sys.exit(_serve_role(topology, select_profile(E("MODEL_PROFILE", ""))))
    sys.argv = [sys.argv[0], "TestPdTpResharding"]
    unittest.main(verbosity=2)
