"""GPT-OSS-120B block-major KV layout gate on a raiden P/D pair with KV offload.

Both engines run with SGLANG_TPU_USE_BLOCK_MAJOR_KV=1, one host each, and the
prefill engine adds the raiden HiCache host tier. The layout is env-driven and
invisible from the API, so each pod reads the pool allocation lines back from
its own tee'd server log before anything else is scored. The prefill pod then
checks the engine configuration through /server_info, that the raiden host
tier came up, runs the device-hit / churn / host-reload probe through the mini
router, and evaluates GSM8K.
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

import msgspec

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


# Expected runtime: the single-host P/D GSM8K gate plus the churn probe.
register_tpu_ci(
    est_time=3600,
    suite="tpu-integration",
    nightly=True,
    labels=["singlehost", "gpt-oss-120b", "pd", "kv-offload", "block-major"],
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
SELF_HOST_ENV = "PD_SELF_HOST"
DECODE_HOST_ENV = "PD_DECODE_HOST"

# Each engine uses one eight-chiplet TPU host.
HOSTS_PER_ENGINE = 1
CHIPLETS_PER_HOST = 8
TP_SIZE = HOSTS_PER_ENGINE * CHIPLETS_PER_HOST
EP_SIZE = TP_SIZE

# Separate pods may reuse the same service and Raiden ports.
PREFILL_PORT = int(E("PREFILL_PORT", "30000"))
DECODE_PORT = int(E("DECODE_PORT", "30000"))
LB_PORT = int(E("LB_PORT", "8000"))
BOOTSTRAP_PORT = int(E("BOOTSTRAP_PORT", "8998"))
RAIDEN_KV_PORT = int(E("SGLANG_RAIDEN_KV_PORT", "31000"))
LOCAL_HOST = "127.0.0.1"

WAIT_SECS = int(E("WAIT_SECS", "3600"))
KEEP_UP = E("KEEP_UP", "0") == "1"
PYTHON = E("PYTHON", "python3")

# The runtime image excludes the JobSet renderer.
RENDER_JOBSET = os.path.join(REPO, ".github", "tpu-ci", "render_jobset.py")

# Pod name prefix within Kubernetes' 63-character limit;
# role words are omitted because peer mapping swaps the role segment.
RUN_NAME = "gpt-oss-120b-pd-sh-bm"

PREFILL_LOG = "/tmp/pd_block_major_prefill.log"
DECODE_LOG = "/tmp/pd_block_major_decode.log"
LB_LOG = "/tmp/pd_block_major_lb.log"

PAGE_SIZE = 64
MEM_FRACTION_STATIC = "0.35"
# Use checkpoint-provided MXFP4 weights and FP8 KV cache.
QUANTIZATION = ""
KV_CACHE_DTYPE = "fp8_e4m3"
# One host has half the HBM of the two-host P/D configuration.
MAX_TOTAL_TOKENS = 393216
DISAGG_TRANSFER_BACKEND = "raiden"
# Raiden handles P/D KV transfer, the prefill host tier and decode host-pool I/O.
HICACHE_IO_BACKEND = "raiden"
# Holds the 2x-pool churn plus the probe target; the host tier evicts LRU too.
HICACHE_RATIO = E("HICACHE_RATIO", "3")
EXTRA_SERVER_ARGS: Tuple[str, ...] = ("--reasoning-parser", "gpt-oss")
# GPT-OSS does not support the SparseCore fused-MoE gather path.
EXTRA_LAUNCH_ENV: Dict[str, str] = {"SGLANG_TPU_FUSED_MOE_SC_GATHER": "0"}

BLOCK_MAJOR_ENV: Dict[str, str] = {"SGLANG_TPU_USE_BLOCK_MAJOR_KV": "1"}
# Logged by hardware_backend/tpu/kv_pool.py once per pool per rank.
BUNDLE_LINE = "Allocating block-major TPU KV bundle"
FUSED_LINE = "Allocating fused TPU KV buffers"
# Logged by hardware_backend/tpu/raiden_transfer.py when the host tier is up.
RAIDEN_READY_MARKER = "Raiden KV transfer ready"

GENERATE_TIMEOUT = 900

# Measured 0.97 accuracy on this topology; 0.80 serves as the CI passing floor.
GSM8K_BASELINE = float(E("GSM8K_THRESHOLD_GPT_OSS_120B", E("GSM8K_THRESHOLD", "0.80")))
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
        return  # HF repo ids and bucket URIs resolve remotely
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
    """This pod's own DNS name."""
    values = os.environ if env is None else env
    if values.get(SELF_HOST_ENV):
        return values[SELF_HOST_ENV]
    hostnames = [h for h in values.get("TPU_WORKER_HOSTNAMES", "").split(",") if h]
    return hostnames[0] if hostnames else socket.gethostname()


def peer_hostname(hostname: str, *, from_role: str, to_role: str) -> str:
    """Map this pod's DNS name onto the pod of the peer role."""
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
        # The block-major bundle and raiden both require the RPA v3 fused pool.
        "ATTENTION_BACKEND": "pallas_tpu_rpa_v3",
        "QUANTIZATION": QUANTIZATION,
        "KV_CACHE_DTYPE": KV_CACHE_DTYPE,
        "DISABLE_RADIX_CACHE": "0",
        "ENABLE_TORCH_COMPILE": values.get("ENABLE_TORCH_COMPILE", "1"),
        "MEM_FRACTION_STATIC": values.get("MEM_FRACTION_STATIC", MEM_FRACTION_STATIC),
        "SGLANG_RAIDEN_KV_PORT": str(RAIDEN_KV_PORT),
        "PYTHON": PYTHON,
        **EXTRA_LAUNCH_ENV,
        **BLOCK_MAJOR_ENV,
    }


def _shared_server_args() -> List[str]:
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


def hicache_server_args() -> List[str]:
    """Flags that put the raiden host tier under the prefill engine's radix cache."""
    return [
        "--enable-hierarchical-cache",
        "--hicache-ratio",
        HICACHE_RATIO,
        "--hicache-io-backend",
        HICACHE_IO_BACKEND,
        "--hicache-mem-layout",
        "page_first_fused",
        "--hicache-write-policy",
        "write_through",
        # cached_tokens_details in /generate replies needs the cache report.
        "--enable-cache-report",
    ]


def build_prefill_env(env: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """Construct the environment for the prefill engine, host tier on."""
    values = os.environ if env is None else env
    extra_args = _shared_server_args() + hicache_server_args()
    if values.get("EXTRA_PREFILL_ARGS"):
        extra_args.append(values["EXTRA_PREFILL_ARGS"])
    return {
        **_common_launch_env(values),
        "PORT": str(PREFILL_PORT),
        "BOOTSTRAP_PORT": str(BOOTSTRAP_PORT),
        "EXTRA_ARGS": " ".join(extra_args).strip(),
    }


def _decode_host_pool_args() -> List[str]:
    """Make the decode retraction host pool buildable on a fused TPU KV pool."""
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


def _post_json(url: str, payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _wait_http_200(
    *, url: str, secs: int, procs: Tuple[Tuple[str, subprocess.Popen], ...] = ()
) -> None:
    """Poll url until it answers 200, failing early if a local process dies."""
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


def _read_log(log_path: str) -> str:
    with open(log_path, "rb") as f:
        return f.read().decode(errors="replace")


class PoolAllocationCounts(msgspec.Struct, frozen=True, kw_only=True):
    bundles: int
    fused: int


def count_pool_allocations(log_text: str) -> PoolAllocationCounts:
    """Count the block-major and layer-major pool allocations a server logged."""
    return PoolAllocationCounts(
        bundles=log_text.count(BUNDLE_LINE), fused=log_text.count(FUSED_LINE)
    )


def layout_failures(*, role: str, counts: PoolAllocationCounts) -> List[str]:
    """Reject an engine that allocated any pool layer-major, or no bundle at all."""
    failures = []
    if counts.bundles == 0:
        failures.append(
            f"{role} engine allocated no block-major bundle: "
            "SGLANG_TPU_USE_BLOCK_MAJOR_KV did not reach it"
        )
    if counts.fused:
        failures.append(
            f"{role} engine allocated {counts.fused} pool(s) layer-major "
            "beside the bundles"
        )
    return failures


def check_layout(*, role: str, log_path: str) -> List[str]:
    """Read one engine's tee'd log and report its pool layout."""
    counts = count_pool_allocations(_read_log(log_path))
    failures = layout_failures(role=role, counts=counts)
    print(
        f"[pd-block-major] {role} layout: bundle allocations={counts.bundles} "
        f"fused allocations={counts.fused} ({log_path})",
        flush=True,
    )
    return failures


def block_major_layout_runner(*, log_path: str):
    """Gate on the prefill engine having allocated every pool as a bundle."""

    def run(_base_url: str) -> Dict[str, Any]:
        failures = check_layout(role=PREFILL_ROLE, log_path=log_path)
        return {
            "score": 0.0 if failures else 1.0,
            "_passed": not failures,
            "_failures": failures,
        }

    return run


def engine_configuration_failures(
    *, prefill_info: Mapping[str, Any], decode_info: Mapping[str, Any]
) -> List[str]:
    """Reject a run whose engines are not a raiden P/D pair with a prefill host tier."""
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
    # The host-reload leg of the probe is void without the prefill host tier.
    if prefill_info.get("enable_hierarchical_cache") is not True:
        failures.append(
            "prefill engine reports enable_hierarchical_cache="
            f"{prefill_info.get('enable_hierarchical_cache')!r}, expected True"
        )
    io_backend = prefill_info.get("hicache_io_backend")
    if io_backend != HICACHE_IO_BACKEND:
        failures.append(
            f"prefill engine reports hicache_io_backend={io_backend!r}, "
            f"expected {HICACHE_IO_BACKEND!r}"
        )
    return failures


def engine_configuration_runner(*, prefill_url: str, decode_url: str):
    """Verify both servers report the expected P/D roles, backend and host tier."""

    def run(base_url: str) -> Dict[str, Any]:
        prefill_info = _get_json(f"{prefill_url}/server_info")
        decode_info = _get_json(f"{decode_url}/server_info")
        failures = engine_configuration_failures(
            prefill_info=prefill_info, decode_info=decode_info
        )
        print(
            f"[pd-block-major] lb={base_url} prefill="
            f"{prefill_info.get('disaggregation_mode')}/"
            f"{prefill_info.get('disaggregation_transfer_backend')} decode="
            f"{decode_info.get('disaggregation_mode')}/"
            f"{decode_info.get('disaggregation_transfer_backend')} prefill_hicache="
            f"{prefill_info.get('enable_hierarchical_cache')}/"
            f"{prefill_info.get('hicache_io_backend')}",
            flush=True,
        )
        return {
            "score": 0.0 if failures else 1.0,
            "_passed": not failures,
            "_failures": failures,
        }

    return run


def raiden_engaged_runner(*, log_path: str):
    """Gate on the raiden host tier having initialized on the prefill engine."""

    def run(_base_url: str) -> Dict[str, Any]:
        engaged = RAIDEN_READY_MARKER in _read_log(log_path)
        print(
            f"[pd-block-major] raiden host tier ready={engaged} ({log_path})",
            flush=True,
        )
        failures = (
            []
            if engaged
            else [
                f"{RAIDEN_READY_MARKER!r} never appeared in {log_path}: the raiden "
                "host tier did not initialize on the prefill engine"
            ]
        )
        return {
            "score": 1.0 if engaged else 0.0,
            "_passed": engaged,
            "_failures": failures,
        }

    return run


class CacheProbeConfig(msgspec.Struct, frozen=True, kw_only=True):
    page_size: int
    device_pool_tokens: int
    target_words: int
    filler_words: int
    # None sizes the churn from the first filler's measured token count.
    num_fillers: Optional[int]
    max_new_tokens: int

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "CacheProbeConfig":
        values = os.environ if env is None else env
        fillers_raw = values.get("CACHE_PROBE_NUM_FILLERS")
        return cls(
            page_size=PAGE_SIZE,
            device_pool_tokens=int(
                values.get("CACHE_PROBE_DEVICE_TOKENS", str(MAX_TOTAL_TOKENS))
            ),
            target_words=int(values.get("CACHE_PROBE_TARGET_WORDS", "2048")),
            filler_words=int(values.get("CACHE_PROBE_FILLER_WORDS", "2048")),
            num_fillers=None if fillers_raw is None else int(fillers_raw),
            # Room for all four copied item lines plus any preamble.
            max_new_tokens=int(values.get("CACHE_PROBE_MAX_NEW_TOKENS", "64")),
        )


def churn_fillers_needed(
    *, measured_filler_tokens: int, device_pool_tokens: int
) -> int:
    """Fillers that push 2x the device pool through the cache.

    Sized from a measured filler's token count; a churn that runs short of the
    pool never evicts, and the host-reload leg then tests nothing. The 2x margin
    absorbs slack between the configured pool bound and the allocator's capacity.
    """
    if measured_filler_tokens <= 0:
        raise AssertionError(
            f"measured filler prompt_tokens={measured_filler_tokens}; cannot "
            "size the churn"
        )
    return (2 * device_pool_tokens) // measured_filler_tokens + 1


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
    return f"Document {index} header alpha{index} beta{index * 7}. " + " ".join(words)


_ITEM_NOUNS = (
    "lantern kettle walnut anchor violet magnet saddle donkey harbor pencil "
    "quartz turnip magpie helmet donkey copper barrel falcon garnet hammer "
    "island jasper kitten locket mirror needle onward parrot quiver ribbon "
    "sombra tundra"
).split()


def make_target_prompt(nwords: int) -> Tuple[str, Tuple[int, ...]]:
    """A numbered item list ending in a verbatim copy instruction.

    The greedy continuation is forced: the model repeats four distinctive lines
    read through full attention from the queried positions' KV, so a text
    difference between cached passes means the KV values themselves changed.
    """
    num_items = max(40, nwords // 6)
    lines = []
    for i in range(1, num_items + 1):
        noun = _ITEM_NOUNS[(i * 2654435761 & 0xFFFFFFFF) % len(_ITEM_NOUNS)]
        lines.append(f"item {i}: {noun}-{i * 7 % 97}")
    queried = (
        num_items // 8,
        num_items // 3,
        (2 * num_items) // 3,
        num_items - 5,
    )
    ask = ", ".join(str(q) for q in queried)
    prompt = (
        "Below is a list of items.\n"
        + "\n".join(lines)
        + f"\nOutput items {ask} from the list above exactly as written, "
        "one per line, and nothing else.\nAnswer:\nitem "
    )
    return prompt, queried


class GenerationRecord(msgspec.Struct, frozen=True, kw_only=True):
    text: str
    prompt_tokens: int
    completion_tokens: int
    device_cached_tokens: int
    host_cached_tokens: int
    latency_seconds: float


def _generate(base_url: str, prompt: str, max_new_tokens: int) -> GenerationRecord:
    started = time.perf_counter()
    payload = _post_json(
        f"{base_url}/generate",
        {
            "text": prompt,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": max_new_tokens},
        },
        timeout=GENERATE_TIMEOUT,
    )
    meta = payload.get("meta_info") or {}
    details = meta.get("cached_tokens_details") or {}
    return GenerationRecord(
        text=payload["text"],
        prompt_tokens=int(meta.get("prompt_tokens") or 0),
        completion_tokens=int(meta.get("completion_tokens") or 0),
        device_cached_tokens=int(details.get("device") or 0),
        host_cached_tokens=int(details.get("host") or 0),
        latency_seconds=round(time.perf_counter() - started, 3),
    )


def _flush_cache(url: str) -> None:
    with urllib.request.urlopen(f"{url}/flush_cache", timeout=60) as r:
        r.read()


def validate_cache_probe(
    *,
    cold: GenerationRecord,
    device_hit: GenerationRecord,
    host_reload: GenerationRecord,
    post_reload_hit: GenerationRecord,
    page_size: int,
) -> List[str]:
    """The device-hit, churn, host-reload oracle.

    The three cached passes run the same extend shape over the same stored
    prefix KV, so their greedy text must be identical. The cold pass runs a
    different extend shape and is held to cache-state and liveness only.
    """
    if cold.prompt_tokens < 2 * page_size:
        return [f"target prompt too short: {cold.prompt_tokens} tokens"]
    if cold.device_cached_tokens or cold.host_cached_tokens:
        return [
            "cold request unexpectedly hit cache: "
            f"device={cold.device_cached_tokens} host={cold.host_cached_tokens}"
        ]
    if not cold.text.strip() or cold.completion_tokens <= 0:
        return ["cold request produced no decode output: P/D KV transfer broken"]

    failures = []
    expected = max(page_size, (cold.prompt_tokens // page_size - 1) * page_size)
    if device_hit.device_cached_tokens < expected:
        failures.append(
            "immediate resend missed the device prefix cache: "
            f"device={device_hit.device_cached_tokens} expected>={expected}"
        )
    if device_hit.host_cached_tokens:
        failures.append(
            f"immediate resend loaded from host: {device_hit.host_cached_tokens}"
        )
    if not device_hit.text.strip() or device_hit.completion_tokens <= 0:
        failures.append("device-hit request produced no decode output")

    if host_reload.host_cached_tokens <= 0:
        failures.append(
            "target was not reloaded from the host tier after churn: "
            "KV offload never engaged"
        )
    if host_reload.device_cached_tokens + host_reload.host_cached_tokens < expected:
        failures.append(
            "host reload did not recover the cached prefix: "
            f"device={host_reload.device_cached_tokens} "
            f"host={host_reload.host_cached_tokens} expected>={expected}"
        )
    # A reload that raced its own host->device DMA reads clean once the pages
    # settle; corrupt values stay wrong on the post-reload pass.
    if host_reload.text != device_hit.text:
        if post_reload_hit.text == device_hit.text:
            failures.append(
                "host-reloaded copy output diverged from the device-hit pass but "
                "the post-reload resend matches: the reload raced its own "
                "host->device transfer"
            )
        else:
            failures.append(
                "host-reloaded copy output diverged from the device-hit pass: "
                "the offloaded KV pages did not survive the host round trip"
            )
    elif post_reload_hit.text != device_hit.text:
        failures.append(
            "post-reload resend diverged from the device-hit pass: the reloaded "
            "device-resident pages hold different KV values"
        )
    return failures


def run_cache_probe(
    *, base_url: str, flush_url: str, config: CacheProbeConfig
) -> Dict[str, Any]:
    """Run the four probe passes and the churn between them through the LB."""
    _flush_cache(flush_url)
    target, queried = make_target_prompt(config.target_words)
    print(f"[pd-block-major] cache probe target queries items {queried}", flush=True)
    cold = _generate(base_url, target, config.max_new_tokens)
    device_hit = _generate(base_url, target, config.max_new_tokens)

    first_filler = _generate(base_url, make_prompt(1000, config.filler_words), 1)
    num_fillers = config.num_fillers
    if num_fillers is None:
        num_fillers = churn_fillers_needed(
            measured_filler_tokens=first_filler.prompt_tokens,
            device_pool_tokens=config.device_pool_tokens,
        )
    print(
        f"[pd-block-major] churn: {num_fillers} fillers x "
        f"{first_filler.prompt_tokens} tokens against a "
        f"{config.device_pool_tokens}-token device pool",
        flush=True,
    )
    for index in range(1, num_fillers):
        _generate(base_url, make_prompt(1000 + index, config.filler_words), 1)
    host_reload = _generate(base_url, target, config.max_new_tokens)
    post_reload_hit = _generate(base_url, target, config.max_new_tokens)

    # Printed before validation so a failed oracle still carries every pass.
    print(
        json.dumps(
            {
                "cache_probe": "pd_block_major",
                "cold": msgspec.to_builtins(cold),
                "device_hit": msgspec.to_builtins(device_hit),
                "host_reload": msgspec.to_builtins(host_reload),
                "post_reload_hit": msgspec.to_builtins(post_reload_hit),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    failures = validate_cache_probe(
        cold=cold,
        device_hit=device_hit,
        host_reload=host_reload,
        post_reload_hit=post_reload_hit,
        page_size=config.page_size,
    )
    print(
        f"[pd-block-major] cache probe device_hit={device_hit.device_cached_tokens} "
        f"host_reload={host_reload.host_cached_tokens} "
        f"failures={len(failures)}",
        flush=True,
    )
    return {
        "score": float(host_reload.host_cached_tokens),
        "_passed": not failures,
        "_failures": failures,
    }


def cache_probe_runner(*, flush_url: str, config: CacheProbeConfig):
    """Score the device-hit, churn, host-reload round trip through the LB."""

    def run(base_url: str) -> Dict[str, Any]:
        return run_cache_probe(base_url=base_url, flush_url=flush_url, config=config)

    return run


def build_model_config(*, decode_host: str) -> Dict[str, Any]:
    """Assemble the gate list, which needs the decode pod's resolved name."""
    prefill_url = f"http://{LOCAL_HOST}:{PREFILL_PORT}"
    return {
        "model": SERVED_MODEL_NAME,
        "test_name": "gpt-oss-120b-pd-singlehost-block-major",
        "variant": f"PD-singlehost-TP{TP_SIZE}+EP{EP_SIZE}-raiden-hicache-block-major",
        "launch": {
            # Role runners launch servers and router before this runner attaches.
            "external": True,
            "health_timeout": HEALTH_TIMEOUT,
            "tag": "gpt-oss-120b-pd-singlehost-block-major-e2e",
        },
        # Keep this specialized P/D result off the standard accuracy dashboard.
        # Gates run in list order; the layout goes first because every later
        # gate would otherwise validate the layer-major pool under this name.
        "gates": [
            {
                "dataset": "block_major_layout",
                "label": "block-major-layout",
                "runner": block_major_layout_runner(log_path=PREFILL_LOG),
            },
            {
                "dataset": "pd_engine_configuration",
                "label": "pd-engine-configuration",
                "runner": engine_configuration_runner(
                    prefill_url=prefill_url,
                    decode_url=f"http://{decode_host}:{DECODE_PORT}",
                ),
            },
            {
                "dataset": "raiden_host_tier",
                "label": "raiden-host-tier",
                "runner": raiden_engaged_runner(log_path=PREFILL_LOG),
            },
            {
                "dataset": "prefix_cache_kv_offload",
                "label": "prefix-cache-kv-offload",
                "runner": cache_probe_runner(
                    flush_url=prefill_url, config=CacheProbeConfig.from_env()
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
    """Serve prefill, wait for the decode pod, then gate and score through the LB."""
    this_host = self_hostname()
    # CI role pod names resolve by swapping the replicated-job name segment;
    # retained CDK debug pods share one generic segment and set peer explicitly.
    decode_host = E(DECODE_HOST_ENV) or peer_hostname(
        this_host, from_role=PREFILL_ROLE, to_role=DECODE_ROLE
    )
    procs: List[Tuple[str, subprocess.Popen]] = []
    try:
        procs.append((PREFILL_ROLE, execute_prefill()))
        _wait_http_200(
            url=f"http://{LOCAL_HOST}:{PREFILL_PORT}/health",
            secs=WAIT_SECS,
            procs=tuple(procs),
        )
        # Decode runs in a separate pod; only its HTTP health endpoint is visible.
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
    """Serve decode until the engine exits, once its own pool layout is verified."""
    proc = execute_decode()
    try:
        _wait_http_200(
            url=f"http://{LOCAL_HOST}:{DECODE_PORT}/health",
            secs=WAIT_SECS,
            procs=((DECODE_ROLE, proc),),
        )
        # The decode log exists on this pod only; a layer-major decode pool
        # fails this replicated job, and the JobSet's failurePolicy ends the run.
        failures = check_layout(role=DECODE_ROLE, log_path=DECODE_LOG)
        if failures:
            for failure in failures:
                print(f"[pd-block-major] {failure}", flush=True)
            return 1
        while True:
            rc = proc.poll()
            if rc is not None:
                return rc
            time.sleep(10)
    finally:
        _kill(proc)


def run_integration() -> int:
    """Dispatch on the role this pod was given."""
    if SGLANG_IMPORT_ERROR is not None:
        # Fallback stubs only support host-side unittests;
        # integration runs require real imports to execute the gate pipeline.
        raise SGLANG_IMPORT_ERROR
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
        # Names the dashboard row and the TIMINGS record.
        "--files",
        rel_path,
        "--exec",
        f"python3 {rel_path} integration",
    ]
    return subprocess.check_output(cmd, text=True)


class TestPdSingleHostBlockMajorGptOss120bConfiguration(unittest.TestCase):
    """Host-side gate tests, runnable without TPU hardware."""

    def test_both_engines_switch_on_the_block_major_layout(self) -> None:
        # A one-sided switch validates a layer-major pool under this test's name.
        for env in (build_prefill_env(env={}), build_decode_env(env={})):
            for key, value in BLOCK_MAJOR_ENV.items():
                self.assertEqual(env.get(key), value)
            self.assertEqual(env["ATTENTION_BACKEND"], "pallas_tpu_rpa_v3")

    def test_prefill_carries_the_raiden_host_tier_and_decode_does_not(self) -> None:
        prefill = build_prefill_env(env={})["EXTRA_ARGS"]
        for flag in (
            "--enable-hierarchical-cache",
            f"--hicache-io-backend {HICACHE_IO_BACKEND}",
            "--hicache-mem-layout page_first_fused",
            "--hicache-write-policy write_through",
            "--enable-cache-report",
        ):
            self.assertIn(flag, prefill)
        decode = build_decode_env(env={})["EXTRA_ARGS"]
        self.assertNotIn("--enable-hierarchical-cache", decode)

    def test_layout_gate_reads_pool_allocations(self) -> None:
        bundles = "\n".join(
            (
                "[TP0] Allocating block-major TPU KV bundle: shape=(6144, 36, 64, "
                "1, 4, 128), dtype=torch.float8_e4m3fn",
                "[TP0] Allocating block-major TPU KV bundle: shape=(6144, 36, 64, "
                "1, 4, 128), dtype=torch.float8_e4m3fn",
            )
        )
        fused = (
            "[TP0] Allocating fused TPU KV buffers: layers=36, shape per layer="
            "(6144, 64, 1, 4, 128), dtype=torch.float8_e4m3fn"
        )
        counts = count_pool_allocations(bundles)
        self.assertEqual((counts.bundles, counts.fused), (2, 0))
        self.assertEqual(layout_failures(role=PREFILL_ROLE, counts=counts), [])

        # One pool left layer-major beside the bundles must still fail.
        mixed = layout_failures(
            role=DECODE_ROLE, counts=count_pool_allocations(bundles + "\n" + fused)
        )
        self.assertEqual(len(mixed), 1)
        self.assertIn("layer-major", mixed[0])

        # A log with neither line is a server the switch never reached.
        none = layout_failures(
            role=PREFILL_ROLE, counts=count_pool_allocations("server ready")
        )
        self.assertEqual(len(none), 1)
        self.assertIn("no block-major bundle", none[0])

    # Skip in runtime images, which do not contain the renderer.
    @unittest.skipUnless(
        os.path.exists(RENDER_JOBSET), "renderer is not shipped in the runtime image"
    )
    def test_dry_run_render_jobset_yaml(self) -> None:
        # Verify the workflow name, roles, and command remain aligned.
        yaml_content = render_jobset_yaml()
        self.assertIn(f"generateName: tpu-ci-{RUN_NAME}-", yaml_content)
        self.assertIn("name: prefill", yaml_content)
        self.assertIn("name: decode", yaml_content)
        self.assertNotIn("name: router", yaml_content)
        self.assertIn("CI_PD_ROLE", yaml_content)
        self.assertIn(
            "python3 test/registered/tpu/e2e/"
            "test_tpu_gpt_oss_120b_pd_singlehost_block_major.py integration",
            yaml_content,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Single-host-per-engine P/D block-major KV layout gate"
    )
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
        # Remove the mode before model_config_main passes argv to unittest.
        sys.argv = [sys.argv[0], *extra_args]
        sys.exit(run_integration())
    sys.argv = [
        sys.argv[0],
        "TestPdSingleHostBlockMajorGptOss120bConfiguration",
        *extra_args,
    ]
    unittest.main(verbosity=2)
