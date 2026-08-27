"""Shared harness for the GPT-OSS-120B P/D multihost lanes.

Several lanes differ only in which features ride along with disaggregation:
speculative decoding, prefix cache + KV offload, or both. The topology, the role
dispatch, the launchers and the mini-LB are identical across all of them, so
they live here once and each lane contributes a ``LaneConfig`` plus its gates.

Every lane is 2 prefill hosts and 2 decode hosts, TP16 per engine, on separate
TPU slices. Prefill rank 0 is the leader: it brings up its own shard, waits for
the decode group, starts the mini-LB locally, and runs the gates. The LB rides
the leader rather than a pod of its own because the Kueue quota covers only the
two TPU slices, so a third pod would never be admitted.

raiden carries everything that crosses a boundary. The P/D hop is the obvious
one. Less obvious is that a P/D decode server switches on a host_pool retraction
backup by itself and builds it from the device pool, which under RPA v3 is
fused. gpt-oss is hybrid-SWA, and a fused host pool mirrors the full-attention
sub-pool alone, so server_args refuses it outright and raiden -- one host pool
per sub-pool -- is the only backend that lets the decode engine finish booting.

HiCache goes on the prefill engine only. Prefill is where prefix reuse happens,
and the tier breakdown still reaches the caller: the prefill engine packs
cached_tokens_device / _host into the P/D metadata buffer and the decode engine
reports them back, so a probe driven through the LB reads the prefill engine's
cache accounting. ``/flush_cache`` reaches both because the mini-LB fans it out.

Nothing here imports sglang or msgspec at module scope, and the containers below
are NamedTuples rather than the repo-default Struct for the same reason the
sibling P/D lanes hand-roll theirs: the host-side configuration tests have to
run on a checkout with no runtime dependencies installed. The sglang imports the
integration path needs are function-local.
"""

import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Dict, List, Mapping, NamedTuple, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, *[".."] * 4))
BENCH = os.path.join(REPO, "benchmark", "tpu")

E = os.environ.get

SERVED_MODEL_NAME = "gpt-oss-120b"
DEFAULT_MODEL_PATH = "/weights/gpt-oss-120b"
HF_REPO_ID = "openai/gpt-oss-120b"
MODEL_ENV = "GPT_OSS_MODEL"
DRAFT_MODEL_PATH = "lmsys/EAGLE3-gpt-oss-120b-bf16"

PREFILL_ROLE = "prefill"
DECODE_ROLE = "decode"
ROLE_ENV = "CI_PD_ROLE"

# Two hosts per engine is the shape these lanes gate. The override exists so the
# same lane can be re-run at one host per engine, which is what separates a fault
# in the TP16 fabric from a fault in the P/D handover: the smaller shape keeps
# every other flag identical and only drops the second host. It has to agree with
# the JobSet's pd_prefill_hosts/pd_decode_hosts, and render_jobset_yaml below
# reads the same value so a locally rendered JobSet cannot disagree with it.
HOSTS_PER_ENGINE = int(E("PD_HOSTS_PER_ENGINE", "2"))
CHIPLETS_PER_HOST = 8
TP_SIZE = HOSTS_PER_ENGINE * CHIPLETS_PER_HOST
EP_SIZE = TP_SIZE

PREFILL_PORT = int(E("PREFILL_PORT", "30000"))
DECODE_PORT = int(E("DECODE_PORT", "30000"))
LB_PORT = int(E("LB_PORT", "8000"))
BOOTSTRAP_PORT = int(E("BOOTSTRAP_PORT", "8998"))
RAIDEN_KV_PORT = int(E("SGLANG_RAIDEN_KV_PORT", "31000"))
RAIDEN_PORTS_PER_RANK = 4

WAIT_SECS = int(E("WAIT_SECS", "3600"))
KEEP_UP = E("KEEP_UP", "0") == "1"
PYTHON = E("PYTHON", "python3")
HEALTH_TIMEOUT = float(E("HEALTH_TIMEOUT", "3600"))

# The JobSet renderer, which lives under .dockerignore's .github/ entry and so
# is absent from the runtime image. Anything that shells out to it only works
# from a checkout, which is why the dry-run test below is guarded on this path.
RENDER_JOBSET = os.path.join(REPO, ".github", "tpu-ci", "render_jobset.py")

PAGE_SIZE = 64
MEM_FRACTION_STATIC = "0.35"
QUANTIZATION = ""
KV_CACHE_DTYPE = "fp8_e4m3"
ATTENTION_BACKEND = "pallas_tpu_rpa_v3"
DISAGG_TRANSFER_BACKEND = "raiden"
HICACHE_IO_BACKEND = "raiden"
HICACHE_MEM_LAYOUT = "page_first_fused"

# Carried over from the plain P/D multihost lane, which is green with it. The
# single-chip gather path in the fused MoE kernel is not used at EP16 there.
EXTRA_LAUNCH_ENV: Dict[str, str] = {"SGLANG_TPU_FUSED_MOE_SC_GATHER": "0"}

# Offload probe sizing, shared by the prefix-cache lanes. Same numbers the
# single-host gpt-oss offload lane is green on: a device pool small enough that
# the fillers evict the probe, and a host pool 3x larger so the reload still
# finds it. plan_eviction re-derives and re-checks the arithmetic, so a bad edit
# here fails on the host instead of passing vacuously on the hardware.
DEVICE_POOL_TOKENS = 16384
PROBE_TOKENS = 2048
HICACHE_RATIO = 3.0

# EAGLE3 on the recipe the single-host gpt-oss spec lane is green on.
SPEC_NUM_STEPS = 3
SPEC_EAGLE_TOPK = 1
SPEC_NUM_DRAFT_TOKENS = 4


class LaneConfig(NamedTuple):
    """What one lane varies on top of the shared 2P+2D topology.

    ``hicache_ratio`` set turns on prefix caching plus raiden KV offload on the
    prefill engine, ``spec`` turns on EAGLE3 on both. ``max_total_tokens`` bounds
    the device pool on both engines: the decode side needs it as much as prefill,
    since an auto-derived pool is materialized as unaliased compile args and
    outputs and OOMs at precompile.

    ``run_name`` is separate from ``test_name`` and deliberately terse: it
    becomes a Kubernetes label, and render_jobset.py's dns_safe truncates it at
    40 characters. Two lanes whose descriptive names agree in their first 40
    characters would otherwise render to the same JobSet name.
    """

    test_name: str
    variant: str
    tag: str
    run_name: str
    max_total_tokens: int
    hicache_ratio: Optional[float] = None
    spec: bool = False

    @property
    def log_prefix(self) -> str:
        return f"/tmp/{self.run_name}"


class PodTopology(NamedTuple):
    """This pod's place in the JobSet, read off the slice-local environment."""

    role: str
    group_rank: int
    group_hostnames: Tuple[str, ...]

    @property
    def group_hosts(self) -> int:
        return len(self.group_hostnames)

    @property
    def is_prefill_leader(self) -> bool:
        return self.role == PREFILL_ROLE and self.group_rank == 0

    @property
    def self_hostname(self) -> str:
        return self.group_hostnames[0]

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "PodTopology":
        values = os.environ if env is None else env
        role = values.get(ROLE_ENV, "")
        if role not in (PREFILL_ROLE, DECODE_ROLE):
            raise ValueError(
                f"{ROLE_ENV} must be {PREFILL_ROLE!r} or {DECODE_ROLE!r}, got {role!r}"
            )
        hostnames = tuple(
            h for h in values.get("TPU_WORKER_HOSTNAMES", "").split(",") if h
        )
        return cls(
            role=role,
            group_rank=int(values.get("TPU_WORKER_ID", "0")),
            group_hostnames=hostnames or (socket.gethostname(),),
        )


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


def peer_hostname(*, hostname: str, from_role: str, to_role: str) -> str:
    """Map this pod's DNS name onto the pod of the peer role.

    The two roles are replicated jobs of one JobSet, so their pod names differ
    only in that segment and enableDNSHostnames makes the swapped name resolve.
    render_jobset.py rejects a run name containing either word, which is what
    keeps the swap unambiguous.
    """
    needle, replacement = f"-{from_role}-", f"-{to_role}-"
    if needle not in hostname:
        raise ValueError(f"hostname {hostname!r} has no {needle!r} segment")
    return hostname.replace(needle, replacement)


def sampling_backend_args(*, lane: LaneConfig) -> List[str]:
    """Pick the sampler, which is not the same choice with and without EAGLE3.

    The non-spec P/D lanes pin ``pytorch`` and are green with it at both one and
    two hosts per engine, so it stays their default. The spec lanes are a
    different case. On TPU the platform default is ``tpu``, and the only green
    EAGLE3 lane on this model, the single-host offload+spec gate at 0.785, runs
    on that default. Pinning ``pytorch`` under EAGLE3 is a pairing no lane has
    ever scored above zero on, and it also contradicts this harness itself,
    which sets SGLANG_TPU_ALLOW_SPEC_RECOMPILE for the TPU distributed top-k
    path that ``pytorch`` switches off, since LogitsProcessor gates
    use_tpu_distributed_top_k on the backend being ``tpu``.

    So a spec lane defers to the platform default by passing nothing.
    PD_SAMPLING_BACKEND forces a value either way, which is what makes the
    backend an A/B axis rather than a rebuild.
    """
    override = E("PD_SAMPLING_BACKEND", "")
    if override:
        return ["--sampling-backend", override]
    if lane.spec:
        return []
    return ["--sampling-backend", "pytorch"]


def spec_args() -> List[str]:
    """EAGLE3 flags, identical on both engines."""
    return [
        "--speculative-algorithm",
        "EAGLE3",
        "--speculative-draft-model-path",
        DRAFT_MODEL_PATH,
        "--speculative-num-steps",
        str(SPEC_NUM_STEPS),
        "--speculative-eagle-topk",
        str(SPEC_EAGLE_TOPK),
        "--speculative-num-draft-tokens",
        str(SPEC_NUM_DRAFT_TOKENS),
    ]


def hicache_args(*, ratio: float) -> List[str]:
    """Prefix caching plus raiden KV offload, for the prefill engine."""
    return [
        "--enable-hierarchical-cache",
        "--hicache-ratio",
        str(ratio),
        "--hicache-mem-layout",
        HICACHE_MEM_LAYOUT,
        "--hicache-io-backend",
        HICACHE_IO_BACKEND,
        # Write through on every prefill so a prefix is already on host by the
        # time later requests evict it. write_back would race that eviction.
        "--hicache-write-policy",
        "write_through",
    ]


def retraction_host_pool_args() -> List[str]:
    """Let the decode engine build the host pool it turns on for itself.

    Not hierarchical cache. The retraction backup is implicit on any P/D decode
    server, and the GPU-shaped defaults abort the boot on a fused TPU pool.
    """
    return [
        "--hicache-mem-layout",
        HICACHE_MEM_LAYOUT,
        "--hicache-io-backend",
        HICACHE_IO_BACKEND,
    ]


def common_launch_env(
    *, lane: LaneConfig, env: Optional[Mapping[str, str]] = None
) -> Dict[str, str]:
    """Launcher environment both engines share."""
    values = os.environ if env is None else env
    launch_env = {
        "MODEL": resolve_model_path(values),
        "TP_SIZE": str(TP_SIZE),
        "EP_SIZE": str(EP_SIZE),
        "PAGE_SIZE": str(PAGE_SIZE),
        # raiden reads device pages through the fused KV pool, which only the
        # RPA v3 backend allocates. Pinned so a launcher default cannot take the
        # transport under test away from the gate.
        "ATTENTION_BACKEND": ATTENTION_BACKEND,
        "QUANTIZATION": QUANTIZATION,
        "KV_CACHE_DTYPE": KV_CACHE_DTYPE,
        # Prefix caching is the substrate HiCache sits on, and the launcher
        # defaults it off.
        "DISABLE_RADIX_CACHE": "0",
        "ENABLE_TORCH_COMPILE": values.get("ENABLE_TORCH_COMPILE", "1"),
        "MEM_FRACTION_STATIC": values.get("MEM_FRACTION_STATIC", MEM_FRACTION_STATIC),
        "SGLANG_RAIDEN_KV_PORT": str(RAIDEN_KV_PORT),
        "PYTHON": PYTHON,
        **EXTRA_LAUNCH_ENV,
    }
    if lane.spec:
        # The TPU distributed top-k path is precompiled for fewer token buckets
        # than the decoder, so the first request on an unseen shape would trip
        # fail_on_recompile and take every rank's scheduler down. This is the
        # escape hatch spec decoding already provides: recompile instead of die.
        launch_env["SGLANG_TPU_ALLOW_SPEC_RECOMPILE"] = "1"
    return launch_env


def shared_server_args(*, lane: LaneConfig) -> List[str]:
    """Server flags both engines carry."""
    # The launchers hardcode the native "tpu" transport then append EXTRA_ARGS,
    # so naming raiden here is what selects it -- argparse keeps the last one.
    args = [
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--reasoning-parser",
        "gpt-oss",
        "--disaggregation-transfer-backend",
        DISAGG_TRANSFER_BACKEND,
        "--max-total-tokens",
        str(lane.max_total_tokens),
        "--skip-server-warmup",
        *sampling_backend_args(lane=lane),
        # cached_tokens_details in meta_info is how an offload gate tells a
        # device hit from a host reload, and it rides the P/D metadata buffer
        # from prefill to decode only when both engines report it.
        "--enable-cache-report",
    ]
    if lane.spec:
        # On both engines, not just decode: the draft model brings its own KV
        # pool, and the prefill engine has to lay one out so the pages it hands
        # over line up with what the decode engine expects.
        args += spec_args()
    return args


def build_prefill_env(
    *, lane: LaneConfig, env: Optional[Mapping[str, str]] = None
) -> Dict[str, str]:
    """Construct the environment for the prefill engine."""
    values = os.environ if env is None else env
    extra_args = shared_server_args(lane=lane)
    if lane.hicache_ratio is not None:
        extra_args += hicache_args(ratio=lane.hicache_ratio)
    if values.get("EXTRA_PREFILL_ARGS"):
        extra_args.append(values["EXTRA_PREFILL_ARGS"])
    return {
        **common_launch_env(lane=lane, env=values),
        "PORT": str(PREFILL_PORT),
        "BOOTSTRAP_PORT": str(BOOTSTRAP_PORT),
        "EXTRA_ARGS": " ".join(extra_args).strip(),
    }


def build_decode_env(
    *, lane: LaneConfig, env: Optional[Mapping[str, str]] = None
) -> Dict[str, str]:
    """Construct the environment for the decode engine."""
    values = os.environ if env is None else env
    extra_args = shared_server_args(lane=lane) + retraction_host_pool_args()
    if values.get("EXTRA_DECODE_ARGS"):
        extra_args.append(values["EXTRA_DECODE_ARGS"])
    return {
        **common_launch_env(lane=lane, env=values),
        "PORT": str(DECODE_PORT),
        "EXTRA_ARGS": " ".join(extra_args).strip(),
    }


def _tee_command(argv: Tuple[str, ...], *, log_path: str) -> Tuple[str, ...]:
    quoted = " ".join(f"'{a}'" for a in argv)
    return ("bash", "-c", f"set -o pipefail; {quoted} 2>&1 | tee -a '{log_path}'")


def _launch(
    cmd: Tuple[str, ...],
    *,
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


def execute_prefill(*, lane: LaneConfig, worker_id: int) -> subprocess.Popen:
    """Launch this pod's shard of the prefill engine."""
    launch_env = build_prefill_env(lane=lane)
    launch_env["NODE_RANK"] = str(worker_id)
    launch_env["TPU_WORKER_ID"] = str(worker_id)
    return _launch(
        ("bash", os.path.join(BENCH, "launch_prefill.sh")),
        log_path=f"{lane.log_prefix}_prefill.log",
        extra_env=launch_env,
        cwd=BENCH,
    )


def execute_decode(*, lane: LaneConfig, worker_id: int) -> subprocess.Popen:
    """Launch this pod's shard of the decode engine."""
    launch_env = build_decode_env(lane=lane)
    launch_env["NODE_RANK"] = str(worker_id)
    launch_env["TPU_WORKER_ID"] = str(worker_id)
    return _launch(
        ("bash", os.path.join(BENCH, "launch_decode.sh")),
        log_path=f"{lane.log_prefix}_decode.log",
        extra_env=launch_env,
        cwd=BENCH,
    )


def execute_router(
    *, lane: LaneConfig, prefill_host: str, decode_host: str
) -> subprocess.Popen:
    """Start the mini-LB.

    Prefill is advertised by its cluster-reachable name, never 127.0.0.1: the LB
    injects that hostname as the KV bootstrap address the decode pod dials back.
    """
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
    return _launch(cmd, log_path=f"{lane.log_prefix}_lb.log")


def kill(proc: Optional[subprocess.Popen]) -> None:
    """Stop a launched engine and everything in its process group."""
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


def http_ok(url: str, *, timeout: int = 5) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, TimeoutError, OSError, ConnectionError):
        return False


def wait_http_200(*, url: str, secs: int, procs: Tuple = ()) -> None:
    deadline = time.time() + secs
    while time.time() < deadline:
        for proc in procs:
            if proc is not None and proc.poll() is not None:
                raise RuntimeError(f"process for {url} exited rc={proc.returncode}")
        if http_ok(url):
            return
        time.sleep(10)
    raise TimeoutError(f"{url} not ready after {secs}s")


def get_json(url: str) -> Dict:
    import json

    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read())


def transport_failures(*, prefill_info: Mapping, decode_info: Mapping) -> List[str]:
    """Reject a run where the two engines are not a raiden P/D pair.

    Without this a mis-launched pair still answers every request -- one engine
    would simply serve them end to end -- and the gates downstream would pass
    while measuring nothing about disaggregation.
    """
    failures = []
    for label, info, expected_mode in (
        ("prefill", prefill_info, PREFILL_ROLE),
        ("decode", decode_info, DECODE_ROLE),
    ):
        mode = info.get("disaggregation_mode")
        if mode != expected_mode:
            failures.append(
                f"{label} engine reports disaggregation_mode={mode!r}, "
                f"expected {expected_mode!r}"
            )
        backend = info.get("disaggregation_transfer_backend")
        if backend != DISAGG_TRANSFER_BACKEND:
            failures.append(
                f"{label} engine reports disaggregation_transfer_backend="
                f"{backend!r}, expected {DISAGG_TRANSFER_BACKEND!r}"
            )
    return failures


def hicache_failures(*, prefill_info: Mapping) -> List[str]:
    """Reject a prefill engine that did not come up on raiden HiCache.

    The offload probe reads tier counts, and a server that silently booted
    without hierarchical cache reports zero host tokens -- indistinguishable
    from a transport that moved nothing. Reading the config back separates them.
    """
    failures = []
    if not prefill_info.get("enable_hierarchical_cache"):
        failures.append(
            "prefill engine came up without hierarchical cache, so nothing "
            "would be offloaded and the probe below could not mean anything"
        )
    backend = prefill_info.get("hicache_io_backend")
    if backend != HICACHE_IO_BACKEND:
        failures.append(
            f"prefill engine reports hicache_io_backend={backend!r}, expected "
            f"{HICACHE_IO_BACKEND!r}"
        )
    return failures


def pd_transport_runner(*, lane: LaneConfig, prefill_url: str, decode_url: str):
    """Build a gate that reads both engines' modes and backends back.

    Runs ahead of the feature gates so a red run says whether the pair was wired
    wrong or the answers were wrong.
    """

    def run(base_url: str) -> Dict:
        prefill_info = get_json(f"{prefill_url}/server_info")
        decode_info = get_json(f"{decode_url}/server_info")
        failures = transport_failures(
            prefill_info=prefill_info, decode_info=decode_info
        )
        if lane.hicache_ratio is not None:
            failures += hicache_failures(prefill_info=prefill_info)
        print(
            f"[{lane.test_name}] lb={base_url} "
            f"prefill={prefill_info.get('disaggregation_mode')}/"
            f"{prefill_info.get('disaggregation_transfer_backend')} "
            f"decode={decode_info.get('disaggregation_mode')}/"
            f"{decode_info.get('disaggregation_transfer_backend')}",
            flush=True,
        )
        return {
            "score": 0.0 if failures else 1.0,
            "_passed": not failures,
            "_failures": failures,
        }

    return run


def spec_runner(*, test_name: str, min_accept_length: float, num_reads: int = 3):
    """Build a gate proving EAGLE3 survived disaggregation.

    The offload lanes get this for free out of their three probe reads, but a
    lane with no probe has to drive its own. Both checks matter and fail for
    different reasons: no verify forwards means the decode engine quietly
    dropped to plain decoding, which a GSM8K score alone would never reveal,
    while verify forwards at an accept length of one mean the draft model runs
    and contributes nothing.

    Reads are summed rather than judged one at a time, since a single short read
    landing no drafts is noise while a mean at the floor is not.
    """

    def run(base_url: str) -> Dict:
        # Imported here rather than in the factory so assembling a lane's gate
        # list needs nothing installed, which is what lets the host-side tests
        # build it on a bare checkout and catch a bad constant before a lane
        # spends forty minutes booting engines to find it.
        from sglang.test.test_utils import dump_metric
        from sglang.test.tpu.kv_offload_probe import (
            generate_with_cache_report,
            probe_prompt,
        )

        reads = [
            generate_with_cache_report(
                base_url=base_url,
                prompt=probe_prompt(index=i, approx_tokens=512),
                max_new_tokens=32,
            )
            for i in range(num_reads)
        ]
        verifies = sum(r.spec_verify_ct for r in reads)
        completions = sum(r.completion_tokens for r in reads)
        accept_length = completions / verifies if verifies else 0.0
        print(
            f"[{test_name}] spec verify forwards {verifies} over {completions} "
            f"completion tokens, mean accept length {accept_length:.2f} against "
            f"floor {min_accept_length}",
            flush=True,
        )
        dump_metric(
            f"{test_name}.spec_accept_length",
            accept_length,
            labels={"model": test_name, "kind": "spec"},
        )

        failures = []
        if not verifies:
            failures.append(
                "speculative decoding never ran through the P/D pair: "
                f"spec_verify_ct=0 over {completions} completion tokens"
            )
        elif accept_length <= min_accept_length:
            failures.append(
                f"mean accept length {accept_length:.2f} is not above "
                f"{min_accept_length:.2f}, so the draft model ran but its tokens "
                "were never accepted"
            )
        return {"score": accept_length, "_passed": not failures, "_failures": failures}

    return run


# One short GSM8K-shaped word problem. The question itself is not the variable
# under test -- run 32986325885 scored 0.0000 on all 100 real examples -- so a
# single self-contained prompt is enough and keeps the gate off the dataset.
_TERMINATION_PROMPT = (
    "Natalia sold clips to 48 of her friends in April, and then she sold half "
    "as many clips in May. How many clips did Natalia sell altogether in April "
    "and May? Answer with the number."
)
# Enough of the discarded reasoning to tell a repetition loop from a long but
# coherent chain. A loop repeats inside a few hundred characters, and anything
# longer would bury the rest of the pod log.
_REASONING_TAIL_CHARS = 400
# Enough tokens to recognise a harmony header, a repeated single token, or
# byte-level junk, without turning the gate into a token dump.
_RAW_TOKENS = 24


def termination_runner(*, test_name: str, max_tokens: int, num_prompts: int = 2):
    """Build a gate proving a response actually finishes and carries an answer.

    GSM8K scores the last integer in the assistant's ``content``. With
    ``--reasoning-parser gpt-oss`` the analysis channel is split off into
    ``reasoning_content``, so a model that never reaches its final channel hands
    the scorer an empty string, which has no integers and grades wrong on every
    example. That is exactly 0.0000 with no error anywhere, which is what the
    EAGLE3-over-P/D lane returns, and the accuracy number alone cannot tell it
    apart from a model that answered and answered badly.

    So this asks the one question the score cannot: did the generation stop on
    its own, and did anything land in the final channel. It runs ahead of GSM8K
    because it is two requests against a hundred, and it prints the tail of the
    discarded reasoning, which separates the two shapes the failure can take --
    a degenerate repetition loop points at the KV the verify step attends over,
    while coherent reasoning that simply runs long points at the sampling path.
    """

    def run(base_url: str) -> Dict:
        import json

        from sglang.test.test_utils import dump_metric

        terminated = 0
        failures = []
        for index in range(num_prompts):
            payload = json.dumps(
                {
                    "model": SERVED_MODEL_NAME,
                    "messages": [{"role": "user", "content": _TERMINATION_PROMPT}],
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                    # Run 33022233492 came back with 1024 tokens and both
                    # channels empty, which the string fields alone cannot
                    # explain: no text at all is neither a repetition loop nor
                    # a long chain. The tokens say whether the model emitted
                    # control tokens the harmony parser then dropped, or
                    # something that is not language in the first place.
                    "logprobs": True,
                }
            ).encode()
            request = urllib.request.Request(
                base_url.rstrip("/") + "/v1/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=900) as response:
                body = json.loads(response.read())
            choice = body["choices"][0]
            finish_reason = choice.get("finish_reason")
            # Absent rather than empty on a build with no reasoning parser, and
            # both read as "nothing was split off", which is what is reported.
            content = choice["message"].get("content") or ""
            reasoning = choice["message"].get("reasoning_content") or ""
            completion_tokens = body.get("usage", {}).get("completion_tokens", 0)
            print(
                f"[{test_name}] termination read {index}: "
                f"finish_reason={finish_reason} completion_tokens="
                f"{completion_tokens} of {max_tokens}, content {len(content)} "
                f"chars, reasoning {len(reasoning)} chars",
                flush=True,
            )
            print(
                f"[{test_name}] read {index} content: {content[:200]!r}",
                flush=True,
            )
            # Absent when the build does not return logprobs, which reads the
            # same as "no tokens to show" and is reported as such.
            logprobs = choice.get("logprobs") or {}
            tokens = [entry["token"] for entry in logprobs.get("content") or []]
            print(
                f"[{test_name}] read {index} first {_RAW_TOKENS} of "
                f"{len(tokens)} tokens: {tokens[:_RAW_TOKENS]!r}",
                flush=True,
            )
            print(
                f"[{test_name}] read {index} reasoning tail: "
                f"{reasoning[-_REASONING_TAIL_CHARS:]!r}",
                flush=True,
            )
            if finish_reason == "length":
                failures.append(
                    f"read {index} hit the {max_tokens} token cap instead of "
                    f"stopping, so it never reached a final channel"
                )
            elif not content.strip():
                failures.append(
                    f"read {index} stopped on {finish_reason} with an empty "
                    f"final channel after {completion_tokens} tokens, so GSM8K "
                    "has no text to score"
                )
            else:
                terminated += 1

        score = terminated / num_prompts
        dump_metric(
            f"{test_name}.termination_rate",
            score,
            labels={"model": test_name, "kind": "termination"},
        )
        return {"score": score, "_passed": not failures, "_failures": failures}

    return run


def run_prefill_leader(
    *, lane: LaneConfig, topology: PodTopology, model_config_fn
) -> int:
    """Serve prefill rank 0, bring up the LB, and run the lane's gates.

    ``model_config_fn`` is a factory rather than a dict because a gate that
    reads the decode engine directly needs its hostname, and that is only known
    once this pod knows its own.
    """
    from sglang.test.tpu import model_config_main

    procs = []
    self_host = topology.self_hostname
    decode_host = peer_hostname(
        hostname=self_host, from_role=PREFILL_ROLE, to_role=DECODE_ROLE
    )
    try:
        procs.append(execute_prefill(lane=lane, worker_id=0))
        wait_http_200(
            url=f"http://127.0.0.1:{PREFILL_PORT}/health",
            secs=WAIT_SECS,
            procs=tuple(procs),
        )
        wait_http_200(url=f"http://{decode_host}:{DECODE_PORT}/health", secs=WAIT_SECS)

        procs.append(
            execute_router(lane=lane, prefill_host=self_host, decode_host=decode_host)
        )
        wait_http_200(
            url=f"http://127.0.0.1:{LB_PORT}/health", secs=300, procs=tuple(procs)
        )

        model_config_main(
            model_config_fn(decode_host=decode_host),
            base_url=f"http://127.0.0.1:{LB_PORT}",
        )
        return 0
    finally:
        if not KEEP_UP:
            for proc in procs:
                kill(proc)


def run_worker(*, lane: LaneConfig, topology: PodTopology) -> int:
    """Serve this pod's shard until it exits, which the JobSet deadline bounds."""
    if topology.role == DECODE_ROLE:
        proc = execute_decode(lane=lane, worker_id=topology.group_rank)
    else:
        proc = execute_prefill(lane=lane, worker_id=topology.group_rank)
    try:
        while True:
            rc = proc.poll()
            if rc is not None:
                return rc
            time.sleep(10)
    finally:
        kill(proc)


def serve_role(*, lane: LaneConfig, model_config_fn) -> int:
    """Dispatch on the role and rank this pod was given."""
    # The gates run under model_config_main, which ends in a bare
    # unittest.main() and so reads sys.argv. The pod is started as
    # "python3 <script> integration", and left in place that mode word is
    # taken for a test name: the loader fails to find it, reports one error
    # and exits before a single gate runs, on a pair of engines that booted
    # fine. Dropping it here makes the loader collect the synthesised gate
    # case instead, which is what the run is for.
    sys.argv = [sys.argv[0]]
    validate_model_source(resolve_model_path())
    topology = PodTopology.from_env()
    if topology.is_prefill_leader:
        return run_prefill_leader(
            lane=lane, topology=topology, model_config_fn=model_config_fn
        )
    return run_worker(lane=lane, topology=topology)


def render_jobset_yaml(
    *,
    lane: LaneConfig,
    script_path: str,
    image: str = "us-central1-docker.pkg.dev/sgl-project/tpu-ci/sglang:latest",
    kueue_lane: str = "nightly",
    weights_pvc: str = "model-weights-rox",
) -> str:
    """Render the JobSet for a lane: two hosts of prefill, two of decode."""
    rel_path = os.path.relpath(script_path, REPO)
    cmd = [
        PYTHON,
        RENDER_JOBSET,
        "--pd-prefill-hosts",
        str(HOSTS_PER_ENGINE),
        "--pd-decode-hosts",
        str(HOSTS_PER_ENGINE),
        "--lane",
        kueue_lane,
        "--image",
        image,
        "--weights-pvc",
        weights_pvc,
        "--name",
        lane.run_name,
        # Names the dashboard row and the TIMINGS record. The renderer rejects
        # P/D mode without it rather than emit a record with an empty `file`,
        # so this has to match what --exec actually runs.
        "--files",
        rel_path,
        "--exec",
        f"python3 {rel_path} integration",
    ]
    return subprocess.check_output(cmd, text=True)
