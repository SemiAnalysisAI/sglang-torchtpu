"""Router argv for TPU prefill/decode disaggregation tests.

Two front ends can sit in front of a P/D pair, and both ship in the same
``sglang-router`` wheel that ``pyproject_tpu.toml`` already installs:

* ``mini_lb`` -- ``MiniLoadBalancer``, a debug-only Python FastAPI app. It
  announces itself as debug-only, force-overrides whatever ``--policy`` you pass
  to ``random``, and picks a prefill and a decode independently with
  ``random.randint``. This is what every TPU P/D test has used to date, so it is
  the default here and its argv is unchanged.
* ``smg`` -- the SGLang Model Gateway's real PD router (compiled Rust). It runs
  an independent, configurable policy over the prefill pool and the decode pool,
  and exports its own Prometheus metrics.
* ``sgl_router`` -- upstream's ``experimental/sgl-router``, the single-model
  router that is replacing the gateway. A standalone Rust binary (built into the
  TPU image by ``docker/tpu.Dockerfile``, name overridable with
  ``SGLANG_TPU_SGL_ROUTER_BIN``), not a Python module. It takes one flat
  ``--worker-urls`` list and classifies each worker as prefill or decode from the
  ``disaggregation_mode`` the engine reports on ``/server_info``, reads the
  prefill's bootstrap port from the same place, tokenizes at the router and
  forwards ``input_ids`` so ``cache_aware`` routes on tokens over plain HTTP.

All three speak the identical bootstrap contract to the engines --
``bootstrap_host``, ``bootstrap_port`` and ``bootstrap_room`` injected into the
request body -- so switching between them needs no server-side change. What
does differ is the client surface: sgl-router serves only the OpenAI endpoints
plus ``/healthz``, ``/readyz``, ``/metrics`` and ``/flush_cache``. It has no
``/health``, ``/generate`` or ``/server_info``; see :func:`router_health_path`
and :func:`spec_server_info_env` for how the harness adapts.

Selecting one
-------------
``SGLANG_TPU_PD_ROUTER=mini_lb|smg|sgl_router`` (default ``mini_lb``), or pass
``mode=`` explicitly.
Policies come from ``TPU_PD_PREFILL_POLICY`` / ``TPU_PD_DECODE_POLICY`` so a CDK
recipe can sweep them without editing a test.

Policy notes specific to TPU
----------------------------
``cache_aware`` is the reason to run SMG at all: it keeps an approximate radix
tree per ``(pool, model)`` and routes a request to the worker that already holds
the longest prefix of it, falling back to shortest-queue when the pools drift out
of balance.

It belongs on **prefill only**. A decode engine in P/D runs with the radix cache
forced off -- ``arg_groups/pd_disaggregation_hook.py`` sets
``disable_radix_cache`` whenever ``disaggregation_mode == "decode"`` and
``--disaggregation-decode-enable-radix-cache`` is unset, which no TPU test sets --
so decode-side prefix affinity would maintain a tree whose matches can never
correspond to a real cache hit.

``power_of_two`` is deliberately **not** the decode default. It is fed by a
LoadMonitor that polls ``GET {worker}/v1/loads?include=core`` and reads
``aggregate.total_tokens``; the SGLang server's ``/v1/loads`` returns
``{timestamp, version, accelerator, num_accelerators, loads: [...]}`` with no
``aggregate`` key, so every worker reports ``-1`` and the policy silently
degenerates to a random pick between two random candidates. ``round_robin``
is honest about doing the same amount of thinking, so it is the default until
the server grows an ``aggregate`` block.

``prefix_hash`` cannot work over HTTP at all: the HTTP PD router passes
``tokens: None`` to the policy, so it always takes its ``no_tokens`` branch.
It becomes usable on the gRPC path, which carries real token ids.

sgl-router has neither of those problems and a different vocabulary: prefill
policies are ``round_robin``, ``random``, ``power_of_two``, ``load_based``,
``cache_aware``, ``session_aware`` and ``prefix_hash``; decode policies are
``power_of_two`` and ``legacy_host_affinity``. Its ``power_of_two`` reads engine
load from ``/server_info`` rather than ``/v1/loads``, so it is the decode
default there. Its ``prefix_hash`` works over HTTP because the router
tokenizes; its two knobs come from ``TPU_PD_PREFIX_TOKEN_COUNT`` and
``TPU_PD_PREFIX_HASH_LOAD_FACTOR`` (the gateway's tunables of the same names).
``SGLANG_TPU_OTLP_ENDPOINT`` turns on span export and router-to-engine trace
propagation.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Sequence, Tuple

MINI_LB = "mini_lb"
SMG = "smg"
SGL_ROUTER = "sgl_router"
ROUTER_MODES = (MINI_LB, SMG, SGL_ROUTER)

# Mirrors ``_POLICY_CHOICES`` in sglang_router.router_args. Note that the Rust
# binary's own value_parser accepts a subset; we always launch through
# ``python -m sglang_router.launch_router``, which accepts all of these.
POLICY_CHOICES = (
    "random",
    "round_robin",
    "cache_aware",
    "power_of_two",
    "bucket",
    "manual",
    "consistent_hashing",
    "prefix_hash",
)

# See the module docstring: affinity on prefill, no false precision on decode.
DEFAULT_PREFILL_POLICY = "cache_aware"
DEFAULT_DECODE_POLICY = "round_robin"

# experimental/sgl-router's own vocabulary (config/types.rs: PolicyKind and
# DecodePolicyKind). Kept apart from POLICY_CHOICES so a typo is caught here
# rather than by the binary at launch.
SGL_ROUTER_PREFILL_POLICIES = (
    "round_robin",
    "random",
    "power_of_two",
    "load_based",
    "cache_aware",
    "session_aware",
    "prefix_hash",
)
SGL_ROUTER_DECODE_POLICIES = ("power_of_two", "legacy_host_affinity")
# power_of_two IS sound here: sgl-router scores decode load from /server_info,
# not from the /v1/loads aggregate the gateway never receives.
SGL_ROUTER_DEFAULT_DECODE_POLICY = "power_of_two"
SGL_ROUTER_DEFAULT_BIN = "sgl-router"

# SMG binds its Prometheus exporter on its own listener, not the gateway port.
DEFAULT_PROMETHEUS_PORT = 29000


def _env(values: Optional[Dict[str, str]]) -> Dict[str, str]:
    return dict(os.environ) if values is None else values


def resolve_mode(*, env: Optional[Dict[str, str]] = None) -> str:
    """Router selected by ``SGLANG_TPU_PD_ROUTER``, defaulting to the legacy mini-LB."""
    mode = _env(env).get("SGLANG_TPU_PD_ROUTER", MINI_LB).strip().lower()
    if mode not in ROUTER_MODES:
        raise ValueError(
            f"SGLANG_TPU_PD_ROUTER={mode!r} is not one of {ROUTER_MODES}. "
            f"Use {MINI_LB!r} for the legacy debug load balancer, {SMG!r} "
            f"for the SGLang Model Gateway PD router or {SGL_ROUTER!r} for "
            f"upstream's experimental/sgl-router binary."
        )
    return mode


def _resolve_policy(
    *,
    name: str,
    explicit: Optional[str],
    env_key: str,
    default: str,
    env: Dict[str, str],
    choices: Sequence[str] = POLICY_CHOICES,
) -> str:
    policy = explicit or env.get(env_key) or default
    if policy not in choices:
        raise ValueError(f"{name} policy {policy!r} is not one of {tuple(choices)}.")
    return policy


def _prefill_arg(*, prefill_url: str, bootstrap_port: Optional[int]) -> Tuple[str, ...]:
    """``--prefill URL [BOOTSTRAP_PORT]``.

    The port is positional and optional. Every TPU test has historically omitted
    it, which makes the router inject ``bootstrap_port: null`` and leaves decode
    to fall back to its own ``--disaggregation-bootstrap-port`` default of 8998 --
    correct only because the tests happen to launch prefill on that same port.
    Passing it makes the pairing independent of that coincidence.
    """
    if bootstrap_port is None:
        return ("--prefill", prefill_url)
    return ("--prefill", prefill_url, str(bootstrap_port))


def build_router_cmd(
    *,
    python: str,
    prefill_url: str,
    decode_url: str,
    host: str,
    port: int,
    bootstrap_port: Optional[int] = None,
    mode: Optional[str] = None,
    prefill_policy: Optional[str] = None,
    decode_policy: Optional[str] = None,
    prometheus_port: Optional[int] = None,
    request_timeout_secs: Optional[int] = None,
    cache_threshold: Optional[float] = None,
    balance_abs_threshold: Optional[int] = None,
    balance_rel_threshold: Optional[float] = None,
    extra: Sequence[str] = (),
    env: Optional[Dict[str, str]] = None,
    model_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    binary: Optional[str] = None,
) -> Tuple[str, ...]:
    """Build the argv that fronts one prefill and one decode engine.

    With ``mode="mini_lb"`` and ``bootstrap_port=None`` this reproduces the argv
    every TPU P/D test used before this helper existed, so the default path is
    unchanged. Policy and Prometheus arguments are only emitted for ``smg``:
    mini-LB ignores the first and does not implement the second.

    ``mode="sgl_router"`` launches the standalone binary instead. It needs
    ``model_path``: sgl-router serves exactly one model and registers each
    worker under the ``served_model_name`` the engine reports, which for a bare
    ``--model-path`` launch is the path itself, so ``--model-id`` must be that
    same string for dispatch to find the workers (and for the ``model`` field
    the eval clients send to match). ``tokenizer_path`` defaults to
    ``model_path``; the router tokenizes there to route on tokens.
    ``bootstrap_port`` is accepted and ignored: sgl-router reads the prefill's
    ``disaggregation_bootstrap_port`` off ``/server_info``.
    """
    values = _env(env)
    mode = mode or resolve_mode(env=values)

    if mode == SGL_ROUTER:
        return _sgl_router_cmd(
            prefill_url=prefill_url,
            decode_url=decode_url,
            host=host,
            port=port,
            prefill_policy=prefill_policy,
            decode_policy=decode_policy,
            request_timeout_secs=request_timeout_secs,
            extra=extra,
            env=values,
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            binary=binary,
        )

    cmd = [python, "-m", "sglang_router.launch_router", "--pd-disaggregation"]
    if mode == MINI_LB:
        cmd.append("--mini-lb")

    cmd += _prefill_arg(prefill_url=prefill_url, bootstrap_port=bootstrap_port)
    cmd += ["--decode", decode_url]
    cmd += ["--host", host, "--port", str(port)]

    if mode == SMG:
        cmd += [
            "--prefill-policy",
            _resolve_policy(
                name="prefill",
                explicit=prefill_policy,
                env_key="TPU_PD_PREFILL_POLICY",
                default=DEFAULT_PREFILL_POLICY,
                env=values,
            ),
            "--decode-policy",
            _resolve_policy(
                name="decode",
                explicit=decode_policy,
                env_key="TPU_PD_DECODE_POLICY",
                default=DEFAULT_DECODE_POLICY,
                env=values,
            ),
        ]
        cmd += _cache_aware_args(
            cache_threshold=cache_threshold,
            balance_abs_threshold=balance_abs_threshold,
            balance_rel_threshold=balance_rel_threshold,
            env=values,
        )
        cmd += _prometheus_args(prometheus_port=prometheus_port, env=values)

    if request_timeout_secs is not None:
        cmd += ["--request-timeout-secs", str(request_timeout_secs)]

    cmd += list(extra)
    return tuple(cmd)


def _sgl_router_cmd(
    *,
    prefill_url: str,
    decode_url: str,
    host: str,
    port: int,
    prefill_policy: Optional[str],
    decode_policy: Optional[str],
    request_timeout_secs: Optional[int],
    extra: Sequence[str],
    env: Dict[str, str],
    model_path: Optional[str],
    tokenizer_path: Optional[str],
    binary: Optional[str],
) -> Tuple[str, ...]:
    """argv for ``experimental/sgl-router`` fronting one prefill and one decode.

    One flat ``--worker-urls`` list: the router asks each worker's
    ``/server_info`` for ``disaggregation_mode`` and pairs the roles itself.
    ``--policy`` governs the prefill pick, ``--decode-policy`` the decode pick.
    """
    if not model_path:
        raise ValueError(
            "mode='sgl_router' needs model_path: sgl-router serves one model and "
            "must be told its id (the engines' served_model_name, i.e. the "
            "--model-path they were launched with)."
        )
    cmd = [
        binary or env.get("SGLANG_TPU_SGL_ROUTER_BIN") or SGL_ROUTER_DEFAULT_BIN,
        "--host",
        host,
        "--port",
        str(port),
        "--model-id",
        model_path,
        "--tokenizer-path",
        tokenizer_path or model_path,
        "--worker-urls",
        prefill_url,
        decode_url,
        "--policy",
        _resolve_policy(
            name="prefill",
            explicit=prefill_policy,
            env_key="TPU_PD_PREFILL_POLICY",
            default=DEFAULT_PREFILL_POLICY,
            env=env,
            choices=SGL_ROUTER_PREFILL_POLICIES,
        ),
        "--decode-policy",
        _resolve_policy(
            name="decode",
            explicit=decode_policy,
            env_key="TPU_PD_DECODE_POLICY",
            default=SGL_ROUTER_DEFAULT_DECODE_POLICY,
            env=env,
            choices=SGL_ROUTER_DECODE_POLICIES,
        ),
    ]
    if cmd[cmd.index("--policy") + 1] == "prefix_hash":
        cmd += _prefix_hash_args(env=env)
    otlp = env.get("SGLANG_TPU_OTLP_ENDPOINT", "").strip()
    if otlp:
        cmd += ["--otlp-endpoint", otlp]
    if request_timeout_secs is not None:
        cmd += ["--request-timeout-secs", str(request_timeout_secs)]
    cmd += list(extra)
    return tuple(cmd)


def _prefix_hash_args(*, env: Dict[str, str]) -> Tuple[str, ...]:
    """sgl-router's prefix_hash knobs, emitted only when overridden.

    Unset, the router applies its own defaults (256 tokens, load factor 1.25),
    which are the gateway's defaults for the same names.
    """
    args: list[str] = []
    for flag, env_key in (
        ("--prefix-token-count", "TPU_PD_PREFIX_TOKEN_COUNT"),
        ("--prefix-hash-load-factor", "TPU_PD_PREFIX_HASH_LOAD_FACTOR"),
    ):
        value = env.get(env_key, "").strip()
        if value:
            args += [flag, value]
    return tuple(args)


def router_health_path(
    *, mode: Optional[str] = None, env: Optional[Dict[str, str]] = None
) -> str:
    """The path a test polls for 200 before sending traffic to the router.

    mini-LB and the gateway proxy ``/health``; sgl-router exposes ``/readyz``,
    which turns 200 only once at least one worker of each role it needs has
    been introspected, so it is the stronger gate of the two.
    """
    mode = mode or resolve_mode(env=env)
    return "/readyz" if mode == SGL_ROUTER else "/health"


# Read by sglang.test.kits.eval_accuracy_kit._check_accept_length when the
# router's own /server_info carries no internal_states.
SPEC_SERVER_INFO_ENV = "SGLANG_TPU_SPEC_SERVER_INFO_URLS"


def spec_server_info_env(
    decode_urls: Sequence[str],
    *,
    mode: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Where the accept-length gate should read ``avg_spec_accept_length``.

    Speculative accept/reject happens on the decode engine, so that is the
    ``/server_info`` that carries ``internal_states``. mini-LB and the gateway
    aggregate it onto the router URL; sgl-router serves no ``/server_info`` at
    all, so under it the gate must go to the decode engines directly. Returns
    the env entries to export before ``model_config_main`` -- empty for the
    routers that proxy it, so exporting unconditionally is safe.
    """
    mode = mode or resolve_mode(env=env)
    if mode != SGL_ROUTER:
        return {}
    return {SPEC_SERVER_INFO_ENV: ",".join(u.rstrip("/") for u in decode_urls)}


def _cache_aware_args(
    *,
    cache_threshold: Optional[float],
    balance_abs_threshold: Optional[int],
    balance_rel_threshold: Optional[float],
    env: Dict[str, str],
) -> Tuple[str, ...]:
    """Cache-aware tuning knobs, emitted only when overridden.

    Left unset, the router applies its own CLI defaults. Those differ from the
    library defaults on every knob, so pinning a value here is the only way to
    make a sweep reproducible across router versions.
    """
    overrides = (
        ("--cache-threshold", cache_threshold, "TPU_PD_CACHE_THRESHOLD"),
        ("--balance-abs-threshold", balance_abs_threshold, "TPU_PD_BALANCE_ABS"),
        ("--balance-rel-threshold", balance_rel_threshold, "TPU_PD_BALANCE_REL"),
    )
    args: list[str] = []
    for flag, explicit, env_key in overrides:
        value = explicit if explicit is not None else env.get(env_key)
        if value is not None and str(value) != "":
            args += [flag, str(value)]
    return tuple(args)


def _prometheus_args(
    *, prometheus_port: Optional[int], env: Dict[str, str]
) -> Tuple[str, ...]:
    """Router-side metrics listener, disabled by setting the port to 0.

    Off by default in the router itself, and the Tier-2/3 measurements read
    ``smg_worker_selection_total`` and ``smg_router_request_duration_seconds``
    from it, so the tests turn it on.
    """
    raw = (
        prometheus_port
        if prometheus_port is not None
        else env.get("TPU_PD_PROMETHEUS_PORT", DEFAULT_PROMETHEUS_PORT)
    )
    resolved = int(raw)
    if resolved == 0:
        return ()
    return ("--prometheus-port", str(resolved), "--prometheus-host", "0.0.0.0")


def router_metrics_url(
    *,
    host: str,
    env: Optional[Dict[str, str]] = None,
    router_port: Optional[int] = None,
) -> Optional[str]:
    """Where ``build_router_cmd`` put the router's ``/metrics``, if anywhere."""
    values = _env(env)
    mode = resolve_mode(env=values)
    if mode == SGL_ROUTER:
        # Same listener as the API; needs the router port, not a metrics port.
        if router_port is None:
            return None
        return f"http://{host}:{router_port}/metrics"
    port = int(values.get("TPU_PD_PROMETHEUS_PORT", DEFAULT_PROMETHEUS_PORT))
    if port == 0 or mode != SMG:
        return None
    return f"http://{host}:{port}/metrics"
