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

Both speak the identical bootstrap contract to the engines -- ``bootstrap_host``,
``bootstrap_port`` and ``bootstrap_room`` injected into the request body -- so
switching between them needs no server-side change.

Selecting one
-------------
``SGLANG_TPU_PD_ROUTER=mini_lb|smg`` (default ``mini_lb``), or pass ``mode=`` explicitly.
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
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Sequence, Tuple

MINI_LB = "mini_lb"
SMG = "smg"
ROUTER_MODES = (MINI_LB, SMG)

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

# SMG binds its Prometheus exporter on its own listener, not the gateway port.
DEFAULT_PROMETHEUS_PORT = 29000


def _env(values: Optional[Dict[str, str]]) -> Dict[str, str]:
    return dict(os.environ) if values is None else values


def resolve_mode(*, env: Optional[Dict[str, str]] = None) -> str:
    """Router selected by ``SGLANG_TPU_PD_ROUTER``, defaulting to the legacy mini-LB.

    NOT FOR MERGE: this branch defaults to SMG so a full nightly exercises the
    router migration end to end before any of the PRs land. The shipped default
    is MINI_LB; see shriharim/tpu-pd-router-seam.
    """
    mode = _env(env).get("SGLANG_TPU_PD_ROUTER", SMG).strip().lower()
    if mode not in ROUTER_MODES:
        raise ValueError(
            f"SGLANG_TPU_PD_ROUTER={mode!r} is not one of {ROUTER_MODES}. "
            f"Use {MINI_LB!r} for the legacy debug load balancer or {SMG!r} "
            f"for the SGLang Model Gateway PD router."
        )
    return mode


def _resolve_policy(
    *,
    name: str,
    explicit: Optional[str],
    env_key: str,
    default: str,
    env: Dict[str, str],
) -> str:
    policy = explicit or env.get(env_key) or default
    if policy not in POLICY_CHOICES:
        raise ValueError(f"{name} policy {policy!r} is not one of {POLICY_CHOICES}.")
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
) -> Tuple[str, ...]:
    """Build the argv that fronts one prefill and one decode engine.

    With ``mode="mini_lb"`` and ``bootstrap_port=None`` this reproduces the argv
    every TPU P/D test used before this helper existed, so the default path is
    unchanged. Policy and Prometheus arguments are only emitted for ``smg``:
    mini-LB ignores the first and does not implement the second.
    """
    values = _env(env)
    mode = mode or resolve_mode(env=values)

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
    *, host: str, env: Optional[Dict[str, str]] = None
) -> Optional[str]:
    """Where ``build_router_cmd`` put the router's ``/metrics``, if anywhere."""
    values = _env(env)
    port = int(values.get("TPU_PD_PROMETHEUS_PORT", DEFAULT_PROMETHEUS_PORT))
    if port == 0 or resolve_mode(env=values) != SMG:
        return None
    return f"http://{host}:{port}/metrics"
