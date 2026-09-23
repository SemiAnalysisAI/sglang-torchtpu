"""Launch and probe helpers for fronting TPU P/D engines with experimental/sgl-router.

``SGLANG_TPU_PD_ROUTER=sgl_router`` selects upstream's single-model router --
a standalone Rust binary built into the TPU image by ``docker/tpu.Dockerfile``
(name overridable with ``SGLANG_TPU_SGL_ROUTER_BIN``) -- instead of the mini-LB
the P/D tests launch by default. Any other value, or none, leaves every test
exactly as it was; this module never validates the variable's other spellings.

What differs from the mini-LB, and what the helpers here absorb:

* One flat ``--worker-urls`` list. The router asks each worker's
  ``/server_info`` for ``disaggregation_mode`` and the prefill's bootstrap
  port, so the prefill/decode split and the bootstrap contract need no flags.
* It serves exactly one model and registers workers under the engine's
  ``served_model_name`` -- for a bare ``--model-path`` launch, the path -- so
  ``--model-id`` must be that same string (:func:`build_cmd` takes ``model_path``).
* Client surface is the OpenAI endpoints plus ``/healthz``, ``/readyz``,
  ``/metrics`` and ``/flush_cache``. No ``/health``, ``/generate`` or
  ``/server_info``: tests poll :func:`health_path`, the accept-length gate
  reads the decode engines directly (:func:`spec_server_info_env`), and the
  cache probes go through ``/v1/chat/completions`` (:func:`chat_generate`),
  reading the same accounting off ``usage`` and SGLang's ``sglext`` extension.
* It tokenizes at the router and forwards ``input_ids``, so ``cache_aware``
  and ``prefix_hash`` are token-accurate over plain HTTP; ``power_of_two`` on
  decode reads real engine load. Those are the defaults here.

Knobs: ``TPU_PD_PREFILL_POLICY`` / ``TPU_PD_DECODE_POLICY``,
``TPU_PD_PREFIX_TOKEN_COUNT`` / ``TPU_PD_PREFIX_HASH_LOAD_FACTOR`` (prefix_hash
only), ``SGLANG_TPU_OTLP_ENDPOINT`` (span export + router->engine trace
propagation). Standard library only, so the argv is unit-testable anywhere.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

SGL_ROUTER = "sgl_router"
DEFAULT_BIN = "sgl-router"

# experimental/sgl-router/src/config/types.rs: PolicyKind and DecodePolicyKind.
PREFILL_POLICIES = (
    "round_robin",
    "random",
    "power_of_two",
    "load_based",
    "cache_aware",
    "session_aware",
    "prefix_hash",
)
DECODE_POLICIES = ("power_of_two", "legacy_host_affinity")
DEFAULT_PREFILL_POLICY = "cache_aware"
DEFAULT_DECODE_POLICY = "power_of_two"

# Read by sglang.test.kits.eval_accuracy_kit._check_accept_length when the
# router URL serves no /server_info.
SPEC_SERVER_INFO_ENV = "SGLANG_TPU_SPEC_SERVER_INFO_URLS"


def _env(values: Optional[Dict[str, str]]) -> Dict[str, str]:
    return dict(os.environ) if values is None else values


def selected(*, env: Optional[Dict[str, str]] = None) -> bool:
    """True when ``SGLANG_TPU_PD_ROUTER=sgl_router``."""
    return _env(env).get("SGLANG_TPU_PD_ROUTER", "").strip().lower() == SGL_ROUTER


def _policy(
    name: str, env_key: str, default: str, choices: Sequence[str], env: Dict[str, str]
) -> str:
    policy = env.get(env_key, "").strip() or default
    if policy not in choices:
        raise ValueError(
            f"{name} policy {policy!r} is not one of {tuple(choices)} for sgl-router"
        )
    return policy


def build_cmd(
    *,
    prefill_urls: Sequence[str],
    decode_urls: Sequence[str],
    host: str,
    port: int,
    model_path: str,
    tokenizer_path: Optional[str] = None,
    request_timeout_secs: Optional[int] = None,
    extra: Sequence[str] = (),
    env: Optional[Dict[str, str]] = None,
) -> Tuple[str, ...]:
    """argv that fronts the given prefill and decode engines with sgl-router."""
    values = _env(env)
    if not model_path:
        raise ValueError(
            "sgl-router needs model_path: it serves one model and must be told its id"
        )
    if not prefill_urls or not decode_urls:
        raise ValueError("sgl-router P/D needs at least one prefill and one decode URL")
    prefill_policy = _policy(
        "prefill",
        "TPU_PD_PREFILL_POLICY",
        DEFAULT_PREFILL_POLICY,
        PREFILL_POLICIES,
        values,
    )
    decode_policy = _policy(
        "decode", "TPU_PD_DECODE_POLICY", DEFAULT_DECODE_POLICY, DECODE_POLICIES, values
    )
    cmd = [
        values.get("SGLANG_TPU_SGL_ROUTER_BIN", "").strip() or DEFAULT_BIN,
        "--host",
        host,
        "--port",
        str(port),
        "--model-id",
        model_path,
        "--tokenizer-path",
        tokenizer_path or model_path,
        "--worker-urls",
        *prefill_urls,
        *decode_urls,
        "--policy",
        prefill_policy,
        "--decode-policy",
        decode_policy,
    ]
    if prefill_policy == "prefix_hash":
        for flag, env_key in (
            ("--prefix-token-count", "TPU_PD_PREFIX_TOKEN_COUNT"),
            ("--prefix-hash-load-factor", "TPU_PD_PREFIX_HASH_LOAD_FACTOR"),
        ):
            value = values.get(env_key, "").strip()
            if value:
                cmd += [flag, value]
    otlp = values.get("SGLANG_TPU_OTLP_ENDPOINT", "").strip()
    if otlp:
        cmd += ["--otlp-endpoint", otlp]
    if request_timeout_secs is not None:
        cmd += ["--request-timeout-secs", str(request_timeout_secs)]
    cmd += list(extra)
    return tuple(cmd)


def health_path(*, env: Optional[Dict[str, str]] = None) -> str:
    """``/readyz`` for sgl-router (200 once every role it needs is introspected), else ``/health``."""
    return "/readyz" if selected(env=env) else "/health"


def spec_server_info_env(
    decode_urls: Sequence[str], *, env: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """Env entries pointing the accept-length gate at the decode engines.

    ``avg_spec_accept_length`` lives on the decode engine's ``/server_info``.
    The mini-LB aggregates it onto the router URL; sgl-router serves no
    ``/server_info`` at all. Empty unless sgl-router is selected, so exporting
    unconditionally changes nothing for the other routers.
    """
    if not selected(env=env):
        return {}
    return {SPEC_SERVER_INFO_ENV: ",".join(u.rstrip("/") for u in decode_urls)}


def metrics_url(*, host: str, port: int) -> str:
    """``/metrics`` shares the API listener."""
    return f"http://{host}:{port}/metrics"


@dataclass(frozen=True)
class ChatGeneration:
    """What the ``/generate`` probes read off ``meta_info``, recovered from a chat completion."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    device_cached_tokens: int
    host_cached_tokens: int
    spec_verify_ct: int


def parse_chat_completion(payload: Dict[str, Any]) -> ChatGeneration:
    """Map a chat-completion body onto the fields the cache and spec probes use.

    ``usage.prompt_tokens_details.cached_tokens`` is the total (engines must run
    with ``--enable-cache-report``; the prefix-cache P/D tests already do).
    ``sglext.cached_tokens_details`` splits it into device/host and
    ``sglext.spec_tokens_details`` carries ``spec_verify_ct``; both are per-request
    opt-ins (``return_cached_tokens_details``, ``return_spec_tokens_details``).
    A missing breakdown with a non-zero total is the contradiction the
    ``/generate`` probe also refuses to pass silently.
    """
    choice = (payload.get("choices") or [{}])[0]
    usage = payload.get("usage") or {}
    details_total = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    if details_total is None:
        raise RuntimeError(
            "chat response carried no usage.prompt_tokens_details.cached_tokens; launch "
            f"the engines with --enable-cache-report: {usage!r}"
        )
    ext = choice.get("sglext") or {}
    details = ext.get("cached_tokens_details")
    if details is None:
        if details_total:
            raise RuntimeError(
                f"the server reported {details_total} cached tokens with no "
                "sglext.cached_tokens_details to break them down (send "
                "return_cached_tokens_details=true)"
            )
        details = {}
    spec = ext.get("spec_tokens_details") or {}
    if isinstance(spec, list):
        spec = spec[0] if spec else {}
    return ChatGeneration(
        text=(choice.get("message") or {}).get("content") or "",
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        cached_tokens=int(details_total),
        device_cached_tokens=int(details.get("device") or 0),
        host_cached_tokens=int(details.get("host") or 0),
        spec_verify_ct=int(spec.get("spec_verify_ct") or 0),
    )


_SERVED_MODEL: Dict[str, str] = {}


def served_model_id(base_url: str, *, timeout: int = 60) -> str:
    """The one model id the router advertises on ``/v1/models``."""
    base = base_url.rstrip("/")
    if base not in _SERVED_MODEL:
        with urllib.request.urlopen(base + "/v1/models", timeout=timeout) as resp:
            data = json.loads(resp.read()).get("data") or []
        if not data:
            raise RuntimeError(
                f"{base}/v1/models lists no model, so no chat request can target it"
            )
        _SERVED_MODEL[base] = data[0]["id"]
    return _SERVED_MODEL[base]


def chat_generate(
    base_url: str, prompt: str, max_new_tokens: int, *, timeout: int = 900
) -> ChatGeneration:
    """One greedy chat completion through the router, with cache and spec accounting.

    Stands in for the ``/generate`` probes when the router has no ``/generate``.
    The prompt goes through the chat template, so token counts differ slightly
    from the raw-text path; the probes assert tiers and ordering, not counts.
    """
    base = base_url.rstrip("/")
    body = {
        "model": served_model_id(base),
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "temperature": 0.0,
        "return_cached_tokens_details": True,
        "return_spec_tokens_details": True,
    }
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return parse_chat_completion(json.loads(resp.read()))
