"""Single-host TPU integration gate for prefix caching plus KV offload.

The feature probe and GSM8K gate share one recipe-launched server boot. Both
large-model profiles run by default; MODEL_PROFILE selects one for CI isolation:

  MODEL_PROFILE=gpt_oss_120b python3 \
    test/registered/tpu/integration/test_prefix_caching_kv_offload.py

  MODEL_PROFILE=qwen3_coder_480b GSM8K_THRESHOLD=0.9 python3 \
    test/registered/tpu/integration/test_prefix_caching_kv_offload.py

Model paths, the quality floor, and cache-pressure sizes are environment
overrides. The Raiden probe verifies an immediate device-tier prefix hit and a
host-tier reload after applying device-cache pressure.
"""

import json
import math
import os
import time
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional, Tuple

import requests

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import dump_metric
from sglang.test.tpu import build_test_case

# A default suite invocation runs both profiles. CI selects one profile per
# JobSet so a model-specific failure does not hide the other.
register_tpu_ci(est_time=14400, suite="tpu-integration")


TPU_TEST_HOST = os.environ.get("SGLANG_TPU_TEST_HOST", "127.0.0.1")
TPU_TEST_PORT = os.environ.get("SGLANG_TPU_TEST_PORT", "30000")
BASE_URL = f"http://{TPU_TEST_HOST}:{TPU_TEST_PORT}"
DEFAULT_HEALTH_TIMEOUT = 2400.0
DEFAULT_GSM8K_THRESHOLD = 0.80


@dataclass(frozen=True)
class ModelProfile:
    key: str
    display_name: str
    served_name: str
    default_model_path: str
    model_env: str
    recipe: str
    default_gsm8k_threshold: float = DEFAULT_GSM8K_THRESHOLD
    reasoning_effort: Optional[str] = None
    chat_template_kwargs: Optional[Dict[str, Any]] = None
    extra_server_args: Tuple[str, ...] = ()


MODEL_PROFILES = {
    "gpt_oss_120b": ModelProfile(
        key="gpt_oss_120b",
        display_name="GPT-OSS-120B",
        served_name="gpt-oss-120b",
        default_model_path="/weights/gpt-oss-120b",
        model_env="GPT_OSS_MODEL",
        recipe="recipe/gpt_oss_120b/server.sh",
        reasoning_effort="low",
        # GPT-OSS has hybrid full/SWA attention. Raiden currently transfers one
        # fused MHA pool, so keep the model's SWA attention semantics while using
        # the non-compressed KV allocation that Raiden supports.
        extra_server_args=(
            "--reasoning-parser",
            "gpt-oss",
            "--disable-hybrid-swa-memory",
        ),
    ),
    "qwen3_coder_480b": ModelProfile(
        key="qwen3_coder_480b",
        display_name="Qwen3-Coder-480B",
        served_name="qwen3-coder-480b",
        # Match the newly merged large-model feature-combo profile. CI can use
        # QWEN3_CODER_MODEL when this checkpoint is mounted under another path.
        default_model_path="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
        model_env="QWEN3_CODER_MODEL",
        recipe="recipe/qwen3_coder_480b/server.sh",
        # The first hardware run scored 0.75 over 100 examples. Keep a
        # conservative regression floor; GSM8K_THRESHOLD[_<PROFILE>] remains
        # the CI escape hatch for re-anchoring after more samples.
        default_gsm8k_threshold=0.70,
        chat_template_kwargs={"enable_thinking": False},
        # Current TPU auto backends collapse this TP8 profile to repeated
        # punctuation. Pin the sampler and legacy Pallas MoE path used by the
        # quality-preserving configuration.
        extra_server_args=(
            "--sampling-backend",
            "pytorch",
            "--moe-runner-backend",
            "pallas_tpu",
        ),
    ),
}


def _parse_int(name: str, raw: str, minimum: int = 1) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _parse_float(
    name: str,
    raw: str,
    minimum: float = 0.0,
    maximum: Optional[float] = None,
) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric, got {raw!r}") from exc
    if value < minimum or (maximum is not None and value > maximum):
        bounds = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{name} must be {bounds}, got {value}")
    return value


def _first_env(
    env: Mapping[str, str], names: Tuple[str, ...], default: str
) -> Tuple[str, str]:
    for name in names:
        if name in env:
            return name, env[name]
    return names[-1], default


def select_profile_keys(raw: str) -> Tuple[str, ...]:
    raw = raw.strip()
    if not raw or raw == "all":
        return tuple(MODEL_PROFILES)

    selected = tuple(part.strip() for part in raw.split(",") if part.strip())
    unknown = sorted(set(selected) - set(MODEL_PROFILES))
    if unknown:
        allowed = ", ".join(("all", *MODEL_PROFILES))
        raise ValueError(
            f"unknown MODEL_PROFILE value(s): {', '.join(unknown)}; choose {allowed}"
        )
    if len(set(selected)) != len(selected):
        raise ValueError("MODEL_PROFILE must not contain duplicate profiles")
    return selected


@dataclass(frozen=True)
class Gsm8kConfig:
    num_examples: int
    num_threads: int
    max_tokens: int
    threshold: float
    atol: float
    data_path: Optional[str]

    @classmethod
    def from_env(
        cls, profile: ModelProfile, env: Optional[Mapping[str, str]] = None
    ) -> "Gsm8kConfig":
        values = os.environ if env is None else env
        suffix = profile.key.upper()

        threshold_name, threshold_raw = _first_env(
            values,
            (f"GSM8K_THRESHOLD_{suffix}", "GSM8K_THRESHOLD"),
            str(profile.default_gsm8k_threshold),
        )
        atol_name, atol_raw = _first_env(
            values,
            (f"GSM8K_ATOL_{suffix}", "GSM8K_ATOL"),
            "0",
        )
        threshold = _parse_float(threshold_name, threshold_raw, 0.0, 1.0)
        atol = _parse_float(atol_name, atol_raw, 0.0, 1.0)
        if threshold - atol < 0:
            raise ValueError(f"{threshold_name} - {atol_name} must be non-negative")

        return cls(
            num_examples=_parse_int(
                "GSM8K_NUM_EXAMPLES", values.get("GSM8K_NUM_EXAMPLES", "100")
            ),
            num_threads=_parse_int(
                "GSM8K_NUM_THREADS", values.get("GSM8K_NUM_THREADS", "8")
            ),
            max_tokens=_parse_int(
                "GSM8K_MAX_TOKENS", values.get("GSM8K_MAX_TOKENS", "2048")
            ),
            threshold=threshold,
            atol=atol,
            data_path=values.get("GSM8K_DATA_PATH") or None,
        )


@dataclass(frozen=True)
class CacheProbeConfig:
    page_size: int
    device_tokens: int
    hicache_ratio: float
    target_words: int
    filler_words: int
    num_fillers: int
    max_new_tokens: int
    kv_cache_dtype: str

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "CacheProbeConfig":
        values = os.environ if env is None else env
        page_size = _parse_int(
            "CACHE_PROBE_PAGE_SIZE", values.get("CACHE_PROBE_PAGE_SIZE", "64")
        )
        device_tokens = _parse_int(
            "CACHE_PROBE_DEVICE_TOKENS",
            values.get("CACHE_PROBE_DEVICE_TOKENS", "32768"),
        )
        target_words = _parse_int(
            "CACHE_PROBE_TARGET_WORDS",
            values.get("CACHE_PROBE_TARGET_WORDS", "2048"),
        )
        filler_words = _parse_int(
            "CACHE_PROBE_FILLER_WORDS",
            values.get("CACHE_PROBE_FILLER_WORDS", "2048"),
        )
        default_fillers = math.ceil(device_tokens / filler_words) + 4

        return cls(
            page_size=page_size,
            device_tokens=device_tokens,
            hicache_ratio=_parse_float(
                "HICACHE_RATIO", values.get("HICACHE_RATIO", "3"), 1.0
            ),
            target_words=target_words,
            filler_words=filler_words,
            num_fillers=_parse_int(
                "CACHE_PROBE_NUM_FILLERS",
                values.get("CACHE_PROBE_NUM_FILLERS", str(default_fillers)),
            ),
            max_new_tokens=_parse_int(
                "CACHE_PROBE_MAX_NEW_TOKENS",
                values.get("CACHE_PROBE_MAX_NEW_TOKENS", "16"),
            ),
            kv_cache_dtype=values.get("KV_CACHE_DTYPE", "fp8_e4m3"),
        )


@dataclass(frozen=True)
class GenerationRecord:
    text: str
    prompt_tokens: int
    device_cached_tokens: int
    host_cached_tokens: int
    latency_seconds: float


@dataclass(frozen=True)
class CacheProbeResult:
    cold: GenerationRecord
    device_hit: GenerationRecord
    host_reload: GenerationRecord
    expected_cached_tokens: int


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
    path = Path(model_path)
    if not path.is_absolute():
        return

    if not path.is_dir():
        raise FileNotFoundError(
            f"{profile.display_name} model directory does not exist: {model_path}. "
            "Provision the checkpoint on the mounted weights volume or set "
            f"{profile.model_env} to an existing local directory or Hugging Face "
            "repository ID."
        )
    config_path = path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"{profile.display_name} model directory is incomplete: "
            f"{config_path} does not exist"
        )


def build_launch_env(
    profile: ModelProfile,
    gsm8k: Gsm8kConfig,
    cache: CacheProbeConfig,
    env: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    values = os.environ if env is None else env
    inherited_extra_args = values.get("EXTRA_ARGS", "")
    server_args = [
        "--hicache-io-backend",
        "raiden",
        "--hicache-mem-layout",
        "page_first_fused",
        "--enable-cache-report",
        "--served-model-name",
        profile.served_name,
        *profile.extra_server_args,
    ]
    if inherited_extra_args:
        server_args.append(inherited_extra_args)

    context_length = max(4096, gsm8k.max_tokens + 2048)
    return {
        "MODEL": resolve_model_path(profile, values),
        "HOST": TPU_TEST_HOST,
        "PORT": TPU_TEST_PORT,
        "ATTENTION_BACKEND": "pallas_tpu_rpa_v3",
        "PAGE_SIZE": str(cache.page_size),
        "DISABLE_RADIX_CACHE": "0",
        "KV_OFFLOAD": "1",
        "HICACHE_RATIO": str(cache.hicache_ratio),
        "KV_CACHE_DTYPE": cache.kv_cache_dtype,
        "MAX_TOTAL_TOKENS": str(cache.device_tokens),
        "ISL": "2048",
        "OSL": str(gsm8k.max_tokens),
        "CONCUR": str(gsm8k.num_threads),
        "MAX_RUNNING": str(gsm8k.num_threads),
        "TORCH_COMPILE_MAX_BS": str(gsm8k.num_threads),
        "CONTEXT_LEN": str(context_length),
        "CHUNK": "2048",
        "CHUNKED_PREFILL_SIZE": "2048",
        "MAX_PREFILL_TOKENS": "2048",
        "ENABLE_TORCH_COMPILE": "1",
        "SGLANG_TPU_CHECK_RECOMPILATION": "1",
        "EXTRA_ARGS": " ".join(server_args),
    }


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


def _flush_cache(base_url: str) -> None:
    response = requests.get(base_url.rstrip("/") + "/flush_cache", timeout=60)
    response.raise_for_status()


def _generate(base_url: str, prompt: str, max_new_tokens: int) -> GenerationRecord:
    started = time.perf_counter()
    response = requests.post(
        base_url.rstrip("/") + "/generate",
        json={
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": max_new_tokens,
                "temperature": 0.0,
            },
        },
        timeout=900,
    )
    response.raise_for_status()
    payload = response.json()
    elapsed = time.perf_counter() - started
    meta = payload.get("meta_info") or {}
    details = meta.get("cached_tokens_details") or {}
    return GenerationRecord(
        text=payload["text"],
        prompt_tokens=int(meta.get("prompt_tokens") or 0),
        device_cached_tokens=int(details.get("device") or 0),
        host_cached_tokens=int(details.get("host") or 0),
        latency_seconds=elapsed,
    )


def validate_cache_probe(
    cold: GenerationRecord,
    device_hit: GenerationRecord,
    host_reload: GenerationRecord,
    page_size: int,
    kv_cache_dtype: str = "bf16",
) -> int:
    if cold.prompt_tokens < 2 * page_size:
        raise AssertionError(f"target prompt is too short: {cold.prompt_tokens} tokens")
    if cold.device_cached_tokens or cold.host_cached_tokens:
        raise AssertionError(
            "cold target request unexpectedly hit cache: "
            f"device={cold.device_cached_tokens}, host={cold.host_cached_tokens}"
        )

    expected_cached_tokens = max(
        page_size, (cold.prompt_tokens // page_size - 1) * page_size
    )
    if device_hit.device_cached_tokens < expected_cached_tokens:
        raise AssertionError(
            "immediate resend did not produce the expected device cache hit: "
            f"device={device_hit.device_cached_tokens}, "
            f"expected>={expected_cached_tokens}"
        )
    if device_hit.host_cached_tokens:
        raise AssertionError(
            "immediate resend unexpectedly loaded from host: "
            f"host={device_hit.host_cached_tokens}"
        )
    if host_reload.host_cached_tokens <= 0:
        raise AssertionError("target was not reloaded from host after cache pressure")
    if (
        host_reload.device_cached_tokens + host_reload.host_cached_tokens
        < expected_cached_tokens
    ):
        raise AssertionError(
            "host reload did not recover the expected cached prefix: "
            f"device={host_reload.device_cached_tokens}, "
            f"host={host_reload.host_cached_tokens}, "
            f"expected>={expected_cached_tokens}"
        )
    if not device_hit.text:
        raise AssertionError("device-hit request returned empty output")
    # Match the merged feature-combo oracle: FP8 KV can shift close greedy
    # token choices between cached and recomputed prefixes (TODO #321). The
    # cache-tier counters remain the mechanism gate; exact output remains an
    # additional correctness oracle for non-FP8 KV.
    fp8_kv_cache = kv_cache_dtype in ("fp8_e4m3", "fp8_e5m2", "mxfp8")
    if not fp8_kv_cache and host_reload.text != device_hit.text:
        raise AssertionError(
            "host-reloaded KV changed greedy output compared with the device hit"
        )
    return expected_cached_tokens


def _record_cache_metrics(profile: ModelProfile, result: CacheProbeResult) -> None:
    labels = {"model": profile.served_name, "test": "prefix_caching_kv_offload"}
    metrics = {
        "prefix_cache_device_hit_tokens": result.device_hit.device_cached_tokens,
        "kv_offload_host_reload_tokens": result.host_reload.host_cached_tokens,
        "prefix_cache_device_hit_latency": result.device_hit.latency_seconds,
        "kv_offload_host_reload_latency": result.host_reload.latency_seconds,
    }
    for name, value in metrics.items():
        dump_metric(name, value, labels=labels)

    print(
        json.dumps(
            {
                "cache_probe": profile.key,
                "expected_cached_tokens": result.expected_cached_tokens,
                "cold": result.cold.__dict__,
                "device_hit": result.device_hit.__dict__,
                "host_reload": result.host_reload.__dict__,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def run_cache_probe(
    base_url: str, profile: ModelProfile, config: CacheProbeConfig
) -> CacheProbeResult:
    _flush_cache(base_url)
    target = make_prompt(0, config.target_words)
    cold = _generate(base_url, target, config.max_new_tokens)
    device_hit = _generate(base_url, target, config.max_new_tokens)

    for index in range(config.num_fillers):
        filler = make_prompt(1000 + index, config.filler_words)
        _generate(base_url, filler, 1)

    host_reload = _generate(base_url, target, config.max_new_tokens)
    expected = validate_cache_probe(
        cold,
        device_hit,
        host_reload,
        config.page_size,
        config.kv_cache_dtype,
    )
    result = CacheProbeResult(cold, device_hit, host_reload, expected)
    _record_cache_metrics(profile, result)
    return result


def make_cache_probe_runner(profile: ModelProfile, config: CacheProbeConfig):
    """Adapt the feature probe to the shared TPU MODEL_CONFIG gate runner."""

    def run(base_url: str) -> Dict[str, Any]:
        result = run_cache_probe(base_url, profile, config)
        return {
            "score": 1.0,
            "_passed": True,
            "device_hit_tokens": result.device_hit.device_cached_tokens,
            "host_reload_tokens": result.host_reload.host_cached_tokens,
        }

    return run


def make_gsm8k_runner(profile: ModelProfile, config: Gsm8kConfig):
    def run(base_url: str) -> Dict[str, Any]:
        from sglang.test.run_eval import run_eval

        _flush_cache(base_url)
        args = SimpleNamespace(
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
            gsm8k_data_path=config.data_path,
        )
        metrics = run_eval(args)
        score = float(metrics["score"])
        floor = config.threshold - config.atol
        passed = score >= floor
        failures = (
            []
            if passed
            else [
                f"gsm8k score {score:.4f} is below floor {floor:.4f} "
                f"(threshold={config.threshold:.4f}, atol={config.atol:.4f})"
            ]
        )
        return {**metrics, "_passed": passed, "_failures": failures}

    return run


def build_model_config(
    profile: ModelProfile, env: Optional[Mapping[str, str]] = None
) -> Dict[str, Any]:
    """Build the declarative two-gate config used by recent TPU e2e tests."""
    values = os.environ if env is None else env
    gsm8k = Gsm8kConfig.from_env(profile, values)
    cache = CacheProbeConfig.from_env(values)
    return {
        "model": profile.served_name,
        "test_name": f"prefix-caching-kv-offload-{profile.display_name}",
        "variant": "single-host-TP8+EP8-Raiden",
        "launch": {
            "recipe": profile.recipe,
            "env": build_launch_env(profile, gsm8k, cache, values),
            "health_timeout": _parse_float(
                "SERVER_HEALTH_TIMEOUT",
                values.get("SERVER_HEALTH_TIMEOUT", str(DEFAULT_HEALTH_TIMEOUT)),
                1.0,
            ),
            "tag": f"prefix-kv-offload-{profile.key}",
        },
        "gates": [
            {
                "dataset": "prefix_caching_kv_offload",
                "label": "cache_hit_and_host_reload",
                "runner": make_cache_probe_runner(profile, cache),
            },
            {
                "dataset": "gsm8k",
                "label": "gsm8k_prefix_caching_kv_offload",
                "runner": make_gsm8k_runner(profile, gsm8k),
            },
        ],
    }


class TestPrefixCachingKvOffloadConfiguration(unittest.TestCase):
    def test_model_selection_is_variable(self) -> None:
        self.assertEqual(
            select_profile_keys("gpt_oss_120b,qwen3_coder_480b"),
            ("gpt_oss_120b", "qwen3_coder_480b"),
        )
        with self.assertRaisesRegex(ValueError, "unknown MODEL_PROFILE"):
            select_profile_keys("unknown")

    def test_gsm8k_threshold_is_configurable(self) -> None:
        profile = MODEL_PROFILES["gpt_oss_120b"]
        config = Gsm8kConfig.from_env(
            profile, {"GSM8K_THRESHOLD": "0.73", "GSM8K_ATOL": "0.02"}
        )
        self.assertEqual(config.threshold, 0.73)
        self.assertEqual(config.atol, 0.02)
        with self.assertRaisesRegex(ValueError, "GSM8K_THRESHOLD"):
            Gsm8kConfig.from_env(profile, {"GSM8K_THRESHOLD": "1.01"})

    def test_model_specific_threshold_wins(self) -> None:
        profile = MODEL_PROFILES["qwen3_coder_480b"]
        config = Gsm8kConfig.from_env(
            profile,
            {
                "GSM8K_THRESHOLD": "0.50",
                "GSM8K_THRESHOLD_QWEN3_CODER_480B": "0.91",
            },
        )
        self.assertEqual(config.threshold, 0.91)

    def test_qwen_uses_measured_default_threshold(self) -> None:
        profile = MODEL_PROFILES["qwen3_coder_480b"]
        self.assertEqual(Gsm8kConfig.from_env(profile, {}).threshold, 0.70)

    def test_qwen_uses_quality_preserving_backends(self) -> None:
        profile = MODEL_PROFILES["qwen3_coder_480b"]
        gsm8k = Gsm8kConfig.from_env(profile, {})
        cache = CacheProbeConfig.from_env({})
        launch_env = build_launch_env(profile, gsm8k, cache, {})
        self.assertIn("--sampling-backend pytorch", launch_env["EXTRA_ARGS"])
        self.assertIn("--moe-runner-backend pallas_tpu", launch_env["EXTRA_ARGS"])

    def test_model_config_shares_one_recipe_across_two_gates(self) -> None:
        config = build_model_config(MODEL_PROFILES["qwen3_coder_480b"], {})
        self.assertEqual(
            config["launch"]["recipe"], "recipe/qwen3_coder_480b/server.sh"
        )
        self.assertEqual(len(config["gates"]), 2)
        self.assertEqual(config["gates"][0]["label"], "cache_hit_and_host_reload")
        self.assertEqual(config["gates"][1]["dataset"], "gsm8k")

    def test_launch_contract_enables_raiden_prefix_and_offload(self) -> None:
        profile = MODEL_PROFILES["qwen3_coder_480b"]
        gsm8k = Gsm8kConfig.from_env(profile, {})
        cache = CacheProbeConfig.from_env({})
        launch_env = build_launch_env(profile, gsm8k, cache, {})

        self.assertEqual(launch_env["ATTENTION_BACKEND"], "pallas_tpu_rpa_v3")
        self.assertEqual(launch_env["DISABLE_RADIX_CACHE"], "0")
        self.assertEqual(launch_env["KV_OFFLOAD"], "1")
        self.assertIn("--hicache-io-backend raiden", launch_env["EXTRA_ARGS"])
        self.assertIn("--hicache-mem-layout page_first_fused", launch_env["EXTRA_ARGS"])
        self.assertIn("--enable-cache-report", launch_env["EXTRA_ARGS"])

    def test_gpt_oss_uses_raiden_compatible_kv_allocation(self) -> None:
        profile = MODEL_PROFILES["gpt_oss_120b"]
        gsm8k = Gsm8kConfig.from_env(profile, {})
        cache = CacheProbeConfig.from_env({})
        launch_env = build_launch_env(profile, gsm8k, cache, {})

        self.assertIn("--disable-hybrid-swa-memory", launch_env["EXTRA_ARGS"])

    def test_cache_probe_validation_accepts_device_then_host_hit(self) -> None:
        cold = GenerationRecord("cold", 1025, 0, 0, 1.0)
        device = GenerationRecord("answer", 1025, 960, 0, 0.1)
        host = GenerationRecord("answer", 1025, 0, 960, 0.2)
        self.assertEqual(validate_cache_probe(cold, device, host, 64), 960)

    def test_cache_probe_validation_rejects_missing_host_reload(self) -> None:
        cold = GenerationRecord("cold", 1025, 0, 0, 1.0)
        device = GenerationRecord("answer", 1025, 960, 0, 0.1)
        missing = GenerationRecord("answer", 1025, 960, 0, 0.1)
        with self.assertRaisesRegex(AssertionError, "not reloaded from host"):
            validate_cache_probe(cold, device, missing, 64)

    def test_fp8_cache_probe_does_not_require_exact_text(self) -> None:
        cold = GenerationRecord("cold", 1025, 0, 0, 1.0)
        device = GenerationRecord("answer one", 1025, 960, 0, 0.1)
        host = GenerationRecord("answer two", 1025, 0, 960, 0.2)
        self.assertEqual(validate_cache_probe(cold, device, host, 64, "fp8_e4m3"), 960)

    def test_bf16_cache_probe_requires_exact_text(self) -> None:
        cold = GenerationRecord("cold", 1025, 0, 0, 1.0)
        device = GenerationRecord("answer one", 1025, 960, 0, 0.1)
        host = GenerationRecord("answer two", 1025, 0, 960, 0.2)
        with self.assertRaisesRegex(AssertionError, "changed greedy output"):
            validate_cache_probe(cold, device, host, 64, "bf16")

    def test_resolve_model_path_precedence(self) -> None:
        profile = MODEL_PROFILES["gpt_oss_120b"]
        self.assertEqual(
            resolve_model_path(
                profile, {"MODEL": "generic", "GPT_OSS_MODEL": "specific"}
            ),
            "specific",
        )

    def test_local_model_path_is_validated_before_launch(self) -> None:
        profile = MODEL_PROFILES["qwen3_coder_480b"]
        missing = "/weights/definitely-missing-qwen3-coder-480b"
        with self.assertRaisesRegex(FileNotFoundError, "QWEN3_CODER_MODEL"):
            validate_model_source(profile, missing)

        # Repository IDs are resolved by Hugging Face and cannot be checked here.
        validate_model_source(profile, "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8")


def _install_integration_cases() -> None:
    """Install one shared-runner TestCase per selected model profile."""
    import __main__

    selected = select_profile_keys(os.environ.get("MODEL_PROFILE", "all"))
    if len(selected) > 1 and os.environ.get("MODEL"):
        raise ValueError(
            "MODEL cannot select two model profiles; set GPT_OSS_MODEL and "
            "QWEN3_CODER_MODEL, or run with one MODEL_PROFILE"
        )
    for profile_key in selected:
        profile = MODEL_PROFILES[profile_key]
        validate_model_source(profile, resolve_model_path(profile))
        case = build_test_case(build_model_config(profile), base_url=BASE_URL)
        setattr(__main__, case.__name__, case)


if __name__ == "__main__":
    _install_integration_cases()
    unittest.main(verbosity=2)
