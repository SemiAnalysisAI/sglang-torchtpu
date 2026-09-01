"""Shared live-server test machinery for the external TPU benchmark matrix
(GPT-OSS-120B / Qwen3-Coder-480B x three recipes x two workloads).

Each registered test file is one (model, recipe, workload) cell: it declares
its launch group (MULTIHOST_LAUNCH + register_tpu_ci, both AST-scanned, so
they stay in the file) and builds its TestCase here. Two files sharing a
(suite, launch_script, weights) triple share one server; the sweep itself is
benchmark/tpu/recipe/external_bench/run_sweep.sh.

WHAT IS ASSERTED, and what is not -- same philosophy as
test_tpu_qwen3_coder_480b_multihost_agentic.py: throughput is *recorded*,
never gated (the dashboard tracks regressions); what is gated is that the run
is trustworthy at all:
  * every planned leg finished,
  * prefix caching actually hit where the workload makes hits reachable
    (multi-turn agentic; a single-turn 8k1k session cannot re-see a prefix,
    so there the rate is recorded but not gated),
  * the server admitted the work without retracting or aborting requests,
  * EAGLE3 actually drafted, on the recipes that claim it (a spec config
    that silently fell back to plain decode would benchmark the wrong thing).
"""

import json
import os
import subprocess
import time
import unittest
import urllib.request

# One source of truth for the reachable prefix-hit formula.
from test_tpu_qwen3_coder_480b_multihost_agentic import _reachable_prefix_hit_rate

from sglang.test.ci.ci_register import register_tpu_ci  # noqa: F401 (re-export)
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, CustomTestCase, dump_metric

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, *[".."] * 4))
SWEEP_SCRIPT = os.path.join(
    REPO_ROOT, "benchmark", "tpu", "recipe", "external_bench", "run_sweep.sh"
)

BASE_URL = os.environ.get("BASE_URL", DEFAULT_URL_FOR_TEST)
HEALTH_TIMEOUT = int(os.environ.get("HEALTH_TIMEOUT", "600"))
# Must clear the ci profile's worst case (8 legs x MAX_LEG_SECONDS=480) and
# stay under the file's timeout_per_file=5400 with room for the summary print.
SWEEP_TIMEOUT = int(os.environ.get("SWEEP_TIMEOUT", "4500"))
PROFILE = os.environ.get(
    "EXTERNAL_BENCH_PROFILE", "e2e"
)  # e2e branch: single point c64-long4
MIN_PREFIX_HIT_FRACTION = float(os.environ.get("MIN_PREFIX_HIT_FRACTION", "0.6"))
MIN_PREFIX_HIT_RATE = os.environ.get("MIN_PREFIX_HIT_RATE")

# The only egress from a TPU JobSet is the streamed pod log; the workflow
# lifts the JSON between these markers into a run artifact + job summary.
RESULT_BEGIN = "=== TPU_BENCH_RESULT_JSON BEGIN ==="
RESULT_END = "=== TPU_BENCH_RESULT_JSON END ==="

# Dashboard series keys: the canonical HF ids, not the gcsfuse mount paths the
# recipes serve from -- restaging the weights must not split a trend in two.
MODEL_LABELS = {
    "gpt_oss_120b": "openai/gpt-oss-120b",
    "qwen3_coder_480b": "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
}
SPEC_RECIPES = {"1host_pc_spec_kvoff", "pd_2p2d_pc_spec_kvoff"}


def _direction(field: str) -> str:
    if "throughput" in field:
        return "higher_is_better"
    if any(s in field for s in ("ttft", "itl", "latency", "duration")):
        return "lower_is_better"
    return "informational"


def _print_workload_digest(summary) -> None:
    """One short line per leg, printed BEFORE the fragile JSON payload so a
    triage never depends on the payload surviving the log stream."""
    for leg in summary.get("legs") or []:
        derived = leg.get("feature_derived") or {}
        metrics = leg.get("metrics") or {}
        print(
            f"[workload] {leg.get('tag')} status={leg.get('status')} "
            f"prefix_cache_hit_rate={derived.get('prefix_cache_hit_rate')} "
            f"kv_offload_written={derived.get('kv_offload_written_tokens')} "
            f"kv_offload_reloaded={derived.get('kv_offload_reloaded_tokens')} "
            f"output_throughput={metrics.get('output_throughput')}",
            flush=True,
        )


def _emit_result_json(summary) -> None:
    """Pretty-printed so no single log line exceeds containerd's 16 KiB CRI
    record cap (the workflow's awk rejoins the block; pretty JSON only breaks
    between tokens, so the rejoined text parses identically)."""
    try:
        blob = json.dumps(summary, sort_keys=True, default=str, indent=1)
    except (TypeError, ValueError) as exc:
        print(f"{RESULT_BEGIN}\n{RESULT_END}", flush=True)
        print(f"[ext-bench] summary not serializable, payload skipped: {exc!r}")
        return
    print(RESULT_BEGIN, flush=True)
    print(blob, flush=True)
    print(RESULT_END, flush=True)


def _spec_metrics_url(bench_recipe: str, base_url: str) -> str:
    """Where the spec gauges live: the engine that decodes.

    For P/D that is the decode leader (hosts[2] of the slice), never the
    mini-LB -- the LB serves no /metrics, and its /generate path hangs against
    this raiden P/D setup (prefill computes, decode never pulls; observed on a
    fresh server, iteration 7), so a request-level probe through it cannot be
    the oracle. For 1-host recipes the one server is the decode.
    """
    if bench_recipe.startswith("pd_"):
        hosts = [h for h in os.environ.get("TPU_WORKER_HOSTNAMES", "").split(",") if h]
        if len(hosts) >= 3:
            port = os.environ.get("PD_DECODE_PORT", "30002")
            return f"http://{hosts[2]}:{port}"
    return base_url


def _read_spec_accept_length(metrics_url: str) -> float:
    """Max sglang:spec_accept_length across ranks; 0.0 when spec never ran."""
    with urllib.request.urlopen(metrics_url.rstrip("/") + "/metrics", timeout=60) as r:
        text = r.read().decode()
    values = [
        float(line.rsplit(" ", 1)[1])
        for line in text.splitlines()
        if line.startswith("sglang:spec_accept_length") and " " in line
    ]
    return max(values, default=0.0)


def build_matrix_test(*, bench_model: str, bench_recipe: str, workload: str):
    """Return the TestCase class for one (model, recipe, workload) cell."""
    model_label = MODEL_LABELS[bench_model]
    cell = f"{bench_model}/{bench_recipe}/{workload}"
    spec_expected = bench_recipe in SPEC_RECIPES
    # A single-turn 8k1k session never re-sees its prefix, so the hit-rate
    # gate only applies to the multi-turn agentic workload.
    gate_prefix_hits = workload == "agentic"

    @unittest.skipUnless(
        os.environ.get("SGLANG_TPU_USE_LIVE_SERVER") == "1",
        "live server not provided (the external JobSet launches it)",
    )
    class ExternalBenchTest(CustomTestCase):
        base_url = BASE_URL
        summary = None

        @classmethod
        def setUpClass(cls):
            cls._wait_healthy()
            cls.summary = cls._run_sweep()

        @classmethod
        def _wait_healthy(cls):
            start = time.perf_counter()
            last_note = start
            deadline = start + HEALTH_TIMEOUT
            while time.perf_counter() < deadline:
                try:
                    import requests

                    if (
                        requests.get(cls.base_url + "/health", timeout=10).status_code
                        == 200
                    ):
                        print(
                            f"[ext-bench] {cell}: server healthy after "
                            f"{time.perf_counter() - start:.0f}s",
                            flush=True,
                        )
                        return
                except Exception:
                    pass
                if time.perf_counter() - last_note >= 60:
                    last_note = time.perf_counter()
                    print(
                        f"[ext-bench] waiting for live server at {cls.base_url} "
                        f"({time.perf_counter() - start:.0f}s elapsed)",
                        flush=True,
                    )
                time.sleep(5)
            raise TimeoutError(
                f"{cls.base_url}/health did not return 200 within {HEALTH_TIMEOUT}s"
            )

        @classmethod
        def _run_sweep(cls):
            run_root = os.environ.get("RUN_ROOT") or (
                f"/tmp/ext-bench-{bench_model}-{bench_recipe}-{workload}"
            )
            env = {
                **os.environ,
                "BENCH_MODEL": bench_model,
                "BENCH_RECIPE": bench_recipe,
                "WORKLOAD": workload,
                "PROFILE": PROFILE,
                "BASE_URL": cls.base_url,
                "RUN_ROOT": run_root,
                "HEALTH_TIMEOUT": str(HEALTH_TIMEOUT),
            }
            print(
                f"[ext-bench] {cell}: running the {PROFILE} sweep against "
                f"{cls.base_url} (deadline {SWEEP_TIMEOUT}s)",
                flush=True,
            )
            proc = subprocess.run(
                ["bash", SWEEP_SCRIPT],
                cwd=os.path.dirname(SWEEP_SCRIPT),
                env=env,
                timeout=SWEEP_TIMEOUT,
            )
            summary_path = os.path.join(run_root, "summary.json")
            if not os.path.exists(summary_path):
                raise AssertionError(
                    f"the sweep produced no summary.json (run_sweep.sh rc="
                    f"{proc.returncode}); see the log above"
                )
            with open(summary_path) as f:
                summary = json.load(f)
            _print_workload_digest(summary)
            _emit_result_json(summary)
            return summary

        def _legs(self):
            return [
                leg
                for leg in self.summary["legs"]
                if not leg["status"].startswith("skipped")
            ]

        def test_every_leg_completed(self):
            """A timed-out or crashed leg fails the run: partial benchmark
            output is indistinguishable from a benchmark nobody looked at."""
            legs = self._legs()
            self.assertGreater(len(legs), 0, "the sweep attempted no legs")
            bad = [(leg["tag"], leg["status"]) for leg in legs if leg["status"] != "ok"]
            self.assertEqual(bad, [], f"legs did not complete: {bad}")
            missing = [leg["tag"] for leg in legs if not leg.get("metrics")]
            self.assertEqual(missing, [], f"legs produced no metrics: {missing}")

        def test_prefix_cache_hit_rate_reported(self):
            """Every leg must RECORD a hit rate (the metrics endpoint is part
            of the contract); the agentic workload is additionally gated
            against what its round count makes reachable."""
            for leg in self._legs():
                rate = (leg.get("feature_derived") or {}).get("prefix_cache_hit_rate")
                self.assertIsNotNone(
                    rate,
                    f"{leg['tag']}: no prefix-cache hit rate recorded -- was the "
                    "server launched with --enable-metrics?",
                )
                if not gate_prefix_hits:
                    continue
                if MIN_PREFIX_HIT_RATE is not None:
                    floor, how = float(MIN_PREFIX_HIT_RATE), "MIN_PREFIX_HIT_RATE"
                else:
                    # These recipes run dp=1, so the reachable rate is the
                    # single-cache ideal; still read dp back off the run so a
                    # future DP recipe cannot silently inherit a wrong floor.
                    dp_size = int((leg.get("server_info") or {}).get("dp_size") or 1)
                    reachable = _reachable_prefix_hit_rate(
                        config=leg.get("config") or {}, dp_size=dp_size
                    )
                    self.assertIsNotNone(
                        reachable,
                        f"{leg['tag']}: leg meta lacks the round counts needed "
                        "to derive a prefix-hit floor; set MIN_PREFIX_HIT_RATE",
                    )
                    self.assertGreater(
                        reachable,
                        0.0,
                        f"{leg['tag']}: no round can reach a cached prefix at "
                        f"dp={dp_size}; the leg measures nothing as it is.",
                    )
                    floor = MIN_PREFIX_HIT_FRACTION * reachable
                    how = (
                        f"{MIN_PREFIX_HIT_FRACTION:.0%} of the {reachable:.1%} "
                        f"this leg's round counts reach at dp={dp_size}"
                    )
                self.assertGreaterEqual(
                    rate,
                    floor,
                    f"{leg['tag']}: prefix-cache hit rate {rate:.1%} is below "
                    f"{floor:.1%} ({how})",
                )

        def test_server_admitted_the_work(self):
            """Retractions and aborts mean the KV pool could not hold the
            admitted batch, so the leg's numbers include recompute."""
            for leg in self._legs():
                counters = leg.get("feature_counters") or {}
                for key in (
                    "sglang:num_retracted_requests_total",
                    "sglang:num_aborted_requests_total",
                ):
                    self.assertEqual(
                        counters.get(key, 0.0),
                        0.0,
                        f"{leg['tag']}: {key} = {counters.get(key)} "
                        f"(warnings: {leg.get('warnings')})",
                    )

        if spec_expected:

            def test_spec_decoding_active(self):
                """The recipe claims EAGLE3; prove drafting actually ran (a
                silent fallback to plain decode would still pass every other
                gate while benchmarking the wrong configuration). Oracle: the
                decoding engine's own sglang:spec_accept_length gauge after
                the sweep -- it stays 0.0 unless spec verify actually ran, and
                needs no request through the mini-LB."""
                url = _spec_metrics_url(bench_recipe, self.base_url)
                accept_length = _read_spec_accept_length(url)
                print(
                    f"[ext-bench] spec_accept_length={accept_length} ({url})",
                    flush=True,
                )
                self.assertGreater(
                    accept_length,
                    0.0,
                    f"sglang:spec_accept_length is 0 on {url}: speculative "
                    "decoding never ran during the sweep",
                )

        def test_throughput_recorded(self):
            """Not a performance gate -- proves the numbers the dashboard
            ingests are present and non-zero, then dumps them for it."""
            for leg in self._legs():
                metrics = leg.get("metrics") or {}
                derived = leg.get("derived") or {}
                self.assertGreater(
                    metrics.get("output_throughput") or 0.0,
                    0.0,
                    f"{leg['tag']}: no output throughput",
                )
                print(
                    f"[ext-bench] {leg['tag']}: "
                    f"out {metrics.get('output_throughput'):.1f} tok/s, "
                    f"prompt {derived.get('prompt_throughput', 0.0):.1f} tok/s, "
                    f"TTFT p50 {metrics.get('median_ttft_ms'):.0f} ms, "
                    f"ITL p50 {metrics.get('median_itl_ms'):.1f} ms",
                    flush=True,
                )
                counters = leg.get("feature_counters") or {}
                clean = leg["status"] == "ok" and not any(
                    counters.get(k)
                    for k in (
                        "sglang:num_retracted_requests_total",
                        "sglang:num_aborted_requests_total",
                    )
                )
                if clean:
                    self._dump_leg_metrics(leg)

        def _dump_leg_metrics(self, leg):
            """One [METRIC] line per number. The leg tag + workload + recipe
            are in the metric name because several cells share a model, and
            the dashboard keys a night's point on (test_file, metric, model)."""
            tag = f"{workload}.{leg['tag']}"
            numbers = {
                **(leg.get("metrics") or {}),
                **{f"derived.{k}": v for k, v in (leg.get("derived") or {}).items()},
                **{
                    f"feature.{k}": v
                    for k, v in (leg.get("feature_derived") or {}).items()
                },
            }
            for field, value in sorted(numbers.items()):
                if not isinstance(value, (int, float)):
                    continue
                labels = {"model": model_label, "direction": _direction(field)}
                if field == "output_throughput":
                    labels["workload"] = tag
                    labels["headline"] = "1"
                else:
                    labels["kind"] = "bench_detail"
                    labels["workload_name"] = tag
                dump_metric(f"{tag}.{field}", value, labels=labels)

    ExternalBenchTest.__qualname__ = ExternalBenchTest.__name__ = (
        f"TestExtBench_{bench_model}_{bench_recipe}_{workload}"
    )
    return ExternalBenchTest
