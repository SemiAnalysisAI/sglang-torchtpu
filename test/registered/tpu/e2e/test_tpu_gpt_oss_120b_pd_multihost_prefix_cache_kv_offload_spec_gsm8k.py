"""GPT-OSS-120B P/D multihost + prefix cache + raiden KV offload + EAGLE3 gate.

The top of the feature matrix: disaggregation, host offload and speculative
decoding all on at once. Each has a lane of its own and each pair has one, and
this exists because the three-way interaction is not the sum of the pairs. With
HiCache on, a speculative boot builds an extra host pool for the draft KV, so
every rank announces three raiden pools instead of two -- full attention,
sliding window, draft sidecar -- and a prefix reloaded from host has to line up
with a draft pool mirrored alongside it before it is handed across the P/D
transport to an engine that will run verify forwards against it.

Topology is the shared 2 prefill hosts + 2 decode hosts at TP16 per engine.
HiCache is on the prefill engine only, where prefix matching happens, and the
draft model is on both, since the prefill engine has to lay out the draft KV
pool for the pages it hands over.

Two gates in order. The offload probe runs first, on a cache it flushes and
fully owns, and it checks both halves at once: which tier served each read, and
that the draft worker still ran on the read whose prefix came back over host.
That last check is the reason this lane exists -- a server that quietly dropped
to plain decoding would pass every cache assertion and still score a normal
GSM8K. GSM8K then scores the same boot with all three features still on.
"""

import argparse
import importlib.util
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, *[".."] * 4))

SGLANG_IMPORT_ERROR = None
try:
    from sglang.test.ci.ci_register import register_tpu_ci
    from sglang.test.tpu.kv_offload_probe import kv_offload_runner
except ModuleNotFoundError as exc:
    SGLANG_IMPORT_ERROR = exc

    def register_tpu_ci(*_args, **_kwargs):
        return None

    def kv_offload_runner(**_kwargs):
        return None


def _load_pd_harness():
    """Import the shared P/D harness, by path when sglang is not installed.

    The harness has no runtime dependencies on purpose, so the configuration
    tests below run on a bare checkout. Importing it through the package pulls
    in sglang's __init__, which does have them.
    """
    try:
        from sglang.test.tpu import pd_multihost

        return pd_multihost
    except ModuleNotFoundError:
        path = os.path.join(REPO, "python", "sglang", "test", "tpu", "pd_multihost.py")
        spec = importlib.util.spec_from_file_location("tpu_pd_multihost", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


pd = _load_pd_harness()

# The offload lane's 8100 plus loading and precompiling the draft model, plus
# the one recompile SGLANG_TPU_ALLOW_SPEC_RECOMPILE buys back on the first
# unseen speculative shape. PROBE_TOKENS lands squarely on such a bucket.
register_tpu_ci(est_time=10200, suite="tpu-integration", nightly=True)

E = os.environ.get

# Short on purpose, and shorter than it reads: this is the only variable part of
# the pod names JobSet generates, which k8s caps at 63 characters. The
# descriptive name below overflows that and gets the whole manifest rejected at
# submission. test_run_name_fits_the_generated_pod_name pins the budget.
RUN_NAME = "gpt-oss-pd-multihost-kvoff-spec"
TEST_NAME = "gpt-oss-120b-pd-multihost-prefix-cache-kv-offload-spec-gsm8k"

# Sized for the probe, not for throughput: it can only prove offloading happened
# if a handful of filler prefills evict it, which needs a device pool it can
# overflow. plan_eviction re-checks that this pool and HICACHE_RATIO leave room
# for the probe plus every filler, and the draft KV pool is carved out of what
# the same mem_fraction leaves behind.
MAX_TOTAL_TOKENS = pd.DEVICE_POOL_TOKENS

# Draft tokens accepted per verify forward, as a strict floor over the probe
# reads. A server that accepts nothing sits at exactly 1.0, so anything above it
# means drafts are landing. Kept at the floor rather than near a measured value
# because the probe reads are short -- they prove the draft worker is alive on a
# host-reloaded prefix, and GSM8K below is what runs speculation at length.
MIN_ACCEPT_LENGTH = 1.0

# Same floor as the two lanes this merges. Neither offloading nor speculation is
# supposed to change what the model computes -- one moves where KV lives, the
# other only changes how many tokens a step proposes -- so a drop here is a
# corrupted host round trip or a draft/target mismatch, not a tuning miss.
GSM8K_BASELINE = float(E("GSM8K_THRESHOLD", "0.70"))
ATOL = float(E("GSM8K_ATOL", "0.03"))
NUM_EXAMPLES = int(E("GSM8K_NUM_EXAMPLES", "100"))
# 8 in flight, matching the pool: more would spend it on retraction instead,
# and the draft tokens each step holds come out of the same budget.
NUM_THREADS = int(E("GSM8K_NUM_THREADS", "8"))
MAX_TOKENS = int(E("GSM8K_MAX_TOKENS", "512"))

LANE = pd.LaneConfig(
    test_name=TEST_NAME,
    variant="PD-2host-TP16+EP16-raiden+HiCache-raiden+EAGLE3",
    tag="gpt-oss-120b-pd-multihost-prefix-cache-kv-offload-spec-e2e",
    run_name=RUN_NAME,
    max_total_tokens=MAX_TOTAL_TOKENS,
    hicache_ratio=pd.HICACHE_RATIO,
    spec=True,
)


def build_model_config(*, decode_host: str) -> dict:
    """Assemble the gate list. Split out so the tests can read it back."""
    return {
        "model": pd.SERVED_MODEL_NAME,
        "test_name": LANE.test_name,
        "variant": LANE.variant,
        "launch": {
            "external": True,
            "health_timeout": pd.HEALTH_TIMEOUT,
            "tag": LANE.tag,
        },
        # No dashboard key: feature validation on a topology no accuracy lane
        # serves, so plotting its score beside one would compare two servers.
        "gates": [
            {
                "dataset": "pd_transport",
                "label": "pd-transport",
                "runner": pd.pd_transport_runner(
                    lane=LANE,
                    prefill_url=f"http://127.0.0.1:{pd.PREFILL_PORT}",
                    decode_url=f"http://{decode_host}:{pd.DECODE_PORT}",
                ),
            },
            {
                "dataset": "prefix_cache_kv_offload",
                "label": "prefix-cache-kv-offload-spec",
                "runner": kv_offload_runner(
                    test_name=LANE.test_name,
                    device_pool_tokens=pd.DEVICE_POOL_TOKENS,
                    page_size=pd.PAGE_SIZE,
                    probe_tokens=pd.PROBE_TOKENS,
                    hicache_ratio=pd.HICACHE_RATIO,
                    # Speculation is checked on the same three reads, so this
                    # gate fails if the draft worker went missing on the
                    # host-reloaded prefix, which is the interaction the whole
                    # lane exists for and costs no extra requests to see.
                    min_accept_length=MIN_ACCEPT_LENGTH,
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


class TestPdMultihostPrefixCacheKvOffloadSpecConfiguration(unittest.TestCase):
    """Host-side configuration tests runnable without TPU hardware."""

    def test_all_three_features_are_on_together(self) -> None:
        # The point of the lane. If any one of them silently fell out of the
        # launch, the remaining gates would still pass and report nothing.
        prefill_args = pd.build_prefill_env(lane=LANE, env={})["EXTRA_ARGS"]
        self.assertIn("--disaggregation-transfer-backend raiden", prefill_args)
        self.assertIn("--enable-hierarchical-cache", prefill_args)
        self.assertIn("--speculative-algorithm EAGLE3", prefill_args)

    def test_hicache_is_on_prefill_only(self) -> None:
        # Prefill owns every prefix match in P/D, so that is the only engine
        # where offloading means anything. Turning it on for decode as well
        # would tier a cache nothing reads and confuse the probe's tier counts.
        self.assertIn(
            "--enable-hierarchical-cache",
            pd.build_prefill_env(lane=LANE, env={})["EXTRA_ARGS"],
        )
        self.assertNotIn(
            "--enable-hierarchical-cache",
            pd.build_decode_env(lane=LANE, env={})["EXTRA_ARGS"],
        )

    def test_draft_model_is_on_both_engines(self) -> None:
        # Prefill needs the spec flags as much as decode. It lays out the draft
        # KV pool for the pages it hands over, and with HiCache on that pool is
        # mirrored to host alongside the two attention pools.
        for build in (pd.build_prefill_env, pd.build_decode_env):
            args = build(lane=LANE, env={})["EXTRA_ARGS"]
            self.assertIn("--speculative-algorithm EAGLE3", args)
            self.assertIn(f"--speculative-draft-model-path {pd.DRAFT_MODEL_PATH}", args)

    def test_both_engines_offload_over_raiden(self) -> None:
        # raiden gives every sub-pool its own host pool. torch_tpu mirrors only
        # full attention, which server_args rejects outright for a hybrid-SWA
        # model rather than half-serving, on either engine.
        for build in (pd.build_prefill_env, pd.build_decode_env):
            args = build(lane=LANE, env={})["EXTRA_ARGS"]
            self.assertIn("--hicache-io-backend raiden", args)
            self.assertIn("--hicache-mem-layout page_first_fused", args)
            self.assertNotIn("torch_tpu", args)

    def test_spec_recompile_escape_hatch_is_on(self) -> None:
        # PROBE_TOKENS lands on a bucket the TPU distributed top-k path is not
        # precompiled for, so without this the first probe read trips
        # fail_on_recompile and takes every rank's scheduler down.
        for build in (pd.build_prefill_env, pd.build_decode_env):
            self.assertEqual(
                build(lane=LANE, env={})["SGLANG_TPU_ALLOW_SPEC_RECOMPILE"], "1"
            )

    def test_device_pool_is_small_enough_to_evict(self) -> None:
        # The offload half of the gate rests on the probe being pushed off
        # device by the fillers. Pinned against the pool the flags actually
        # request, so a later bump for throughput cannot quietly make the probe
        # unevictable and leave the lane passing on nothing.
        for build in (pd.build_prefill_env, pd.build_decode_env):
            self.assertIn(
                f"--max-total-tokens {pd.DEVICE_POOL_TOKENS}",
                build(lane=LANE, env={})["EXTRA_ARGS"],
            )

    def test_gate_list_builds_without_hardware(self) -> None:
        # The gate list is otherwise only ever assembled inside the pod, so a
        # harness constant renamed out from under it surfaces as an
        # AttributeError once all four engines have finished booting -- forty
        # minutes in, on a lane that is red for a reason nothing host-side
        # could see. Building it here costs a second. Off-pod the offload
        # runner is the stub above, so what this pins there is the constants
        # feeding it, not the runner itself.
        config = build_model_config(decode_host="decode-0")
        # Transport first on purpose, so a red run says whether the pair was
        # wired wrong or the answers were wrong.
        self.assertEqual(
            [gate["label"] for gate in config["gates"]],
            ["pd-transport", "prefix-cache-kv-offload-spec", "gsm8k"],
        )
        self.assertTrue(callable(config["gates"][0]["runner"]))

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
        os.path.exists(pd.RENDER_JOBSET), "renderer is not shipped in the runtime image"
    )
    def test_dry_run_render_jobset_yaml(self) -> None:
        # The nightly job passes this run_name and this exec command, so a
        # rename on either side has to land on both.
        yaml_content = pd.render_jobset_yaml(lane=LANE, script_path=__file__)
        self.assertIn(f"generateName: tpu-ci-{RUN_NAME}-", yaml_content)
        self.assertIn("name: prefill", yaml_content)
        self.assertIn("name: decode", yaml_content)
        self.assertNotIn("name: router", yaml_content)
        self.assertIn("CI_PD_ROLE", yaml_content)
        self.assertIn(
            "python3 test/registered/tpu/e2e/"
            "test_tpu_gpt_oss_120b_pd_multihost_prefix_cache_kv_offload_spec_gsm8k.py"
            " integration",
            yaml_content,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P/D multihost KV offload spec test")
    parser.add_argument(
        "mode",
        nargs="?",
        default="unittest",
        choices=["unittest", "integration", "integration_dry_run"],
        help="Execution mode (default: unittest)",
    )
    args, extra_args = parser.parse_known_args()

    if args.mode == "integration_dry_run":
        print(pd.render_jobset_yaml(lane=LANE, script_path=__file__))
        sys.exit(0)
    elif args.mode == "integration":
        sys.exit(pd.serve_role(lane=LANE, model_config_fn=build_model_config))
    else:
        sys.argv = [
            sys.argv[0],
            "TestPdMultihostPrefixCacheKvOffloadSpecConfiguration",
            *extra_args,
        ]
        unittest.main(verbosity=2)
