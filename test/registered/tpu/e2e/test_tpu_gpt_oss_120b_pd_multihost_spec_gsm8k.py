"""GPT-OSS-120B P/D multihost + EAGLE3 speculative decoding GSM8K gate.

The plain P/D multihost lane proves the raiden transport carries KV correctly.
This one adds a draft model on top, because their interaction is not covered by
either the P/D lane or the single-host spec lane. Speculation changes what the
decode engine does with a prefix it did not compute: every verify forward reads
KV that arrived over the transport, and the draft model keeps a KV pool of its
own that has to be laid out on both engines so the handover lines up.

Topology is the shared 2 prefill hosts + 2 decode hosts at TP16 per engine, so
everything except the draft model matches the lane this is derived from.

Two gates in order. The spec gate proves the draft worker actually ran through
the P/D pair and that its tokens land -- a decode engine that silently fell back
to plain decoding would still score a perfectly normal GSM8K, so the accuracy
number alone cannot see it. GSM8K then scores the same boot with speculation on,
which is where a draft/target mismatch shows up as a wrong answer rather than a
slow one.
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
except ModuleNotFoundError as exc:
    SGLANG_IMPORT_ERROR = exc

    def register_tpu_ci(*_args, **_kwargs):
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

# The plain P/D lane's 7200, plus loading and precompiling the draft model, plus
# the one recompile SGLANG_TPU_ALLOW_SPEC_RECOMPILE buys back on the first
# unseen speculative shape.
register_tpu_ci(est_time=9000, suite="tpu-integration", nightly=True)

E = os.environ.get

# Short on purpose, and shorter than it reads: this is the only variable part of
# the pod names JobSet generates, which k8s caps at 63 characters. The
# descriptive name below overflows that and gets the whole manifest rejected at
# submission. test_run_name_fits_the_generated_pod_name pins the budget.
RUN_NAME = "gpt-oss-pd-multihost-spec"
TEST_NAME = "gpt-oss-120b-pd-multihost-spec-gsm8k"

# Half the plain lane's 786432, per host. The draft model needs weights and a KV
# pool of its own on every rank, and the decode engine also keeps a retraction
# host pool mirroring whatever is left, so the device pool has to give that room
# back. Scaled by the host count because the pool is per engine while the HBM
# behind it is per host: the PD_HOSTS_PER_ENGINE=1 re-run has half the memory and
# a fixed literal would OOM it at precompile.
MAX_TOTAL_TOKENS = 196608 * pd.HOSTS_PER_ENGINE

# Draft tokens accepted per verify forward, as a strict floor. A server that
# accepts nothing sits at exactly 1.0, so anything above it means drafts land.
# Kept at the floor rather than near a measured value because the point of this
# gate is liveness of the draft worker across the transport, not its tuning.
MIN_ACCEPT_LENGTH = 1.0

# Same floor as the plain P/D lane. Speculation is not supposed to change what
# the model computes, only how many tokens a step proposes, so a drop here is a
# draft/target mismatch rather than a tuning miss.
GSM8K_BASELINE = float(E("GSM8K_THRESHOLD_GPT_OSS_120B", E("GSM8K_THRESHOLD", "0.72")))
ATOL = float(E("GSM8K_ATOL", "0.03"))
NUM_EXAMPLES = int(E("GSM8K_NUM_EXAMPLES", "100"))
NUM_THREADS = int(E("GSM8K_NUM_THREADS", "8"))
MAX_TOKENS = int(E("GSM8K_MAX_TOKENS", "1024"))

LANE = pd.LaneConfig(
    test_name=TEST_NAME,
    variant="PD-2host-TP16+EP16-raiden+EAGLE3",
    tag="gpt-oss-120b-pd-multihost-spec-e2e",
    run_name=RUN_NAME,
    max_total_tokens=MAX_TOTAL_TOKENS,
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
                "dataset": "spec_decode",
                "label": "spec-decode",
                "runner": pd.spec_runner(
                    test_name=LANE.test_name,
                    min_accept_length=MIN_ACCEPT_LENGTH,
                ),
            },
            {
                "dataset": "termination",
                "label": "termination",
                "runner": pd.termination_runner(
                    test_name=LANE.test_name,
                    max_tokens=MAX_TOKENS,
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


class TestPdMultihostSpecConfiguration(unittest.TestCase):
    """Host-side configuration tests runnable without TPU hardware."""

    def test_both_engines_carry_the_draft_model(self) -> None:
        # Prefill needs the spec flags as much as decode. It lays out the draft
        # KV pool for the pages it hands over, and an engine pair that disagrees
        # on whether a draft pool exists cannot pair its transfers up.
        for build in (pd.build_prefill_env, pd.build_decode_env):
            args = build(lane=LANE, env={})["EXTRA_ARGS"]
            self.assertIn("--speculative-algorithm EAGLE3", args)
            self.assertIn(f"--speculative-draft-model-path {pd.DRAFT_MODEL_PATH}", args)
            self.assertIn("--speculative-num-steps 3", args)
            self.assertIn("--speculative-eagle-topk 1", args)
            self.assertIn("--speculative-num-draft-tokens 4", args)

    def test_spec_recompile_escape_hatch_is_on(self) -> None:
        # Without it the first request on a token bucket the distributed top-k
        # path was not precompiled for trips fail_on_recompile and takes every
        # rank's scheduler down, leaving the gates scoring against a dead server.
        for build in (pd.build_prefill_env, pd.build_decode_env):
            self.assertEqual(
                build(lane=LANE, env={})["SGLANG_TPU_ALLOW_SPEC_RECOMPILE"], "1"
            )

    def test_no_hierarchical_cache_on_this_lane(self) -> None:
        # Offloading is the next lane over. Prefill must stay clean of hicache
        # flags so a red run here is speculation and not cache tiering.
        self.assertNotIn(
            "hierarchical-cache", pd.build_prefill_env(lane=LANE, env={})["EXTRA_ARGS"]
        )
        self.assertNotIn(
            "hicache", pd.build_prefill_env(lane=LANE, env={})["EXTRA_ARGS"]
        )

    def test_spec_lane_leaves_the_sampler_at_the_tpu_default(self) -> None:
        # Pinning pytorch under EAGLE3 is the one flag that separates this lane
        # from the green single-host spec gate, and it is also self-defeating:
        # the harness turns on SGLANG_TPU_ALLOW_SPEC_RECOMPILE for the TPU
        # distributed top-k path, which LogitsProcessor only takes when the
        # backend is tpu. Re-adding the pin here would silently undo both.
        for env_fn in (pd.build_prefill_env, pd.build_decode_env):
            self.assertNotIn(
                "--sampling-backend", env_fn(lane=LANE, env={})["EXTRA_ARGS"]
            )

    def test_non_spec_lane_keeps_pytorch_and_the_override_wins(self) -> None:
        # pytorch is proven on the non-spec P/D lanes, so dropping the pin has
        # to stay scoped to speculation. The override is the A/B handle that
        # makes the backend testable without a rebuild, so it has to beat both.
        plain = LANE._replace(spec=False)
        self.assertEqual(
            pd.sampling_backend_args(lane=plain), ["--sampling-backend", "pytorch"]
        )
        os.environ["PD_SAMPLING_BACKEND"] = "pytorch"
        try:
            self.assertEqual(
                pd.sampling_backend_args(lane=LANE),
                ["--sampling-backend", "pytorch"],
            )
        finally:
            del os.environ["PD_SAMPLING_BACKEND"]

    def test_decode_retraction_host_pool_stays_on_raiden(self) -> None:
        # server_args rejects torch_tpu for a hybrid-SWA model, so this is the
        # one io backend that lets the decode engine finish booting.
        args = pd.build_decode_env(lane=LANE, env={})["EXTRA_ARGS"]
        self.assertIn("--hicache-mem-layout page_first_fused", args)
        self.assertIn("--hicache-io-backend raiden", args)
        self.assertNotIn("torch_tpu", args)

    def test_both_engines_agree_on_transport_and_pool(self) -> None:
        # TP is checked against the pod count the JobSet is rendered with, not
        # against a literal. A TP that outruns the hosts hangs the rendezvous
        # until the deadline with no error, and the two have to stay tied when
        # PD_HOSTS_PER_ENGINE re-runs this lane at one host per engine.
        expected_tp = str(pd.CHIPLETS_PER_HOST * pd.HOSTS_PER_ENGINE)
        for build in (pd.build_prefill_env, pd.build_decode_env):
            env = build(lane=LANE, env={})
            self.assertIn("--disaggregation-transfer-backend raiden", env["EXTRA_ARGS"])
            self.assertIn(f"--max-total-tokens {MAX_TOTAL_TOKENS}", env["EXTRA_ARGS"])
            self.assertEqual(env["TP_SIZE"], expected_tp)
            self.assertEqual(env["EP_SIZE"], expected_tp)
            self.assertEqual(env["ATTENTION_BACKEND"], "pallas_tpu_rpa_v3")

    def test_gate_list_builds_without_hardware(self) -> None:
        # The gate list is otherwise only ever assembled inside the pod, so a
        # harness constant renamed out from under it surfaces as an
        # AttributeError once both engines have finished booting -- forty
        # minutes in, on a lane that is red for a reason nothing host-side
        # could see. Building it here costs a second.
        config = build_model_config(decode_host="decode-0")
        # Transport first on purpose, so a red run says whether the pair was
        # wired wrong or the answers were wrong. Termination sits ahead of
        # GSM8K for the same reason one step further in: two requests say
        # whether the responses even finish, which is the difference between a
        # wrong answer and no answer, and GSM8K's score cannot tell them apart.
        self.assertEqual(
            [gate["label"] for gate in config["gates"]],
            ["pd-transport", "spec-decode", "termination", "gsm8k"],
        )
        self.assertTrue(callable(config["gates"][0]["runner"]))
        self.assertTrue(callable(config["gates"][1]["runner"]))
        self.assertTrue(callable(config["gates"][2]["runner"]))

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
            "test_tpu_gpt_oss_120b_pd_multihost_spec_gsm8k.py integration",
            yaml_content,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P/D multihost spec test")
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
        sys.argv = [sys.argv[0], "TestPdMultihostSpecConfiguration", *extra_args]
        unittest.main(verbosity=2)
