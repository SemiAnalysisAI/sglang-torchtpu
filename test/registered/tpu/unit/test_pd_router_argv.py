"""Unit tests for the TPU P/D router argv builder.

Host-only: :mod:`sglang.test.tpu.pd_router` imports nothing but the standard
library, so these run anywhere and need neither a TPU nor the router wheel.

The load-bearing test is :meth:`TestMiniLbArgv.test_matches_legacy_argv`. Every
TPU P/D test used to build this argv inline; if the builder ever drifts from
that literal, the mini-LB baseline stops being a baseline and the A/B against
the SGLang Model Gateway becomes meaningless.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=30, suite="stage-a-test-tpu-unit")

import unittest

from sglang.test.test_utils import CustomTestCase
from sglang.test.tpu.pd_router import (
    DEFAULT_DECODE_POLICY,
    DEFAULT_PREFILL_POLICY,
    MINI_LB,
    SMG,
    build_router_cmd,
    resolve_mode,
    router_metrics_url,
)

PREFILL_URL = "http://prefill-0:30000"
DECODE_URL = "http://decode-0:30000"
BOOTSTRAP_PORT = 8998
LB_PORT = 8000


def _cmd(**overrides):
    kwargs = dict(
        python="python3",
        prefill_url=PREFILL_URL,
        decode_url=DECODE_URL,
        host="0.0.0.0",
        port=LB_PORT,
        # NOT FOR MERGE: this validation branch defaults resolve_mode to SMG, so
        # the mini-LB cases name their mode. On the PR branch this is the default.
        mode=MINI_LB,
        env={},
    )
    kwargs.update(overrides)
    return build_router_cmd(**kwargs)


class TestMiniLbArgv(CustomTestCase):
    def test_matches_legacy_argv(self):
        """Byte-for-byte the argv the P/D tests built inline before the helper."""
        self.assertEqual(
            _cmd(),
            (
                "python3",
                "-m",
                "sglang_router.launch_router",
                "--pd-disaggregation",
                "--mini-lb",
                "--prefill",
                PREFILL_URL,
                "--decode",
                DECODE_URL,
                "--host",
                "0.0.0.0",
                "--port",
                str(LB_PORT),
            ),
        )

    def test_bootstrap_port_is_positional_after_the_url(self):
        """``--prefill URL PORT``; the router parses the port positionally."""
        cmd = _cmd(bootstrap_port=BOOTSTRAP_PORT)
        i = cmd.index("--prefill")
        self.assertEqual(cmd[i + 1 : i + 3], (PREFILL_URL, str(BOOTSTRAP_PORT)))

    def test_omits_policy_and_prometheus(self):
        """mini-LB forces ``random`` and serves no metrics, so passing either lies."""
        cmd = _cmd(bootstrap_port=BOOTSTRAP_PORT, prefill_policy="cache_aware")
        for flag in ("--prefill-policy", "--decode-policy", "--prometheus-port"):
            self.assertNotIn(flag, cmd)


class TestSmgArgv(CustomTestCase):
    def test_drops_mini_lb_and_sets_both_policies(self):
        cmd = _cmd(mode=SMG, bootstrap_port=BOOTSTRAP_PORT)
        self.assertNotIn("--mini-lb", cmd)
        self.assertIn("--pd-disaggregation", cmd)
        self.assertEqual(cmd[cmd.index("--prefill-policy") + 1], DEFAULT_PREFILL_POLICY)
        self.assertEqual(cmd[cmd.index("--decode-policy") + 1], DEFAULT_DECODE_POLICY)

    def test_defaults_are_affinity_on_prefill_only(self):
        """cache_aware belongs on prefill; decode runs with its radix cache off.

        power_of_two is not the decode default on purpose: it reads
        ``aggregate.total_tokens`` from ``/v1/loads``, which the SGLang server
        does not emit, so it would silently score every worker ``-1``.
        """
        self.assertEqual(DEFAULT_PREFILL_POLICY, "cache_aware")
        self.assertEqual(DEFAULT_DECODE_POLICY, "round_robin")

    def test_enables_the_metrics_listener(self):
        cmd = _cmd(mode=SMG, bootstrap_port=BOOTSTRAP_PORT)
        self.assertEqual(cmd[cmd.index("--prometheus-port") + 1], "29000")
        self.assertEqual(cmd[cmd.index("--prometheus-host") + 1], "0.0.0.0")

    def test_metrics_listener_can_be_disabled(self):
        cmd = _cmd(mode=SMG, bootstrap_port=BOOTSTRAP_PORT, prometheus_port=0)
        self.assertNotIn("--prometheus-port", cmd)

    def test_cache_knobs_emitted_only_when_overridden(self):
        plain = _cmd(mode=SMG, bootstrap_port=BOOTSTRAP_PORT)
        self.assertNotIn("--cache-threshold", plain)

        tuned = _cmd(
            mode=SMG,
            bootstrap_port=BOOTSTRAP_PORT,
            cache_threshold=0.5,
            balance_abs_threshold=32,
            balance_rel_threshold=1.1,
        )
        self.assertEqual(tuned[tuned.index("--cache-threshold") + 1], "0.5")
        self.assertEqual(tuned[tuned.index("--balance-abs-threshold") + 1], "32")
        self.assertEqual(tuned[tuned.index("--balance-rel-threshold") + 1], "1.1")

    def test_request_timeout_applies_to_either_router(self):
        for mode in (MINI_LB, SMG):
            cmd = _cmd(mode=mode, request_timeout_secs=3600)
            self.assertEqual(cmd[cmd.index("--request-timeout-secs") + 1], "3600")


class TestEnvResolution(CustomTestCase):
    def test_default_mode_is_the_legacy_router(self):
        # NOT FOR MERGE: this validation branch defaults to SMG so a full
        # nightly exercises the migration. On the PR branch this asserts MINI_LB.
        self.assertEqual(resolve_mode(env={}), SMG)

    def test_mode_from_env(self):
        self.assertEqual(resolve_mode(env={"SGLANG_TPU_PD_ROUTER": "smg"}), SMG)
        self.assertEqual(resolve_mode(env={"SGLANG_TPU_PD_ROUTER": "  SMG  "}), SMG)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_mode(env={"SGLANG_TPU_PD_ROUTER": "sgl-router"})

    def test_policies_and_knobs_from_env(self):
        cmd = _cmd(
            bootstrap_port=BOOTSTRAP_PORT,
            mode=None,  # NOT FOR MERGE: let env resolution run; _cmd pins MINI_LB here
            env={
                "SGLANG_TPU_PD_ROUTER": "smg",
                "TPU_PD_PREFILL_POLICY": "random",
                "TPU_PD_DECODE_POLICY": "random",
                "TPU_PD_CACHE_THRESHOLD": "0.7",
            },
        )
        self.assertEqual(cmd[cmd.index("--prefill-policy") + 1], "random")
        self.assertEqual(cmd[cmd.index("--decode-policy") + 1], "random")
        self.assertEqual(cmd[cmd.index("--cache-threshold") + 1], "0.7")

    def test_explicit_argument_beats_env(self):
        cmd = _cmd(
            mode=SMG,
            prefill_policy="round_robin",
            env={"TPU_PD_PREFILL_POLICY": "random"},
        )
        self.assertEqual(cmd[cmd.index("--prefill-policy") + 1], "round_robin")

    def test_unknown_policy_is_rejected(self):
        with self.assertRaises(ValueError):
            _cmd(mode=SMG, prefill_policy="cache-aware")

    def test_metrics_url_only_exists_for_smg(self):
        self.assertEqual(
            router_metrics_url(host="127.0.0.1", env={"SGLANG_TPU_PD_ROUTER": "smg"}),
            "http://127.0.0.1:29000/metrics",
        )
        self.assertIsNone(
            # NOT FOR MERGE: env={} resolves to SMG on this branch.
            router_metrics_url(
                host="127.0.0.1", env={"SGLANG_TPU_PD_ROUTER": "mini_lb"}
            )
        )
        self.assertIsNone(
            router_metrics_url(
                host="127.0.0.1",
                env={"SGLANG_TPU_PD_ROUTER": "smg", "TPU_PD_PROMETHEUS_PORT": "0"},
            )
        )


if __name__ == "__main__":
    unittest.main()
