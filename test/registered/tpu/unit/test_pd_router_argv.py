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
    SGL_ROUTER,
    SGL_ROUTER_DEFAULT_DECODE_POLICY,
    SMG,
    SPEC_SERVER_INFO_ENV,
    build_router_cmd,
    resolve_mode,
    router_health_path,
    router_metrics_url,
    spec_server_info_env,
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


MODEL_PATH = "/weights/gpt-oss-120b"


class TestSglRouterArgv(CustomTestCase):
    def _sgl(self, **overrides):
        kwargs = dict(
            mode=SGL_ROUTER, model_path=MODEL_PATH, bootstrap_port=BOOTSTRAP_PORT
        )
        kwargs.update(overrides)
        return _cmd(**kwargs)

    def test_launches_the_binary_not_the_python_module(self):
        cmd = self._sgl()
        self.assertEqual(cmd[0], "sgl-router")
        self.assertNotIn("sglang_router.launch_router", cmd)
        self.assertNotIn("--pd-disaggregation", cmd)
        self.assertNotIn("--mini-lb", cmd)

    def test_binary_overridable_from_env(self):
        cmd = self._sgl(env={"SGLANG_TPU_SGL_ROUTER_BIN": "/opt/bin/sgl-router"})
        self.assertEqual(cmd[0], "/opt/bin/sgl-router")
        self.assertEqual(self._sgl(binary="./sgl-router")[0], "./sgl-router")

    def test_one_flat_worker_list_roles_come_from_server_info(self):
        """No --prefill/--decode: the router classifies each URL itself."""
        cmd = self._sgl()
        i = cmd.index("--worker-urls")
        self.assertEqual(cmd[i + 1 : i + 3], (PREFILL_URL, DECODE_URL))
        for flag in ("--prefill", "--decode"):
            self.assertNotIn(flag, cmd)
        # bootstrap_port is read off the prefill's /server_info, never passed.
        self.assertNotIn(str(BOOTSTRAP_PORT), cmd)

    def test_model_id_and_tokenizer_default_to_the_model_path(self):
        cmd = self._sgl()
        self.assertEqual(cmd[cmd.index("--model-id") + 1], MODEL_PATH)
        self.assertEqual(cmd[cmd.index("--tokenizer-path") + 1], MODEL_PATH)
        cmd = self._sgl(tokenizer_path="/tok")
        self.assertEqual(cmd[cmd.index("--tokenizer-path") + 1], "/tok")

    def test_model_path_is_required(self):
        with self.assertRaises(ValueError):
            self._sgl(model_path=None)

    def test_defaults_are_cache_aware_prefill_power_of_two_decode(self):
        """Unlike the gateway, sgl-router's power_of_two reads real load."""
        cmd = self._sgl()
        self.assertEqual(cmd[cmd.index("--policy") + 1], DEFAULT_PREFILL_POLICY)
        self.assertEqual(
            cmd[cmd.index("--decode-policy") + 1], SGL_ROUTER_DEFAULT_DECODE_POLICY
        )
        self.assertEqual(SGL_ROUTER_DEFAULT_DECODE_POLICY, "power_of_two")
        self.assertNotIn("--prefill-policy", cmd)

    def test_policies_from_env_use_the_routers_vocabulary(self):
        env = {
            "TPU_PD_PREFILL_POLICY": "round_robin",
            "TPU_PD_DECODE_POLICY": "legacy_host_affinity",
        }
        cmd = self._sgl(env=env)
        self.assertEqual(cmd[cmd.index("--policy") + 1], "round_robin")
        self.assertEqual(cmd[cmd.index("--decode-policy") + 1], "legacy_host_affinity")

    def test_prefix_hash_is_rejected_with_a_pointer(self):
        with self.assertRaisesRegex(ValueError, "no prefix_hash"):
            self._sgl(prefill_policy="prefix_hash")
        # A gateway decode policy the binary does not know is rejected too.
        with self.assertRaises(ValueError):
            self._sgl(decode_policy="round_robin")

    def test_no_prometheus_flags_metrics_share_the_listener(self):
        cmd = self._sgl()
        self.assertNotIn("--prometheus-port", cmd)
        env = {"SGLANG_TPU_PD_ROUTER": SGL_ROUTER}
        self.assertEqual(
            router_metrics_url(host="h", env=env, router_port=LB_PORT),
            f"http://h:{LB_PORT}/metrics",
        )
        self.assertIsNone(router_metrics_url(host="h", env=env))

    def test_request_timeout_is_forwarded(self):
        cmd = self._sgl(request_timeout_secs=3600)
        self.assertEqual(cmd[cmd.index("--request-timeout-secs") + 1], "3600")

    def test_smg_and_mini_lb_argv_unchanged_by_the_new_mode(self):
        """Adding a third router must not disturb the two existing argvs."""
        self.assertEqual(_cmd()[:3], ("python3", "-m", "sglang_router.launch_router"))
        self.assertNotIn("--worker-urls", _cmd(mode=SMG, bootstrap_port=BOOTSTRAP_PORT))


class TestHarnessAdapters(CustomTestCase):
    def test_health_path_per_router(self):
        self.assertEqual(router_health_path(mode=MINI_LB), "/health")
        self.assertEqual(router_health_path(mode=SMG), "/health")
        self.assertEqual(router_health_path(mode=SGL_ROUTER), "/readyz")
        self.assertEqual(
            router_health_path(env={"SGLANG_TPU_PD_ROUTER": "sgl_router"}), "/readyz"
        )
        self.assertEqual(router_health_path(env={}), "/health")

    def test_spec_server_info_env_only_for_sgl_router(self):
        urls = ["http://decode-0:30000/", "http://decode-1:30000"]
        self.assertEqual(spec_server_info_env(urls, mode=MINI_LB), {})
        self.assertEqual(spec_server_info_env(urls, mode=SMG), {})
        self.assertEqual(
            spec_server_info_env(urls, mode=SGL_ROUTER),
            {SPEC_SERVER_INFO_ENV: "http://decode-0:30000,http://decode-1:30000"},
        )


class TestEnvResolution(CustomTestCase):
    def test_default_mode_is_the_legacy_router(self):
        self.assertEqual(resolve_mode(env={}), MINI_LB)

    def test_mode_from_env(self):
        self.assertEqual(resolve_mode(env={"SGLANG_TPU_PD_ROUTER": "smg"}), SMG)
        self.assertEqual(resolve_mode(env={"SGLANG_TPU_PD_ROUTER": "  SMG  "}), SMG)
        self.assertEqual(
            resolve_mode(env={"SGLANG_TPU_PD_ROUTER": "sgl_router"}), SGL_ROUTER
        )

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_mode(env={"SGLANG_TPU_PD_ROUTER": "sgl-router"})

    def test_policies_and_knobs_from_env(self):
        cmd = _cmd(
            bootstrap_port=BOOTSTRAP_PORT,
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
        self.assertIsNone(router_metrics_url(host="127.0.0.1", env={}))
        self.assertIsNone(
            router_metrics_url(
                host="127.0.0.1",
                env={"SGLANG_TPU_PD_ROUTER": "smg", "TPU_PD_PROMETHEUS_PORT": "0"},
            )
        )


if __name__ == "__main__":
    unittest.main()
