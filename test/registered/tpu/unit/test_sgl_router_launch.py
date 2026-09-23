"""Unit tests for the sgl-router launch and probe helpers.

Host-only: :mod:`sglang.test.tpu.sgl_router` imports only the standard
library, so these run anywhere and need neither a TPU nor the router binary.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=30, suite="stage-a-test-tpu-unit")

import unittest

from sglang.test.test_utils import CustomTestCase
from sglang.test.tpu import sgl_router

PREFILL = "http://prefill-0:30000"
DECODE = "http://decode-0:30000"
MODEL = "/weights/gpt-oss-120b"
ON = {"SGLANG_TPU_PD_ROUTER": "sgl_router"}


def _cmd(**overrides):
    kwargs = dict(
        prefill_urls=[PREFILL],
        decode_urls=[DECODE],
        host="0.0.0.0",
        port=8000,
        model_path=MODEL,
        env={},
    )
    kwargs.update(overrides)
    return sgl_router.build_cmd(**kwargs)


class TestSelection(CustomTestCase):
    def test_only_the_exact_value_selects(self):
        self.assertFalse(sgl_router.selected(env={}))
        self.assertFalse(sgl_router.selected(env={"SGLANG_TPU_PD_ROUTER": "mini_lb"}))
        self.assertFalse(sgl_router.selected(env={"SGLANG_TPU_PD_ROUTER": "smg"}))
        self.assertTrue(sgl_router.selected(env=ON))
        self.assertTrue(
            sgl_router.selected(env={"SGLANG_TPU_PD_ROUTER": " SGL_ROUTER "})
        )

    def test_health_path_and_spec_env_follow_selection(self):
        self.assertEqual(sgl_router.health_path(env={}), "/health")
        self.assertEqual(sgl_router.health_path(env=ON), "/readyz")
        self.assertEqual(sgl_router.spec_server_info_env([DECODE], env={}), {})
        self.assertEqual(
            sgl_router.spec_server_info_env(
                [DECODE + "/", "http://decode-1:30000"], env=ON
            ),
            {
                sgl_router.SPEC_SERVER_INFO_ENV: "http://decode-0:30000,http://decode-1:30000"
            },
        )


class TestArgv(CustomTestCase):
    def test_binary_flat_worker_list_and_model_id(self):
        cmd = _cmd()
        self.assertEqual(cmd[0], "sgl-router")
        i = cmd.index("--worker-urls")
        self.assertEqual(cmd[i + 1 : i + 3], (PREFILL, DECODE))
        for flag in ("--prefill", "--decode", "--pd-disaggregation", "--mini-lb"):
            self.assertNotIn(flag, cmd)
        self.assertEqual(cmd[cmd.index("--model-id") + 1], MODEL)
        self.assertEqual(cmd[cmd.index("--tokenizer-path") + 1], MODEL)
        self.assertEqual(
            _cmd(tokenizer_path="/tok")[cmd.index("--tokenizer-path") + 1], "/tok"
        )

    def test_two_prefill_two_decode(self):
        cmd = _cmd(
            prefill_urls=["http://p0", "http://p1"],
            decode_urls=["http://d0", "http://d1"],
        )
        i = cmd.index("--worker-urls")
        self.assertEqual(
            cmd[i + 1 : i + 5], ("http://p0", "http://p1", "http://d0", "http://d1")
        )

    def test_binary_overridable(self):
        self.assertEqual(
            _cmd(env={"SGLANG_TPU_SGL_ROUTER_BIN": "/opt/sgl-router"})[0],
            "/opt/sgl-router",
        )

    def test_defaults_cache_aware_prefill_power_of_two_decode(self):
        cmd = _cmd()
        self.assertEqual(cmd[cmd.index("--policy") + 1], "cache_aware")
        self.assertEqual(cmd[cmd.index("--decode-policy") + 1], "power_of_two")

    def test_policies_from_env_in_the_routers_vocabulary(self):
        cmd = _cmd(
            env={
                "TPU_PD_PREFILL_POLICY": "round_robin",
                "TPU_PD_DECODE_POLICY": "legacy_host_affinity",
            }
        )
        self.assertEqual(cmd[cmd.index("--policy") + 1], "round_robin")
        self.assertEqual(cmd[cmd.index("--decode-policy") + 1], "legacy_host_affinity")
        with self.assertRaises(ValueError):
            _cmd(env={"TPU_PD_PREFILL_POLICY": "consistent_hashing"})
        with self.assertRaises(ValueError):
            _cmd(env={"TPU_PD_DECODE_POLICY": "round_robin"})

    def test_prefix_hash_knobs_only_with_prefix_hash(self):
        plain = _cmd(env={"TPU_PD_PREFIX_TOKEN_COUNT": "64"})
        self.assertNotIn("--prefix-token-count", plain)
        cmd = _cmd(env={"TPU_PD_PREFILL_POLICY": "prefix_hash"})
        self.assertEqual(cmd[cmd.index("--policy") + 1], "prefix_hash")
        self.assertNotIn("--prefix-token-count", cmd)
        tuned = _cmd(
            env={
                "TPU_PD_PREFILL_POLICY": "prefix_hash",
                "TPU_PD_PREFIX_TOKEN_COUNT": "64",
                "TPU_PD_PREFIX_HASH_LOAD_FACTOR": "2.0",
            }
        )
        self.assertEqual(tuned[tuned.index("--prefix-token-count") + 1], "64")
        self.assertEqual(tuned[tuned.index("--prefix-hash-load-factor") + 1], "2.0")

    def test_otlp_and_timeout_passthrough(self):
        self.assertNotIn("--otlp-endpoint", _cmd())
        cmd = _cmd(
            env={"SGLANG_TPU_OTLP_ENDPOINT": "http://otel:4317"},
            request_timeout_secs=3600,
        )
        self.assertEqual(cmd[cmd.index("--otlp-endpoint") + 1], "http://otel:4317")
        self.assertEqual(cmd[cmd.index("--request-timeout-secs") + 1], "3600")

    def test_model_path_and_both_roles_required(self):
        with self.assertRaises(ValueError):
            _cmd(model_path="")
        with self.assertRaises(ValueError):
            _cmd(decode_urls=[])


class TestChatCompletionParsing(CustomTestCase):
    def _payload(self, *, cached=0, details=None, spec=None, content="1 2 3"):
        usage = {
            "prompt_tokens": 120,
            "completion_tokens": 16,
            "prompt_tokens_details": {"cached_tokens": cached},
        }
        ext = {}
        if details is not None:
            ext["cached_tokens_details"] = details
        if spec is not None:
            ext["spec_tokens_details"] = spec
        choice = {"message": {"role": "assistant", "content": content}}
        if ext:
            choice["sglext"] = ext
        return {"choices": [choice], "usage": usage}

    def test_cold_read_has_zero_total_and_no_breakdown(self):
        g = sgl_router.parse_chat_completion(self._payload())
        self.assertEqual(
            (g.cached_tokens, g.device_cached_tokens, g.host_cached_tokens), (0, 0, 0)
        )
        self.assertEqual(
            (g.prompt_tokens, g.completion_tokens, g.spec_verify_ct), (120, 16, 0)
        )
        self.assertEqual(g.text, "1 2 3")

    def test_tiers_and_spec_counters_come_from_sglext(self):
        g = sgl_router.parse_chat_completion(
            self._payload(
                cached=100,
                details={"device": 40, "host": 60},
                spec={"spec_verify_ct": 7},
            )
        )
        self.assertEqual(
            (g.cached_tokens, g.device_cached_tokens, g.host_cached_tokens),
            (100, 40, 60),
        )
        self.assertEqual(g.spec_verify_ct, 7)
        # n>1 responses carry a list; the first choice's stats are the probe's.
        g = sgl_router.parse_chat_completion(
            self._payload(cached=0, details={}, spec=[{"spec_verify_ct": 3}])
        )
        self.assertEqual(g.spec_verify_ct, 3)

    def test_contradictions_are_errors(self):
        with self.assertRaisesRegex(RuntimeError, "enable-cache-report"):
            sgl_router.parse_chat_completion(
                {"choices": [{"message": {"content": "x"}}], "usage": {}}
            )
        with self.assertRaisesRegex(RuntimeError, "no sglext.cached_tokens_details"):
            sgl_router.parse_chat_completion(self._payload(cached=50))

    def test_empty_content_is_empty_text(self):
        g = sgl_router.parse_chat_completion(self._payload(content=None))
        self.assertEqual(g.text, "")


if __name__ == "__main__":
    unittest.main()
