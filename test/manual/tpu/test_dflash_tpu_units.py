"""CPU-runnable unit tests for DFlash-on-TPU host-side helpers.

Covers ``validate_dflash_request`` TPU greedy-only admission — non-greedy
(temperature/top-k/top-p) requests must be rejected on TPU rather than silently
degrading to greedy argmax, while greedy (top_k == 1) is admitted. Needs no JAX
/ TPU device. The non-causal draft-block "allow" mask is a fixed-shape all-ones
cache inside the RPA-v3 backend and is validated on TPU in
``test_ragged_paged_attention_v3.py``.
"""

import types
import unittest

import sglang.srt.speculative.dflash_utils as du
from sglang.srt.speculative.dflash_utils import validate_dflash_request
from sglang.srt.speculative.dflash_worker import (
    resolve_dflash_draft_attention_backend,
)
from sglang.test.test_utils import CustomTestCase


class TestResolveDflashDraftBackend(CustomTestCase):
    """DFlash draft attention backend selection (the ENCODER_ONLY path)."""

    def test_tpu_defaults_to_rpa_v3(self):
        self.assertEqual(
            resolve_dflash_draft_attention_backend(None, is_tpu=True, is_hip=False),
            "pallas_tpu_rpa_v3",
        )

    def test_tpu_explicit_rpa_v3_ok(self):
        self.assertEqual(
            resolve_dflash_draft_attention_backend(
                "pallas_tpu_rpa_v3", is_tpu=True, is_hip=False
            ),
            "pallas_tpu_rpa_v3",
        )

    def test_tpu_rejects_gpu_backends(self):
        for bad in ("flashinfer", "fa3", "fa4", "triton", "trtllm_mha"):
            with self.subTest(backend=bad):
                with self.assertRaises(ValueError):
                    resolve_dflash_draft_attention_backend(
                        bad, is_tpu=True, is_hip=False
                    )

    def test_gpu_defaults_and_fallbacks(self):
        self.assertEqual(
            resolve_dflash_draft_attention_backend(None, is_tpu=False, is_hip=False),
            "flashinfer",
        )
        self.assertEqual(
            resolve_dflash_draft_attention_backend(None, is_tpu=False, is_hip=True),
            "triton",
        )
        # trtllm_mha (causal-only) and unknown backends fall back on GPU.
        self.assertEqual(
            resolve_dflash_draft_attention_backend(
                "trtllm_mha", is_tpu=False, is_hip=False
            ),
            "flashinfer",
        )
        self.assertEqual(
            resolve_dflash_draft_attention_backend(
                "unknown_backend", is_tpu=False, is_hip=False
            ),
            "flashinfer",
        )

    def test_gpu_supported_backends_passthrough(self):
        for good in ("flashinfer", "fa3", "fa4", "triton"):
            with self.subTest(backend=good):
                self.assertEqual(
                    resolve_dflash_draft_attention_backend(
                        good, is_tpu=False, is_hip=False
                    ),
                    good,
                )


def _make_req(
    top_k: int,
    return_logprob: bool = False,
    *,
    repetition_penalty: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    min_new_tokens: int = 0,
):
    """Minimal Req stand-in exposing only the fields validate_dflash_request reads."""
    sampling_params = types.SimpleNamespace(
        top_k=top_k,
        json_schema=None,
        regex=None,
        ebnf=None,
        structural_tag=None,
        repetition_penalty=repetition_penalty,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        min_new_tokens=min_new_tokens,
    )
    return types.SimpleNamespace(
        return_logprob=return_logprob,
        sampling_params=sampling_params,
    )


class TestValidateDflashRequestTPUGreedyOnly(CustomTestCase):
    def setUp(self):
        # Force the TPU code path regardless of the host running the test.
        self._saved = du._IS_TPU
        du._IS_TPU = True

    def tearDown(self):
        du._IS_TPU = self._saved

    def test_greedy_request_admitted(self):
        # top_k == 1 is greedy after SamplingParams normalization.
        self.assertIsNone(validate_dflash_request(_make_req(top_k=1)))

    def test_topk_sampling_rejected(self):
        err = validate_dflash_request(_make_req(top_k=50))
        self.assertIsInstance(err, str)
        self.assertIn("greedy", err.lower())

    def test_top_k_all_rejected(self):
        # top_k == -1 normalizes to TOP_K_ALL (a large int > 1): a sampling
        # request (any temperature/top_p sampling) -> rejected on TPU.
        from sglang.srt.sampling.sampling_params import TOP_K_ALL

        self.assertIsNotNone(validate_dflash_request(_make_req(top_k=TOP_K_ALL)))

    def test_return_logprob_still_rejected_first(self):
        err = validate_dflash_request(_make_req(top_k=1, return_logprob=True))
        self.assertIn("return_logprob", err)

    def test_greedy_with_repetition_penalty_rejected(self):
        # Multiplicative repetition_penalty is dropped by the block-verify path
        # (applied to a zero tensor), so a greedy request with it must be rejected
        # rather than silently diverging from the non-spec target.
        err = validate_dflash_request(_make_req(top_k=1, repetition_penalty=1.2))
        self.assertIsInstance(err, str)
        self.assertIn("penalt", err.lower())

    def test_greedy_with_frequency_penalty_rejected(self):
        err = validate_dflash_request(_make_req(top_k=1, frequency_penalty=0.5))
        self.assertIsInstance(err, str)
        self.assertIn("penalt", err.lower())

    def test_greedy_with_presence_penalty_rejected(self):
        err = validate_dflash_request(_make_req(top_k=1, presence_penalty=0.5))
        self.assertIsInstance(err, str)
        self.assertIn("penalt", err.lower())

    def test_greedy_with_min_new_tokens_rejected(self):
        # min_new_tokens is enforced by BatchedMinNewTokensPenalizer (masks EOS
        # while output_len < min_new_tokens) through the same broadcast-once block
        # path, so it must be rejected on TPU.
        err = validate_dflash_request(_make_req(top_k=1, min_new_tokens=8))
        self.assertIsInstance(err, str)
        self.assertIn("min_new_tokens", err)

    def test_plain_greedy_without_penalties_admitted(self):
        # Defaults (repetition 1.0, frequency/presence 0.0, min_new_tokens 0) stay
        # admitted.
        self.assertIsNone(validate_dflash_request(_make_req(top_k=1)))


class TestValidateDflashRequestNonTPU(CustomTestCase):
    def setUp(self):
        self._saved = du._IS_TPU
        du._IS_TPU = False

    def tearDown(self):
        du._IS_TPU = self._saved

    def test_non_tpu_admits_sampling(self):
        # Off TPU, sampling verify is available; non-greedy is not rejected here.
        self.assertIsNone(validate_dflash_request(_make_req(top_k=50)))

    def test_non_tpu_admits_penalties(self):
        # The penalizer reject is TPU-scoped (matches the greedy-only reject).
        self.assertIsNone(
            validate_dflash_request(_make_req(top_k=1, repetition_penalty=1.3))
        )

    def test_non_tpu_admits_min_new_tokens(self):
        # min_new_tokens reject is also TPU-scoped.
        self.assertIsNone(
            validate_dflash_request(_make_req(top_k=1, min_new_tokens=8))
        )


class TestDflashNumCorrectDrafts(CustomTestCase):
    """The verify() correct-draft metric (``dflash_num_correct_drafts``): correct
    drafts exclude the bonus, except when a finishing draft stops the commit loop
    before the bonus is appended.
    """

    def test_bonus_appended_common_case(self):
        # accept_len drafts + 1 bonus committed -> correct = accept_len.
        for acc in (0, 1, 3):
            self.assertEqual(du.dflash_num_correct_drafts(acc, acc + 1), acc)

    def test_finishing_draft_before_bonus(self):
        # Loop broke on an accepted draft (no bonus): all appended are drafts.
        # acc_len=3 but only 2 committed before the finishing draft -> 2, not 1.
        self.assertEqual(du.dflash_num_correct_drafts(3, 2), 2)
        self.assertEqual(du.dflash_num_correct_drafts(3, 3), 3)

    def test_zero_drafts_only_bonus(self):
        # No drafts accepted, only the bonus committed -> 0 correct drafts.
        self.assertEqual(du.dflash_num_correct_drafts(0, 1), 0)


class TestDflashKvCellSizing(CustomTestCase):
    """The shared target+draft KV pool must be sized from the DRAFT config, not
    by assuming the draft's per-layer KV bytes equal the target's.
    """

    def test_explicit_draft_cell_size_is_summed(self):
        from sglang.srt.speculative.dflash_utils import (
            scale_kv_cell_size_per_token_for_dflash,
        )

        got = scale_kv_cell_size_per_token_for_dflash(
            target_cell_size_per_token=1000,
            target_num_layers=32,
            draft_num_layers=5,
            draft_cell_size_per_token=200,
        )
        self.assertEqual(got, 1200)

    def test_layer_count_fallback_when_draft_cell_size_none(self):
        from sglang.srt.speculative.dflash_utils import (
            scale_kv_cell_size_per_token_for_dflash,
        )

        # Layer-count scaling: ceil(1000 * (32+5) / 32) = 1157.
        got = scale_kv_cell_size_per_token_for_dflash(
            target_cell_size_per_token=1000,
            target_num_layers=32,
            draft_num_layers=5,
        )
        self.assertEqual(got, 1157)

    def test_wider_draft_not_undersized_vs_layer_count(self):
        # A draft with larger per-token KV (e.g. more KV heads / bigger head_dim)
        # must be sized from its real bytes, which exceed the layer-count estimate
        # that assumes draft==target per-layer.
        from sglang.srt.speculative.dflash_utils import (
            scale_kv_cell_size_per_token_for_dflash,
        )

        layer_count = scale_kv_cell_size_per_token_for_dflash(
            target_cell_size_per_token=1000, target_num_layers=32, draft_num_layers=5
        )
        exact_wide = scale_kv_cell_size_per_token_for_dflash(
            target_cell_size_per_token=1000,
            target_num_layers=32,
            draft_num_layers=5,
            draft_cell_size_per_token=2000,  # a wide draft
        )
        self.assertGreater(exact_wide, layer_count)


class TestDflashDraftCellSize(CustomTestCase):
    """The draft KV cell size is derived from the DRAFT config (import-light
    helper, no model_runner import) and needs the resolved KV cache dtype.
    """

    @staticmethod
    def _draft_cfg(kv_heads, head_dim, v_head_dim=None, mla=False):
        return types.SimpleNamespace(
            use_mla_backend=mla,
            head_dim=head_dim,
            v_head_dim=head_dim if v_head_dim is None else v_head_dim,
            get_num_kv_heads=lambda tp, _n=kv_heads: _n,
        )

    def test_computes_exact_bytes(self):
        import torch

        from sglang.srt.speculative.dflash_utils import (
            compute_dflash_draft_cell_size_per_token,
        )

        got = compute_dflash_draft_cell_size_per_token(
            self._draft_cfg(kv_heads=8, head_dim=128), 5, torch.bfloat16, 1
        )
        self.assertEqual(got, 8 * (128 + 128) * 5 * 2)  # bf16 = 2 bytes

    def test_requires_resolved_kv_dtype(self):
        # The Round-5 bug ran this before configure_kv_cache_dtype() with an
        # unset dtype; the helper must fail loudly, not silently return a wrong
        # value (guarding the init-order fix in ModelRunner).
        from sglang.srt.speculative.dflash_utils import (
            compute_dflash_draft_cell_size_per_token,
        )

        with self.assertRaises(Exception):
            compute_dflash_draft_cell_size_per_token(
                self._draft_cfg(kv_heads=8, head_dim=128), 5, None, 1
            )

    def test_mla_and_missing_return_none(self):
        import torch

        from sglang.srt.speculative.dflash_utils import (
            compute_dflash_draft_cell_size_per_token,
        )

        self.assertIsNone(
            compute_dflash_draft_cell_size_per_token(
                self._draft_cfg(kv_heads=8, head_dim=128, mla=True),
                5,
                torch.bfloat16,
                1,
            )
        )
        self.assertIsNone(
            compute_dflash_draft_cell_size_per_token(None, 5, torch.bfloat16, 1)
        )
        self.assertIsNone(
            compute_dflash_draft_cell_size_per_token(
                self._draft_cfg(kv_heads=8, head_dim=128), None, torch.bfloat16, 1
            )
        )

    def test_pool_configurator_passes_draft_cell_size_through(self):
        # Prove DefaultPoolConfigurator forwards mr.dflash_draft_cell_size_per_token
        # into scale_kv_cell_size_per_token_for_dflash (not just that the scale
        # function sums correctly). Import-light: pool_configurator does not pull
        # in model_runner.
        import torch
        from unittest import mock

        import sglang.srt.model_executor.pool_configurator as pc

        fake_model_config = types.SimpleNamespace(
            get_num_kv_heads=lambda tp: 8,
            head_dim=128,
            v_head_dim=128,
            hf_config=types.SimpleNamespace(architectures=["LlamaForCausalLM"]),
        )
        fake_mr = types.SimpleNamespace(
            mambaish_config=None,
            num_effective_layers=32,
            model_config=fake_model_config,
            kv_cache_dtype=torch.bfloat16,
            use_mla_backend=False,
            spec_algorithm=types.SimpleNamespace(is_dflash=lambda: True),
            is_draft_worker=False,
            dflash_draft_num_layers=5,
            dflash_draft_cell_size_per_token=999,
        )
        captured = {}

        def _spy(**kwargs):
            captured.update(kwargs)
            return kwargs["target_cell_size_per_token"] + (
                kwargs.get("draft_cell_size_per_token") or 0
            )

        with mock.patch.object(
            pc, "get_attention_tp_size", return_value=1
        ), mock.patch(
            "sglang.srt.speculative.dflash_utils.scale_kv_cell_size_per_token_for_dflash",
            side_effect=_spy,
        ):
            pc.DefaultPoolConfigurator(fake_mr)

        self.assertEqual(captured.get("draft_cell_size_per_token"), 999)


class TestDflashVerifyMaskPolicyAvoidsTriton(CustomTestCase):
    """Verify-routing guard: on the TPU RPA-v3 backend, DFlash verify skips its custom mask,
    so it uses RPA-v3's own metadata builder and never reaches
    DFlashVerifyInput.generate_attn_arg_prefill (the only caller of the Triton
    create_flashinfer_kv_indices_triton kernel).
    """

    @staticmethod
    def _named(name):
        cls = type(name, (), {})
        return cls()

    def test_rpa_v3_backend_skips_custom_mask(self):
        from sglang.srt.speculative.dflash_utils import (
            resolve_dflash_verify_mask_policy,
        )

        backend = self._named("TpuPallasRpaV3AttnBackend")
        backend_name, build_custom_mask = resolve_dflash_verify_mask_policy(backend)
        self.assertEqual(backend_name, "TpuPallasRpaV3AttnBackend")
        self.assertFalse(
            build_custom_mask,
            "RPA-v3 verify must skip the DFlash custom mask so it uses RPA-v3's "
            "own metadata (not the flashinfer/Triton kv-indices path).",
        )

    def test_unknown_backend_still_builds_custom_mask(self):
        from sglang.srt.speculative.dflash_utils import (
            resolve_dflash_verify_mask_policy,
        )

        _, build_custom_mask = resolve_dflash_verify_mask_policy(
            self._named("SomeGpuAttnBackend")
        )
        self.assertTrue(build_custom_mask)

    def test_dflash_info_does_not_import_triton_kv_indices_at_module_scope(self):
        # The Triton kv-indices kernel must NOT be imported at dflash_info module
        # scope (that eagerly loads Triton). It is imported lazily inside
        # generate_attn_arg_prefill, which the TPU RPA-v3 verify path never calls.
        import inspect

        import sglang.srt.speculative.dflash_info as di

        self.assertNotIn(
            "create_flashinfer_kv_indices_triton",
            vars(di),
            "dflash_info must not import create_flashinfer_kv_indices_triton at "
            "module scope (import it lazily inside generate_attn_arg_prefill).",
        )
        # ...and it is imported lazily inside the only method that uses it.
        gen_src = inspect.getsource(di.DFlashVerifyInput.generate_attn_arg_prefill)
        self.assertIn(
            "from sglang.srt.layers.attention.utils import", gen_src
        )
        self.assertIn("create_flashinfer_kv_indices_triton[", gen_src)

    def test_dflash_owned_modules_import_no_triton_at_module_scope(self):
        # Robust, deterministic (no subprocess, no framework backend init) proof
        # that DFlash's OWN source files add no Triton import: none of the
        # DFlash-owned modules may import Triton (or the flashinfer kv-indices
        # kernel) at module scope. This is the DFlash-controllable half of the
        # no-Triton requirement; any Triton in sys.modules on the verify route
        # comes from shared framework deps (torch itself, logits_processor, ...),
        # not from DFlash.
        import ast
        import pathlib

        import sglang.srt.speculative.dflash_info as di

        pkg_dir = pathlib.Path(di.__file__).parent
        dflash_files = sorted(pkg_dir.glob("dflash*.py"))
        self.assertTrue(dflash_files, "expected DFlash-owned source files")

        offenders = []
        for path in dflash_files:
            tree = ast.parse(path.read_text(), filename=str(path))
            # Only inspect module-scope (top-level) import statements; lazy imports
            # inside functions/methods are allowed (that is how the kv-indices
            # kernel is confined to generate_attn_arg_prefill).
            for node in tree.body:
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""] + [a.name for a in node.names]
                for name in names:
                    if name.split(".")[0] == "triton" or (
                        "create_flashinfer_kv_indices_triton" in name
                    ):
                        offenders.append(f"{path.name}: {name}")
        self.assertEqual(
            offenders,
            [],
            "DFlash-owned modules must not import Triton at module scope: "
            f"{offenders}",
        )

    def test_dflash_verify_route_adds_no_triton_beyond_framework_baseline(self):
        # Strongest ACHIEVABLE "no Triton import" guard for the DFlash verify
        # route on this stack, checked in CLEAN subprocesses via sys.modules.
        #
        # A literal "zero Triton modules" is impossible here: a bare `import torch`
        # already pulls Triton into sys.modules (torch._inductor.runtime imports
        # triton.language.extra.libdevice at import-torch time). Every torch-
        # importing module therefore has Triton in sys.modules. What DFlash CAN
        # guarantee is that its OWN code imports no Triton beyond the framework
        # modules it already depends on. We assert that against a deterministic,
        # CPU-runnable baseline: `torch` + `sglang.srt.layers.logits_processor`.
        #
        # Baseline choice rationale (robustness): dflash_info imports
        # logits_processor at module scope, so logits_processor is guaranteed
        # importable wherever dflash_info is. It is also import-light (unlike
        # eagle_info, whose import triggers TPU backend/port init and raises
        # portpicker.NoFreePortFoundError in a constrained env), and it is the
        # module that pulls the sole non-torch Triton module on the route
        # (triton.tools.tensor_descriptor, via vocab_parallel_embedding ->
        # quantization -> fp8_kernel).
        import json
        import subprocess
        import sys

        probe = (
            "import sys, json\n"
            "{stmt}\n"
            "tri = sorted(m for m in sys.modules "
            "if m == 'triton' or m.startswith('triton.'))\n"
            "sys.stderr.write('TRITONSET:' + json.dumps(tri) + '\\n')\n"
        )

        def triton_modules(stmt):
            out = subprocess.run(
                [sys.executable, "-c", probe.format(stmt=stmt)],
                capture_output=True,
                text=True,
            )
            tail = "\n".join(
                (out.stdout.splitlines()[-15:] + out.stderr.splitlines()[-15:])
            )
            self.assertEqual(
                out.returncode,
                0,
                f"child import `{stmt}` exited {out.returncode}; output tail:\n{tail}",
            )
            for line in out.stderr.splitlines():
                if line.startswith("TRITONSET:"):
                    return set(json.loads(line[len("TRITONSET:") :]))
            self.fail(
                f"child import `{stmt}` did not report a triton set; tail:\n{tail}"
            )

        torch_only = triton_modules("import torch")
        baseline = triton_modules(
            "import torch; import sglang.srt.layers.logits_processor"
        )
        dflash = triton_modules("import sglang.srt.speculative.dflash_info")

        # 1. A bare `import torch` already loads Triton -> literal zero is impossible.
        self.assertGreater(
            len(torch_only),
            0,
            "expected `import torch` to load Triton on this stack (documents that a "
            "literal zero-Triton-import guard is a torch property, not DFlash's).",
        )
        # 2. DFlash's own code imports NO Triton module beyond the framework
        #    baseline (torch + logits_processor) it already depends on.
        dflash_beyond_baseline = dflash - baseline
        self.assertEqual(
            dflash_beyond_baseline,
            set(),
            "DFlash verify route imports Triton module(s) beyond the framework "
            f"baseline (torch + logits_processor): {sorted(dflash_beyond_baseline)}. "
            "DFlash must add no Triton of its own.",
        )

    def test_triton_kv_indices_only_reachable_via_generate_attn_arg_prefill(self):
        # The Triton kernel is only invoked from generate_attn_arg_prefill, so
        # skipping custom mask + using RPA-v3 metadata provably avoids it.
        import inspect

        import sglang.srt.speculative.dflash_info as di

        src = inspect.getsource(di)
        # It is imported and called exactly once, inside generate_attn_arg_prefill.
        self.assertEqual(src.count("create_flashinfer_kv_indices_triton["), 1)
        gen_src = inspect.getsource(di.DFlashVerifyInput.generate_attn_arg_prefill)
        self.assertIn("create_flashinfer_kv_indices_triton[", gen_src)


if __name__ == "__main__":
    unittest.main()
