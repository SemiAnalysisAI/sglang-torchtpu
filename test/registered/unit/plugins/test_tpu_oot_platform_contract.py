# SPDX-License-Identifier: Apache-2.0
"""Contract test for the out-of-tree TPU platform (wave 0 of the plugin migration).

What this pins
--------------
``sglang_torchtpu.platform.TpuOOTPlatform`` declares ``PlatformEnum.OOT``, which
turns on every ``is_out_of_tree()`` gate in ``python/sglang/srt``. Those gates
call platform methods that the in-tree TPU platform never had to answer, because
under ``PlatformEnum.TPU`` they were unreachable. The base ``SRTPlatform`` /
``DeviceMixin`` implementations of several are bare ``raise NotImplementedError``.

So the contract is: **for every platform method reachable from an OOT gate, the
TPU side of the class hierarchy must have an opinion.** An opinion may well be
"not supported on TPU" -- ``get_piecewise_backend_cls`` raises exactly that, and
that is correct, because the alternative is silently falling through to the CUDA
piecewise backend. What is not acceptable is inheriting the *base* stub, which
means nobody considered the method at all.

The gate inventory is recomputed from source on every run by
``tools/plugin/gate_audit.py``, so an upstream sync that adds a gate calling a new
platform method fails this test. That alarm is the point of the file.

Runs on CPU; needs no TPU hardware and does not import ``torch_tpu``.

Run locally:
    SGLANG_TEST_RETRY_NUM=1 PYTHONPATH=$PWD/python \\
        python -m unittest test.registered.unit.plugins.test_tpu_oot_platform_contract
"""

import ast
import importlib.util
import os
import sys
import unittest
from typing import Optional
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, os.pardir)
)
_PLUGIN_ROOT = os.path.join(_REPO_ROOT, "hardware_backend", "tpu")

# The plugin distribution is installed only in CI mode B (`pip install -e
# hardware_backend/tpu/`). On the CPU gate it is not installed, so import it from
# the worktree. Appending rather than prepending lets a real install win.
if _PLUGIN_ROOT not in sys.path:
    sys.path.append(_PLUGIN_ROOT)


def _load_tool(filename: str, name: str):
    """Import a ``tools/plugin/`` script without putting tools/ on sys.path."""
    path = os.path.join(_REPO_ROOT, "tools", "plugin", filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # @dataclass resolves annotations through sys.modules[cls.__module__], so the
    # module has to be registered before its body runs.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gate_audit = _load_tool("gate_audit.py", "_tpu_gate_audit")
tpu_debt = _load_tool("check_tpu_branches.py", "_tpu_debt_ratchet")


# ---------------------------------------------------------------------------
# Known divergences: gate-reachable methods the TPU platform does not answer.
#
# Every entry is a path that breaks under SGLANG_PLATFORM=tpu_oot. The list must
# only ever shrink -- `test_known_unanswered_list_is_not_stale` fails if an entry
# has since been implemented, and nothing may be added without a reason here.
# ---------------------------------------------------------------------------
KNOWN_UNANSWERED = {
    "get_device_core_count": (
        "Defined on no platform class at all, yet utils/common.py:1012 calls it "
        "behind `is_out_of_tree() or is_tpu()` -- so it is already an AttributeError "
        "on the in-tree TPU platform, not a regression introduced by the flip. Its "
        "only callers are triton_backend, wave_backend and batch_invariant_ops, "
        "none of which TPU reaches, which is why nobody has hit it. Belongs upstream."
    ),
    "get_torch_profiler_activity": (
        "`[Planned]` stub on DeviceMixin with no implementer on any platform. "
        "Reached from profile_utils.py:332 under a pure-OOT gate, so torch-profiler "
        "startup raises in mode B. Needs a torch_tpu ProfilerActivity decision on "
        "hardware (the runtime logs that native TPU profiling is off below torch "
        "2.12), so it is not something to guess at from a CPU box."
    ),
    "get_torch_profiler_activity_str": (
        "Same `[Planned]` DeviceMixin stub, reached from profile_utils.py:251 and "
        "profiler_manager.py:201. Pairs with get_torch_profiler_activity above."
    ),
}


def _defining_class(cls: type, name: str) -> Optional[type]:
    for klass in cls.__mro__:
        if name in klass.__dict__:
            return klass
    return None


class TestTpuOOTPlatformContract(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang_torchtpu.platform import TpuOOTPlatform

        from sglang.srt.hardware_backend.tpu.tpu_platform import TpuSRTPlatform
        from sglang.srt.platforms.device_mixin import PlatformEnum
        from sglang.srt.platforms.interface import SRTPlatform

        cls.PlatformEnum = PlatformEnum
        cls.SRTPlatform = SRTPlatform
        cls.TpuSRTPlatform = TpuSRTPlatform
        cls.TpuOOTPlatform = TpuOOTPlatform
        cls.platform = TpuOOTPlatform()

        # Classes that count as "the TPU side has an opinion". Anything resolved
        # above these in the MRO is an unconsidered base-class default.
        cls.tpu_owned = (TpuOOTPlatform, TpuSRTPlatform)

        cls.gates = gate_audit.audit_tree(_REPO_ROOT)
        cls.required = gate_audit.required_platform_methods(cls.gates)

    # -- identity ----------------------------------------------------------

    def test_declares_out_of_tree(self):
        self.assertEqual(self.TpuOOTPlatform._enum, self.PlatformEnum.OOT)
        self.assertTrue(self.platform.is_out_of_tree())

    def test_still_reports_as_tpu(self):
        """~300 `is_tpu()` sites in core must keep seeing True through the flip."""
        self.assertTrue(self.platform.is_tpu())
        self.assertFalse(self.platform.is_cuda())
        self.assertFalse(self.platform.is_cpu())

    def test_subclasses_the_in_tree_platform(self):
        """Wave 0 moves no code: every factory is inherited, not reimplemented."""
        self.assertTrue(issubclass(self.TpuOOTPlatform, self.TpuSRTPlatform))

    # -- the gate contract -------------------------------------------------

    def test_gate_audit_finds_the_core_gates(self):
        """A sanity floor: the audit must still be looking at the right tree."""
        self.assertGreater(len(self.gates), 0, "gate audit found nothing")
        self.assertGreater(len(self.required), 0, "no platform methods reached")

    def test_every_gate_reachable_method_has_a_tpu_opinion(self):
        unanswered = {}
        for method, sites in self.required.items():
            if method in KNOWN_UNANSWERED:
                continue
            owner = _defining_class(self.TpuOOTPlatform, method)
            if owner is None:
                unanswered[method] = (f"not defined on any class in the MRO", sites)
            elif owner not in self.tpu_owned:
                unanswered[method] = (
                    f"resolves to {owner.__name__}, not the TPU platform",
                    sites,
                )

        if unanswered:
            lines = [
                "Platform methods reached from an is_out_of_tree() gate that the "
                "TPU platform does not answer:"
            ]
            for method, (why, sites) in sorted(unanswered.items()):
                lines.append(f"  {method}: {why}")
                for site in sites:
                    lines.append(f"      gate at {site}")
            lines.append(
                "Implement each on TpuOOTPlatform (hardware_backend/tpu/), or add it "
                "to KNOWN_UNANSWERED in this file with the reason it cannot be."
            )
            self.fail("\n".join(lines))

    def test_known_unanswered_list_is_not_stale(self):
        """Entries must be removed once implemented -- the list only shrinks."""
        now_answered = []
        for method in KNOWN_UNANSWERED:
            owner = _defining_class(self.TpuOOTPlatform, method)
            if owner in self.tpu_owned:
                now_answered.append(method)
        self.assertEqual(
            now_answered,
            [],
            "these are implemented now; delete them from KNOWN_UNANSWERED",
        )

    def test_known_unanswered_entries_are_still_reachable(self):
        """A method no gate reaches any more does not belong in the list."""
        unreachable = [m for m in KNOWN_UNANSWERED if m not in self.required]
        self.assertEqual(
            unreachable,
            [],
            "no is_out_of_tree() gate reaches these; drop them from KNOWN_UNANSWERED",
        )

    # -- DSA / sparse attention -------------------------------------------

    def test_dsa_kv_pool_cls_fails_with_a_tpu_specific_message(self):
        """The DSA path becomes reachable under OOT, and TPU has no DSA pool.

        ``kv_cache_configurator._build_oot_dsa_kv_pool`` calls this; under the
        in-tree platform that arm never fired. There is no TPU DSA pool class to
        return (the fork's ``get_nsa_kv_pool_cls`` is itself a raising stub), so
        the contract is a refusal that names TPU rather than a bare
        ``NotImplementedError`` from ``SRTPlatform``.
        """
        owner = _defining_class(self.TpuOOTPlatform, "get_dsa_kv_pool_cls")
        self.assertIn(owner, self.tpu_owned)

        with self.assertRaises(NotImplementedError) as caught:
            self.platform.get_dsa_kv_pool_cls()
        message = str(caught.exception)
        self.assertTrue(message.strip(), "refusal must explain itself, not raise bare")
        self.assertIn("TPU", message)

    def test_nsa_kv_pool_cls_is_also_a_refusal(self):
        """Pins the premise of the test above: neither spelling has a pool."""
        with self.assertRaises(NotImplementedError):
            self.platform.get_nsa_kv_pool_cls()

    # -- speculative decoding ---------------------------------------------

    def test_supported_speculative_algorithms_are_real_algorithms(self):
        """Catches an upstream rename of a SpeculativeAlgorithm member."""
        from sglang_torchtpu.platform import _SUPPORTED_SPEC_ALGORITHMS

        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        known = {member.name for member in SpeculativeAlgorithm}
        self.assertTrue(
            _SUPPORTED_SPEC_ALGORITHMS <= known,
            f"unknown algorithms claimed: {sorted(_SUPPORTED_SPEC_ALGORITHMS - known)}",
        )

    def test_dspark_is_deliberately_not_claimed(self):
        """DSPARK is in flight (fork PR #707); a capability that lies is worse
        than a missing one. Delete this test when it lands and is tested."""
        self.assertFalse(self.platform.supports_speculative_algorithm("DSPARK"))

    def test_claimed_draft_backends_are_registered_by_draft_utils(self):
        """The two pallas backends must be keys draft_utils actually dispatches."""
        from sglang_torchtpu.platform import _SUPPORTED_DRAFT_ATTENTION_BACKENDS

        registered = self._draft_backend_keys()
        missing = _SUPPORTED_DRAFT_ATTENTION_BACKENDS - registered
        self.assertEqual(
            missing,
            set(),
            f"claimed draft backends draft_utils does not register: {sorted(missing)}",
        )

    def test_default_draft_backend_is_one_we_claim(self):
        from sglang_torchtpu.platform import (
            _DEFAULT_DRAFT_ATTENTION_BACKEND,
            _SUPPORTED_DRAFT_ATTENTION_BACKENDS,
        )

        self.assertIn(
            _DEFAULT_DRAFT_ATTENTION_BACKEND, _SUPPORTED_DRAFT_ATTENTION_BACKENDS
        )
        self.assertEqual(
            self.platform.get_default_speculative_draft_attention_backend("EAGLE3"),
            _DEFAULT_DRAFT_ATTENTION_BACKEND,
        )

    def test_default_draft_backend_matches_the_core_default(self):
        """speculative_hook.py picks pallas_tpu_rpa_v3 when is_tpu(); stay in step."""
        from sglang_torchtpu.platform import _DEFAULT_DRAFT_ATTENTION_BACKEND

        source = self._read_core("arg_groups/speculative_hook.py")
        self.assertIn(f'"{_DEFAULT_DRAFT_ATTENTION_BACKEND}"', source)

    def test_speculative_queries_reject_unsupported_inputs(self):
        self.assertFalse(self.platform.supports_speculative_algorithm("NGRAM"))
        self.assertTrue(self.platform.supports_speculative_algorithm("eagle3"))
        self.assertFalse(
            self.platform.supports_speculative_draft_attention_backend(
                "EAGLE3", "flashinfer"
            )
        )
        self.assertFalse(
            self.platform.supports_speculative_draft_attention_backend(
                "NGRAM", "pallas_tpu_rpa_v3"
            )
        )
        with self.assertRaises(ValueError):
            self.platform.get_default_speculative_draft_attention_backend("NGRAM")

    def test_speculative_cache_locs_fn_is_not_overridden(self):
        """Returning one here while core still branches on `_is_tpu` would give
        two implementations of the same decision. Revisit when speculative/ moves."""
        self.assertIsNone(self.platform.get_speculative_cache_locs_fn())

    # -- activation --------------------------------------------------------

    def test_activate_declines_unless_selected_by_name(self):
        from sglang_torchtpu import activate

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SGLANG_PLATFORM", None)
            self.assertIsNone(activate())
        with patch.dict(os.environ, {"SGLANG_PLATFORM": "tpu"}):
            self.assertIsNone(activate())

    def test_advertised_class_path_resolves_to_this_class(self):
        import pkgutil

        from sglang_torchtpu import PLATFORM_CLASS_PATH

        self.assertIs(pkgutil.resolve_name(PLATFORM_CLASS_PATH), self.TpuOOTPlatform)

    def test_entry_point_name_matches_the_packaging_metadata(self):
        from sglang_torchtpu import ENTRY_POINT_NAME

        pyproject = os.path.join(_PLUGIN_ROOT, "pyproject.toml")
        with open(pyproject, "r", encoding="utf-8") as handle:
            content = handle.read()
        self.assertIn(f"\n{ENTRY_POINT_NAME} = ", content)

    def test_does_not_shadow_the_in_tree_entry_point(self):
        """Wave 0 leaves the default path untouched; `tpu` stays the in-tree one."""
        with open(
            os.path.join(_REPO_ROOT, "python", "pyproject.toml"), "r", encoding="utf-8"
        ) as handle:
            content = handle.read()
        self.assertIn(
            'tpu = "sglang.srt.hardware_backend.tpu:activate"',
            content,
        )

    # -- hooks are debt ----------------------------------------------------

    def test_no_hooks_are_registered(self):
        """Wave 0 ships zero monkeypatches. Adding one requires the documented
        header, and `tools/plugin/check_tpu_branches.py` counts what lands here."""
        from sglang_torchtpu.hooks import HOOKS

        self.assertEqual(HOOKS, [], "a hook was added without updating this test")

    # -- helpers -----------------------------------------------------------

    def _read_core(self, relpath: str) -> str:
        path = os.path.join(_REPO_ROOT, "python", "sglang", "srt", relpath)
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()

    def _draft_backend_keys(self) -> set:
        """String keys of the backend dispatch dicts in speculative/draft_utils.py.

        Parsed rather than imported: importing the speculative stack pulls model
        and attention code this CPU gate has no reason to load.
        """
        tree = ast.parse(self._read_core("speculative/draft_utils.py"))
        keys = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                for key in node.keys:
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        keys.add(key.value)
        return keys


class TestTpuDebtRatchet(CustomTestCase):
    """``tools/plugin/check_tpu_branches.py`` must actually catch an increase.

    The ratchet is the only thing stopping the shared-core TPU pile from growing
    while waves 1-3 drain it, and its failure mode is a silent pass: a predicate
    that stops matching, or a comparison that comes out empty, reads exactly like
    a clean tree. These pin the decisions it makes.
    """

    def test_tpu_is_recognised_only_as_a_whole_path_word(self):
        """`output.py` contains the letters t-p-u; it is not a TPU module."""
        self.assertTrue(
            tpu_debt.names_tpu("python/sglang/srt/model_executor/tpu_graph_runner.py")
        )
        self.assertTrue(
            tpu_debt.names_tpu("python/sglang/srt/disaggregation/tpu/wire.py")
        )
        self.assertTrue(
            tpu_debt.names_tpu("python/sglang/multimodal_gen/runtime/platforms/tpu.py")
        )
        self.assertFalse(
            tpu_debt.names_tpu("python/sglang/srt/managers/auxiliary_output.py")
        )
        self.assertFalse(tpu_debt.names_tpu("python/sglang/srt/beam_search/output.py"))

    def test_imports_are_not_counted_as_branches(self):
        """Importing the symbol is how a file gets it, not a place it branches."""
        source = "from sglang.srt.utils import is_tpu\nimport sglang.srt.utils as u\n"
        self.assertEqual(self._count(source), 0)

    def test_calls_and_attribute_reads_are_counted(self):
        source = (
            "from sglang.srt.utils import is_tpu\n"
            "def f(p):\n"
            "    if is_tpu():\n"
            "        return p.is_tpu()\n"
            "    return _is_tpu\n"
        )
        self.assertEqual(self._count(source), 3)

    def test_an_added_branch_is_a_violation(self):
        baseline = tpu_debt.Debt(branches={"python/sglang/srt/a.py": 2})
        current = tpu_debt.Debt(branches={"python/sglang/srt/a.py": 3})
        violations, _ = tpu_debt.compare(current, baseline, partial=False)
        self.assertEqual(len(violations), 1)
        self.assertIn("baseline 2", violations[0])

    def test_a_branch_in_a_previously_clean_file_is_a_violation(self):
        current = tpu_debt.Debt(branches={"python/sglang/srt/new.py": 1})
        violations, _ = tpu_debt.compare(current, tpu_debt.Debt(), partial=False)
        self.assertEqual(len(violations), 1)

    def test_a_removed_branch_is_an_improvement_not_a_violation(self):
        baseline = tpu_debt.Debt(branches={"python/sglang/srt/a.py": 2})
        current = tpu_debt.Debt(branches={"python/sglang/srt/a.py": 1})
        violations, improvements = tpu_debt.compare(current, baseline, partial=False)
        self.assertEqual(violations, [])
        self.assertEqual(len(improvements), 1)

    def test_a_new_tpu_module_is_a_violation_even_under_files_mode(self):
        """pre-commit runs `--files`; the module check must survive that path."""
        current = tpu_debt.Debt(modules=["python/sglang/srt/layers/tpu_thing.py"])
        for partial in (False, True):
            violations, _ = tpu_debt.compare(current, tpu_debt.Debt(), partial=partial)
            self.assertEqual(len(violations), 1, f"partial={partial}")

    def test_a_missing_file_is_not_read_as_progress_under_files_mode(self):
        """Under `--files` an unstaged file is simply unexamined."""
        baseline = tpu_debt.Debt(
            branches={"python/sglang/srt/a.py": 2},
            modules=["python/sglang/srt/b/tpu.py"],
        )
        _, improvements = tpu_debt.compare(tpu_debt.Debt(), baseline, partial=True)
        self.assertEqual(improvements, [])

    def test_committed_baseline_matches_the_tree(self):
        """Catches a baseline edited by hand, or a stale one after a sync."""
        current = tpu_debt.collect(_REPO_ROOT)
        baseline = tpu_debt.load_baseline(_REPO_ROOT)
        violations, _ = tpu_debt.compare(current, baseline, partial=False)
        self.assertEqual(
            violations,
            [],
            "run `python tools/plugin/check_tpu_branches.py --write-baseline`",
        )

    def _count(self, source: str) -> int:
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
            handle.write(source)
            path = handle.name
        try:
            return tpu_debt.count_tpu_branches(path)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
