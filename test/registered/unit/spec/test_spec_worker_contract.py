"""Every spec worker and relayed draft input exposes what core reads from it.

A worker the platform swaps in through ``SRTPlatform.get_spec_worker_cls`` is
reached only through ``SpecWorkerContract``, and its ``next_draft_input`` only
through ``RelayedDraftInput``; a missing member fails at the first scheduler
step that reads it, on the platform that owns the worker. The checks are
class-level, so nothing is constructed.
"""

import ast
import dataclasses
import inspect
import unittest
from pathlib import Path

from sglang.srt.speculative import spec_info
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (
    DSparkWorkerV2,
)
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2
from sglang.srt.speculative.frozen_kv_mtp_info import FrozenKVMTPDraftInput
from sglang.srt.speculative.frozen_kv_mtp_worker_v2 import FrozenKVMTPWorkerV2
from sglang.srt.speculative.multi_layer_eagle_worker_v2 import (
    MultiLayerEagleWorkerV2,
)
from sglang.srt.speculative.spec_worker_contract import (
    SPEC_WORKER_INIT_KWARGS,
    RelayedDraftInput,
    SpecWorkerContract,
)
from sglang.srt.speculative.standalone_worker_v2 import StandaloneWorkerV2
from sglang.srt.speculative.uno_info import UnoDraftInput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.experimental.speculative.base_spec_worker import TpuSpecWorker
from sglang_tpu.experimental.speculative.block_spec_worker import (
    TpuBlockSpecWorker,
)
from sglang_tpu.experimental.speculative.dflash_info import HostPaddedDraftInput

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

CORE_WORKERS = (
    EAGLEWorkerV2,
    MultiLayerEagleWorkerV2,
    StandaloneWorkerV2,
    FrozenKVMTPWorkerV2,
    DFlashWorkerV2,
    DSparkWorkerV2,
)
# Their modules import CUDA-only kernels (sgl_kernel, flashinfer) at load time.
CORE_WORKERS_NOT_IMPORTABLE_ON_CPU = frozenset({"NGRAMWorker", "UnoWorkerV2"})
PLUGIN_WORKERS = (TpuSpecWorker, TpuBlockSpecWorker)
# The plugin's abstract skeleton; only its subclasses are handed to the scheduler.
NOT_CONSTRUCTED_BY_SCHEDULER = frozenset({TpuSpecWorker})
# check_pipeline_parallel_compat admits only EAGLE under PP, so the scheduler
# never passes these workers pp_proxy_tensors.
PP_INCAPABLE_WORKERS = frozenset({FrozenKVMTPWorkerV2})

DRAFT_INPUTS = (
    EagleDraftInput,
    FrozenKVMTPDraftInput,
    DFlashDraftInputV2,
    UnoDraftInput,
    HostPaddedDraftInput,
)

_PROTOCOL_INTERNALS = frozenset({"_is_protocol", "_is_runtime_protocol", "_abc_impl"})


def _protocol_members(protocol: type) -> set[str]:
    names = {
        name
        for name in vars(protocol)
        if not (name.startswith("__") and name.endswith("__"))
    }
    return (names | set(protocol.__annotations__)) - _PROTOCOL_INTERNALS


def _class_has(cls: type, name: str) -> bool:
    """Class attribute, method, property, or a dataclass field without a default."""
    if hasattr(cls, name):
        return True
    return dataclasses.is_dataclass(cls) and any(
        field.name == name for field in dataclasses.fields(cls)
    )


def _keyword_callable(function, names) -> list[str]:
    """Names in ``names`` that ``function`` does not accept as a keyword."""
    parameters = inspect.signature(function).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return []
    return [
        name
        for name in names
        if name not in parameters
        or parameters[name].kind is inspect.Parameter.POSITIONAL_ONLY
    ]


def _default_worker_class_names() -> set[str]:
    """Worker classes ``SpeculativeAlgorithm._default_worker_cls`` can return."""
    tree = ast.parse(Path(spec_info.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_default_worker_cls":
            return {
                alias.name
                for child in ast.walk(node)
                if isinstance(child, ast.ImportFrom)
                for alias in child.names
            }
    raise AssertionError("_default_worker_cls not found in spec_info.py")


class TestSpecWorkerContract(CustomTestCase):
    def test_every_core_default_worker_is_checked(self):
        checked = {cls.__name__ for cls in CORE_WORKERS}
        self.assertEqual(
            _default_worker_class_names(),
            checked | CORE_WORKERS_NOT_IMPORTABLE_ON_CPU,
        )

    def test_workers_expose_every_member_core_reads(self):
        members = _protocol_members(SpecWorkerContract)
        self.assertIn("forward_batch_generation", members)
        for cls in CORE_WORKERS + PLUGIN_WORKERS:
            with self.subTest(worker=cls.__name__):
                missing = sorted(name for name in members if not _class_has(cls, name))
                self.assertEqual(missing, [])

    def test_workers_take_the_calls_the_scheduler_makes(self):
        for cls in CORE_WORKERS + PLUGIN_WORKERS:
            with self.subTest(worker=cls.__name__):
                if cls not in NOT_CONSTRUCTED_BY_SCHEDULER:
                    self.assertEqual(
                        _keyword_callable(cls.__init__, SPEC_WORKER_INIT_KWARGS), []
                    )
                forward_kwargs = ("on_publish", "grammar_barrier")
                if cls not in PP_INCAPABLE_WORKERS:
                    forward_kwargs += ("pp_proxy_tensors",)
                self.assertEqual(
                    _keyword_callable(cls.forward_batch_generation, forward_kwargs),
                    [],
                )


class TestRelayedDraftInputContract(CustomTestCase):
    def test_draft_inputs_expose_every_member_core_reads(self):
        members = _protocol_members(RelayedDraftInput)
        self.assertIn("future_indices", members)
        for cls in DRAFT_INPUTS:
            with self.subTest(draft_input=cls.__name__):
                missing = sorted(name for name in members if not _class_has(cls, name))
                self.assertEqual(missing, [])

    def test_draft_inputs_take_the_filter_call_schedule_batch_makes(self):
        for cls in DRAFT_INPUTS:
            with self.subTest(draft_input=cls.__name__):
                self.assertEqual(
                    _keyword_callable(
                        cls.filter_batch, ("new_indices", "new_indices_cpu")
                    ),
                    [],
                )


if __name__ == "__main__":
    unittest.main()
