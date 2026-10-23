"""CPU numerical and request-lifetime coverage for host Engram staging."""

import ast
import importlib.util
import mmap
import os
import sys
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.engram import (
    HostEngramCoordinator,
    HostEngramInputs,
    PendingTokens,
    capture_inputs,
    hash_on_host,
    pack_rows,
    unpack_rows,
)
from sglang_tpu.result_copy_worker import TpuAsyncCopyEvent

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[5]


def _request(prompt, output=()):
    return SimpleNamespace(origin_input_ids=list(prompt), output_ids=list(output))


def _tables():
    return dict(
        token_map=np.asarray([3, 1, 6, 0, 2, 5, 4, 7], dtype=np.int64),
        multipliers=np.asarray(
            [[100000000000000003, 13, 31], [7, 99999999999999999, 19]],
            dtype=np.int64,
        ),
        primes=np.asarray([[[11, 13], [17, 19]], [[23, 29], [31, 37]]]),
        offsets=np.asarray([[0, 11, 24, 41], [0, 23, 52, 83]]),
        pad_id=3,
    )


def _scalar_hash(inputs, tables):
    """Independent Python-integer oracle; no NumPy broadcasting or int casts."""
    out = []
    for tokens, position in zip(inputs.tokens, inputs.positions, strict=True):
        layer_rows = []
        for layer, multipliers in enumerate(tables["multipliers"]):
            rolling = 0
            row = []
            for shift, multiplier in enumerate(multipliers):
                mapped = (
                    tables["pad_id"]
                    if shift > position
                    else tables["token_map"][tokens[shift]]
                )
                rolling ^= int(mapped) * int(multiplier)
                if shift:
                    for head, prime in enumerate(tables["primes"][layer, shift - 1]):
                        column = (shift - 1) * 2 + head
                        row.append(
                            rolling % int(prime) + tables["offsets"][layer, column]
                        )
            layer_rows.append(row)
        out.append(layer_rows)
    return torch.tensor(out, dtype=torch.int64)


def test_hash_matches_integer_oracle_and_chunked_decode():
    req = _request([1, 2, 3, 4, 5, 6])
    whole = capture_inputs([req], [0], [6], 3)
    expected = _scalar_hash(whole, _tables())
    torch.testing.assert_close(hash_on_host(whole, **_tables()), expected)
    parts = [
        capture_inputs([req], [start], [length], 3)
        for start, length in [(0, 2), (2, 3), (5, 1)]
    ]
    actual = torch.cat([hash_on_host(part, **_tables()) for part in parts])
    torch.testing.assert_close(actual, expected)
    assert req.output_ids == []


def test_hash_empty_and_invalid_token():
    empty = capture_inputs([], [], [], 3)
    assert hash_on_host(empty, **_tables()).shape == (0, 2, 4)
    with pytest.raises(ValueError, match="outside"):
        hash_on_host(
            HostEngramInputs(np.asarray([[8, 0, 0]]), np.asarray([0])), **_tables()
        )


def test_decode_uses_latest_sample_after_copy_and_respects_reordering():
    a, b = _request([1, 2], [3]), _request([4, 5])
    result = SimpleNamespace(next_token_ids=None)
    event = Mock()

    def complete():
        result.next_token_ids = torch.tensor([6, 7])

    event.synchronize.side_effect = complete
    result.copy_done = event
    pending = PendingTokens([a, b], [3, 2], result)
    inputs = capture_inputs([b, a], [2, 3], [1, 1], 3, pending)
    np.testing.assert_array_equal(inputs.tokens, [[7, 5, 4], [6, 3, 2]])
    event.synchronize.assert_called_once()
    assert a.output_ids == [3] and b.output_ids == []


def test_result_submission_and_transfer_both_precede_host_hashing():
    req = _request([1, 2])
    submitted = Future()
    waiting, transferred = Event(), Event()
    result = SimpleNamespace(next_token_ids=None)

    def synchronize():
        waiting.set()
        assert transferred.wait(5)
        result.next_token_ids = torch.tensor([3])

    event = SimpleNamespace(synchronize=synchronize)
    result.copy_done = TpuAsyncCopyEvent(event, submitted)
    pending = PendingTokens([req], [2], result)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(capture_inputs, [req], [2], [1], 3, pending)
        try:
            assert not waiting.is_set()
            assert not future.done()
            submitted.set_result(None)
            assert waiting.wait(5)
            assert not future.done()
            transferred.set()
            np.testing.assert_array_equal(future.result(5).tokens, [[3, 2, 1]])
        finally:
            if not submitted.done():
                submitted.set_result(None)
            transferred.set()


def test_copy_failure_propagates():
    req = _request([1])
    future = Future()
    future.set_exception(RuntimeError("copy failed"))
    result = SimpleNamespace(
        copy_done=TpuAsyncCopyEvent(Mock(), future), next_token_ids=None
    )
    with pytest.raises(RuntimeError, match="copy failed"):
        capture_inputs([req], [1], [1], 3, PendingTokens([req], [1], result))


def test_slot_reuse_or_wrong_position_cannot_consume_old_result():
    old, new = _request([1, 2]), _request([1, 2])
    old.rid = new.rid = "reused"
    old.req_pool_idx = new.req_pool_idx = 4
    result = SimpleNamespace(copy_done=Mock(), next_token_ids=torch.tensor([7]))
    pending = PendingTokens([old], [2], result)
    with pytest.raises(RuntimeError, match="position 2"):
        capture_inputs([new], [2], [1], 3, pending)
    with pytest.raises(RuntimeError, match="position 3"):
        pending.read(old, 3)
    result.copy_done.synchronize.assert_not_called()


def test_chunked_prefill_ignores_pseudo_sample_and_uses_batch_prefix():
    req = _request([1, 2, 3, 4, 5, 6])
    # Stale request range must not override the current batch's prefix.
    req.extend_range = SimpleNamespace(start=0)
    result = SimpleNamespace(copy_done=Mock(), next_token_ids=torch.tensor([7]))
    pending = PendingTokens([req], [2], result)
    inputs = capture_inputs([req], [2], [2], 3, pending)
    np.testing.assert_array_equal(inputs.tokens, [[3, 2, 1], [4, 3, 2]])
    result.copy_done.synchronize.assert_not_called()


def test_snapshot_does_not_alias_request_history():
    req = _request([1, 2], [3, 4])
    inputs = capture_inputs([req], [3], [1], 3)
    req.output_ids[:] = [7, 7]
    np.testing.assert_array_equal(inputs.tokens, [[4, 3, 2]])


def _table(rows=7, dim=64):
    generator = torch.Generator().manual_seed(42)
    weight = (torch.randn(rows, dim, generator=generator) * 2).to(torch.float8_e4m3fn)
    scale = torch.randint(
        123, 131, (rows, dim // 32), generator=generator, dtype=torch.uint8
    )
    return weight, scale


@pytest.mark.parametrize("padded_tokens", [2, 4, 8])
def test_packed_lookup_matches_reference_and_masks_unowned_rows(padded_tokens):
    weight, scale = _table()
    ids = torch.tensor([[100, 106, -1], [103, 2**62, 99]])
    packed = pack_rows(weight, scale, ids, row_start=100, padded_tokens=padded_tokens)
    assert packed.ndim == 1 and packed.dtype == torch.int32
    result = unpack_rows(packed, tokens=padded_tokens, heads=3, dim=64)
    expected = torch.zeros(padded_tokens, 3, 64, dtype=torch.bfloat16)
    for t in range(2):
        for h in range(3):
            row = int(ids[t, h]) - 100
            if 0 <= row < len(weight):
                values = weight[row].float()
                powers = torch.exp2(scale[row].float() - 127).repeat_interleave(32)
                expected[t, h] = (values * powers).to(torch.bfloat16)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_invalid_scale_masks_nan_and_scale_zero_matches_vllm_flush():
    weight = torch.full((1, 32), float("nan")).to(torch.float8_e4m3fn)
    scale = torch.tensor([[127]], dtype=torch.uint8)
    packed = pack_rows(
        weight, scale, torch.tensor([[-1]]), row_start=0, padded_tokens=2
    )
    result = unpack_rows(packed, tokens=2, heads=1, dim=32)
    assert torch.equal(result, torch.zeros_like(result))
    weight.fill_(1)
    scale.zero_()
    packed = pack_rows(weight, scale, torch.tensor([[0]]), row_start=0, padded_tokens=1)
    assert unpack_rows(packed, tokens=1, heads=1, dim=32).count_nonzero() == 0


def test_empty_shard_and_empty_batch():
    for tokens in (0, 3):
        packed = pack_rows(
            torch.empty(0, 32, dtype=torch.float8_e4m3fn),
            torch.empty(0, 1, dtype=torch.uint8),
            torch.empty(0, 2, dtype=torch.int64),
            row_start=0,
            padded_tokens=tokens,
        )
        result = unpack_rows(packed, tokens=tokens, heads=2, dim=32)
        assert result.shape == (tokens, 2, 32)
        assert result.count_nonzero() == 0


def test_tensor_parallel_shards_reassemble_the_same_embedding():
    weight, scale = _table()
    ids = torch.tensor([[0, 6], [3, 4]])
    expected = unpack_rows(
        pack_rows(weight, scale, ids, row_start=0, padded_tokens=2),
        tokens=2,
        heads=2,
        dim=64,
    )
    shards = []
    for start, stop in [(0, 2), (2, 4), (4, 7)]:
        words = pack_rows(
            weight[start:stop], scale[start:stop], ids, row_start=start, padded_tokens=2
        )
        shards.append(unpack_rows(words, tokens=2, heads=2, dim=64))
    torch.testing.assert_close(sum(shards), expected, rtol=0, atol=0)


def test_fullgraph_reads_updated_staging_buffer_after_warmup():
    weight, scale = _table()
    heads, tokens = 2, 4

    class Lookup(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer(
                "packed",
                pack_rows(
                    weight,
                    scale,
                    torch.empty(0, heads, dtype=torch.int64),
                    row_start=0,
                    padded_tokens=tokens,
                ),
            )

        def forward(self):
            return unpack_rows(self.packed, tokens=tokens, heads=heads, dim=64)

    module = Lookup()
    # This checks graph capture and buffer reuse on CPU, not TPU lowering.
    compiled = torch.compile(module, backend="eager", fullgraph=True)
    assert compiled().count_nonzero() == 0
    for ids in (torch.tensor([[0, 6]]), torch.tensor([[3, 4], [1, 2]])):
        words = pack_rows(weight, scale, ids, row_start=0, padded_tokens=tokens)
        module.packed.copy_(words)
        torch.testing.assert_close(
            compiled(),
            unpack_rows(words, tokens=tokens, heads=heads, dim=64),
            rtol=0,
            atol=0,
        )


def _coordinator():
    tables = _tables()
    hasher = SimpleNamespace(
        max_ngram_size=3,
        **{k: torch.as_tensor(v) for k, v in tables.items() if k != "pad_id"},
        pad_id=tables["pad_id"],
    )
    weight, scale = _table(rows=130)
    embedding = SimpleNamespace(
        weight=weight, scale=scale, row_start=0, _tpu_engram_buffers={}
    )
    return HostEngramCoordinator(hasher, [(0, embedding)], "cpu"), embedding


def _batch(requests, *, decode=False, starts=None, lengths=None):
    mode = SimpleNamespace(
        is_decode=lambda: decode,
        is_extend_without_speculative=lambda: not decode,
        is_idle=lambda: False,
    )
    return SimpleNamespace(
        reqs=requests,
        spec_algorithm=SimpleNamespace(is_none=lambda: True),
        forward_mode=mode,
        prefix_lens=starts,
        extend_lens=lengths,
        seq_lens_cpu=torch.tensor([s + 1 for s in starts]) if decode else None,
    )


def test_coordinator_warmup_preserves_inflight_buffer_and_rejects_missing_step():
    coordinator, embedding = _coordinator()
    coordinator.stage(4)
    warmup = unpack_rows(embedding._tpu_engram_packed, tokens=4, heads=4, dim=64)
    assert warmup.count_nonzero() == 0
    req = _request([1, 2, 3])
    coordinator.prepare(_batch([req], starts=[0], lengths=[3]))
    coordinator.stage(4)
    first = embedding._tpu_engram_packed
    first_data = first.clone()
    coordinator.clear_step()
    with pytest.raises(RuntimeError, match="no prepared"):
        coordinator.stage(4)
    coordinator.prepare(_batch([req], starts=[1], lengths=[1]))
    coordinator.stage(8)
    assert embedding._tpu_engram_packed.shape != first.shape
    coordinator.stage(4)
    assert embedding._tpu_engram_packed is not first
    torch.testing.assert_close(first_data, first)
    assert not torch.equal(first_data, embedding._tpu_engram_packed)


def test_coordinator_prefill_then_overlapped_decode():
    coordinator, embedding = _coordinator()
    req = _request([1, 2, 3])
    requests, positions = coordinator.prepare(_batch([req], starts=[0], lengths=[3]))
    coordinator.stage(4)
    coordinator.remember(
        requests,
        positions,
        SimpleNamespace(copy_done=Mock(), next_token_ids=torch.tensor([4])),
    )
    coordinator.clear_step()
    coordinator.prepare(_batch([req], decode=True, starts=[3]))
    expected = hash_on_host(
        capture_inputs([_request([1, 2, 3, 4])], [3], [1], 3), **_tables()
    )
    torch.testing.assert_close(coordinator.hashes, expected)
    coordinator.stage(2)
    result = unpack_rows(embedding._tpu_engram_packed, tokens=2, heads=4, dim=64)
    assert result[1].count_nonzero() == 0


def _load_hooks():
    # Load the leaf adapter without booting unrelated TPU plugin hooks.
    name = "_host_engram_test_hooks"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "sglang_tpu/hooks/engram.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_core_embedding(monkeypatch, hooks, parallel):
    from sglang.srt.environ import envs

    runtime = ModuleType("sglang.srt.runtime_context")
    runtime.get_parallel = lambda: parallel
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    # Execute the core constructors and checkpoint methods without importing
    # unrelated GPU ops. Host mapping is replaced by the actual TPU hook.
    path = ROOT / "python/sglang/srt/layers/engram.py"
    tree = ast.parse(path.read_text())
    classes = [
        node
        for node in tree.body
        if getattr(node, "name", None) in ("_HostTable", "EngramEmbedding")
    ]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            *classes,
        ],
        type_ignores=[],
    )
    namespace = {
        "nn": torch.nn,
        "torch": torch,
        "os": os,
        "mmap": mmap,
        "envs": envs,
        "get_parallel": lambda: parallel,
        "FP8_BLOCK_SIZE": 32,
        "logger": Mock(),
        # Huge-page advice is Linux-specific and outside these layout tests.
        "_huge_pages_backing": lambda ptr: (1, 1),
    }
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    table = namespace["_HostTable"]
    original = table.__init__
    monkeypatch.setattr(
        table,
        "__init__",
        lambda self, *args: hooks.init_host_table(original, self, *args),
    )
    return namespace["EngramEmbedding"], table


def test_existing_host_flag_controls_adapters_and_disabled_path(monkeypatch):
    hooks = _load_hooks()
    monkeypatch.delenv("SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE", raising=False)
    assert not hooks._enabled()
    assert hooks.hook_specs() == []
    original, embedding = Mock(return_value="core"), object()
    assert hooks.init_embedding(original, embedding, 7, 64, 2) == "core"
    original.assert_called_once_with(embedding, 7, 64, 2)
    monkeypatch.setenv("SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE", "1")
    assert hooks._enabled()
    assert len(hooks.hook_specs()) == 10
    assert any(
        target == "sglang.srt.layers.engram.engram_gate" and hook is hooks.gate_forward
        for target, hook, _ in hooks.hook_specs()
    )
    monkeypatch.setenv("SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE", "0")
    assert not hooks._enabled()


def test_embedding_adapter_rejects_row_shards_before_allocation(monkeypatch):
    hooks = _load_hooks()
    monkeypatch.setenv("SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE", "1")
    monkeypatch.setenv("SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT", "per_rank")
    parallel = SimpleNamespace(
        pp_size=1, attn_dp_size=8, attn_cp_size=1, attn_dcp_size=1
    )
    module = ModuleType("sglang.srt.runtime_context")
    module.get_parallel = lambda: parallel
    monkeypatch.setitem(sys.modules, module.__name__, module)
    original = Mock()
    with pytest.raises(ValueError, match="shared full host table"):
        hooks.init_embedding(original, SimpleNamespace(), 7, 64, 2)
    original.assert_not_called()


def test_default_shared_layout_loads_each_rank_and_looks_up_without_reduce(monkeypatch):
    hooks = _load_hooks()
    monkeypatch.setenv("SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE", "1")
    monkeypatch.delenv("SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT", raising=False)
    parallel = SimpleNamespace(
        tp_size=2,
        tp_rank=0,
        pp_size=1,
        attn_dp_size=2,
        attn_cp_size=1,
        attn_dcp_size=1,
        tp_group=Mock(),
    )
    cls, table = _load_core_embedding(monkeypatch, hooks, parallel)
    weight, scale = _table()
    embeddings = []
    # macOS has no memfd_create; two real shared mmap views still exercise
    # shared storage and row offsets. Linux uses the core memfd helper.
    with tempfile.TemporaryFile() as backing:
        backing.truncate(weight.numel() + scale.numel())
        monkeypatch.setattr(
            table,
            "_open_shared_fd",
            lambda self, nbytes, name: os.dup(backing.fileno()),
        )
        try:
            for rank in range(2):
                parallel.tp_rank = rank
                embedding = cls.__new__(cls)
                hooks.init_embedding(cls.__init__, embedding, 7, 64, 2)
                embeddings.append(embedding)
                assert embedding._shared
                embedding.weight.weight_loader(embedding.weight, weight)
                embedding.scale.weight_loader(
                    embedding.scale, scale.view(torch.float8_e8m0fnu)
                )
            ids = torch.tensor([[0, 6, 3]])
            expected = unpack_rows(
                pack_rows(weight, scale, ids, row_start=0, padded_tokens=2),
                tokens=2,
                heads=3,
                dim=64,
            )
            for embedding in embeddings:
                embedding.finish_load("test")
                torch.testing.assert_close(
                    embedding.weight.view(torch.uint8), weight.view(torch.uint8)
                )
                coordinator, _ = _coordinator()
                coordinator.embeddings = [(0, embedding)]
                coordinator.by_layer = {0: embedding}
                coordinator.hashes = ids[:, None, :]
                coordinator.stage(2)
                embedding._reduce_owned_rows = Mock()
                actual = hooks.embedding_forward(Mock(), embedding, ids.expand(2, -1))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                embedding._reduce_owned_rows.assert_not_called()
            assert parallel.tp_group.barrier.call_count == 4
        finally:
            for embedding in embeddings:
                os.close(embedding.host_table.fd)


def test_unknown_upstream_host_layout_is_rejected(monkeypatch):
    hooks = _load_hooks()
    monkeypatch.setenv("SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE", "1")
    with pytest.raises(ValueError, match="SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT"):
        hooks.init_host_table(Mock(), SimpleNamespace(), "unknown", 16, "test", Mock())


def test_worker_clears_step_after_failure_and_disabled_path_is_unchanged():
    hooks = _load_hooks()
    original = Mock(return_value="unchanged")
    worker = SimpleNamespace(model_runner=SimpleNamespace())
    assert hooks.worker_forward(original, worker, None) == "unchanged"
    coordinator, _ = _coordinator()
    worker.model_runner.host_engram = coordinator
    original.side_effect = RuntimeError("forward failed")
    with pytest.raises(RuntimeError, match="forward failed"):
        hooks.worker_forward(
            original, worker, _batch([_request([1])], starts=[0], lengths=[1])
        )
    assert coordinator.hashes is None
    assert coordinator.pending is None


def test_hook_targets_and_checkpoint_block_size_match_current_source():
    """Catch integration drift without importing the model's CUDA modules."""
    tree = ast.parse((ROOT / "sglang_tpu/hooks/engram.py").read_text())
    targets = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Tuple) and len(node.elts) == 3:
            target, hook, _ = node.elts
            if (
                isinstance(target, ast.Constant)
                and isinstance(target.value, str)
                and target.value.startswith("sglang.")
            ):
                targets.append(ast.literal_eval(target))
    assert len(targets) == 10
    for target in targets:
        parts = target.split(".")
        for split in range(len(parts) - 1, 0, -1):
            path = ROOT / "python" / ("/".join(parts[:split]) + ".py")
            if path.exists():
                node = ast.parse(path.read_text())
                for name in parts[split:]:
                    node = next(
                        child
                        for child in node.body
                        if getattr(child, "name", None) == name
                    )
                break
        else:
            pytest.fail(f"Hook target does not exist: {target}")
    source = ast.parse(
        (ROOT / "python/sglang/kernels/ops/attention/dsv4/torch_quant.py").read_text()
    )
    block_size = next(
        ast.literal_eval(n.value)
        for n in source.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "FP8_BLOCK_SIZE" for t in n.targets)
    )
    assert block_size == 32
