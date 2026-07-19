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

import torch

import sglang.srt.speculative.dflash_utils as du
from sglang.srt.speculative.dflash_utils import (
    validate_dflash_draft_startup,
    validate_dflash_request,
)
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


def _valid_dflash_startup_kwargs(**overrides):
    """A canonical-shaped (Llama-3.1-8B + z-lab DFlash draft) compatible set."""
    kwargs = dict(
        target_vocab_size=128256,
        target_hidden_size=4096,
        target_num_layers=32,
        draft_hidden_size=4096,
        draft_num_layers=1,
        draft_total_num_heads=32,
        draft_total_num_kv_heads=8,
        tp_size=4,
        block_size=8,
        target_layer_ids=[0, 15, 31],
        mask_token_id=128002,
    )
    kwargs.update(overrides)
    return kwargs


class TestValidateDflashDraftStartup(CustomTestCase):
    """Aggregate startup draft/target compatibility validation
    (``validate_dflash_draft_startup``): every failure returns a clear message.
    """

    def _errs(self, **overrides):
        return validate_dflash_draft_startup(**_valid_dflash_startup_kwargs(**overrides))

    def test_compatible_config_has_no_errors(self):
        self.assertEqual(self._errs(), [])

    def test_none_mask_token_id_allowed(self):
        # A missing config mask_token_id is NOT an error: the worker resolves it
        # from the tokenizer (preserves the non-TPU tokenizer-fallback path).
        self.assertEqual(self._errs(mask_token_id=None), [])

    def test_mask_token_id_out_of_vocab(self):
        errs = self._errs(mask_token_id=128256)  # == vocab_size -> out of range
        self.assertTrue(any("out of range" in e for e in errs), errs)

    def test_hidden_size_mismatch(self):
        errs = self._errs(draft_hidden_size=2048)
        self.assertTrue(any("hidden_size" in e for e in errs), errs)

    def test_nonpositive_draft_layers(self):
        self.assertTrue(any("num_hidden_layers" in e for e in self._errs(draft_num_layers=0)))
        self.assertTrue(
            any("num_hidden_layers" in e for e in self._errs(draft_num_layers=None))
        )

    def test_nonpositive_block_size(self):
        errs = self._errs(block_size=0)
        self.assertTrue(any("block_size" in e for e in errs), errs)
        # None block_size is allowed (resolved elsewhere).
        self.assertEqual(self._errs(block_size=None), [])

    def test_empty_target_layer_ids(self):
        errs = self._errs(target_layer_ids=[])
        self.assertTrue(any("empty" in e for e in errs), errs)

    def test_out_of_range_target_layer_ids(self):
        errs = self._errs(target_layer_ids=[0, 99])  # 99 >= 32
        self.assertTrue(any("out-of-range layer id" in e for e in errs), errs)

    def test_heads_not_divisible_by_tp(self):
        errs = self._errs(draft_total_num_heads=30, tp_size=4)
        self.assertTrue(any("attention heads" in e for e in errs), errs)

    def test_kv_heads_ge_tp_not_divisible(self):
        errs = self._errs(draft_total_num_kv_heads=6, tp_size=4)
        self.assertTrue(any("num_kv_heads" in e for e in errs), errs)

    def test_kv_heads_lt_tp_not_dividing(self):
        errs = self._errs(draft_total_num_kv_heads=3, tp_size=4)
        self.assertTrue(any("num_kv_heads" in e for e in errs), errs)

    def test_kv_heads_lt_tp_dividing_ok(self):
        # 2 < 4 and 4 % 2 == 0 -> valid.
        self.assertEqual(self._errs(draft_total_num_kv_heads=2, tp_size=4), [])

    def test_nonpositive_tp(self):
        errs = self._errs(tp_size=0)
        self.assertTrue(any("tp_size" in e for e in errs), errs)

    def test_multiple_errors_aggregated(self):
        # Several independent problems must all be reported in one pass.
        errs = self._errs(
            mask_token_id=999999, draft_hidden_size=1, target_layer_ids=[]
        )
        self.assertGreaterEqual(len(errs), 3)


class TestDflashMissingWeightKeys(CustomTestCase):
    """The draft-checkpoint completeness check (``dflash_missing_weight_keys``):
    every required parameter (direct + all packed q/k/v and gate/up shards) must
    receive a weight, else it is reported as missing.
    """

    # A minimal but representative DFlash draft param set: one attention block
    # (packed qkv/gate_up + direct o_proj/down_proj + norms) and the DFlash head.
    PARAMS = [
        "layers.0.self_attn.qkv_proj.weight",
        "layers.0.self_attn.o_proj.weight",
        "layers.0.mlp.gate_up_proj.weight",
        "layers.0.mlp.down_proj.weight",
        "layers.0.input_layernorm.weight",
        "layers.0.post_attention_layernorm.weight",
        "fc.weight",
        "hidden_norm.weight",
        "norm.weight",
    ]

    def _all_loaded(self):
        qkv = "layers.0.self_attn.qkv_proj.weight"
        gate_up = "layers.0.mlp.gate_up_proj.weight"
        direct = {
            p
            for p in self.PARAMS
            if "qkv_proj" not in p and "gate_up_proj" not in p
        }
        packed = {qkv: {"q", "k", "v"}, gate_up: {0, 1}}
        return direct, packed

    def test_all_present_no_missing(self):
        direct, packed = self._all_loaded()
        self.assertEqual(du.dflash_missing_weight_keys(self.PARAMS, direct, packed), [])

    def test_missing_direct_fc_weight(self):
        direct, packed = self._all_loaded()
        direct.discard("fc.weight")
        missing = du.dflash_missing_weight_keys(self.PARAMS, direct, packed)
        self.assertIn("fc.weight", missing)

    def test_missing_hidden_norm_and_norm(self):
        direct, packed = self._all_loaded()
        direct.discard("hidden_norm.weight")
        direct.discard("norm.weight")
        missing = du.dflash_missing_weight_keys(self.PARAMS, direct, packed)
        self.assertIn("hidden_norm.weight", missing)
        self.assertIn("norm.weight", missing)

    def test_missing_qkv_shard(self):
        direct, packed = self._all_loaded()
        packed["layers.0.self_attn.qkv_proj.weight"] = {"q", "k"}  # missing v
        missing = du.dflash_missing_weight_keys(self.PARAMS, direct, packed)
        self.assertTrue(any("qkv_proj" in m and "v" in m for m in missing), missing)

    def test_missing_gate_up_shard(self):
        direct, packed = self._all_loaded()
        packed["layers.0.mlp.gate_up_proj.weight"] = {0}  # missing up (1)
        missing = du.dflash_missing_weight_keys(self.PARAMS, direct, packed)
        self.assertTrue(any("gate_up_proj" in m for m in missing), missing)

    def test_fully_absent_packed_param_reported(self):
        # A packed param that received NO shard at all is missing, not silently ok.
        direct, packed = self._all_loaded()
        packed.pop("layers.0.self_attn.qkv_proj.weight")
        missing = du.dflash_missing_weight_keys(self.PARAMS, direct, packed)
        self.assertTrue(any("qkv_proj" in m for m in missing), missing)

    def test_fused_on_disk_packed_params_accepted(self):
        # A native/SGLang checkpoint may store qkv_proj/gate_up_proj FUSED on disk;
        # the direct branch loads them (into loaded_direct) and the linear layer's
        # weight_loader splits internally. Empty packed_shards_seen must NOT flag
        # them missing.
        qkv = "layers.0.self_attn.qkv_proj.weight"
        gate_up = "layers.0.mlp.gate_up_proj.weight"
        direct = {
            p for p in self.PARAMS if "qkv_proj" not in p and "gate_up_proj" not in p
        }
        direct |= {qkv, gate_up}  # fused-on-disk direct loads
        self.assertEqual(du.dflash_missing_weight_keys(self.PARAMS, direct, {}), [])

    def test_fused_qkv_does_not_mask_missing_gate_up_shard(self):
        # Fused-on-disk qkv accepted, but gate_up still needs its shards: a missing
        # gate/up shard must still be reported.
        qkv = "layers.0.self_attn.qkv_proj.weight"
        gate_up = "layers.0.mlp.gate_up_proj.weight"
        direct = {
            p for p in self.PARAMS if "qkv_proj" not in p and "gate_up_proj" not in p
        }
        direct |= {qkv}  # only qkv fused-on-disk
        packed = {gate_up: {0}}  # gate_up split but missing shard 1 (up)
        missing = du.dflash_missing_weight_keys(self.PARAMS, direct, packed)
        self.assertTrue(any("gate_up_proj" in m for m in missing), missing)
        self.assertFalse(any("qkv_proj" in m for m in missing), missing)


class TestDflashConfigEarlyBoundaryFailures(CustomTestCase):
    """Config failures caught EARLIER than validate_dflash_draft_startup in the
    real ModelRunner path (parse_dflash_draft_config -> require_num_layers ->
    resolve_target_layer_ids all raise before the aggregate validator).
    """

    def _cfg(self, **overrides):
        from sglang.srt.speculative.dflash_utils import DFlashDraftConfig

        base = dict(
            num_hidden_layers=1,
            num_target_layers=32,
            block_size=8,
            target_layer_ids=[0, 15, 31],
            mask_token="<mask>",
            mask_token_id=128002,
        )
        base.update(overrides)
        return DFlashDraftConfig(**base)

    def test_require_num_layers_raises_on_none(self):
        with self.assertRaises(ValueError):
            self._cfg(num_hidden_layers=None).require_num_layers()

    def test_resolve_target_layer_ids_raises_on_empty(self):
        with self.assertRaises(ValueError):
            self._cfg(target_layer_ids=[]).resolve_target_layer_ids(
                target_num_layers=32, draft_num_layers=1
            )

    def test_resolve_target_layer_ids_raises_on_out_of_range(self):
        with self.assertRaises(ValueError):
            self._cfg(target_layer_ids=[0, 99]).resolve_target_layer_ids(
                target_num_layers=32, draft_num_layers=1
            )


class TestDflashFusedKvMaterializeDecision(CustomTestCase):
    """The fused-vs-sequential KV-materialization gate (``dflash_fused_kv_materialize_decision``).

    The fused path is a CUDA Triton kernel; on TPU (non-CUDA) it must be disabled
    with an observable reason so the worker falls back to the sequential
    pure-torch path intentionally, not silently.
    """

    def test_non_cuda_disabled_with_reason(self):
        enabled, reason = du.dflash_fused_kv_materialize_decision(is_cuda=False)
        self.assertFalse(enabled)
        self.assertIsInstance(reason, str)
        self.assertIn("sequential", reason.lower())

    def test_cuda_enabled_no_reason(self):
        enabled, reason = du.dflash_fused_kv_materialize_decision(is_cuda=True)
        self.assertTrue(enabled)
        self.assertIsNone(reason)


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


def _ref_accept(candidates_row, target_row):
    """Independent Python reference for the DFlash greedy accept rule.

    Accept drafts while candidates[i+1] == target_predict[i], stopping at the
    first mismatch; bonus is the target prediction at the first non-accepted
    position (== len(accepted) drafts).
    """
    block = len(candidates_row)
    correct = 0
    for i in range(block - 1):
        if candidates_row[i + 1] == target_row[i]:
            correct += 1
        else:
            break
    return correct, target_row[correct]


class TestDflashGreedyAcceptParity(CustomTestCase):
    """Forced-logit greedy accept parity for ``compute_dflash_correct_drafts_and_bonus``
    (the core DFlash verify rule producing accept_len + bonus), checked against an
    independent Python reference across zero / partial / full / mixed-batch accepts.
    candidates[:, 0] is the current token; candidates[:, 1:] are the drafts;
    target_predict[:, t] is the target argmax at block position t.
    """

    def _run(self, candidates, target_predict):
        c = torch.tensor(candidates, dtype=torch.int64)
        t = torch.tensor(target_predict, dtype=torch.int64)
        correct_len, bonus = du.compute_dflash_correct_drafts_and_bonus(
            candidates=c, target_predict=t
        )
        return correct_len, bonus

    def test_zero_accept(self):
        # First draft (candidates[1]) mismatches target_predict[0] -> 0 accepted;
        # bonus is target_predict[0].
        candidates = [[100, 7, 8, 9]]
        target = [[999, 55, 66, 77]]  # target_predict[0]=999 != draft 7
        correct_len, bonus = self._run(candidates, target)
        self.assertEqual(correct_len.tolist(), [0])
        self.assertEqual(bonus.tolist(), [999])

    def test_partial_accept(self):
        # drafts [7,8,9]; target_predict[:3]=[7,8,55] -> first two match, third
        # mismatches -> correct_len=2, bonus=target_predict[2]=55.
        candidates = [[100, 7, 8, 9]]
        target = [[7, 8, 55, 77]]
        correct_len, bonus = self._run(candidates, target)
        self.assertEqual(correct_len.tolist(), [2])
        self.assertEqual(bonus.tolist(), [55])

    def test_full_accept(self):
        # All block-1 drafts match -> correct_len = block-1; bonus is the last
        # target prediction (target_predict[block-1]).
        candidates = [[100, 7, 8, 9]]
        target = [[7, 8, 9, 42]]
        correct_len, bonus = self._run(candidates, target)
        self.assertEqual(correct_len.tolist(), [3])  # block_size 4 -> 3 drafts
        self.assertEqual(bonus.tolist(), [42])

    def test_first_mismatch_stops_cumprod(self):
        # A later match AFTER an earlier mismatch must NOT count: drafts [7,8,9],
        # target_predict[:3]=[7,999,9] -> stop at pos1 -> correct_len=1, bonus=999.
        candidates = [[100, 7, 8, 9]]
        target = [[7, 999, 9, 11]]
        correct_len, bonus = self._run(candidates, target)
        self.assertEqual(correct_len.tolist(), [1])
        self.assertEqual(bonus.tolist(), [999])

    def test_mixed_batch(self):
        # One call, rows with zero / partial / full accept lengths.
        candidates = [
            [100, 7, 8, 9],  # zero: draft 7 != target 999
            [100, 7, 8, 9],  # partial: [7,8] match, 9 vs 55 mismatch -> 2
            [100, 7, 8, 9],  # full: [7,8,9] all match -> 3
        ]
        target = [
            [999, 1, 2, 3],
            [7, 8, 55, 4],
            [7, 8, 9, 42],
        ]
        correct_len, bonus = self._run(candidates, target)
        self.assertEqual(correct_len.tolist(), [0, 2, 3])
        self.assertEqual(bonus.tolist(), [999, 55, 42])

    def test_matches_independent_reference_exhaustively(self):
        # Enumerate a small deterministic space and compare accept_len + bonus to
        # the independent Python reference row-by-row.
        block = 4
        vocab = [5, 6, 7]  # small alphabet so accepts/mismatches both occur
        rows_c, rows_t, ref = [], [], []
        # Deterministic sweep: vary each draft/target position over the alphabet.
        for a in vocab:
            for b in vocab:
                for tt0 in vocab:
                    for tt1 in vocab:
                        cand = [0, a, b, 6]  # current token + 3 drafts
                        targ = [tt0, tt1, a, b]
                        rows_c.append(cand)
                        rows_t.append(targ)
                        ref.append(_ref_accept(cand, targ))
        correct_len, bonus = self._run(rows_c, rows_t)
        self.assertEqual(correct_len.tolist(), [r[0] for r in ref])
        self.assertEqual(bonus.tolist(), [r[1] for r in ref])

    def test_bonus_index_in_range_and_dtypes(self):
        candidates = [[100, 7, 8, 9], [100, 1, 2, 3]]
        target = [[7, 8, 9, 42], [999, 5, 6, 7]]
        correct_len, bonus = self._run(candidates, target)
        # correct_len is int64 (torch.sum promotes int32->int64), and int64 is
        # REQUIRED: verify() cats it with the int64 `candidates` in the packed D2H
        # transfer, so an int32 would raise a dtype-mismatch in torch.cat.
        self.assertEqual(correct_len.dtype, torch.int64)
        self.assertEqual(bonus.dtype, torch.int64)
        # bonus[i] == target_predict[i, correct_len[i]] and index in [0, block).
        for i in range(2):
            k = int(correct_len[i].item())
            self.assertTrue(0 <= k < 4)
            self.assertEqual(int(bonus[i].item()), target[i][k])

    def test_validation_errors(self):
        with self.assertRaises(ValueError):  # not 2D
            du.compute_dflash_correct_drafts_and_bonus(
                candidates=torch.tensor([1, 2, 3]),
                target_predict=torch.tensor([1, 2, 3]),
            )
        with self.assertRaises(ValueError):  # shape mismatch
            du.compute_dflash_correct_drafts_and_bonus(
                candidates=torch.tensor([[1, 2, 3]]),
                target_predict=torch.tensor([[1, 2]]),
            )
        with self.assertRaises(ValueError):  # bs == 0
            du.compute_dflash_correct_drafts_and_bonus(
                candidates=torch.zeros((0, 4), dtype=torch.int64),
                target_predict=torch.zeros((0, 4), dtype=torch.int64),
            )


class _FakeReq:
    """Minimal Req stand-in exposing only what DFlashVerifyInput.verify reads."""

    def __init__(self, eos_id, origin_last=1):
        self.output_ids = []
        self.origin_input_ids = [origin_last]
        self.grammar = None
        self.spec_verify_ct = 0
        self.spec_num_correct_drafts = 0
        self.kv_committed_len = 0
        self.kv_allocated_len = 0
        self._eos = eos_id
        self._finished = False

    def update_finish_state(self):
        # Finish exactly when the most recent output token is EOS.
        self._finished = bool(self.output_ids) and self.output_ids[-1] == self._eos

    def finished(self):
        return self._finished


class _FakeAllocator:
    def __init__(self):
        self.freed = []

    def free(self, locs):
        self.freed.append(locs)


class TestDflashVerifyHostSemantics(CustomTestCase):
    """CPU-runnable parity for ``DFlashVerifyInput.verify()`` host-side semantics
    (commit_len, EOS early-stop, output_ids, new_bonus_tokens, num_correct_drafts,
    hidden-segment selection, seq_lens/seq_lens_sum, req KV counters, free/keep +
    out_cache_loc), driven with fakes and forced-argmax logits -- no TPU server.
    """

    EOS = 2
    DTN = 4  # block size: current token + 3 drafts
    VOCAB = 1000
    H = 2
    PAGE = 64

    def _build(self):
        import types as _types

        # 5 rows: partial / zero / full / EOS-in-accepted-draft / EOS-as-bonus.
        # candidates[:,0]=current token (100), [:,1:]=drafts.
        cand_rows = [
            [100, 7, 8, 9],  # 0 partial
            [100, 7, 8, 9],  # 1 zero
            [100, 7, 8, 9],  # 2 full
            [100, 7, self.EOS, 9],  # 3 EOS is an accepted draft (before bonus)
            [100, 7, 8, 9],  # 4 EOS is the bonus
        ]
        # target_predict per (row, pos) = argmax of the constructed logits.
        tgt_rows = [
            [7, 8, 55, 0],  # correct_len 2, bonus 55
            [999, 1, 2, 3],  # correct_len 0, bonus 999
            [7, 8, 9, 42],  # correct_len 3, bonus 42
            [7, self.EOS, 9, 50],  # correct_len 3, bonus 50 -> EOS draft stops at 2
            [self.EOS, 1, 2, 3],  # correct_len 0, bonus EOS
        ]
        bs = len(cand_rows)

        draft_token = torch.tensor(
            [t for row in cand_rows for t in row], dtype=torch.int64
        )
        logits = torch.zeros((bs * self.DTN, self.VOCAB), dtype=torch.float32)
        hidden = torch.zeros((bs * self.DTN, self.H), dtype=torch.float32)
        for r in range(bs):
            for p in range(self.DTN):
                logits[r * self.DTN + p, tgt_rows[r][p]] = 1.0
                hidden[r * self.DTN + p, 0] = float(r)
                hidden[r * self.DTN + p, 1] = float(p)

        fake_self = _types.SimpleNamespace(
            draft_token=draft_token, draft_token_num=self.DTN
        )
        logits_output = _types.SimpleNamespace(
            next_token_logits=logits, hidden_states=hidden
        )
        reqs = [_FakeReq(self.EOS) for _ in range(bs)]
        allocator = _FakeAllocator()
        batch = _types.SimpleNamespace(
            forward_mode=_types.SimpleNamespace(is_idle=lambda: False),
            batch_size=lambda: bs,
            sampling_info=None,
            reqs=reqs,
            seq_lens=torch.full((bs,), self.PAGE, dtype=torch.int64),
            seq_lens_cpu=torch.full((bs,), self.PAGE, dtype=torch.int64),
            seq_lens_sum=self.PAGE * bs,
            out_cache_loc=torch.arange(
                1000, 1000 + bs * self.DTN, dtype=torch.int64
            ),
            token_to_kv_pool_allocator=allocator,
            req_pool_indices=torch.arange(bs, dtype=torch.int64),
            req_to_token_pool=_types.SimpleNamespace(
                req_to_token=torch.zeros((bs, 256), dtype=torch.int64)
            ),
            device="cpu",
        )
        return fake_self, batch, logits_output, reqs, allocator, bs

    def test_verify_host_semantics_mixed_batch(self):
        import sglang.srt.speculative.dflash_info as di

        fake_self, batch, logits_output, reqs, allocator, bs = self._build()

        # Patch the pool updater to a recorder (its effect is out of scope here;
        # the paged free/keep accounting is covered by the KV-pool tests).
        calls = []
        orig = di.assign_req_to_token_pool_func
        di.assign_req_to_token_pool_func = lambda *a, **k: calls.append((a, k))
        try:
            new_bonus, commit_lens, next_hidden, num_correct = (
                di.DFlashVerifyInput.verify(
                    fake_self,
                    batch=batch,
                    logits_output=logits_output,
                    page_size=self.PAGE,
                )
            )
        finally:
            di.assign_req_to_token_pool_func = orig

        # Accept accounting.
        self.assertEqual(commit_lens.tolist(), [3, 1, 4, 2, 1])
        self.assertEqual(num_correct, [2, 0, 3, 2, 0])
        self.assertEqual(new_bonus.tolist(), [55, 999, 42, self.EOS, self.EOS])
        self.assertEqual(commit_lens.dtype, torch.int32)
        self.assertEqual(new_bonus.dtype, torch.int64)

        # Per-req output_ids (EOS stops before the bonus in row 3; EOS is the only
        # token in row 4).
        self.assertEqual(reqs[0].output_ids, [7, 8, 55])
        self.assertEqual(reqs[1].output_ids, [999])
        self.assertEqual(reqs[2].output_ids, [7, 8, 9, 42])
        self.assertEqual(reqs[3].output_ids, [7, self.EOS])  # stopped at EOS draft
        self.assertEqual(reqs[4].output_ids, [self.EOS])

        # Per-req counters.
        for r, (cl, nc) in enumerate(zip([3, 1, 4, 2, 1], [2, 0, 3, 2, 0])):
            self.assertEqual(reqs[r].spec_verify_ct, 1)
            self.assertEqual(reqs[r].spec_num_correct_drafts, nc)
            self.assertEqual(reqs[r].kv_committed_len, cl)
            self.assertEqual(reqs[r].kv_allocated_len, cl)

        # next_target_hidden is now the FULL fixed-width [bs*draft_token_num, feat]
        # grid (row r*DTN+p == verify hidden[r, p]); the decode materialization
        # masks uncommitted rows (p >= commit_r) to the reserved dummy KV slot.
        # Fixed shape regardless of accept count (was sum(commit_lens) -> the
        # dominant AC-7 recompile source). Committed rows carry the same encoded
        # (row, pos) the old compacted path wrote.
        self.assertEqual(next_hidden.shape[0], bs * self.DTN)
        for r, cl in enumerate([3, 1, 4, 2, 1]):
            for p in range(cl):
                self.assertEqual(
                    next_hidden[r * self.DTN + p, :2].tolist(), [float(r), float(p)]
                )

        # seq_lens advanced by commit; seq_lens_sum kept in sync.
        self.assertEqual(batch.seq_lens.tolist(), [67, 65, 68, 66, 65])
        self.assertEqual(batch.seq_lens_cpu.tolist(), [67, 65, 68, 66, 65])
        self.assertEqual(batch.seq_lens_sum, self.PAGE * bs + 11)

        # KV compaction: allocator.free called; out_cache_loc replaced with int64
        # kept locs; pool updater invoked once.
        self.assertTrue(allocator.freed, "allocator.free should be called")
        self.assertEqual(batch.out_cache_loc.dtype, torch.int64)
        self.assertEqual(len(calls), 1)

        # hidden_states cleared to avoid confusing downstream consumers.
        self.assertIsNone(logits_output.hidden_states)

    def test_verify_idle_returns_empty(self):
        import types as _types

        import sglang.srt.speculative.dflash_info as di

        fake_self = _types.SimpleNamespace(
            draft_token=torch.empty((0,), dtype=torch.int64), draft_token_num=self.DTN
        )
        batch = _types.SimpleNamespace(
            forward_mode=_types.SimpleNamespace(is_idle=lambda: True), device="cpu"
        )
        logits_output = _types.SimpleNamespace(
            next_token_logits=torch.empty((0, self.VOCAB)), hidden_states=None
        )
        new_bonus, commit_lens, next_hidden, num_correct = di.DFlashVerifyInput.verify(
            fake_self, batch=batch, logits_output=logits_output, page_size=self.PAGE
        )
        self.assertEqual(new_bonus.numel(), 0)
        self.assertEqual(commit_lens.numel(), 0)
        self.assertEqual(commit_lens.dtype, torch.int32)
        self.assertEqual(num_correct, [])


class TestDflashStreamParityVerdict(CustomTestCase):
    """The token-ID greedy stream-parity verdict (`dflash_greedy_parity.diff_token_streams`),
    used by the e2e harness to diff DFlash vs non-spec output token IDs.
    """

    def _diff(self, a, b):
        import importlib.util
        import pathlib

        # Import the manual harness by path (it lives under test/manual/tpu).
        p = pathlib.Path(__file__).parent / "dflash_greedy_parity.py"
        spec = importlib.util.spec_from_file_location("dflash_greedy_parity", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.diff_token_streams(a, b)

    def test_identical_streams_match(self):
        v = self._diff([1, 2, 3, 4], [1, 2, 3, 4])
        self.assertTrue(v["match"])
        self.assertEqual(v["first_divergence"], -1)

    def test_first_divergence_index(self):
        v = self._diff([1, 2, 9, 4], [1, 2, 3, 4])
        self.assertFalse(v["match"])
        self.assertEqual(v["first_divergence"], 2)

    def test_prefix_but_shorter_is_divergence(self):
        # spec stopped early (e.g. EOS) but ref continued -> not a match; divergence
        # reported at the shorter length.
        v = self._diff([1, 2, 3], [1, 2, 3, 4, 5])
        self.assertFalse(v["match"])
        self.assertEqual(v["first_divergence"], 3)
        self.assertEqual(v["spec_len"], 3)
        self.assertEqual(v["ref_len"], 5)

    def test_longer_spec_is_divergence(self):
        v = self._diff([1, 2, 3, 4, 5], [1, 2, 3])
        self.assertFalse(v["match"])
        self.assertEqual(v["first_divergence"], 3)

    def test_empty_streams_match(self):
        v = self._diff([], [])
        self.assertTrue(v["match"])

    def test_first_token_divergence(self):
        v = self._diff([7], [8])
        self.assertFalse(v["match"])
        self.assertEqual(v["first_divergence"], 0)


class TestDflashDivergenceClassifier(CustomTestCase):
    """The DECISIVE bf16 tie-break tolerance policy (`classify_divergence` +
    `_tolerated_divergence`): a divergence is an accepted tie-break ONLY when the
    non-spec decode top-2 and DFlash's own verify-forward top-2 are the SAME token
    pair in swapped order, both margins are within the bf16 noise scale, and the
    emitted token is DFlash's verify top-1. `_tolerated_divergence` recomputes this
    from manifest evidence bound to the current dumps — never trusting a stored
    verdict — so stale/forged/looser-epsilon manifests are rejected.
    """

    # Observed live case: non-spec top-1=706(-0.670), top-2=374(-0.795), margin
    # 0.125; DFlash verify top-1=374, top-2=706, margin 0.10 -> a swapped-pair tie.
    NOSPEC_TOPK = [[706, -0.66965], [374, -0.79465]]
    VERIFY_TOP2 = [[374, -0.10], [706, -0.20]]

    def _mod(self):
        import importlib.util
        import pathlib

        p = pathlib.Path(__file__).parent / "dflash_greedy_parity.py"
        spec = importlib.util.spec_from_file_location("dflash_greedy_parity", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _tie_entry(self, m):
        return m.classify_divergence(self.NOSPEC_TOPK, 26, 374, verify_top2=self.VERIFY_TOP2)

    def test_decisive_swapped_pair_is_tie(self):
        m = self._mod()
        d = self._tie_entry(m)
        self.assertEqual(d["verdict"], "bf16-tie")
        self.assertEqual(d["reasons"], [])
        self.assertEqual(d["dflash_rank_in_nospec_topk"], 1)
        self.assertEqual(d["nospec_token"], 706)
        self.assertAlmostEqual(d["top2_margin"], 0.125, places=5)

    def test_no_verify_trace_is_not_tie(self):
        # Non-spec runner-up evidence alone is not decisive.
        m = self._mod()
        d = m.classify_divergence(self.NOSPEC_TOPK, 26, 374)
        self.assertEqual(d["verdict"], "MISMATCH")

    def test_verify_top1_not_emitted_is_mismatch(self):
        # DFlash verify argmax is 706 but it emitted 374 -> accept bug.
        m = self._mod()
        d = m.classify_divergence(self.NOSPEC_TOPK, 26, 374, verify_top2=[[706, -0.1], [374, -0.2]])
        self.assertEqual(d["verdict"], "MISMATCH")

    def test_verify_pair_not_a_swap_is_mismatch(self):
        # verify top-2 is a third token, not the non-spec top-1 -> not a swap.
        m = self._mod()
        d = m.classify_divergence(self.NOSPEC_TOPK, 26, 374, verify_top2=[[374, -0.1], [999, -0.2]])
        self.assertEqual(d["verdict"], "MISMATCH")

    def test_verify_margin_too_large_is_mismatch(self):
        m = self._mod()
        d = m.classify_divergence(self.NOSPEC_TOPK, 26, 374, verify_top2=[[374, -0.1], [706, -1.6]])
        self.assertEqual(d["verdict"], "MISMATCH")

    def test_nospec_large_margin_is_mismatch(self):
        m = self._mod()
        d = m.classify_divergence([[706, -0.10], [374, -1.60]], 26, 374, verify_top2=self.VERIFY_TOP2)
        self.assertEqual(d["verdict"], "MISMATCH")

    def test_lower_rank_token_is_mismatch(self):
        m = self._mod()
        d = m.classify_divergence(
            [[706, -0.10], [374, -0.15], [999, -0.20]], 5, 999, verify_top2=[[999, -0.1], [706, -0.2]]
        )
        self.assertEqual(d["dflash_rank_in_nospec_topk"], 2)
        self.assertEqual(d["verdict"], "MISMATCH")

    def test_tolerated_recomputes_and_binds_to_current_dumps(self):
        m = self._mod()
        manifest = {"P": self._tie_entry(m)}
        # exact match to current dumps -> tolerated
        ok, _ = m._tolerated_divergence(manifest, "P", 26, 374, 706)
        self.assertTrue(ok)
        # current DFlash token changed -> NOT tolerated
        self.assertFalse(m._tolerated_divergence(manifest, "P", 26, 999, 706)[0])
        # current non-spec token changed -> NOT tolerated
        self.assertFalse(m._tolerated_divergence(manifest, "P", 26, 374, 999)[0])
        # stale index -> NOT tolerated
        self.assertFalse(m._tolerated_divergence(manifest, "P", 25, 374, 706)[0])
        # unknown prompt -> NOT tolerated
        self.assertFalse(m._tolerated_divergence(manifest, "Z", 0, 1, 2)[0])

    def test_tolerated_rejects_oversized_epsilon(self):
        m = self._mod()
        entry = self._tie_entry(m)
        entry = dict(entry)
        entry["epsilon"] = 10.0  # forged looser epsilon
        self.assertFalse(m._tolerated_divergence({"P": entry}, "P", 26, 374, 706)[0])

    def test_tolerated_rejects_forged_topk(self):
        m = self._mod()
        entry = dict(self._tie_entry(m))
        # tamper nospec_topk so the margin blows past eps; recompute must reject
        entry["nospec_topk"] = [[706, -0.10], [374, -1.60]]
        self.assertFalse(m._tolerated_divergence({"P": entry}, "P", 26, 374, 706)[0])

    def test_tolerated_rejects_scalar_nospec_token_forgery(self):
        # Codex R2 forgery: edit ONLY the scalar nospec_token to a changed current
        # ref token (999) while leaving nospec_topk top-1 as the old 706. The
        # recomputed-evidence binding must reject it (recomputed top-1 706 != 999).
        m = self._mod()
        entry = dict(self._tie_entry(m))
        entry["nospec_token"] = 999
        ok, why = m._tolerated_divergence({"P": entry}, "P", 26, 374, 999)
        self.assertFalse(ok)
        self.assertIn("top-1", why)

    def test_tolerated_rejects_runner_up_not_current_dflash(self):
        # nospec_topk runner-up is a third token, not the current DFlash token.
        m = self._mod()
        entry = dict(self._tie_entry(m))
        entry["nospec_topk"] = [[706, -0.66965], [555, -0.79465]]
        entry["verify_top2"] = [[374, -0.10], [706, -0.20]]
        self.assertFalse(m._tolerated_divergence({"P": entry}, "P", 26, 374, 706)[0])


class TestDflashTpVerifyInvariant(CustomTestCase):
    """All-ranks-identical invariant helper (`dflash_tp_ranks_agree`): after
    the rank-0 broadcast + host rebuild, an all-gathered per-rank vector
    (correct_len | commit_lens | new_bonus_tokens) must be identical across TP
    ranks. The helper is pure and CPU-testable; the live verify path all-gathers
    and raises on any divergence flagged here.
    """

    def _fn(self):
        from sglang.srt.speculative.dflash_utils import dflash_tp_ranks_agree

        return dflash_tp_ranks_agree

    def test_all_ranks_agree(self):
        agree = self._fn()
        local = torch.tensor([2, 1, 0, 5, 7], dtype=torch.int64)  # cl|commit|bonus
        gathered = local.repeat(4)  # 4 identical ranks
        self.assertEqual(agree(gathered, 4), (True, None))

    def test_divergent_rank_is_flagged(self):
        agree = self._fn()
        local = torch.tensor([2, 1, 0, 5, 7], dtype=torch.int64)
        diff = torch.tensor([2, 1, 0, 5, 8], dtype=torch.int64)  # bonus differs
        gathered = torch.cat([local, local, diff, local])  # rank 2 diverges
        ok, bad = agree(gathered, 4)
        self.assertFalse(ok)
        self.assertEqual(bad, 2)

    def test_lowest_differing_rank_reported(self):
        agree = self._fn()
        local = torch.tensor([1, 1], dtype=torch.int64)
        d = torch.tensor([1, 9], dtype=torch.int64)
        gathered = torch.cat([local, d, d])  # ranks 1 and 2 differ; report 1
        ok, bad = agree(gathered, 3)
        self.assertFalse(ok)
        self.assertEqual(bad, 1)

    def test_world_size_one_trivially_agrees(self):
        agree = self._fn()
        self.assertEqual(agree(torch.tensor([1, 2, 3]), 1), (True, None))

    def test_world_size_two(self):
        # Dedicated ws=2 case: live TP=2 is infra-blocked (torch_tpu set_device),
        # so the helper's TP=2 behavior is only covered here.
        agree = self._fn()
        local = torch.tensor([3, 2, 5], dtype=torch.int64)  # cl|commit|bonus
        self.assertEqual(agree(local.repeat(2), 2), (True, None))
        diff = torch.tensor([3, 2, 6], dtype=torch.int64)  # rank 1 bonus differs
        ok, bad = agree(torch.cat([local, diff]), 2)
        self.assertFalse(ok)
        self.assertEqual(bad, 1)


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


class TestDflashPrecompileBucketPlan(CustomTestCase):
    """The draft-precompile shape planner is a pure function (no device/model
    state) deriving (bs, prefix_len) from the configured batch buckets x the
    512-mask buckets up to the context envelope. CPU-testable so the precompile
    coverage/accounting can be verified without a TPU.
    """

    def test_covers_every_bs_x_mask_bucket(self):
        from sglang.srt.speculative.dflash_utils import (
            dflash_precompile_bucket_plan,
        )

        # bs buckets [1, 8, 16], mask stride 512, context 2048, block 512.
        plan = dflash_precompile_bucket_plan(
            bs_buckets=[1, 8, 16],
            mask_stride=512,
            max_context=2048,
            block_size=512,
        )
        # kv buckets 512,1024,1536,2048 -> prefix = kv-512 = 0,512,1024,1536.
        # prefix<=0 dropped -> 3 mask shapes per bs; 3 bs -> 9 shapes.
        self.assertEqual(len(plan), 9)
        by_bs = {}
        for bs, prefix in plan:
            by_bs.setdefault(bs, []).append(prefix)
        self.assertEqual(sorted(by_bs), [1, 8, 16])
        for bs in (1, 8, 16):
            self.assertEqual(sorted(by_bs[bs]), [512, 1024, 1536])

    def test_prefix_makes_mask_bucket_exact(self):
        # Every planned (bs, prefix) must land prefix+block on a mask-bucket
        # boundary (the metadata build rounds kv_len up to the 512 stride), so
        # each planned shape warms exactly one distinct mask bucket.
        from sglang.srt.speculative.dflash_utils import (
            dflash_precompile_bucket_plan,
        )

        stride, block = 512, 512
        plan = dflash_precompile_bucket_plan(
            bs_buckets=[1], mask_stride=stride, max_context=2048, block_size=block
        )
        for _bs, prefix in plan:
            self.assertEqual((prefix + block) % stride, 0)
            self.assertGreater(prefix, 0)

    def test_always_includes_bs1_and_dedups(self):
        from sglang.srt.speculative.dflash_utils import (
            dflash_precompile_bucket_plan,
        )

        # bs=1 injected even if not listed; duplicates collapse.
        plan = dflash_precompile_bucket_plan(
            bs_buckets=[8, 8, 16],
            mask_stride=512,
            max_context=1024,
            block_size=512,
        )
        bss = sorted({bs for bs, _ in plan})
        self.assertEqual(bss, [1, 8, 16])
        # kv buckets 512,1024 -> prefix 0(drop),512 -> 1 shape per bs, 3 bs.
        self.assertEqual(len(plan), 3)

    def test_empty_when_context_below_first_prefix(self):
        # context 512 with block 512: only kv bucket 512 -> prefix 0 -> dropped.
        from sglang.srt.speculative.dflash_utils import (
            dflash_precompile_bucket_plan,
        )

        plan = dflash_precompile_bucket_plan(
            bs_buckets=[1, 8],
            mask_stride=512,
            max_context=512,
            block_size=512,
        )
        self.assertEqual(plan, [])


class TestDflashDecodeMaterializeGrid(CustomTestCase):
    """The decode KV-materialization position grid is a pure function so it stays
    a single static [bs, block] shape regardless of accept count. Committed cells
    (valid==True) must map to exactly the (req, position) pairs the old
    accept-count-compacted path produced; uncommitted cells are masked to the
    reserved dummy slot by the worker.
    """

    def test_grid_positions_and_validity(self):
        from sglang.srt.speculative.dflash_utils import (
            dflash_decode_materialize_grid,
        )

        ctx_start = torch.tensor([5, 20], dtype=torch.int64)
        commit_lens = torch.tensor([3, 1], dtype=torch.int32)
        pos2d, valid = dflash_decode_materialize_grid(ctx_start, commit_lens, 4)
        self.assertEqual(pos2d.tolist(), [[5, 6, 7, 8], [20, 21, 22, 23]])
        self.assertEqual(
            valid.tolist(),
            [[True, True, True, False], [True, False, False, False]],
        )

    def test_all_reject_and_full_accept(self):
        from sglang.srt.speculative.dflash_utils import (
            dflash_decode_materialize_grid,
        )

        # all-reject: commit_lens=0 -> no valid cells (worker sends all to slot 0).
        _pos, valid = dflash_decode_materialize_grid(
            torch.tensor([7], dtype=torch.int64),
            torch.tensor([0], dtype=torch.int32),
            4,
        )
        self.assertFalse(bool(valid.any()))
        # full-accept: commit_lens==block -> every cell valid.
        _pos, valid_full = dflash_decode_materialize_grid(
            torch.tensor([7], dtype=torch.int64),
            torch.tensor([4], dtype=torch.int32),
            4,
        )
        self.assertTrue(bool(valid_full.all()))

    def test_committed_cells_match_old_compaction(self):
        # The old path emitted, per req, positions ctx_start+[0..commit_len) as a
        # flat cat. The fixed grid's committed cells (row-major) must equal that.
        from sglang.srt.speculative.dflash_utils import (
            dflash_decode_materialize_grid,
        )

        ctx_start = torch.tensor([5, 20, 100], dtype=torch.int64)
        commit_lens = torch.tensor([2, 4, 1], dtype=torch.int32)
        block = 4
        pos2d, valid = dflash_decode_materialize_grid(ctx_start, commit_lens, block)
        fixed_committed = pos2d[valid].tolist()
        old_compacted = []
        for i, cl in enumerate(commit_lens.tolist()):
            old_compacted.extend(int(ctx_start[i]) + j for j in range(cl))
        self.assertEqual(fixed_committed, old_compacted)


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
