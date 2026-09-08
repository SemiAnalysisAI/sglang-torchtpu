"""TPU patch-budget batching of the EPD encode server's vision forward."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.disaggregation import encode_server
from sglang.srt.disaggregation.encode_server import MMEncoder
from sglang.srt.managers.schedule_batch import Modality
from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=5, suite="stage-a-test-tpu-unit")

_BUDGET_PATH = (
    "sglang.srt.hardware_backend.tpu.graph_runner."
    "vit_tpu_compile_runner.precompiled_max_patch_count"
)
_MERGE_SIZE = 2
_MERGE_SQ = _MERGE_SIZE**2
# 16 patches per image, which the merger folds to 4 embedding rows.
_GRIDS = torch.tensor([[1, 4, 4]] * 6)
_PATCHES_PER_IMAGE = 16
_TOKENS_PER_IMAGE = _PATCHES_PER_IMAGE // _MERGE_SQ
_HIDDEN = 3


def _make_encoder():
    encoder = MMEncoder.__new__(MMEncoder)
    encoder.model_type = "qwen3_vl"
    encoder.model_config = SimpleNamespace(hf_config=SimpleNamespace())
    encoder.image_processor = SimpleNamespace(merge_size=_MERGE_SIZE)
    return encoder


def _mm_inputs(grids=_GRIDS):
    rows = int(grids.prod(dim=1).sum())
    return {
        "pixel_values": torch.arange(rows * _HIDDEN, dtype=torch.float32).reshape(
            rows, _HIDDEN
        ),
        "image_grid_thw": grids,
    }


class _RecordingEncoderFn:
    """Stands in for ``model.get_image_feature``, merger included."""

    def __init__(self):
        self.calls = []

    def __call__(self, items):
        self.calls.append(items)
        feature = torch.cat([item.feature for item in items], dim=0)
        return feature[::_MERGE_SQ].clone()


class TestEncodeServerTpuVisionSplit(unittest.TestCase):
    def _run(self, encoder, fn, grids=_GRIDS):
        mm_inputs = _mm_inputs(grids)
        return encoder._encode_missing_tpu(
            mm_inputs["pixel_values"],
            mm_inputs,
            list(range(len(grids))),
            Modality.IMAGE,
            fn,
            grids,
        )

    def _reference(self, encoder, grids=_GRIDS):
        """What a single unsplit call produces."""
        mm_inputs = _mm_inputs(grids)
        items = encoder._build_mm_data_items(
            mm_inputs["pixel_values"],
            mm_inputs,
            list(range(len(grids))),
            Modality.IMAGE,
            grids,
        )
        return _RecordingEncoderFn()(items)

    def test_aggregate_items_collapse_to_one_row_of_n_by_3_grid(self):
        # Why the split has to regroup grid rows: the item-level split serving
        # uses would see a single item here and never fire.
        encoder = _make_encoder()
        mm_inputs = _mm_inputs()
        items = encoder._build_mm_data_items(
            mm_inputs["pixel_values"],
            mm_inputs,
            list(range(len(_GRIDS))),
            Modality.IMAGE,
            _GRIDS,
        )

        self.assertEqual(len(items), 1)
        self.assertEqual(tuple(items[0].image_grid_thw.shape), (len(_GRIDS), 3))

    def test_splits_grid_rows_to_fit_the_budget(self):
        encoder = _make_encoder()
        fn = _RecordingEncoderFn()

        with patch(_BUDGET_PATH, return_value=2 * _PATCHES_PER_IMAGE):
            slices = self._run(encoder, fn)

        self.assertEqual(len(fn.calls), 3)
        for call_items in fn.calls:
            patch_rows = sum(item.feature.shape[0] for item in call_items)
            self.assertEqual(patch_rows, 2 * _PATCHES_PER_IMAGE)
        # Splitting is a regrouping, not a change of result.
        self.assertEqual(
            [tuple(s.shape) for s in slices],
            [(_TOKENS_PER_IMAGE, _HIDDEN)] * len(_GRIDS),
        )
        torch.testing.assert_close(torch.cat(slices, dim=0), self._reference(encoder))

    def test_a_batch_within_the_budget_runs_as_one_call(self):
        encoder = _make_encoder()
        fn = _RecordingEncoderFn()

        with patch(_BUDGET_PATH, return_value=10**6):
            slices = self._run(encoder, fn)

        self.assertEqual(len(fn.calls), 1)
        torch.testing.assert_close(torch.cat(slices, dim=0), self._reference(encoder))

    def test_an_oversized_image_still_gets_its_own_call(self):
        # One image above the budget cannot be split further; it must not be
        # merged with its neighbours on top of that.
        encoder = _make_encoder()
        fn = _RecordingEncoderFn()

        with patch(_BUDGET_PATH, return_value=_PATCHES_PER_IMAGE // 2):
            self._run(encoder, fn)

        self.assertEqual(len(fn.calls), len(_GRIDS))

    def test_log_metrics_gates_the_forward_observation(self):
        # /encode observes the forward itself, and health checks must not land
        # in the latency histogram at all.
        encoder = _make_encoder()
        collector = MagicMock()

        with (
            patch.object(encode_server, "encoder_metrics_collector", collector),
            patch(_BUDGET_PATH, return_value=10**6),
        ):
            self._run(encoder, _RecordingEncoderFn())
            self.assertEqual(collector.observe_model_forward.call_count, 1)

            mm_inputs = _mm_inputs()
            encoder._encode_missing_tpu(
                mm_inputs["pixel_values"],
                mm_inputs,
                list(range(len(_GRIDS))),
                Modality.IMAGE,
                _RecordingEncoderFn(),
                _GRIDS,
                log_metrics=False,
            )

        self.assertEqual(collector.observe_model_forward.call_count, 1)

    def test_audio_never_splits(self):
        # Audio grids are scalar frame counts, not patch grids.
        encoder = _make_encoder()

        with patch(_BUDGET_PATH, side_effect=AssertionError("vision-only")):
            groups = encoder._split_indices_by_patch_budget(
                [0, 1], Modality.AUDIO, torch.tensor([8, 8])
            )

        self.assertEqual(groups, [[0, 1]])


class TestEncodeMissingDispatch(unittest.TestCase):
    """_encode_missing must reach the TPU path only on TPU."""

    def _encode_missing(self, encoder, fn):
        mm_inputs = _mm_inputs()
        return encoder._encode_missing(
            mm_inputs["pixel_values"],
            mm_inputs,
            list(range(len(_GRIDS))),
            Modality.IMAGE,
            fn,
            _GRIDS,
        )

    def test_tpu_short_circuits_into_the_tpu_path(self):
        encoder = _make_encoder()
        with (
            patch.object(encode_server, "_is_tpu", True),
            patch.object(
                MMEncoder, "_encode_missing_tpu", return_value=["sentinel"]
            ) as tpu_path,
        ):
            result = self._encode_missing(encoder, _RecordingEncoderFn())

        self.assertEqual(result, ["sentinel"])
        self.assertEqual(tpu_path.call_count, 1)

    def test_off_tpu_keeps_the_single_call_path(self):
        encoder = _make_encoder()
        fn = _RecordingEncoderFn()
        with (
            patch.object(encode_server, "_is_tpu", False),
            patch(_BUDGET_PATH, side_effect=AssertionError("TPU-only")),
        ):
            slices = self._encode_missing(encoder, fn)

        self.assertEqual(len(fn.calls), 1)
        self.assertEqual(len(slices), len(_GRIDS))


if __name__ == "__main__":
    unittest.main()
