"""Which backend reads the checkpoint, and why that choice is not a boolean."""

import pytest

from sglang.multimodal_gen.runtime.loader import weight_readers
from sglang.multimodal_gen.runtime.loader.weight_readers import (
    FALLBACK_READER,
    RunaiStreamerReader,
    SafetensorsMmapReader,
    available_reader_names,
    select_weight_reader,
)


class TestCapabilities:
    def test_the_streamer_cannot_skip_keys(self):
        # it materializes every tensor before yielding any of them
        assert not RunaiStreamerReader.supports_key_filter

    def test_only_the_mapping_backend_leaves_pages_reclaimable(self):
        assert SafetensorsMmapReader.retains_file_mapping
        assert not RunaiStreamerReader.retains_file_mapping

    def test_the_fallback_is_always_available(self):
        assert FALLBACK_READER.is_available()
        assert FALLBACK_READER.name in available_reader_names()


class TestSelection:
    def test_an_explicit_request_is_honoured(self):
        assert select_weight_reader(requested="safetensors").name == "safetensors"

    def test_an_unknown_name_is_an_error_not_a_silent_fallback(self):
        with pytest.raises(ValueError, match="unknown weight reader"):
            select_weight_reader(requested="does_not_exist")

    def test_a_key_filter_passes_over_a_backend_that_cannot_filter(self):
        # reading the whole checkpoint to discard most of it is worse than
        # reading the requested part more slowly
        chosen = select_weight_reader(requested="runai_streamer", needs_key_filter=True)
        assert chosen.name == FALLBACK_READER.name

    def test_a_key_filter_leaves_a_capable_backend_alone(self):
        chosen = select_weight_reader(requested="safetensors", needs_key_filter=True)
        assert chosen.name == "safetensors"

    def test_an_unavailable_backend_falls_back(self, monkeypatch):
        monkeypatch.setattr(
            RunaiStreamerReader, "is_available", classmethod(lambda cls: False)
        )
        assert (
            select_weight_reader(requested="runai_streamer").name
            == FALLBACK_READER.name
        )

    def test_the_environment_decides_when_nothing_is_requested(self, monkeypatch):
        monkeypatch.setattr(
            weight_readers.envs, "SGLANG_USE_RUNAI_MODEL_STREAMER", False
        )
        assert select_weight_reader().name == FALLBACK_READER.name

    def test_the_environment_can_ask_for_the_streamer(self, monkeypatch):
        if not RunaiStreamerReader.is_available():
            pytest.skip("run:ai model streamer is not installed")
        monkeypatch.setattr(
            weight_readers.envs, "SGLANG_USE_RUNAI_MODEL_STREAMER", True
        )
        assert select_weight_reader().name == "runai_streamer"


class TestSafetensorsDevicePlacement:
    def test_a_tpu_target_reads_on_the_host_and_moves(self, monkeypatch, tmp_path):
        import torch
        from safetensors.torch import save_file

        from sglang.multimodal_gen.runtime.loader.weight_readers import (
            safetensors_mmap,
        )

        path = tmp_path / "w.safetensors"
        save_file({"w": torch.arange(4.0)}, str(path))
        opened, moved = [], []
        real_open = safetensors_mmap.safe_open

        def recording_open(file, framework, device):
            opened.append(device)
            return real_open(file, framework=framework, device=device)

        monkeypatch.setattr(safetensors_mmap, "safe_open", recording_open)
        monkeypatch.setattr(
            torch.Tensor, "to", lambda self, device: moved.append(device) or self
        )
        weights = dict(
            SafetensorsMmapReader().iter_weights(
                [str(path)], device="tpu:0", to_cpu=False, show_progress=False
            )
        )
        assert opened == ["cpu"]
        assert moved == ["tpu:0"]
        assert weights["w"].tolist() == [0.0, 1.0, 2.0, 3.0]
