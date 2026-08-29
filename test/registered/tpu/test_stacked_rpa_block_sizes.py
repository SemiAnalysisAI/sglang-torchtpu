"""Pure-Python coverage for stacked-RPA decode block-size policy."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit")

import dataclasses

import jax.numpy as jnp
import pytest

from sglang.srt.layers.attention.pallas.stacked_rpa import (
    configs,
    tuned_block_sizes,
)
from sglang.srt.layers.attention.pallas.stacked_rpa.decode import (
    block_sizes,
)
from sglang.srt.layers.attention.pallas.stacked_rpa.decode import (
    config as decode_config,
)


class _FakeTpuInfo:
    num_lanes = 128
    num_sublanes = 8
    mxu_column_size = 128


def _model(*, sliding_window=None):
    return configs.ModelConfigs(
        num_q_heads=8,
        num_kv_heads=2,
        head_dim=128,
        mask_value=-1e30,
        sliding_window=sliding_window,
    )


def _serve(*, page_size=128, total_q_tokens=128, pages_per_seq=8):
    return configs.ServingConfigs(
        num_seqs=4,
        page_size=page_size,
        total_q_tokens=total_q_tokens,
        num_page_indices=4 * pages_per_seq,
        dtype_q=jnp.bfloat16,
        dtype_kv=jnp.bfloat16,
        dtype_out=jnp.bfloat16,
    )


def _block(bq=1, bkv=256, batch=4, n_buffer=2):
    return block_sizes.BlockSizes(bq, bkv, batch, n_buffer)


@pytest.fixture(autouse=True)
def _block_size_environment(monkeypatch):
    monkeypatch.setattr(block_sizes.pltpu, "get_tpu_info", lambda: _FakeTpuInfo())


class TestDecodeConfiguration:
    def test_types_live_with_decode_kernel(self):
        assert block_sizes.BlockSizes.__module__ == block_sizes.__name__
        assert decode_config.DecodeConfig.__module__ == decode_config.__name__
        assert "decode_q_len" in {
            field.name for field in dataclasses.fields(decode_config.DecodeConfig)
        }

    @pytest.mark.parametrize("n_buffer", [1, 3])
    def test_requires_double_buffering(self, n_buffer):
        with pytest.raises(ValueError, match="requires double buffering"):
            _block(n_buffer=n_buffer)

    def test_vmem_uses_seq_along_lane_staging_shape(self):
        assert (
            block_sizes.calculate_vmem_usage(
                _model(), _serve(), _block(bkv=128, batch=1)
            )
            == 864_256
        )


class TestTunedBlockSizes:
    def test_table_contains_decode_entries_only(self):
        assert tuned_block_sizes.TUNED_BLOCK_SIZES
        assert {key[5] for key in tuned_block_sizes.TUNED_BLOCK_SIZES} == {"d"}
        assert {
            values[-1] for values in tuned_block_sizes.TUNED_BLOCK_SIZES.values()
        } == {2}

    @pytest.mark.parametrize(
        "entries,expected",
        [
            ({"exact", "windowless", "base"}, (1, 1, 256, 4, 2)),
            ({"windowless", "base"}, (1, 1, 512, 4, 2)),
            ({"base"}, (1, 1, 1024, 4, 2)),
        ],
    )
    def test_lookup_fallback(self, monkeypatch, entries, expected):
        model = _model(sliding_window=2048)
        serve = _serve(total_q_tokens=128)
        base = dict(
            device_name="TPU7x",
            page_size=serve.page_size,
            q_dtype=serve.dtype_q,
            kv_dtype=serve.dtype_kv,
            num_q_heads=model.num_q_heads,
            num_kv_heads=model.num_kv_heads,
            head_dim=model.head_dim,
            max_model_len=1024,
            mode_symbol=configs.RpaCase.DECODE.symbol,
        )
        keys = {
            "exact": tuned_block_sizes.make_lookup_key(
                **base, batched_tokens_bucket=128, sliding_window=2048
            ),
            "windowless": tuned_block_sizes.make_lookup_key(
                **base, batched_tokens_bucket=128, sliding_window=None
            ),
            "base": tuned_block_sizes.make_lookup_key(**base),
        }
        values = {
            "exact": (1, 1, 256, 4, 2),
            "windowless": (1, 1, 512, 4, 2),
            "base": (1, 1, 1024, 4, 2),
        }
        monkeypatch.setattr(tuned_block_sizes, "get_device_name", lambda: "TPU7x")
        monkeypatch.setattr(
            tuned_block_sizes,
            "TUNED_BLOCK_SIZES",
            {keys[name]: values[name] for name in entries},
        )

        assert (
            tuned_block_sizes.get_tuned_block_sizes(
                model, serve, 1024, configs.RpaCase.DECODE
            )
            == expected
        )

    def test_non_decode_mode_has_no_tuning(self):
        assert (
            tuned_block_sizes.get_tuned_block_sizes(
                _model(), _serve(), 1024, configs.RpaCase.MIXED
            )
            is None
        )


class TestGlobalDecodePolicy:
    @pytest.fixture
    def default(self, monkeypatch):
        default = _block()
        monkeypatch.setattr(block_sizes, "choose_block_sizes", lambda *args: default)
        return default

    def test_heuristic_reduces_batch_while_staying_double_buffered(self, monkeypatch):
        monkeypatch.setattr(
            block_sizes,
            "calculate_vmem_usage",
            lambda _, __, block: 50 if block.batch_size == 1 else 90,
        )
        chosen = block_sizes.choose_block_sizes(_model(), _serve(), 100, decode_q_len=1)
        assert chosen == _block(bkv=1024, batch=1)

    def test_override_skips_tuned_lookup(self, monkeypatch, default):
        override = _block(bq=2, bkv=512, batch=6)
        monkeypatch.setattr(
            block_sizes.tuned_block_sizes,
            "get_tuned_block_sizes",
            lambda *args: pytest.fail("explicit override must skip tuning"),
        )
        assert (
            block_sizes.choose_non_sliding_window_block_sizes(
                _model(), _serve(), 128 * 1024**2, override=override
            )
            is override
        )

    def test_tuned_entry_wins(self, monkeypatch, default):
        monkeypatch.setattr(
            block_sizes.tuned_block_sizes,
            "get_tuned_block_sizes",
            lambda *args: (1, 1, 1024, 8, 2),
        )
        chosen = block_sizes.choose_non_sliding_window_block_sizes(
            _model(), _serve(), 128 * 1024**2
        )
        assert chosen == _block(bkv=1024, batch=8)

    def test_spec_decode_caps_single_token_tuning(self, monkeypatch, default):
        monkeypatch.setattr(
            block_sizes.tuned_block_sizes,
            "get_tuned_block_sizes",
            lambda *args: (1, 1, 8192, 8, 2),
        )
        monkeypatch.setattr(block_sizes, "calculate_vmem_usage", lambda *args: 0)
        chosen = block_sizes.choose_non_sliding_window_block_sizes(
            _model(),
            _serve(page_size=512),
            128 * 1024**2,
            decode_q_len=4,
        )
        assert chosen == _block(bq=4, bkv=3072, batch=8)


class TestSlidingWindowDecodePolicy:
    @pytest.fixture(autouse=True)
    def _heuristic_default(self, monkeypatch):
        monkeypatch.setattr(
            block_sizes,
            "choose_block_sizes",
            lambda *args: _block(),
        )

    def test_window_uses_one_anchored_block(self, monkeypatch):
        monkeypatch.setattr(block_sizes, "calculate_vmem_usage", lambda *args: 0)
        chosen = block_sizes.choose_sliding_window_block_sizes(
            _model(sliding_window=2048),
            _serve(page_size=512),
            128 * 1024**2,
            decode_q_len=4,
        )
        assert chosen == _block(bq=4, bkv=3072, batch=8)

    def test_window_caps_to_sequence_capacity(self, monkeypatch):
        monkeypatch.setattr(block_sizes, "calculate_vmem_usage", lambda *args: 0)
        chosen = block_sizes.choose_sliding_window_block_sizes(
            _model(sliding_window=2048),
            _serve(page_size=512, pages_per_seq=2),
            128 * 1024**2,
            decode_q_len=4,
        )
        assert chosen.bkv_sz == 1024

    def test_undersized_override_is_rejected(self, monkeypatch):
        monkeypatch.setattr(block_sizes, "calculate_vmem_usage", lambda *args: 0)
        with pytest.raises(ValueError, match="one anchored KV block"):
            block_sizes.choose_sliding_window_block_sizes(
                _model(sliding_window=2048),
                _serve(page_size=512),
                128 * 1024**2,
                decode_q_len=4,
                override=_block(bkv=1024),
            )
