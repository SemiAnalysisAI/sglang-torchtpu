from unittest.mock import MagicMock, patch

from sglang.srt.layers.sampler import Sampler


def test_tpu_sampled_tokens_are_broadcast_across_tp() -> None:
    token_ids = MagicMock()
    token_ids.device.type = "tpu"
    sampling_info = MagicMock()

    for dp_attention_enabled in (False, True):
        tp_group = MagicMock()
        attention_tp_group = MagicMock()
        with (
            patch(
                "sglang.srt.layers.sampler.is_dp_attention_enabled",
                return_value=dp_attention_enabled,
            ),
            patch("sglang.srt.layers.sampler.get_tp_group", return_value=tp_group),
            patch(
                "sglang.srt.layers.sampler.get_attention_tp_group",
                return_value=attention_tp_group,
            ),
        ):
            Sampler._sync_token_ids_across_tp(None, token_ids, sampling_info)

        expected_group = attention_tp_group if dp_attention_enabled else tp_group
        expected_group.broadcast.assert_called_once_with(token_ids, src=0)
