"""Unit tests for request construction in the encode-disaggregation path."""

import unittest
from array import array
from http import HTTPStatus
from types import MethodType, SimpleNamespace

from fastapi import HTTPException

from sglang.srt.disaggregation.encode_receiver import MMReceiverBase
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.utils import ImageData
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestEncodeReceiverRequestConstruction(CustomTestCase):
    def test_extra_key_and_cache_salt_are_forwarded(self):
        scheduler = SimpleNamespace(
            model_config=SimpleNamespace(hf_eos_token_id={2}, vocab_size=128),
            disaggregation_mode=DisaggregationMode.NULL,
            metrics_reporter=SimpleNamespace(enable_metrics=False),
            metrics_collector=None,
            dllm_config=None,
            tokenizer=object(),
        )
        receiver = SimpleNamespace(scheduler=scheduler)
        recv_req = SimpleNamespace(
            rid="request-1",
            input_text="hello",
            input_ids=array("q", [1, 2]),
            sampling_params=SamplingParams(max_new_tokens=1),
            return_logprob=False,
            top_logprobs_num=0,
            token_ids_logprob=None,
            stream=False,
            lora_id=None,
            input_embeds=None,
            custom_logit_processor=None,
            require_reasoning=False,
            return_hidden_states=False,
            return_routed_experts=False,
            routed_experts_start_len=0,
            bootstrap_host=None,
            bootstrap_port=None,
            bootstrap_room=None,
            routed_dp_rank=None,
            disagg_prefill_dp_rank=None,
            priority=None,
            extra_key="classification",
            cache_salt="tenant-a",
            http_worker_ipc=None,
        )

        req = MMReceiverBase.create_req(receiver, recv_req)

        self.assertEqual(req.extra_key, "classification")
        self.assertEqual(req.cache_salt, "tenant-a")


class TestEncodeDispatchWithoutEncoder(CustomTestCase):
    """Dispatch must fail loudly when no encoder is registered.

    Regression: health checks evicted the only encoder, the dispatch quietly
    cleared need_wait_for_mm_inputs, and the language-only server encoded the
    image itself. That server precompiles no ViT patch buckets, so the call
    recompiled and every scheduler died under fail_on_recompile.
    """

    @staticmethod
    def _receiver(encode_urls, backend="mooncake"):
        receiver = SimpleNamespace(
            encode_urls=encode_urls,
            encoder_transfer_backend=backend,
            _assign_items_by_modality=lambda mm_data, count: {},
        )
        receiver._extract_url_data = MethodType(
            MMReceiverBase._extract_url_data, receiver
        )
        return receiver

    @staticmethod
    def _request(image_data):
        return SimpleNamespace(
            rid="request-1",
            mm_content_hashes=None,
            image_data=image_data,
            video_data=None,
            audio_data=None,
            need_wait_for_mm_inputs=True,
        )

    def test_rejects_mm_request_when_no_encoder_is_registered(self):
        receiver = self._receiver(encode_urls=[])
        obj = self._request([ImageData(url="http://example.invalid/a.png")])

        with self.assertRaises(HTTPException) as raised:
            MMReceiverBase._send_encode_request(receiver, obj)

        self.assertEqual(raised.exception.status_code, HTTPStatus.SERVICE_UNAVAILABLE)

    def test_text_only_request_is_not_rejected(self):
        # Nothing to dispatch is not an encoder outage: the request must go
        # through with the wait flag cleared.
        receiver = self._receiver(encode_urls=[])
        obj = self._request(None)

        MMReceiverBase._send_encode_request(receiver, obj)

        self.assertFalse(obj.need_wait_for_mm_inputs)

    def test_dispatches_when_an_encoder_is_registered(self):
        receiver = self._receiver(encode_urls=["http://encoder.invalid:47904"])
        obj = self._request([ImageData(url="http://example.invalid/a.png")])

        MMReceiverBase._send_encode_request(receiver, obj)

        self.assertTrue(obj.need_wait_for_mm_inputs)
        self.assertEqual(obj.encoder_urls, ["http://encoder.invalid:47904"])


if __name__ == "__main__":
    unittest.main()
