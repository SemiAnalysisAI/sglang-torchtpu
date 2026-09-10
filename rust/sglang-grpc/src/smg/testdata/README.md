# Recorded `sglang.grpc.scheduler` request bytes

Each `.hex` file is one protobuf message exactly as a live sgl-model-gateway
router put it on the wire, captured against a throwaway `SglangScheduler`
recorder. They are recordings, not fixtures: the numbers in them are what SMG
sends, not what we think it should send.

The corpus is request-side only. No `GenerateResponse` recordings exist yet —
capturing them needs a CPU-backed engine, and this fork's CPU path needs an
`sgl_kernel` build that PyPI ships CUDA-only. The response-side tests in
`../response_map/tests.rs` are therefore written from the producer source and
say so.

| file | what it pins |
|---|---|
| `01_chat_completion_nonstream` | the ordinary chat request; `stream: false` |
| `02_chat_completion_stream` | the same with `stream: true` |
| `03_chat_with_tools` | `skip_special_tokens: false`, set whenever tools are present |
| `04_chat_logprobs_top3` | `return_logprob: true`, `logprob_start_len: -1`, `top_logprobs_num: 3` |
| `05_chat_n2` | `n: 2` reaching the wire from ordinary chat traffic |
| `06_stop_strings_json_schema` | `stop`, and the `json_schema` constraint arm |
| `07_embeddings` | `/v1/embeddings` |
| `09_maximal_sampling_params` | every sampling field set at once, plus `ebnf_grammar` |
| `10_v1_classify` | `/v1/classify`, which reuses `Embed` |
| `11_vision_model_card` | an image_url chat against a vision model card: `mm_inputs: None` |
| `12_harmony_gpt_oss` | the Harmony streaming path |
| `13_abort_on_disconnect` | the implicit abort from `AbortOnDropStream::Drop` |
| `20_pd_prefill_leg` / `21_pd_decode_leg` | a PD pair; the two are byte-identical |
