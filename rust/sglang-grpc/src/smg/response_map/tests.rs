//! These are hand-constructed from the **producer source**, not recorded.
//!
//! Phase 0's request-side corpus is real bytes (`../testdata/`), but no
//! `GenerateResponse` was ever captured from our own engine: this fork's CPU
//! path needs an `sgl_kernel` build that PyPI ships CUDA-only, so the run that
//! would have produced one could not start. Each test below therefore pins what
//! `tokenizer_manager.py` is *written* to emit. Where that is an inference
//! rather than a reading, the test says so.

use std::collections::HashMap;

use tonic::Code;

use super::{StreamMode, StreamTranslator, error_messages, with_request_id};
use crate::bridge::ResponseData;
use crate::smg_proto;

// ---------- fixtures ----------

/// `meta_info` as the bridge builds it: every value `json.dumps`-encoded.
fn meta(pairs: &[(&str, &str)]) -> HashMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

/// A streaming, non-terminal `meta_info`: `finish_reason` is the literal
/// four-character `null`.
fn streaming_meta(prompt: i32, completion: i32) -> HashMap<String, String> {
    meta(&[
        ("id", "\"chatcmpl-0\""),
        ("finish_reason", "null"),
        ("prompt_tokens", &prompt.to_string()),
        ("completion_tokens", &completion.to_string()),
        ("cached_tokens", "0"),
        ("weight_version", "\"default\""),
    ])
}

fn data(output_ids: &[i32], meta_info: HashMap<String, String>) -> ResponseData {
    ResponseData {
        text: None,
        output_ids: Some(output_ids.to_vec()),
        embedding: None,
        json_bytes: None,
        meta_info,
    }
}

fn chunk_of(response: &smg_proto::GenerateResponse) -> &smg_proto::GenerateStreamChunk {
    match response.response.as_ref() {
        Some(smg_proto::generate_response::Response::Chunk(chunk)) => chunk,
        other => panic!("expected a chunk, got {:?}", other),
    }
}

fn complete_of(response: &smg_proto::GenerateResponse) -> &smg_proto::GenerateComplete {
    match response.response.as_ref() {
        Some(smg_proto::generate_response::Response::Complete(complete)) => complete,
        other => panic!("expected a complete, got {:?}", other),
    }
}

// ---------- cumulative vs incremental ----------

/// T20. The headline. Passing cumulative ids through makes SMG render
/// `"a" "ab" "abc"` concatenated, with `completion_tokens` correct throughout,
/// so no counter and no log ever shows a problem.
#[test]
fn cumulative_chunks_are_differenced() {
    let mut translator = StreamTranslator::new(StreamMode::Cumulative);

    let first = translator
        .on_data(&data(&[1], streaming_meta(3, 1)))
        .unwrap()
        .unwrap();
    let second = translator
        .on_data(&data(&[1, 2], streaming_meta(3, 2)))
        .unwrap()
        .unwrap();
    let third = translator
        .on_data(&data(&[1, 2, 3], streaming_meta(3, 3)))
        .unwrap()
        .unwrap();

    assert_eq!(chunk_of(&first).token_ids, vec![1]);
    assert_eq!(chunk_of(&second).token_ids, vec![2]);
    assert_eq!(chunk_of(&third).token_ids, vec![3]);
}

/// T21. The converse bug: double-differencing an already-incremental stream
/// drops every token after the first.
#[test]
fn incremental_chunks_are_passed_through() {
    let mut translator = StreamTranslator::new(StreamMode::Incremental);

    for (input, expected) in [(1, 1), (2, 2), (3, 3)] {
        let message = translator
            .on_data(&data(&[input], streaming_meta(3, input)))
            .unwrap()
            .unwrap();
        assert_eq!(chunk_of(&message).token_ids, vec![expected as u32]);
    }

    let complete = translator
        .on_finished(&data(&[], streaming_meta(3, 3)))
        .unwrap();
    assert_eq!(
        complete_of(complete.last().unwrap()).output_ids,
        vec![1, 2, 3]
    );
}

/// T22. Someone flipping `--incremental-streaming-output` on one worker turns
/// from silent truncation into a loud error.
#[test]
fn non_prefix_chunk_in_cumulative_mode_errors() {
    let mut translator = StreamTranslator::new(StreamMode::Cumulative);
    translator
        .on_data(&data(&[1, 2], streaming_meta(3, 2)))
        .unwrap();

    let status = translator
        .on_data(&data(&[9, 9], streaming_meta(3, 4)))
        .unwrap_err();
    assert_eq!(status.code(), Code::Internal);
    assert!(
        status.message().contains("misdetected"),
        "{}",
        status.message()
    );
}

/// T23. Our `Finished` carries the whole cumulative sequence, so without the
/// residual chunk the stream is truncated by the final delta while
/// `completion_tokens` stays right.
#[test]
fn terminal_frame_emits_residual_chunk_then_complete() {
    let mut translator = StreamTranslator::new(StreamMode::Cumulative);
    translator
        .on_data(&data(&[1, 2], streaming_meta(3, 2)))
        .unwrap();

    let messages = translator
        .on_finished(&data(
            &[1, 2, 3, 4],
            meta(&[
                ("finish_reason", r#"{"type": "length", "length": 4}"#),
                ("prompt_tokens", "3"),
                ("completion_tokens", "4"),
                ("cached_tokens", "0"),
            ]),
        ))
        .unwrap();

    assert_eq!(messages.len(), 2, "residual chunk then complete");
    assert_eq!(chunk_of(&messages[0]).token_ids, vec![3, 4]);
    assert_eq!(complete_of(&messages[1]).output_ids, vec![1, 2, 3, 4]);
}

/// T24. Non-streaming SMG reconstructs the whole answer from
/// `complete.output_ids` alone, so a delta here truncates every non-streaming
/// response.
#[test]
fn complete_output_ids_is_the_full_sequence() {
    let mut translator = StreamTranslator::new(StreamMode::Cumulative);

    // The non-streaming path sends exactly one chunk, the terminal one.
    let messages = translator
        .on_finished(&data(
            &[7, 8, 9],
            meta(&[
                ("finish_reason", r#"{"type": "stop", "matched": 13}"#),
                ("prompt_tokens", "3"),
                ("completion_tokens", "3"),
            ]),
        ))
        .unwrap();

    assert_eq!(
        complete_of(messages.last().unwrap()).output_ids,
        vec![7, 8, 9]
    );
}

/// T25. Harmony adds one completion token per chunk regardless of content, so
/// a zero-token chunk inflates its usage.
#[test]
fn empty_delta_chunks_are_suppressed() {
    let mut translator = StreamTranslator::new(StreamMode::Cumulative);
    translator
        .on_data(&data(&[1], streaming_meta(3, 1)))
        .unwrap();

    assert!(
        translator
            .on_data(&data(&[1], streaming_meta(3, 1)))
            .unwrap()
            .is_none()
    );
}

/// T26. The literal four characters `null` are truthy to `contains_key` and
/// non-empty as a String; treating that as completion ends the stream after one
/// token.
#[test]
fn finish_reason_null_string_is_not_a_completion() {
    let mut translator = StreamTranslator::new(StreamMode::Cumulative);

    let message = translator
        .on_data(&data(&[1], streaming_meta(3, 1)))
        .unwrap()
        .unwrap();

    assert!(
        matches!(
            message.response,
            Some(smg_proto::generate_response::Response::Chunk(_))
        ),
        "a Data chunk is a chunk whatever finish_reason says"
    );
}

// ---------- finish_reason ----------

/// T27 and T28.
#[test]
fn finish_reason_object_maps_to_openai_strings() {
    use smg_proto::generate_complete::MatchedStop;

    let cases: Vec<(&str, &str, Option<MatchedStop>)> = vec![
        (
            r#"{"type": "stop", "matched": 13}"#,
            "stop",
            Some(MatchedStop::MatchedTokenId(13)),
        ),
        (
            r#"{"type": "stop", "matched": "</s>"}"#,
            "stop",
            Some(MatchedStop::MatchedStopStr("</s>".to_string())),
        ),
        // FINISH_MATCHED_TOKEN accepts a list and a single uint32 cannot carry
        // it; matched[0] would name the wrong stop token.
        (r#"{"type": "stop", "matched": [13, 14]}"#, "stop", None),
        (r#"{"type": "length", "length": 128}"#, "length", None),
        (
            r#"{"type": "abort", "message": "Aborted", "status_code": null, "err_type": null}"#,
            "abort",
            None,
        ),
    ];

    for (raw, expected_reason, expected_matched) in cases {
        let mut translator = StreamTranslator::new(StreamMode::Cumulative);
        let messages = translator
            .on_finished(&data(&[1], meta(&[("finish_reason", raw)])))
            .unwrap();
        let complete = complete_of(messages.last().unwrap());

        assert_eq!(complete.finish_reason, expected_reason, "{}", raw);
        assert_eq!(complete.matched_stop, expected_matched, "{}", raw);
    }
}

/// T29. `meta_info["id"]` is `json.dumps`-encoded, so it arrives with literal
/// quote characters in it; using it as the response id breaks every log join.
#[test]
fn meta_info_id_is_not_used_as_request_id() {
    let mut translator = StreamTranslator::new(StreamMode::Cumulative);
    let message = translator
        .on_data(&data(&[1], streaming_meta(3, 1)))
        .unwrap()
        .unwrap();

    assert_eq!(
        message.request_id, "",
        "the translator must not invent an id"
    );
    assert_eq!(
        with_request_id(message, "chatcmpl-0").request_id,
        "chatcmpl-0",
        "the id must come from the request, quote-free"
    );
}

// ---------- counters ----------

/// T31. SMG deliberately does not re-accumulate counters for SGLang, so
/// summing them here would inflate usage with plausible numbers.
#[test]
fn counts_are_passed_through_not_summed() {
    let mut translator = StreamTranslator::new(StreamMode::Cumulative);

    let mut seen = Vec::new();
    for (n, ids) in [(1, vec![1]), (2, vec![1, 2]), (3, vec![1, 2, 3])] {
        let message = translator
            .on_data(&data(&ids, streaming_meta(3, n)))
            .unwrap()
            .unwrap();
        seen.push(chunk_of(&message).completion_tokens);
    }

    assert_eq!(seen, vec![1, 2, 3]);
}

/// A3. A sentinel capture put `{111, 999, 222}` on the chunks and
/// `{555, 777, 333}` on the complete; the client reported 555 and 777. These
/// three are the ones that must be right.
#[test]
fn complete_counters_are_the_ones_smg_reports() {
    let mut translator = StreamTranslator::new(StreamMode::Cumulative);
    let messages = translator
        .on_finished(&data(
            &[1, 2],
            meta(&[
                ("finish_reason", r#"{"type": "length", "length": 2}"#),
                ("prompt_tokens", "555"),
                ("completion_tokens", "777"),
                ("cached_tokens", "333"),
            ]),
        ))
        .unwrap();
    let complete = complete_of(messages.last().unwrap());

    assert_eq!(complete.prompt_tokens, 555);
    assert_eq!(complete.completion_tokens, 777);
    assert_eq!(complete.cached_tokens, 333);
}

/// T30. Every `meta_info` value is JSON, so `"12"` is the encoding of the
/// integer 12; a value that is not an integer means the producer's encoding
/// changed and must not be rounded into a plausible number.
#[test]
fn counts_are_json_decoded_not_string_parsed() {
    let mut translator = StreamTranslator::new(StreamMode::Cumulative);
    let messages = translator
        .on_finished(&data(
            &[1],
            meta(&[
                ("finish_reason", r#"{"type": "length", "length": 1}"#),
                ("prompt_tokens", "12"),
            ]),
        ))
        .unwrap();
    assert_eq!(complete_of(messages.last().unwrap()).prompt_tokens, 12);
    assert_eq!(
        complete_of(messages.last().unwrap()).cached_tokens,
        0,
        "an absent counter is 0; cached_tokens is genuinely not emitted for embeddings"
    );

    let mut translator = StreamTranslator::new(StreamMode::Cumulative);
    let status = translator
        .on_finished(&data(&[1], meta(&[("prompt_tokens", "1e3")])))
        .unwrap_err();
    assert_eq!(status.code(), Code::Internal);
    assert!(status.message().contains("int32"), "{}", status.message());
}

// ---------- logprobs ----------

/// The JSON shape below is an **inference, not an observation**.
///
/// `detokenize_logprob_tokens` (`tokenizer_manager.py`) builds
/// `[logprob, token_id, text_or_null]` triples and
/// `detokenize_top_logprobs_tokens` wraps them one list per position, with a
/// JSON `null` for a position that has no top-k. Nothing has been captured off
/// our producer to confirm it: Phase 0's C1 run needed a CPU engine and this
/// fork's CPU path needs an `sgl_kernel` build that PyPI ships CUDA-only.
///
/// What would confirm it: one `sglang.runtime.v1.SglangService/Generate` call
/// against any CPU-backed server with `return_logprob: true` and
/// `top_logprobs_num: 3`, dumping `meta_info["output_token_logprobs"]` and
/// `["output_top_logprobs"]` verbatim into `../testdata/`.
///
/// The consumer half *is* pinned: SMG reads `OutputLogProbs.token_logprobs`,
/// `.token_ids` and `.top_logprobs.{values, token_ids}`, and those render into
/// OpenAI logprobs correctly.
fn inferred_logprob_meta(entries: usize) -> HashMap<String, String> {
    let triples: Vec<serde_json::Value> = (0..entries)
        .map(|i| serde_json::json!([-0.5 - i as f64, 100 + i as i64, null]))
        .collect();
    let tops: Vec<serde_json::Value> = (0..entries)
        .map(|i| serde_json::json!([[-0.5 - i as f64, 100 + i as i64, null], [-2.0, 999, null]]))
        .collect();

    meta(&[
        ("finish_reason", "null"),
        ("prompt_tokens", "3"),
        ("completion_tokens", &entries.to_string()),
        (
            "output_token_logprobs",
            &serde_json::Value::Array(triples).to_string(),
        ),
        (
            "output_top_logprobs",
            &serde_json::Value::Array(tops).to_string(),
        ),
    ])
}

/// T32. If this fails on a real capture, the producer's shape differs from the
/// inference documented above `inferred_logprob_meta` -- fix the parser, not
/// this test.
#[test]
fn output_logprobs_are_sliced_like_token_ids() {
    let mut translator = StreamTranslator::new(StreamMode::Cumulative);

    let first = translator
        .on_data(&data(&[100], inferred_logprob_meta(1)))
        .unwrap()
        .unwrap();
    let second = translator
        .on_data(&data(&[100, 101], inferred_logprob_meta(2)))
        .unwrap()
        .unwrap();

    let first_logprobs = chunk_of(&first).output_logprobs.as_ref().unwrap();
    assert_eq!(first_logprobs.token_ids, vec![100]);
    assert_eq!(first_logprobs.token_logprobs, vec![-0.5]);
    assert_eq!(first_logprobs.top_logprobs.len(), 1);
    assert_eq!(first_logprobs.top_logprobs[0].token_ids, vec![100, 999]);

    let second_logprobs = chunk_of(&second).output_logprobs.as_ref().unwrap();
    assert_eq!(
        second_logprobs.token_ids,
        vec![101],
        "the second chunk must carry only the new position, not both"
    );
    assert_eq!(second_logprobs.token_logprobs, vec![-1.5]);

    let messages = translator
        .on_finished(&data(&[100, 101, 102], inferred_logprob_meta(3)))
        .unwrap();
    let complete_logprobs = complete_of(messages.last().unwrap())
        .output_logprobs
        .as_ref()
        .unwrap();
    assert_eq!(
        complete_logprobs.token_ids,
        vec![100, 101, 102],
        "complete carries the cumulative list"
    );
    assert_eq!(complete_logprobs.top_logprobs.len(), 3);
}

/// T33. A logprob of exactly 0.0 means probability 1 -- a confident lie.
#[test]
fn input_logprobs_first_value_is_none() {
    let mut translator = StreamTranslator::new(StreamMode::Cumulative);
    let messages = translator
        .on_finished(&data(
            &[1],
            meta(&[
                ("finish_reason", r#"{"type": "length", "length": 1}"#),
                (
                    "input_token_logprobs",
                    r#"[[null, 151644, null], [-3.25, 8948, null]]"#,
                ),
            ]),
        ))
        .unwrap();

    let input = complete_of(messages.last().unwrap())
        .input_logprobs
        .as_ref()
        .unwrap();
    assert_eq!(input.token_ids, vec![151644, 8948]);
    assert_eq!(input.token_logprobs[0].value, None);
    assert_eq!(input.token_logprobs[1].value, Some(-3.25));
}

#[test]
fn absent_logprobs_leave_the_fields_empty() {
    let mut translator = StreamTranslator::new(StreamMode::Cumulative);
    let messages = translator
        .on_finished(&data(
            &[1],
            meta(&[("finish_reason", r#"{"type": "length", "length": 1}"#)]),
        ))
        .unwrap();

    let complete = complete_of(messages.last().unwrap());
    assert!(complete.output_logprobs.is_none());
    assert!(complete.input_logprobs.is_none());
}

// ---------- errors and hygiene ----------

/// The dual-emit error frame: a 1.0.0 client reads field 4 and stops, a 1.12.0
/// client skips it as reserved and takes the status. Neither hangs.
#[test]
fn errors_emit_both_a_message_and_a_status() {
    let (response, status) = error_messages("out of memory".to_string());

    match response.response {
        Some(smg_proto::generate_response::Response::Error(error)) => {
            assert_eq!(error.message, "out of memory");
        }
        other => panic!("expected an error frame, got {:?}", other),
    }
    assert_eq!(status.code(), Code::Internal);
    assert_eq!(status.message(), "out of memory");
}

/// SMG's proto types `token_ids` as `uint32`; the bridge hands us `i32`. A
/// negative would wrap into a huge token id.
#[test]
fn negative_token_ids_are_an_error_not_a_wrap() {
    let mut translator = StreamTranslator::new(StreamMode::Cumulative);
    let status = translator
        .on_data(&data(&[-1], streaming_meta(3, 1)))
        .unwrap_err();

    assert_eq!(status.code(), Code::Internal);
    assert!(
        status.message().contains("negative"),
        "{}",
        status.message()
    );
}

/// SMG hard-codes `None` for hidden states everywhere and never reads
/// `input_logprobs` off the chunk variant; filling them would be pure cost.
#[test]
fn chunk_leaves_the_fields_smg_never_reads_empty() {
    let mut translator = StreamTranslator::new(StreamMode::Cumulative);
    let message = translator
        .on_data(&data(&[1], streaming_meta(3, 1)))
        .unwrap()
        .unwrap();

    let chunk = chunk_of(&message);
    assert!(chunk.hidden_states.is_empty());
    assert!(chunk.input_logprobs.is_none());
    assert_eq!(chunk.index, 0, "n > 1 is rejected, so there is one stream");
}

// ---------- Embed ----------

fn embed_data(embedding: Option<Vec<f32>>) -> ResponseData {
    ResponseData {
        text: None,
        output_ids: None,
        embedding,
        json_bytes: None,
        meta_info: meta(&[("id", "\"embed-0\""), ("prompt_tokens", "2")]),
    }
}

/// A8. Confirmed safe: a 1.0.0 client skips the higher field numbers as
/// unknown. Emitting only the 1.12.0 layout produced a loud
/// `HTTP 500 embedding_no_response`; emitting only the 1.0.0 layout would give
/// a 1.12.0 client an empty embedding and no error at all.
#[test]
fn embed_response_carries_both_wire_layouts() {
    let response = super::embed_response("embed-0", &embed_data(Some(vec![0.25, -0.5]))).unwrap();

    match response.response.as_ref() {
        Some(smg_proto::embed_response::Response::Complete(complete)) => {
            assert_eq!(complete.embedding, vec![0.25, -0.5]);
            assert_eq!(complete.embedding_dim, 2, "derived from the vector length");
            assert_eq!(complete.prompt_tokens, 2);
            assert_eq!(
                complete.cached_tokens, 0,
                "unknown, not zero: not emitted for embeddings"
            );
        }
        other => panic!("expected the 1.0.0 layout, got {:?}", other),
    }

    assert_eq!(response.embedding_v112, vec![0.25, -0.5]);
    assert_eq!(response.embedding_dim_v112, 2);
    assert_eq!(response.prompt_tokens_v112, 2);
    assert_eq!(response.request_id, "embed-0");
}

/// An absent embedding is forwarded as empty, matching the native Embed's
/// `unwrap_or_default()` rather than erroring where it would not.
#[test]
fn embed_absent_embedding_matches_the_native_path() {
    let response = super::embed_response("embed-0", &embed_data(None)).unwrap();

    match response.response.as_ref() {
        Some(smg_proto::embed_response::Response::Complete(complete)) => {
            assert!(complete.embedding.is_empty());
            assert_eq!(complete.embedding_dim, 0);
        }
        other => panic!("expected the 1.0.0 layout, got {:?}", other),
    }
    assert!(response.embedding_v112.is_empty());
}
