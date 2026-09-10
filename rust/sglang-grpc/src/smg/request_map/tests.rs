//! Two kinds of test live here.
//!
//! The `recorded_*` tests decode real bytes from `../testdata/`, captured off a
//! live sgl-model-gateway router, and assert on what the mapper does with them.
//! Everything else is hand-constructed to reach a case the recordings do not
//! contain — a `SamplingParams::default()`, an out-of-range token id, a
//! multimodal request — and is a claim about the *producer source*, not an
//! observation.

use std::collections::HashMap;

use prost::Message;
use serde_json::{Value, json};
use tonic::Code;

use super::{MappedGenerate, generate_to_dict};
use crate::smg_proto;

// ---------- fixtures ----------

/// What SMG's builders actually seed: never `SamplingParams::default()`.
fn smg_sampling_params() -> smg_proto::SamplingParams {
    smg_proto::SamplingParams {
        temperature: 1.0,
        top_p: 1.0,
        top_k: -1,
        repetition_penalty: 1.0,
        n: 1,
        skip_special_tokens: true,
        spaces_between_special_tokens: true,
        max_new_tokens: Some(8),
        ..Default::default()
    }
}

fn smg_generate_request() -> smg_proto::GenerateRequest {
    smg_proto::GenerateRequest {
        request_id: "chatcmpl-0".to_string(),
        tokenized: Some(smg_proto::TokenizedInput {
            original_text: "hi".to_string(),
            input_ids: vec![151644, 872, 198],
        }),
        sampling_params: Some(smg_sampling_params()),
        logprob_start_len: -1,
        ..Default::default()
    }
}

fn recorded_generate(name: &str) -> smg_proto::GenerateRequest {
    smg_proto::GenerateRequest::decode(recorded_bytes(name).as_slice())
        .unwrap_or_else(|err| panic!("{} is not a GenerateRequest: {}", name, err))
}

fn recorded_bytes(name: &str) -> Vec<u8> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/smg/testdata")
        .join(name);
    let hex = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("{} is readable: {}", path.display(), err));
    let hex = hex.trim();
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("testdata is hex"))
        .collect()
}

fn sampling(mapped: &MappedGenerate) -> &Value {
    mapped
        .dict
        .get("sampling_params")
        .expect("sampling_params was mapped")
}

fn map_ok(req: &smg_proto::GenerateRequest) -> MappedGenerate {
    generate_to_dict(req).expect("request maps cleanly")
}

// ---------- recorded corpus ----------

/// T1. The single highest-value field in the request table: SMG sets it false
/// whenever tools are present, and dropping it strips the tool-call special
/// tokens so the tool parser finds nothing and the model appears not to call
/// tools. No exception, no log.
#[test]
fn recorded_skip_special_tokens_false_reaches_the_dict() {
    let req = recorded_generate("03_chat_with_tools.generate_request.hex");
    assert!(
        !req.sampling_params.as_ref().unwrap().skip_special_tokens,
        "the recording is the tools case; it must carry skip_special_tokens: false"
    );

    let mapped = map_ok(&req);
    assert_eq!(sampling(&mapped)["skip_special_tokens"], json!(false));
}

/// T2 and T4. `skip_special_tokens` is field 11 in SMG's message and
/// `stop_token_ids` is field 11 in ours; `spaces_between_special_tokens` is 12
/// there and `ignore_eos` is 12 here. Both cross-wirings are silent: one runs
/// generation past EOS to max_new_tokens, the other truncates every generation
/// at token 1.
#[test]
fn recorded_field_11_and_12_do_not_cross_wire() {
    let mapped = map_ok(&recorded_generate(
        "01_chat_completion_nonstream.generate_request.hex",
    ));
    let sampling = sampling(&mapped);

    assert_eq!(sampling["skip_special_tokens"], json!(true));
    assert_eq!(sampling["spaces_between_special_tokens"], json!(true));
    assert_eq!(sampling["ignore_eos"], json!(false));
    assert!(
        sampling.get("stop_token_ids").is_none(),
        "skip_special_tokens must not leak into stop_token_ids"
    );
}

/// T10. The abort key. A substituted UUID makes every `AbortOnDropStream::Drop`
/// a no-op, so each disconnected client leaves a request generating to
/// max_new_tokens.
#[test]
fn recorded_request_id_is_used_verbatim_as_rid() {
    let req = recorded_generate("01_chat_completion_nonstream.generate_request.hex");
    let expected = req.request_id.clone();
    assert!(expected.starts_with("chatcmpl-"));

    let mapped = map_ok(&req);
    assert_eq!(mapped.rid, expected);
    assert_eq!(mapped.dict["rid"], json!(expected));
}

/// T18's amended form. SMG already sends -1 when logprobs are off, so the
/// mapper forwards the field rather than special-casing it; the failure this
/// guards is a mapper that defaults the key to 0 and asks the engine for
/// full-prompt logprobs on every request.
#[test]
fn recorded_logprob_fields_are_forwarded_verbatim() {
    let off = map_ok(&recorded_generate(
        "01_chat_completion_nonstream.generate_request.hex",
    ));
    assert_eq!(off.dict["return_logprob"], json!(false));
    assert_eq!(off.dict["logprob_start_len"], json!(-1));
    assert_eq!(off.dict["top_logprobs_num"], json!(0));

    let on = map_ok(&recorded_generate(
        "04_chat_logprobs_top3.generate_request.hex",
    ));
    assert_eq!(on.dict["return_logprob"], json!(true));
    assert_eq!(on.dict["logprob_start_len"], json!(-1));
    assert_eq!(on.dict["top_logprobs_num"], json!(3));
}

/// T15. `n: 2` reaches the wire from ordinary chat traffic, so this is a real
/// functional loss, taken deliberately: the alternative is returning one
/// stream's tokens as if they were two samples.
#[test]
fn recorded_n_greater_than_one_is_rejected() {
    let req = recorded_generate("05_chat_n2.generate_request.hex");
    assert_eq!(req.sampling_params.as_ref().unwrap().n, 2);

    let status = generate_to_dict(&req).unwrap_err();
    assert_eq!(status.code(), Code::Unimplemented);
    assert!(status.message().contains("n > 1"), "{}", status.message());
}

/// T5 and T6. `ebnf_grammar` must land on `ebnf`, not `regex`: usually loud,
/// but silent whenever the grammar text also parses as a regex.
#[test]
fn recorded_constraint_arms_land_on_their_own_keys() {
    let ebnf = map_ok(&recorded_generate(
        "09_maximal_sampling_params.generate_request.hex",
    ));
    assert_eq!(sampling(&ebnf)["ebnf"], json!("root ::= \"a\""));
    assert!(sampling(&ebnf).get("regex").is_none());

    let json_schema = map_ok(&recorded_generate(
        "06_stop_strings_json_schema.generate_request.hex",
    ));
    assert_eq!(
        sampling(&json_schema)["json_schema"],
        json!("{\"type\":\"object\"}")
    );
    assert!(sampling(&json_schema).get("ebnf").is_none());
    assert_eq!(sampling(&json_schema)["stop"], json!(["\n\n", "END"]));
}

/// The one recording that touches every sampling field at once. Values come
/// from the capture, so a rename or a dropped key shows up here first.
#[test]
fn recorded_maximal_sampling_params_map_completely() {
    let mapped = map_ok(&recorded_generate(
        "09_maximal_sampling_params.generate_request.hex",
    ));
    let sampling = sampling(&mapped);

    assert_eq!(sampling["temperature"], json!(0.3f32));
    assert_eq!(sampling["top_p"], json!(0.8f32));
    assert_eq!(sampling["top_k"], json!(7));
    assert_eq!(sampling["min_p"], json!(0.02f32));
    assert_eq!(sampling["frequency_penalty"], json!(0.1f32));
    assert_eq!(sampling["presence_penalty"], json!(0.2f32));
    assert_eq!(sampling["repetition_penalty"], json!(1.05f32));
    assert_eq!(sampling["min_new_tokens"], json!(2));
    assert_eq!(sampling["ignore_eos"], json!(true));
    assert_eq!(sampling["no_stop_trim"], json!(true));
    assert_eq!(sampling["skip_special_tokens"], json!(false));
    assert_eq!(sampling["stop"], json!(["X"]));

    assert_eq!(mapped.dict["return_hidden_states"], json!(true));
    assert_eq!(mapped.dict["logprob_start_len"], json!(2));
}

/// A9. SMG drops images before the wire even for a vision model card, so the
/// multimodal rejection has never fired in practice. Recorded so that a future
/// SMG that *does* send `mm_inputs` is a change in this assertion, not a
/// silent text-only answer to an image prompt.
#[test]
fn recorded_vision_request_carries_no_mm_inputs() {
    let req = recorded_generate("11_vision_model_card.generate_request.hex");
    assert!(
        req.mm_inputs.is_none(),
        "SMG dropped images before the wire when this was recorded"
    );
    assert!(generate_to_dict(&req).is_ok());
}

/// T11 and T12 cover the widening; this covers the plumbing, and pins that the
/// two PD legs map identically. The recorded pair is byte-identical on the
/// wire, so any per-call state in the mapper would show up as a mismatch.
#[test]
fn recorded_pd_legs_map_to_the_same_bootstrap_room() {
    let prefill = map_ok(&recorded_generate("20_pd_prefill_leg.generate_request.hex"));
    let decode = map_ok(&recorded_generate("21_pd_decode_leg.generate_request.hex"));

    assert_eq!(prefill.bootstrap_room, Some(507_686_309));
    assert_eq!(prefill.bootstrap_room, decode.bootstrap_room);
    assert_eq!(prefill.dict["bootstrap_room"], json!(507_686_309));
    assert_eq!(prefill.dict["bootstrap_host"], json!("127.0.0.1"));
    assert_eq!(prefill.dict["bootstrap_port"], json!(9001));
    assert_eq!(
        decode.dict["bootstrap_room"],
        prefill.dict["bootstrap_room"]
    );
}

#[test]
fn recorded_non_pd_request_has_no_bootstrap_keys() {
    let mapped = map_ok(&recorded_generate(
        "01_chat_completion_nonstream.generate_request.hex",
    ));

    assert_eq!(mapped.bootstrap_room, None);
    for key in ["bootstrap_host", "bootstrap_port", "bootstrap_room"] {
        assert!(!mapped.dict.contains_key(key), "{} must be absent", key);
    }
}

#[test]
fn recorded_stream_flag_round_trips() {
    let non_streaming = map_ok(&recorded_generate(
        "01_chat_completion_nonstream.generate_request.hex",
    ));
    assert!(!non_streaming.stream);
    assert_eq!(non_streaming.dict["stream"], json!(false));

    let streaming = map_ok(&recorded_generate(
        "02_chat_completion_stream.generate_request.hex",
    ));
    assert!(streaming.stream);
    assert_eq!(streaming.dict["stream"], json!(true));
}

/// Every recording must map or reject deliberately; nothing may panic. The
/// Harmony recording has no assertion of its own, and this is what covers it.
#[test]
fn every_recorded_generate_request_maps_or_rejects_deliberately() {
    let expected_rejections: HashMap<&str, Code> =
        HashMap::from([("05_chat_n2.generate_request.hex", Code::Unimplemented)]);

    for name in [
        "01_chat_completion_nonstream.generate_request.hex",
        "02_chat_completion_stream.generate_request.hex",
        "03_chat_with_tools.generate_request.hex",
        "04_chat_logprobs_top3.generate_request.hex",
        "05_chat_n2.generate_request.hex",
        "06_stop_strings_json_schema.generate_request.hex",
        "09_maximal_sampling_params.generate_request.hex",
        "11_vision_model_card.generate_request.hex",
        "12_harmony_gpt_oss.generate_request.hex",
        "20_pd_prefill_leg.generate_request.hex",
        "21_pd_decode_leg.generate_request.hex",
    ] {
        let result = generate_to_dict(&recorded_generate(name));
        match expected_rejections.get(name) {
            Some(code) => assert_eq!(result.err().map(|s| s.code()), Some(*code), "{}", name),
            None => {
                let mapped = result.unwrap_or_else(|s| panic!("{} rejected: {}", name, s));
                assert!(!mapped.rid.is_empty(), "{}", name);
                assert!(mapped.dict.contains_key("input_ids"), "{}", name);
            }
        }
    }
}

/// The dict is splatted into `GenerateReqInput(**req_dict)` /
/// `EmbeddingReqInput(**req_dict)`, both dataclasses, so an unknown key is a
/// TypeError at request time rather than a mapping bug found in review. These
/// two lists are the fields this layer claims exist on the Python side; adding
/// a key without adding it here is what the test is for.
///
/// `EmbeddingReqInput` in particular has neither `routed_dp_rank` nor
/// `data_parallel_rank`, which is why `embed_to_dict` rejects a non-zero rank
/// instead of forwarding it the way `generate_to_dict` does.
#[test]
fn the_dict_only_uses_keys_the_python_dataclasses_declare() {
    const GENERATE_REQ_INPUT_FIELDS: &[&str] = &[
        "rid",
        "input_ids",
        "sampling_params",
        "return_logprob",
        "logprob_start_len",
        "top_logprobs_num",
        "token_ids_logprob",
        "return_hidden_states",
        "bootstrap_host",
        "bootstrap_port",
        "bootstrap_room",
        "custom_logit_processor",
        "lora_path",
        "routed_dp_rank",
        "stream",
        "received_time",
    ];
    const EMBEDDING_REQ_INPUT_FIELDS: &[&str] = &["rid", "input_ids", "received_time"];
    const SAMPLING_PARAMS_FIELDS: &[&str] = &[
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "frequency_penalty",
        "presence_penalty",
        "repetition_penalty",
        "max_new_tokens",
        "stop",
        "stop_token_ids",
        "skip_special_tokens",
        "spaces_between_special_tokens",
        "min_new_tokens",
        "ignore_eos",
        "no_stop_trim",
        "stream_interval",
        "logit_bias",
        "custom_params",
        "regex",
        "json_schema",
        "ebnf",
        "structural_tag",
    ];

    let mut maximal = recorded_generate("09_maximal_sampling_params.generate_request.hex");
    maximal.data_parallel_rank = 2;
    maximal.lora_id = "adapter".to_string();
    maximal.custom_logit_processor = "{}".to_string();
    maximal.disaggregated_params = Some(smg_proto::DisaggregatedParams {
        bootstrap_host: "127.0.0.1".to_string(),
        bootstrap_port: 9001,
        bootstrap_room: 7,
    });
    maximal.sampling_params = Some(smg_proto::SamplingParams {
        stop_token_ids: vec![13],
        stream_interval: Some(1),
        logit_bias: [("13".to_string(), -1.0f32)].into_iter().collect(),
        custom_params: Some(prost_types::Struct::default()),
        ..maximal.sampling_params.clone().unwrap()
    });

    let mapped = map_ok(&maximal);
    for key in mapped.dict.keys() {
        assert!(
            GENERATE_REQ_INPUT_FIELDS.contains(&key.as_str()),
            "{} is not a GenerateReqInput field",
            key
        );
    }
    for key in sampling(&mapped).as_object().unwrap().keys() {
        assert!(
            SAMPLING_PARAMS_FIELDS.contains(&key.as_str()),
            "{} is not a SamplingParams field",
            key
        );
    }

    let embed = super::embed_to_dict(&recorded_embed("07_embeddings.embed_request.hex")).unwrap();
    for key in embed.dict.keys() {
        assert!(
            EMBEDDING_REQ_INPUT_FIELDS.contains(&key.as_str()),
            "{} is not an EmbeddingReqInput field",
            key
        );
    }
}

// ---------- constructed cases the recordings do not reach ----------

/// T3, the converse of the field-12 collision.
#[test]
fn ignore_eos_true_maps_to_ignore_eos() {
    let mut req = smg_generate_request();
    req.sampling_params = Some(smg_proto::SamplingParams {
        ignore_eos: true,
        spaces_between_special_tokens: false,
        ..smg_sampling_params()
    });

    let mapped = map_ok(&req);
    assert_eq!(sampling(&mapped)["ignore_eos"], json!(true));
    assert_eq!(
        sampling(&mapped)["spaces_between_special_tokens"],
        json!(false)
    );
}

/// T7. Dormant today because SMG's builders seed semantic defaults; it wakes
/// the first time an upstream path uses `..Default::default()`, and then
/// temperature 0 means greedy decoding for every request with no complaint.
#[test]
fn default_sampling_params_are_rejected() {
    let mut req = smg_generate_request();
    req.sampling_params = Some(smg_proto::SamplingParams::default());

    let status = generate_to_dict(&req).unwrap_err();
    assert_eq!(status.code(), Code::InvalidArgument);
    assert!(
        status.message().contains("proto3 defaults"),
        "{}",
        status.message()
    );
}

/// T8. Guards against an `unwrap_or_default()` slipping in later, which would
/// hit the same all-zeros case the sentinel above exists to catch.
#[test]
fn absent_sampling_params_omits_the_key() {
    let mut req = smg_generate_request();
    req.sampling_params = None;

    let mapped = map_ok(&req);
    assert!(!mapped.dict.contains_key("sampling_params"));
}

/// T9. Every recording has `data_parallel_rank: 0`, which is both "unset" and
/// a valid rank; forwarding it unconditionally pins every request to rank 0,
/// collapsing throughput without a single error.
#[test]
fn data_parallel_rank_zero_is_not_forwarded() {
    let mut req = smg_generate_request();
    req.data_parallel_rank = 0;
    assert!(!map_ok(&req).dict.contains_key("routed_dp_rank"));

    req.data_parallel_rank = 2;
    assert_eq!(map_ok(&req).dict["routed_dp_rank"], json!(2));
}

/// T10, the empty half.
#[test]
fn empty_request_id_is_rejected_rather_than_replaced() {
    let mut req = smg_generate_request();
    req.request_id = String::new();

    let status = generate_to_dict(&req).unwrap_err();
    assert_eq!(status.code(), Code::InvalidArgument);
    assert!(
        status.message().contains("abort key"),
        "{}",
        status.message()
    );
}

/// T14.
#[test]
fn multimodal_and_input_embeds_are_rejected() {
    let mut with_mm = smg_generate_request();
    with_mm.mm_inputs = Some(smg_proto::MultimodalInputs {
        image_urls: vec!["https://example/cat.png".to_string()],
        ..Default::default()
    });
    assert_eq!(
        generate_to_dict(&with_mm).unwrap_err().code(),
        Code::Unimplemented
    );

    let mut with_embeds = smg_generate_request();
    with_embeds.input_embeds = vec![0.1, 0.2];
    assert_eq!(
        generate_to_dict(&with_embeds).unwrap_err().code(),
        Code::Unimplemented
    );
}

/// T17. `as i32` here would turn u32::MAX into -1 and hand the engine a
/// negative token id.
#[test]
fn token_ids_are_widened_not_wrapped() {
    let mut req = smg_generate_request();
    req.tokenized = Some(smg_proto::TokenizedInput {
        original_text: String::new(),
        input_ids: vec![u32::MAX],
    });

    let status = generate_to_dict(&req).unwrap_err();
    assert_eq!(status.code(), Code::InvalidArgument);
    assert!(status.message().contains("int32"), "{}", status.message());
}

#[test]
fn missing_tokenized_input_is_rejected() {
    let mut req = smg_generate_request();
    req.tokenized = None;

    assert_eq!(
        generate_to_dict(&req).unwrap_err().code(),
        Code::InvalidArgument
    );
}

/// A9's other half: `lora_id` and `custom_logit_processor` are empty strings on
/// every recording, and an empty string is not the same as "set to empty".
#[test]
fn empty_optional_strings_are_omitted_not_forwarded() {
    let mapped = map_ok(&smg_generate_request());

    assert!(!mapped.dict.contains_key("lora_path"));
    assert!(!mapped.dict.contains_key("custom_logit_processor"));

    let mut req = smg_generate_request();
    req.lora_id = "adapter-a".to_string();
    req.custom_logit_processor = "{\"callable\": \"x\"}".to_string();
    let mapped = map_ok(&req);
    assert_eq!(mapped.dict["lora_path"], json!("adapter-a"));
    assert_eq!(
        mapped.dict["custom_logit_processor"],
        json!("{\"callable\": \"x\"}")
    );
}

/// Python's `CustomParamValue` admits a scalar, a flat list, or a flat dict.
/// Truncating a deeper value would send the scheduler something it cannot use
/// while reporting success.
#[test]
fn custom_params_reject_nesting_rather_than_truncating() {
    let scalar = prost_types::Value {
        kind: Some(prost_types::value::Kind::NumberValue(2.0)),
    };
    let flat_list = prost_types::Value {
        kind: Some(prost_types::value::Kind::ListValue(
            prost_types::ListValue {
                values: vec![scalar.clone()],
            },
        )),
    };
    let nested_list = prost_types::Value {
        kind: Some(prost_types::value::Kind::ListValue(
            prost_types::ListValue {
                values: vec![flat_list.clone()],
            },
        )),
    };

    let mut req = smg_generate_request();
    req.sampling_params = Some(smg_proto::SamplingParams {
        custom_params: Some(prost_types::Struct {
            fields: [("a".to_string(), flat_list)].into_iter().collect(),
        }),
        ..smg_sampling_params()
    });
    let mapped = map_ok(&req);
    assert_eq!(sampling(&mapped)["custom_params"], json!({"a": [2.0]}));

    req.sampling_params = Some(smg_proto::SamplingParams {
        custom_params: Some(prost_types::Struct {
            fields: [("a".to_string(), nested_list)].into_iter().collect(),
        }),
        ..smg_sampling_params()
    });
    let status = generate_to_dict(&req).unwrap_err();
    assert_eq!(status.code(), Code::InvalidArgument);
    assert!(
        status.message().contains("nests deeper"),
        "{}",
        status.message()
    );
}

/// An absent `max_new_tokens` means 128 on the Python side, not "unbounded".
/// The mapper must not invent a value to paper over that.
#[test]
fn absent_max_new_tokens_is_left_absent() {
    let mut req = smg_generate_request();
    req.sampling_params = Some(smg_proto::SamplingParams {
        max_new_tokens: None,
        ..smg_sampling_params()
    });

    let mapped = map_ok(&req);
    assert!(sampling(&mapped).get("max_new_tokens").is_none());
}

#[test]
fn logit_bias_and_stream_interval_are_mapped_when_present() {
    let mut req = smg_generate_request();
    req.sampling_params = Some(smg_proto::SamplingParams {
        logit_bias: [("13".to_string(), -100.0f32)].into_iter().collect(),
        stream_interval: Some(4),
        stop_token_ids: vec![13, 14],
        ..smg_sampling_params()
    });

    let mapped = map_ok(&req);
    assert_eq!(sampling(&mapped)["logit_bias"], json!({"13": -100.0f32}));
    assert_eq!(sampling(&mapped)["stream_interval"], json!(4));
    assert_eq!(sampling(&mapped)["stop_token_ids"], json!([13, 14]));
}

// ---------- Embed ----------

fn recorded_embed(name: &str) -> smg_proto::EmbedRequest {
    smg_proto::EmbedRequest::decode(recorded_bytes(name).as_slice())
        .unwrap_or_else(|err| panic!("{} is not an EmbedRequest: {}", name, err))
}

#[test]
fn recorded_embed_requests_map() {
    for name in [
        "07_embeddings.embed_request.hex",
        "10_v1_classify.embed_request.hex",
    ] {
        let req = recorded_embed(name);
        let mapped = super::embed_to_dict(&req).unwrap_or_else(|s| panic!("{}: {}", name, s));

        assert_eq!(mapped.dict["rid"], json!(req.request_id), "{}", name);
        assert!(mapped.dict["input_ids"].as_array().is_some(), "{}", name);
        assert!(
            !mapped.dict.contains_key("sampling_params"),
            "{}: the proto calls these dummy params for compatibility",
            name
        );
        assert!(!mapped.dict.contains_key("routed_dp_rank"), "{}", name);
    }
}

/// T16. Each of these silently changes the question being answered: a relevance
/// score computed without segment ids is a plausible wrong number.
#[test]
fn cross_encoder_embed_is_rejected() {
    let base = recorded_embed("07_embeddings.embed_request.hex");

    let mut with_token_types = base.clone();
    with_token_types.token_type_ids = vec![0, 1];

    let mut cross_encoder = base.clone();
    cross_encoder.is_cross_encoder = true;

    let mut with_texts = base.clone();
    with_texts.texts = vec!["query".to_string()];

    let mut with_mm = base.clone();
    with_mm.mm_inputs = Some(smg_proto::MultimodalInputs::default());

    for req in [with_token_types, cross_encoder, with_texts, with_mm] {
        assert_eq!(
            super::embed_to_dict(&req).unwrap_err().code(),
            Code::Unimplemented
        );
    }
}

#[test]
fn embed_data_parallel_rank_is_rejected_because_the_dict_has_no_key() {
    let mut req = recorded_embed("07_embeddings.embed_request.hex");
    assert_eq!(req.data_parallel_rank, 0, "every recording has it unset");
    assert!(
        !super::embed_to_dict(&req)
            .unwrap()
            .dict
            .contains_key("routed_dp_rank")
    );

    // EmbeddingReqInput has no routed_dp_rank and no data_parallel_rank, so
    // forwarding it raises a TypeError from the dataclass and dropping it
    // silently discards SMG's routing decision.
    req.data_parallel_rank = 3;
    assert_eq!(
        super::embed_to_dict(&req).unwrap_err().code(),
        Code::Unimplemented
    );
}
