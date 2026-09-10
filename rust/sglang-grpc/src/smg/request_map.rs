//! SMG `GenerateRequest` / `EmbedRequest` -> the Python request dict.
//!
//! The dict, not `sglang.runtime.v1`, is the internal representation: eleven
//! SMG fields have no `runtime.v1` counterpart but are accepted by
//! `GenerateReqInput` / `SamplingParams`, and one of them is
//! `skip_special_tokens`, which SMG sets false whenever tools are present.
//!
//! Every function here is a free function over plain data. `PyBridge` is a
//! concrete struct with no mock, so this is what makes the mapping testable at
//! all — not a testing convenience but a constraint on the shape of the module.

use std::collections::HashMap;

use serde_json::Value;
use tonic::Status;

use crate::smg::rooms::widen_room;
use crate::smg_proto;
use crate::utils::now_timestamp;

/// `sglang.runtime.v1.GenerateResponse` carries `output_ids`, `meta_info` and
/// `finished` and no `index`, so the native response protocol cannot say which
/// of N sample streams a message belongs to; one stream is the limit the
/// service already has. This rejects rather than returning that one stream as
/// if it were N, which reads as a single fluent wrong completion.
///
/// Carrying `index` would mean adding a field upstream does not define, so the
/// limit is matched, not worked around. If upstream ever adds one, honouring it
/// is reading it in `bridge.rs` and keying `response_map::StreamTranslator` by
/// it; the translator already stamps the index it holds on every message.
const MAX_SUPPORTED_PARALLEL_SAMPLES: i32 = 1;

/// A mapped `GenerateRequest`, ready for `PyBridge::submit_request`.
#[derive(Debug)]
pub struct MappedGenerate {
    /// SMG's `request_id` verbatim. Anything else and `Abort` matches nothing.
    pub rid: String,
    pub dict: HashMap<String, Value>,
    /// The widened room, when the request is a PD leg. Kept out of the dict's
    /// shadow so the service can register it against concurrent reuse.
    pub bootstrap_room: Option<i64>,
    pub stream: bool,
}

pub fn generate_to_dict(req: &smg_proto::GenerateRequest) -> Result<MappedGenerate, Status> {
    let rid = req.request_id.trim();
    if rid.is_empty() {
        // Substituting a UUID here would produce a request no Abort can find.
        return Err(Status::invalid_argument(
            "GenerateRequest.request_id is required; it is the abort key",
        ));
    }

    reject_unsupported_inputs(req)?;

    let tokenized = req.tokenized.as_ref().ok_or_else(|| {
        Status::invalid_argument("GenerateRequest.tokenized is required; SMG owns tokenization")
    })?;

    let mut dict = HashMap::new();
    dict.insert("rid".into(), Value::from(rid));
    dict.insert(
        "input_ids".into(),
        Value::from(to_token_ids(&tokenized.input_ids)?),
    );
    // `tokenized.original_text` is deliberately dropped: SMG owning
    // tokenization is the point of the gRPC path, and re-tokenizing here would
    // make the router's prefix-hash routing decisions disagree with the engine.

    if let Some(params) = req.sampling_params.as_ref() {
        dict.insert("sampling_params".into(), sampling_params_to_json(params)?);
    }
    // Absent sampling_params means "engine defaults"; never unwrap_or_default(),
    // whose proto3 zeros are not the semantic defaults.

    dict.insert("return_logprob".into(), Value::from(req.return_logprob));
    // Forwarded verbatim. SMG already sends -1 when logprobs are off, which is
    // the same value the native path defaults to.
    dict.insert(
        "logprob_start_len".into(),
        Value::from(req.logprob_start_len),
    );
    dict.insert("top_logprobs_num".into(), Value::from(req.top_logprobs_num));
    if !req.token_ids_logprob.is_empty() {
        dict.insert(
            "token_ids_logprob".into(),
            Value::from(to_token_ids(&req.token_ids_logprob)?),
        );
    }
    // Forwarded, not forced false: a chat request can set it, and dropping a
    // field the client explicitly asked for is worse than paying for it.
    dict.insert(
        "return_hidden_states".into(),
        Value::from(req.return_hidden_states),
    );

    let bootstrap_room = req.disaggregated_params.as_ref().map(|params| {
        let room = widen_room(params.bootstrap_room);
        dict.insert(
            "bootstrap_host".into(),
            Value::from(params.bootstrap_host.clone()),
        );
        dict.insert("bootstrap_port".into(), Value::from(params.bootstrap_port));
        dict.insert("bootstrap_room".into(), Value::from(room));
        room
    });

    if !req.custom_logit_processor.is_empty() {
        dict.insert(
            "custom_logit_processor".into(),
            Value::from(req.custom_logit_processor.clone()),
        );
    }
    if !req.lora_id.is_empty() {
        // SMG 1.0.0 says "id"; the dict's caller-facing key is `lora_path`
        // (`lora_id` is documented as tokenizer-manager-initialised state). An
        // unknown adapter raises Python-side, which is loud.
        dict.insert("lora_path".into(), Value::from(req.lora_id.clone()));
    }
    if req.data_parallel_rank > 0 {
        // Proto3 gives 0 for "unset" and 0 is also a valid rank; forwarding it
        // unconditionally pins every request to DP rank 0 with no error.
        dict.insert("routed_dp_rank".into(), Value::from(req.data_parallel_rank));
    }
    dict.insert("stream".into(), Value::from(req.stream));

    // `timestamp` is dropped: `received_time` comes from the server clock, and
    // trusting a client clock would corrupt e2e_latency under skew.
    // `log_metrics` is dropped: metrics are governed by --enable-metrics.
    dict.insert("received_time".into(), Value::from(now_timestamp()));

    Ok(MappedGenerate {
        rid: rid.to_string(),
        dict,
        bootstrap_room,
        stream: req.stream,
    })
}

/// A mapped `EmbedRequest`. Also serves `/v1/classify`, which reuses this RPC
/// and reinterprets the embedding as logits.
#[derive(Debug)]
pub struct MappedEmbed {
    pub rid: String,
    pub dict: HashMap<String, Value>,
}

pub fn embed_to_dict(req: &smg_proto::EmbedRequest) -> Result<MappedEmbed, Status> {
    let rid = req.request_id.trim();
    if rid.is_empty() {
        return Err(Status::invalid_argument(
            "EmbedRequest.request_id is required",
        ));
    }

    if req.mm_inputs.is_some() {
        return Err(Status::unimplemented(
            "Multimodal embedding inputs are not supported on sglang.grpc.scheduler",
        ));
    }
    // A cross-encoder scored without segment ids returns a plausible wrong
    // number, so all three of these reject rather than ignore.
    if !req.token_type_ids.is_empty() {
        return Err(Status::unimplemented(
            "token_type_ids has no request-dict key; a cross-encoder scored \
             without segment ids returns a plausible wrong number",
        ));
    }
    if req.is_cross_encoder {
        return Err(Status::unimplemented(
            "Cross-encoder embedding is not supported on sglang.grpc.scheduler",
        ));
    }
    if !req.texts.is_empty() {
        return Err(Status::unimplemented(
            "EmbedRequest.texts is the cross-encoder batch input and is not supported",
        ));
    }
    // Unlike GenerateReqInput, EmbeddingReqInput has neither `routed_dp_rank`
    // nor `data_parallel_rank`, so there is nowhere to put this. Passing it
    // would raise a TypeError from the dataclass constructor, and dropping it
    // would silently ignore SMG's DP-aware routing decision.
    if req.data_parallel_rank > 0 {
        return Err(Status::unimplemented(
            "data_parallel_rank has no EmbeddingReqInput field; DP-aware routing \
             is not supported for embeddings on sglang.grpc.scheduler",
        ));
    }

    let tokenized = req
        .tokenized
        .as_ref()
        .ok_or_else(|| Status::invalid_argument("EmbedRequest.tokenized is required"))?;

    let mut dict = HashMap::new();
    dict.insert("rid".into(), Value::from(rid));
    dict.insert(
        "input_ids".into(),
        Value::from(to_token_ids(&tokenized.input_ids)?),
    );
    dict.insert("received_time".into(), Value::from(now_timestamp()));
    // `sampling_params` is dropped: the proto itself calls it "dummy sampling
    // params for compatibility". `log_metrics` is dropped as for Generate.

    Ok(MappedEmbed {
        rid: rid.to_string(),
        dict,
    })
}

/// The inputs this layer cannot carry. Each would otherwise produce a fluent
/// answer to a different question than the one asked.
fn reject_unsupported_inputs(req: &smg_proto::GenerateRequest) -> Result<(), Status> {
    // SMG drops images before the wire even for a vision model card, so this
    // has never fired. It is a guard, not a mitigation.
    if req.mm_inputs.is_some() {
        return Err(Status::unimplemented(
            "Multimodal inputs are not supported on sglang.grpc.scheduler; \
             a text-only completion for an image prompt is fluent and wrong",
        ));
    }
    if !req.input_embeds.is_empty() {
        return Err(Status::unimplemented(
            "input_embeds is not supported on sglang.grpc.scheduler; \
             the request dict would silently complete the token prompt instead",
        ));
    }
    Ok(())
}

fn sampling_params_to_json(params: &smg_proto::SamplingParams) -> Result<Value, Status> {
    if params.n > MAX_SUPPORTED_PARALLEL_SAMPLES {
        return Err(Status::unimplemented(format!(
            "n > {} is not supported on sglang.grpc.scheduler: the per-sample \
             index never reaches this layer, so the samples cannot be \
             demultiplexed (got n = {})",
            MAX_SUPPORTED_PARALLEL_SAMPLES, params.n
        )));
    }

    let mut map = serde_json::Map::new();
    // Verbatim, including the zeros. temperature < 1e-6 means greedy on the
    // Python side, which is what SMG's HTTP path means by temperature: 0;
    // top_p / top_k / repetition_penalty zeros raise in verify(), loudly.
    map.insert("temperature".into(), Value::from(params.temperature));
    map.insert("top_p".into(), Value::from(params.top_p));
    map.insert("top_k".into(), Value::from(params.top_k));
    map.insert("min_p".into(), Value::from(params.min_p));
    map.insert(
        "frequency_penalty".into(),
        Value::from(params.frequency_penalty),
    );
    map.insert(
        "presence_penalty".into(),
        Value::from(params.presence_penalty),
    );
    map.insert(
        "repetition_penalty".into(),
        Value::from(params.repetition_penalty),
    );

    if let Some(max_new_tokens) = params.max_new_tokens {
        map.insert("max_new_tokens".into(), Value::from(max_new_tokens));
    } else {
        // Absent is not harmless: Python defaults to 128, so a request meaning
        // "unbounded" is silently capped. Do not invent a value.
        tracing::debug!("SMG request has no max_new_tokens; the engine will cap it at 128");
    }
    if !params.stop.is_empty() {
        map.insert("stop".into(), Value::from(params.stop.clone()));
    }
    if !params.stop_token_ids.is_empty() {
        map.insert(
            "stop_token_ids".into(),
            Value::from(to_token_ids(&params.stop_token_ids)?),
        );
    }
    // Field 11. Python defaults it true and SMG sets it false whenever tools
    // are present; dropping it strips the tool-call special tokens and the tool
    // parser finds nothing, so the model appears to just not call tools.
    map.insert(
        "skip_special_tokens".into(),
        Value::from(params.skip_special_tokens),
    );
    map.insert(
        "spaces_between_special_tokens".into(),
        Value::from(params.spaces_between_special_tokens),
    );
    map.insert("min_new_tokens".into(), Value::from(params.min_new_tokens));
    map.insert("ignore_eos".into(), Value::from(params.ignore_eos));
    map.insert("no_stop_trim".into(), Value::from(params.no_stop_trim));
    if let Some(stream_interval) = params.stream_interval {
        map.insert("stream_interval".into(), Value::from(stream_interval));
    }
    if !params.logit_bias.is_empty() {
        map.insert("logit_bias".into(), serde_json::json!(params.logit_bias));
    }
    if let Some(custom_params) = params.custom_params.as_ref() {
        map.insert(
            "custom_params".into(),
            custom_params_to_json(custom_params)?,
        );
    }

    if let Some(constraint) = params.constraint.as_ref() {
        use smg_proto::sampling_params::Constraint;
        let (key, value) = match constraint {
            Constraint::Regex(value) => ("regex", value),
            Constraint::JsonSchema(value) => ("json_schema", value),
            // SMG calls it ebnf_grammar; the Python key is `ebnf`. A grammar
            // routed to `regex` usually fails to compile, but not always --
            // some grammar texts are also valid regexes.
            Constraint::EbnfGrammar(value) => ("ebnf", value),
            Constraint::StructuralTag(value) => ("structural_tag", value),
        };
        map.insert(key.into(), Value::from(value.clone()));
    }

    Ok(Value::Object(map))
}

/// `google.protobuf.Struct` -> the dict's `custom_params`.
///
/// Python's `CustomParamValue` admits a JSON scalar, a flat list of scalars, or
/// a flat dict of scalars. Anything deeper is rejected rather than truncated.
fn custom_params_to_json(params: &prost_types::Struct) -> Result<Value, Status> {
    let mut map = serde_json::Map::new();
    for (key, value) in &params.fields {
        map.insert(key.clone(), custom_param_value_to_json(value, key)?);
    }
    Ok(Value::Object(map))
}

fn custom_param_value_to_json(value: &prost_types::Value, key: &str) -> Result<Value, Status> {
    use prost_types::value::Kind;
    match value.kind.as_ref() {
        None | Some(Kind::NullValue(_)) => Ok(Value::Null),
        Some(Kind::BoolValue(v)) => Ok(Value::from(*v)),
        Some(Kind::NumberValue(v)) => Ok(Value::from(*v)),
        Some(Kind::StringValue(v)) => Ok(Value::from(v.clone())),
        Some(Kind::ListValue(list)) => {
            let items = list
                .values
                .iter()
                .map(|item| custom_param_scalar_to_json(item, key))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Value::Array(items))
        }
        Some(Kind::StructValue(nested)) => {
            let mut map = serde_json::Map::new();
            for (nested_key, nested_value) in &nested.fields {
                map.insert(
                    nested_key.clone(),
                    custom_param_scalar_to_json(nested_value, key)?,
                );
            }
            Ok(Value::Object(map))
        }
    }
}

fn custom_param_scalar_to_json(value: &prost_types::Value, key: &str) -> Result<Value, Status> {
    use prost_types::value::Kind;
    match value.kind.as_ref() {
        None | Some(Kind::NullValue(_)) => Ok(Value::Null),
        Some(Kind::BoolValue(v)) => Ok(Value::from(*v)),
        Some(Kind::NumberValue(v)) => Ok(Value::from(*v)),
        Some(Kind::StringValue(v)) => Ok(Value::from(v.clone())),
        Some(Kind::ListValue(_) | Kind::StructValue(_)) => Err(Status::invalid_argument(format!(
            "custom_params[\"{}\"] nests deeper than Python's CustomParamValue \
             allows (scalar, flat list, or flat dict)",
            key
        ))),
    }
}

/// Widen SMG's `uint32` token ids to the `i32` the dict carries, erroring
/// rather than wrapping. `as i32` would turn u32::MAX into -1 silently.
fn to_token_ids(ids: &[u32]) -> Result<Vec<i32>, Status> {
    ids.iter()
        .map(|&id| {
            i32::try_from(id).map_err(|_| {
                Status::invalid_argument(format!("token id {} does not fit in int32", id))
            })
        })
        .collect()
}

#[cfg(test)]
mod tests;
