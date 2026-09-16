//! `GetModelInfo` — SMG calls this exactly once, at worker registration.
//!
//! SMG never names a field: `ModelInfo::to_labels()` serde-serialises the whole
//! message into an opaque label map and drops every proto3 default (empty
//! string, zero, false, empty array). So an unavailable field left at its
//! default is invisible, while a *fabricated* one would be trusted. Everything
//! we cannot source is therefore left at the default and named below, never
//! guessed.
//!
//! `bridge.get_model_info()` supplies ten keys
//! (`python/sglang/srt/entrypoints/grpc_bridge.py::get_model_info`); six of them
//! map here, `max_context_length` comes from Rust state, and the remaining ten
//! proto fields have no source.

use crate::smg_proto;

/// Read a `json_info` key as a string, treating JSON `null`, a missing key and
/// a non-string value alike as "absent".
///
/// `serde_json::Value::to_string()` on a null yields the four characters
/// `null`, which `to_labels()` would keep because it is a non-empty string.
fn info_str(info: &serde_json::Value, key: &str) -> String {
    info.get(key)
        .and_then(|value| value.as_str())
        .unwrap_or_default()
        .to_string()
}

pub fn build_model_info(json_info: &str, context_len: i32) -> smg_proto::GetModelInfoResponse {
    let info = serde_json::from_str::<serde_json::Value>(json_info).unwrap_or_else(|err| {
        tracing::warn!("Failed to parse model info JSON: {}", err);
        serde_json::Value::Null
    });

    let model_path = info_str(&info, "model_path");
    // With neither a tokenizer path nor a model path SMG warns and *skips* the
    // tokenizer job (submit_tokenizer_job.rs), and gRPC routing needs a local
    // tokenizer for every request. Mirrors lib.rs::extract_tokenizer_info.
    let tokenizer_path = match info_str(&info, "tokenizer_path") {
        path if path.is_empty() => model_path.clone(),
        path => path,
    };
    let architectures = info
        .get("architectures")
        .and_then(|value| value.as_array())
        .map(|entries| {
            entries
                .iter()
                .filter_map(|entry| entry.as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default();

    smg_proto::GetModelInfoResponse {
        model_path,
        tokenizer_path,
        is_generation: info
            .get("is_generation")
            .and_then(|value| value.as_bool())
            .unwrap_or(false),
        weight_version: info_str(&info, "weight_version"),
        served_model_name: info_str(&info, "served_model_name"),
        model_type: info_str(&info, "model_type"),
        architectures,
        max_context_length: context_len,

        // Unavailable from the bridge. Left at the proto3 default so
        // `to_labels()` drops the key and SMG falls back to its own source.
        // `vocab_size`: SMG reads it off its own tokenizer instead.
        vocab_size: 0,
        // `supports_vision`: SMG detects vision from `model_type`.
        supports_vision: false,
        // `preferred_sampling_params`: not in `json_info`; the only other
        // source is `get_server_info()`, and nothing downstream reads it.
        preferred_sampling_params: String::new(),
        // `eos_token_ids`: stop handling comes from the request and from
        // `GenerateComplete.matched_stop`.
        eos_token_ids: Vec::new(),
        pad_token_id: 0,
        bos_token_id: 0,
        // `max_req_input_len`: SMG does no context-length validation.
        max_req_input_len: 0,
        // `id2label_json` / `num_labels`: HuggingFace config.json fields the
        // bridge does not expose. Only /v1/classify reads them.
        id2label_json: String::new(),
        num_labels: 0,
    }
}

#[cfg(test)]
mod tests;
