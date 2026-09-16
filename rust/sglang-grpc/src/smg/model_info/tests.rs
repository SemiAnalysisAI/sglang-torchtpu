use super::build_model_info;

/// The shape `grpc_bridge.get_model_info()` returns for a generation model.
fn json_info() -> String {
    serde_json::json!({
        "model_path": "/models/qwen3-0.6b",
        "served_model_name": "qwen3",
        "tokenizer_path": "/models/qwen3-0.6b",
        "is_generation": true,
        // ServerArgs.weight_version is a non-Optional str defaulting to
        // "default", so the CLI cannot produce a null here.
        "weight_version": "default",
        "load_format": "auto",
        "reasoning_parser": null,
        "tool_call_parser": null,
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
    })
    .to_string()
}

/// T34.
#[test]
fn model_info_required_labels_are_non_empty() {
    let info = build_model_info(&json_info(), 32768);

    assert!(
        !info.tokenizer_path.is_empty(),
        "an empty tokenizer_path makes SMG skip the tokenizer job entirely"
    );
    assert!(
        !info.served_model_name.is_empty(),
        "an empty served_model_name collapses every route onto UNKNOWN_MODEL_ID"
    );
    assert_eq!(info.model_path, "/models/qwen3-0.6b");
    assert_eq!(info.max_context_length, 32768);
    assert_eq!(info.architectures, vec!["Qwen3ForCausalLM".to_string()]);
    assert!(info.is_generation);
    assert_eq!(info.weight_version, "default");
}

/// T34, the fallback arm: `get_serving().tokenizer_path` is None whenever the
/// operator did not pass `--tokenizer-path`, which is the common case.
#[test]
fn null_tokenizer_path_falls_back_to_model_path() {
    let json = serde_json::json!({
        "model_path": "/models/qwen3-0.6b",
        "served_model_name": "qwen3",
        "tokenizer_path": null,
    })
    .to_string();

    assert_eq!(
        build_model_info(&json, 4096).tokenizer_path,
        "/models/qwen3-0.6b"
    );
}

/// T35, on the two keys that can genuinely be null: `model_type` is
/// `getattr(hf_config, "model_type", None)` and `tokenizer_path` is Optional.
/// `weight_version` is not among them -- see the fixture.
#[test]
fn model_info_json_null_does_not_become_the_string_null() {
    let json = serde_json::json!({
        "model_path": "/models/m",
        "served_model_name": "m",
        "model_type": null,
    })
    .to_string();

    let info = build_model_info(&json, 4096);
    assert_eq!(
        info.model_type, "",
        "a literal \"null\" is non-empty, so to_labels() would keep it and SMG \
         would build a ModelCard around a model type that does not exist"
    );
    assert_eq!(build_model_info(&json_info(), 4096).model_type, "qwen3");
}

/// T36. Named individually so a future `..Default::default()` cannot make the
/// gap invisible.
#[test]
fn model_info_unavailable_fields_are_explicit() {
    let info = build_model_info(&json_info(), 4096);

    assert_eq!(
        info.vocab_size, 0,
        "vocab_size is unavailable from the bridge"
    );
    assert_eq!(
        info.eos_token_ids,
        Vec::<i32>::new(),
        "eos_token_ids is unavailable from the bridge"
    );
    assert_eq!(info.pad_token_id, 0, "pad_token_id is unavailable");
    assert_eq!(info.bos_token_id, 0, "bos_token_id is unavailable");
    assert_eq!(
        info.max_req_input_len, 0,
        "max_req_input_len is unavailable"
    );
    assert_eq!(info.num_labels, 0, "num_labels is unavailable");
    assert_eq!(info.id2label_json, "", "id2label_json is unavailable");
    assert_eq!(
        info.preferred_sampling_params, "",
        "preferred_sampling_params is unavailable"
    );
    assert!(!info.supports_vision, "supports_vision is unavailable");
}

#[test]
fn unparseable_model_info_yields_defaults_rather_than_panicking() {
    let info = build_model_info("not json", 4096);

    assert_eq!(info.model_path, "");
    assert_eq!(info.max_context_length, 4096);
}
