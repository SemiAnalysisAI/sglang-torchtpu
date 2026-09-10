//! Pins on the vendored contract itself, not on the mapping.
//!
//! The mapping tests live next to the code they cover
//! (`request_map`, `response_map`, `model_info`, `rooms`).

use super::abort_rid;
use crate::smg_proto;
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use tonic::Code;

fn repo_path(relative: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(relative)
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|b| format!("{:02x}", b)).collect()
}

/// T39. Every field number, oneof arm and default in this layer is chosen for
/// smg-grpc-client 1.0.0. 1.12.0 restructured `EmbedResponse` incompatibly, so
/// a pin bump without a re-vendor is silently wrong, not loudly broken.
#[test]
fn smg_client_pin_matches_the_vendored_proto() {
    let manifest = std::fs::read_to_string(repo_path("sgl-model-gateway/Cargo.toml"))
        .expect("sgl-model-gateway/Cargo.toml is readable");

    assert!(
        manifest.contains("smg-grpc-client = \"=1.0.0\""),
        "sgl-model-gateway pins a different smg-grpc-client than the vendored proto \
         (proto/sglang/grpc/scheduler/sglang_scheduler.proto is 1.0.0). Re-vendor the \
         proto and re-check src/smg/ before bumping: 1.12.0 reserves EmbedResponse 1-3."
    );
}

/// T40. The vendored file is a contract, not source we own. A local edit to it
/// changes what we promise SMG with no other signal.
#[test]
fn vendored_proto_digest_is_unchanged() {
    // sha256 of proto/sglang/grpc/scheduler/sglang_scheduler.proto as vendored,
    // i.e. upstream 1.0.0 plus the provenance header and the EmbedResponse
    // 1.12.0 forward-compat fields, both marked inline in the file.
    const EXPECTED: &str = "9b4e5444f7f6e2ce8249807a9cfdc1387bed3ddec643caaea6d5df087514f1ab";

    let bytes = std::fs::read(repo_path(
        "proto/sglang/grpc/scheduler/sglang_scheduler.proto",
    ))
    .expect("vendored proto is readable");

    assert_eq!(
        sha256_hex(&bytes),
        EXPECTED,
        "the vendored sglang_scheduler.proto changed. If that was deliberate, \
         update EXPECTED and re-read src/smg/ for the fields you moved."
    );
}

/// T38, request half. The recorded abort is
/// `AbortRequest { request_id: "chatcmpl-...", reason: "Stream dropped" }`, so
/// the id SMG aborts with is byte-identical to the one it generated with; see
/// `request_map`'s `request_id_is_used_verbatim_as_rid`.
#[test]
fn abort_uses_the_request_id_verbatim() {
    let req = smg_proto::AbortRequest {
        request_id: "chatcmpl-bbd2aa96-d42d-4870-b9b5-36e10adfc71f".to_string(),
        reason: "Stream dropped".to_string(),
    };

    assert_eq!(
        abort_rid(&req).unwrap(),
        "chatcmpl-bbd2aa96-d42d-4870-b9b5-36e10adfc71f"
    );
}

#[test]
fn abort_without_a_request_id_is_rejected() {
    for request_id in ["", "   "] {
        let req = smg_proto::AbortRequest {
            request_id: request_id.to_string(),
            reason: String::new(),
        };

        assert_eq!(
            abort_rid(&req).unwrap_err().code(),
            Code::InvalidArgument,
            "an empty request_id cannot identify anything; aborting nothing \
             silently would leave the request generating"
        );
    }
}
