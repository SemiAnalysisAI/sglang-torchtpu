//! `sglang.grpc.scheduler.SglangScheduler` — the sgl-model-gateway (SMG) wire
//! contract, served alongside `sglang.runtime.v1.SglangService` on the same
//! `--grpc-port` listener.
//!
//! SMG's pinned client (`smg-grpc-client = "=1.0.0"`) speaks this service and
//! nothing else, so without it a gRPC worker cannot register and the
//! `prefix_hash` routing policy — which needs token ids and therefore the gRPC
//! transport — is unreachable.
//!
//! Requests are translated straight to the Python request dict
//! (`utils::request_utils`'s representation), not via `sglang.runtime.v1`:
//! eleven SMG fields, `skip_special_tokens` among them, have no `runtime.v1`
//! counterpart and would be dropped in a proto-to-proto hop.
//!
//! Two engine behaviours a reader should know before touching this module:
//! absent `max_new_tokens` means 128 on the Python side, not "unbounded"; and
//! `--incremental-streaming-output` defaults false, so our `output_ids` are
//! cumulative while SMG's chunk contract is incremental (see `response_map`).

// Every fallible function in this module reports through `tonic::Status`,
// which is 176 bytes, because that is what the service trait returns. Boxing it
// would only move the unboxing to each tonic boundary.
#![allow(clippy::result_large_err)]

use std::pin::Pin;
use std::sync::Arc;

use tokio::time::Duration;
use tokio_stream::Stream;
use tonic::{Request, Response, Status};

use crate::bridge::{PyBridge, ResponseChunk};
use crate::server::{
    RequestAbortGuard, closed_stream_status, pyerr_to_status, recv_chunk_with_timeout,
    recv_terminal_chunk_for_request,
};
use crate::smg::response_map::{StreamMode, StreamTranslator};
use crate::smg_proto;

pub mod model_info;
pub mod request_map;
pub mod response_map;
pub mod rooms;

type StreamResult<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send + 'static>>;

pub struct SmgSchedulerServiceImpl {
    pub bridge: Arc<PyBridge>,
    pub response_timeout: Duration,
    /// Whether the engine emits cumulative or incremental `output_ids`; read
    /// once at startup because it is a server argument, not a per-request one.
    pub stream_mode: StreamMode,
    pub rooms: Arc<rooms::RoomRegistry>,
}

#[tonic::async_trait]
impl smg_proto::sglang_scheduler_server::SglangScheduler for SmgSchedulerServiceImpl {
    type GenerateStream = StreamResult<smg_proto::GenerateResponse>;

    async fn generate(
        &self,
        request: Request<smg_proto::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let mapped = request_map::generate_to_dict(&request.into_inner())?;
        let rid = mapped.rid;

        // Claimed before submitting, released by the guard when the response
        // stream drops. A concurrent room collision becomes ALREADY_EXISTS
        // instead of this request's decode consuming another's KV.
        let room_guard = match mapped.bootstrap_room {
            Some(room) => Some(self.rooms.claim(room, &rid)?),
            None => None,
        };

        let mut receiver = self
            .bridge
            .submit_request(&rid, "generate", mapped.dict)
            .map_err(|e| pyerr_to_status(e, "Failed to submit request"))?;

        let bridge = self.bridge.clone();
        let response_timeout = self.response_timeout;
        let mut translator = StreamTranslator::new(self.stream_mode);

        let stream = async_stream::stream! {
            let _room_guard = room_guard;
            let mut abort_guard = RequestAbortGuard::new(bridge.clone(), rid.clone());
            loop {
                match recv_chunk_with_timeout(&mut receiver, response_timeout, || "Stream chunk timed out".to_string()).await {
                    Ok(Some(ResponseChunk::Data(data))) => {
                        match translator.on_data(&data) {
                            Ok(Some(message)) => yield Ok(response_map::with_request_id(message, &rid)),
                            // An empty delta is not an error; it just has
                            // nothing SMG can use, and a zero-token chunk
                            // inflates Harmony's per-chunk token count.
                            Ok(None) => {}
                            Err(status) => {
                                abort_guard.abort_now();
                                yield Err(status);
                                break;
                            }
                        }
                    }
                    Ok(Some(ResponseChunk::Finished(data))) => {
                        abort_guard.disarm();
                        match translator.on_finished(&data) {
                            Ok(messages) => {
                                for message in messages {
                                    yield Ok(response_map::with_request_id(message, &rid));
                                }
                            }
                            Err(status) => yield Err(status),
                        }
                        break;
                    }
                    Ok(Some(ResponseChunk::Error(msg))) => {
                        abort_guard.disarm();
                        let (message, status) = response_map::error_messages(msg);
                        yield Ok(response_map::with_request_id(message, &rid));
                        yield Err(status);
                        break;
                    }
                    Ok(None) => {
                        let (status, should_abort) = closed_stream_status(&bridge, &rid);
                        if should_abort {
                            abort_guard.abort_now();
                        } else {
                            abort_guard.disarm();
                        }
                        yield Err(status);
                        break;
                    }
                    Err(status) => {
                        abort_guard.abort_now();
                        yield Err(status);
                        break;
                    }
                }
            }
        };

        Ok(Response::new(Box::pin(stream)))
    }

    /// Serves both `/v1/embeddings` and `/v1/classify`; the latter reuses this
    /// RPC and reinterprets the embedding as logits.
    async fn embed(
        &self,
        request: Request<smg_proto::EmbedRequest>,
    ) -> Result<Response<smg_proto::EmbedResponse>, Status> {
        let mapped = request_map::embed_to_dict(&request.into_inner())?;
        let rid = mapped.rid;

        let mut receiver = self
            .bridge
            .submit_request(&rid, "embed", mapped.dict)
            .map_err(|e| pyerr_to_status(e, "Failed to submit request"))?;

        // Same unary receive as the native Embed: a non-terminal Data chunk is a
        // protocol violation that must abort the request rather than answer from
        // whatever arrived first.
        let chunk = recv_terminal_chunk_for_request(
            &self.bridge,
            &rid,
            &mut receiver,
            self.response_timeout,
        )
        .await?;

        match chunk {
            ResponseChunk::Data(data) | ResponseChunk::Finished(data) => {
                Ok(Response::new(response_map::embed_response(&rid, &data)?))
            }
            ResponseChunk::Error(msg) => Ok(Response::new(response_map::embed_error(&rid, msg))),
        }
    }

    async fn health_check(
        &self,
        _request: Request<smg_proto::HealthCheckRequest>,
    ) -> Result<Response<smg_proto::HealthCheckResponse>, Status> {
        let healthy = tokio::task::spawn_blocking({
            let bridge = self.bridge.clone();
            move || bridge.health_check()
        })
        .await
        .map_err(|e| Status::internal(format!("Task join error: {}", e)))?
        .map_err(|e| Status::internal(format!("Health check failed: {}", e)))?;

        Ok(Response::new(smg_proto::HealthCheckResponse {
            healthy,
            message: String::new(),
        }))
    }

    /// SMG never calls this explicitly; `AbortOnDropStream::Drop` fires it on
    /// every client disconnect, every `GenerateError`, every tonic stream error
    /// and every cancellation, because SMG calls `mark_completed()` only on
    /// clean success. Leaving it `unimplemented` would leave each of those
    /// requests generating to `max_new_tokens` and, in PD, strand the
    /// bootstrap room.
    ///
    /// An abort for an already-finished request is the normal case — Drop races
    /// the terminal frame — so `PyBridge::abort` treats an unknown rid as a
    /// no-op success and we must not turn that into an error.
    async fn abort(
        &self,
        request: Request<smg_proto::AbortRequest>,
    ) -> Result<Response<smg_proto::AbortResponse>, Status> {
        let req = request.into_inner();
        let rid = abort_rid(&req)?;
        tracing::debug!(rid, reason = %req.reason, "SMG abort");

        // `abort_all` is hard-coded false: AbortRequest has no such field, and
        // this listener is unauthenticated (see run_grpc_server's TODO), so the
        // blast radius stays at one request.
        self.bridge
            .abort(rid, false)
            .map_err(|e| Status::internal(format!("Failed to abort: {}", e)))?;

        Ok(Response::new(smg_proto::AbortResponse {
            success: true,
            message: String::new(),
        }))
    }

    async fn get_model_info(
        &self,
        _request: Request<smg_proto::GetModelInfoRequest>,
    ) -> Result<Response<smg_proto::GetModelInfoResponse>, Status> {
        let json_info = tokio::task::spawn_blocking({
            let bridge = self.bridge.clone();
            move || bridge.get_model_info()
        })
        .await
        .map_err(|e| Status::internal(format!("Task join error: {}", e)))?
        .map_err(|e| Status::internal(format!("Failed to get model info: {}", e)))?;

        Ok(Response::new(model_info::build_model_info(
            &json_info,
            self.bridge.context_len(),
        )))
    }

    /// Never called by SMG: `/server_info` goes to the worker's HTTP endpoint
    /// (`core/steps/worker/local/discover_metadata.rs`). Zero calls in 9,346
    /// recorded RPCs.
    async fn get_server_info(
        &self,
        _request: Request<smg_proto::GetServerInfoRequest>,
    ) -> Result<Response<smg_proto::GetServerInfoResponse>, Status> {
        Err(Status::unimplemented(
            "GetServerInfo is not served over sglang.grpc.scheduler; use HTTP /get_server_info",
        ))
    }

    /// Never called by SMG: `/get_loads` hard-codes `-1` for gRPC workers
    /// (`core/worker_manager.rs`). Zero calls in 9,346 recorded RPCs.
    async fn get_loads(
        &self,
        _request: Request<smg_proto::GetLoadsRequest>,
    ) -> Result<Response<smg_proto::GetLoadsResponse>, Status> {
        Err(Status::unimplemented(
            "GetLoads is not served over sglang.grpc.scheduler; SMG reports -1 for gRPC workers",
        ))
    }
}

/// The rid an `AbortRequest` targets, which must be the verbatim
/// `GenerateRequest.request_id` or the abort matches nothing.
fn abort_rid(req: &smg_proto::AbortRequest) -> Result<&str, Status> {
    let rid = req.request_id.trim();
    if rid.is_empty() {
        return Err(Status::invalid_argument(
            "Abort requires a non-empty request_id",
        ));
    }
    Ok(rid)
}

#[cfg(test)]
mod tests;
