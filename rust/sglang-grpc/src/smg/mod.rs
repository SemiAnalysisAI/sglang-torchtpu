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

use crate::bridge::PyBridge;
use crate::smg_proto;

pub mod model_info;

type StreamResult<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send + 'static>>;

pub struct SmgSchedulerServiceImpl {
    pub bridge: Arc<PyBridge>,
    pub response_timeout: Duration,
}

#[tonic::async_trait]
impl smg_proto::sglang_scheduler_server::SglangScheduler for SmgSchedulerServiceImpl {
    type GenerateStream = StreamResult<smg_proto::GenerateResponse>;

    async fn generate(
        &self,
        _request: Request<smg_proto::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        Err(Status::unimplemented("Generate is not implemented yet"))
    }

    async fn embed(
        &self,
        _request: Request<smg_proto::EmbedRequest>,
    ) -> Result<Response<smg_proto::EmbedResponse>, Status> {
        Err(Status::unimplemented("Embed is not implemented yet"))
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
