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

use std::pin::Pin;
use std::sync::Arc;

use tokio::time::Duration;
use tokio_stream::Stream;
use tonic::{Request, Response, Status};

use crate::bridge::PyBridge;
use crate::smg_proto;

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
        Err(Status::unimplemented("HealthCheck is not implemented yet"))
    }

    async fn abort(
        &self,
        _request: Request<smg_proto::AbortRequest>,
    ) -> Result<Response<smg_proto::AbortResponse>, Status> {
        Err(Status::unimplemented("Abort is not implemented yet"))
    }

    async fn get_model_info(
        &self,
        _request: Request<smg_proto::GetModelInfoRequest>,
    ) -> Result<Response<smg_proto::GetModelInfoResponse>, Status> {
        Err(Status::unimplemented("GetModelInfo is not implemented yet"))
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

#[cfg(test)]
mod tests;
