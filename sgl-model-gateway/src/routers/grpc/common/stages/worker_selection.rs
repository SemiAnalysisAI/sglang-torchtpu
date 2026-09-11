//! Worker selection stage: Select appropriate worker(s) based on routing mode

use std::sync::Arc;

use async_trait::async_trait;
use axum::response::Response;
use tracing::{error, warn};

use super::PipelineStage;
use crate::{
    core::{ConnectionMode, Worker, WorkerRegistry, WorkerType, UNKNOWN_MODEL_ID},
    observability::metrics::{metrics_labels, Metrics},
    policies::{PolicyRegistry, SelectWorkerInfo},
    routers::{
        error,
        grpc::context::{RequestContext, WorkerSelection},
    },
};

/// Worker selection stage: Select appropriate worker(s) based on routing mode
pub(crate) struct WorkerSelectionStage {
    worker_registry: Arc<WorkerRegistry>,
    policy_registry: Arc<PolicyRegistry>,
    mode: WorkerSelectionMode,
}

pub(crate) enum WorkerSelectionMode {
    /// Regular mode: select single worker
    Regular,
    /// PD mode: select prefill + decode workers
    PrefillDecode,
}

impl WorkerSelectionStage {
    pub fn new(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        mode: WorkerSelectionMode,
    ) -> Self {
        Self {
            worker_registry,
            policy_registry,
            mode,
        }
    }
}

#[async_trait]
impl PipelineStage for WorkerSelectionStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        let prep = ctx.state.preparation.as_ref().ok_or_else(|| {
            error!(
                function = "WorkerSelectionStage::execute",
                "Preparation stage not completed"
            );
            error::internal_error(
                "preparation_stage_not_completed",
                "Preparation stage not completed",
            )
        })?;

        // For Harmony, use selection_text produced during Harmony encoding
        // Otherwise, use original_text from regular preparation
        let text = if prep.harmony_mode {
            prep.selection_text.as_deref()
        } else {
            prep.original_text.as_deref()
        };

        // Get tokens for PrefixHash policy support
        let tokens = if prep.token_ids.is_empty() {
            None
        } else {
            Some(prep.token_ids.as_slice())
        };

        let headers = ctx.input.headers.as_ref();

        let workers = match self.mode {
            WorkerSelectionMode::Regular => {
                match self
                    .select_single_worker(ctx.input.model_id.as_deref(), text, tokens, headers)
                    .await
                {
                    Some(w) => WorkerSelection::Single { worker: w },
                    None => {
                        let model = ctx.input.model_id.as_deref().unwrap_or(UNKNOWN_MODEL_ID);
                        error!(
                            function = "WorkerSelectionStage::execute",
                            mode = "Regular",
                            model_id = %model,
                            "No available workers for model"
                        );
                        return Err(error::service_unavailable(
                            "no_available_workers",
                            format!("No available workers for model: {}", model),
                        ));
                    }
                }
            }
            WorkerSelectionMode::PrefillDecode => {
                match self
                    .select_pd_pair(ctx.input.model_id.as_deref(), text, tokens, headers)
                    .await
                {
                    Some((prefill, decode)) => WorkerSelection::Dual { prefill, decode },
                    None => {
                        let model = ctx.input.model_id.as_deref().unwrap_or(UNKNOWN_MODEL_ID);
                        error!(
                            function = "WorkerSelectionStage::execute",
                            mode = "PrefillDecode",
                            model_id = %model,
                            "No available PD worker pairs for model"
                        );
                        return Err(error::service_unavailable(
                            "no_available_pd_worker_pairs",
                            format!("No available PD worker pairs for model: {}", model),
                        ));
                    }
                }
            }
        };

        ctx.state.workers = Some(workers);
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "WorkerSelection"
    }
}

impl WorkerSelectionStage {
    async fn select_single_worker(
        &self,
        model_id: Option<&str>,
        text: Option<&str>,
        tokens: Option<&[u32]>,
        headers: Option<&http::HeaderMap>,
    ) -> Option<Arc<dyn Worker>> {
        // Get workers for the specified model, filtered by connection mode
        let workers = self.worker_registry.get_workers_filtered(
            model_id,
            Some(WorkerType::Regular),
            Some(ConnectionMode::Grpc { port: None }),
            None,  // any runtime type
            false, // get all workers, we'll filter by is_available() next
        );

        // Use into_iter() to take ownership of Arcs without cloning (avoids atomic inc/dec)
        let available: Vec<Arc<dyn Worker>> =
            workers.into_iter().filter(|w| w.is_available()).collect();

        if available.is_empty() {
            return None;
        }

        // Get the appropriate policy for this model
        let policy = match model_id {
            Some(model) => self.policy_registry.get_policy_or_default(model),
            None => self.policy_registry.get_default_policy(),
        };

        // Get cached hash ring for consistent hashing (O(log n) lookup)
        let hash_ring = self
            .worker_registry
            .get_hash_ring(model_id.unwrap_or(UNKNOWN_MODEL_ID));

        // Select worker using the policy
        let idx = policy
            .select_worker(
                &available,
                &SelectWorkerInfo {
                    request_text: text,
                    tokens,
                    headers,
                    hash_ring,
                },
            )
            .await?;
        let selected = available[idx].clone();

        // Record worker selection metric
        Metrics::record_worker_selection(
            metrics_labels::WORKER_REGULAR,
            metrics_labels::CONNECTION_GRPC,
            model_id.unwrap_or(UNKNOWN_MODEL_ID),
            policy.name(),
        );

        Some(selected)
    }

    async fn select_pd_pair(
        &self,
        model_id: Option<&str>,
        text: Option<&str>,
        tokens: Option<&[u32]>,
        headers: Option<&http::HeaderMap>,
    ) -> Option<(Arc<dyn Worker>, Arc<dyn Worker>)> {
        let all_workers = self.worker_registry.get_workers_filtered(
            model_id,
            None,
            Some(ConnectionMode::Grpc { port: None }), // Match any gRPC worker
            None,                                      // any runtime type
            false,
        );

        let (available_prefill, available_decode): (Vec<_>, Vec<_>) =
            all_workers
                .into_iter()
                .fold((Vec::new(), Vec::new()), |mut acc, w| {
                    if w.is_available() {
                        match w.metadata().worker_type {
                            WorkerType::Prefill { .. } => acc.0.push(w),
                            WorkerType::Decode => acc.1.push(w),
                            _ => {}
                        }
                    }
                    acc
                });

        if available_prefill.is_empty() {
            warn!("No available prefill workers");
            return None;
        }

        if available_decode.is_empty() {
            warn!("No available decode workers");
            return None;
        }

        // Select using policies.
        //
        // PD applies TWO independent policies -- the prefill and decode pools
        // have different characteristics, so `--prefill-policy cache_aware
        // --decode-policy round_robin` is the normal configuration. The HTTP PD
        // router already does this (routers/http/pd_router.rs), but this path
        // previously resolved a single policy and applied it to both pools, so
        // both flags were silently ignored on gRPC: they are populated in the
        // registry by routers/factory.rs, and only the HTTP path read them.
        let prefill_policy = self.policy_registry.get_prefill_policy();
        let decode_policy = self.policy_registry.get_decode_policy();

        // Get cached hash ring for consistent hashing (O(log n) lookup)
        let hash_ring = self
            .worker_registry
            .get_hash_ring(model_id.unwrap_or(UNKNOWN_MODEL_ID));

        let info = SelectWorkerInfo {
            request_text: text,
            tokens,
            headers,
            hash_ring,
        };
        let prefill_idx = prefill_policy
            .select_worker(&available_prefill, &info)
            .await?;
        let decode_idx = decode_policy
            .select_worker(&available_decode, &info)
            .await?;

        let model = model_id.unwrap_or(UNKNOWN_MODEL_ID);

        // Record worker selection metrics for both prefill and decode. Each pool
        // reports the policy that actually chose its worker; labelling both with
        // one name would hide the split that was just fixed above.
        Metrics::record_worker_selection(
            metrics_labels::WORKER_PREFILL,
            metrics_labels::CONNECTION_GRPC,
            model,
            prefill_policy.name(),
        );
        Metrics::record_worker_selection(
            metrics_labels::WORKER_DECODE,
            metrics_labels::CONNECTION_GRPC,
            model,
            decode_policy.name(),
        );

        Some((
            available_prefill[prefill_idx].clone(),
            available_decode[decode_idx].clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::PolicyConfig;
    use crate::core::BasicWorkerBuilder;
    use crate::policies::RoundRobinPolicy;

    fn add_workers(reg: &WorkerRegistry, kind: WorkerType, n: usize, base_port: u16) {
        for i in 0..n {
            let w = BasicWorkerBuilder::new(format!("http://127.0.0.1:{}", base_port + i as u16))
                .worker_type(kind.clone())
                .connection_mode(ConnectionMode::Grpc { port: None })
                .build();
            reg.register(Arc::new(w) as Arc<dyn Worker>);
        }
    }

    /// 2 prefill + 2 decode gRPC workers, with an INDEPENDENT round-robin
    /// policy configured for each pool.
    fn pd_stage() -> WorkerSelectionStage {
        let worker_registry = Arc::new(WorkerRegistry::new());
        add_workers(
            &worker_registry,
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            2,
            9000,
        );
        add_workers(&worker_registry, WorkerType::Decode, 2, 9100);

        let policy_registry = Arc::new(PolicyRegistry::new(PolicyConfig::Random));
        policy_registry.set_prefill_policy(Arc::new(RoundRobinPolicy::new()));
        policy_registry.set_decode_policy(Arc::new(RoundRobinPolicy::new()));

        WorkerSelectionStage::new(
            worker_registry,
            policy_registry,
            WorkerSelectionMode::PrefillDecode,
        )
    }

    /// gRPC PD must apply the prefill and decode policies INDEPENDENTLY.
    ///
    /// Regression test for a real bug: this path used to resolve a single
    /// policy and hand it to both pools, so `--prefill-policy` and
    /// `--decode-policy` were silently ignored on gRPC. They are populated in
    /// the registry by routers/factory.rs, but only the HTTP PD router read
    /// them. Every pre-existing test still passed, because none configured
    /// differing policies and checked they were honoured separately.
    ///
    /// The discriminator is deliberately order-independent, so it does not
    /// depend on the order WorkerRegistry happens to return workers in.
    /// RoundRobin does `count = counter.fetch_add(1); idx = count % len`.
    ///
    /// With TWO independent policies each pool advances its own counter, so
    /// two successive requests land on two different workers:
    ///     prefill: req1 -> idx 0, req2 -> idx 1
    ///     decode:  req1 -> idx 0, req2 -> idx 1
    ///
    /// With ONE shared policy the counter advances twice per request (once per
    /// pool), so prefill always sees an even count and decode an odd one, and
    /// both pools stay pinned to a single worker:
    ///     prefill: req1 -> idx 0, req2 -> idx 0   (counts 0, 2)
    ///     decode:  req1 -> idx 1, req2 -> idx 1   (counts 1, 3)
    ///
    /// "Both pools moved between requests" is exactly what the bug destroyed.
    #[tokio::test]
    async fn grpc_pd_applies_prefill_and_decode_policies_independently() {
        let stage = pd_stage();

        let (p1, d1) = stage
            .select_pd_pair(None, Some("hello"), None, None)
            .await
            .expect("first request should select a prefill/decode pair");
        let (p2, d2) = stage
            .select_pd_pair(None, Some("hello"), None, None)
            .await
            .expect("second request should select a prefill/decode pair");

        assert_ne!(
            p1.url(),
            p2.url(),
            "prefill pool did not advance: both requests chose {}. The prefill \
             policy is not being applied independently (shared-policy bug).",
            p1.url()
        );
        assert_ne!(
            d1.url(),
            d2.url(),
            "decode pool did not advance: both requests chose {}. The decode \
             policy is not being applied independently (shared-policy bug).",
            d1.url()
        );
    }

    /// Setting one pool's policy must not set the other's.
    ///
    /// The two accessors read separate slots, and an unset slot falls back to
    /// the global default -- the same contract the HTTP PD router relies on
    /// (`routers/http/pd_router.rs`). If they ever shared a slot, the bug above
    /// would come back through the registry instead of through the stage.
    #[test]
    fn setting_one_pool_policy_does_not_set_the_other() {
        let registry = PolicyRegistry::new(PolicyConfig::Random);
        assert_eq!(registry.get_prefill_policy().name(), "random");
        assert_eq!(registry.get_decode_policy().name(), "random");

        registry.set_prefill_policy(Arc::new(RoundRobinPolicy::new()));
        assert_eq!(registry.get_prefill_policy().name(), "round_robin");
        assert_eq!(
            registry.get_decode_policy().name(),
            "random",
            "setting the prefill policy must not implicitly set the decode policy"
        );
    }
}
