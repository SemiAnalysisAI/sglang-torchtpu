// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Consistent-hash prefix affinity (`--policy prefix_hash`).
//!
//! Requests whose first `prefix_token_count` tokens agree route to the same
//! worker, so a session's turns -- which share their system prompt and
//! history -- keep hitting the KV cache that already holds them. Unlike
//! `cache_aware`, nothing is learned from the fleet: the target is a pure
//! function of the prefix and the healthy worker set, so N router replicas
//! agree without sharing state and no tree is kept in memory. The price is
//! precision: two prompts that diverge after the hashed window are
//! indistinguishable, and a prompt that shares a long prefix with a worker's
//! cache but differs inside the window is not matched at all.
//!
//! Placement is rendezvous (highest-random-weight) hashing rather than a
//! ring. Each worker scores `mix(prefix_hash ^ hash(worker.url))` and the
//! highest score wins. Adding or removing a worker moves only the keys that
//! belonged to it -- the property a cache-affinity policy needs from
//! consistent hashing -- and there is no ring to build or invalidate when the
//! healthy set changes between requests. Scores are computed per request,
//! O(workers), which is negligible against the tokenization that produced the
//! tokens.
//!
//! The load bound reuses the engine-reported snapshot the router already
//! polls for `power_of_two`: a target carrying more than `mean * load_factor`
//! running+waiting requests is skipped for the next worker in score order.
//! With `load_factor >= 1` the least-loaded worker always passes, so the walk
//! terminates. Without a snapshot the hashed target is taken as-is.
//!
//! A request the router could not tokenize falls back to power-of-two on the
//! same snapshot rather than failing; every branch is counted in
//! `sgl_router_prefix_hash_total{outcome}`.

use std::sync::{Arc, OnceLock};

use crate::config::{PrefixHashConfig, DEFAULT_MIN_LOAD_CHOICES};
use crate::policies::power_of_two::select_k_with_snapshot;
use crate::policies::{Policy, SelectionContext};
use crate::server::metrics::{MetricsRegistry, PrefixHashOutcome};
use crate::state::load_monitor::engine_reported_load::EngineReportedLoadSnapshot;
use crate::workers::Worker;

#[derive(Debug)]
pub struct PrefixHashPolicy {
    config: PrefixHashConfig,
    metrics: OnceLock<Arc<MetricsRegistry>>,
}

impl PrefixHashPolicy {
    pub fn new(config: PrefixHashConfig) -> Self {
        Self {
            config,
            metrics: OnceLock::new(),
        }
    }

    pub fn config(&self) -> &PrefixHashConfig {
        &self.config
    }

    /// Hash of the first `prefix_token_count` tokens.
    pub fn prefix_hash(&self, tokens: &[u32]) -> u64 {
        let window = &tokens[..tokens.len().min(self.config.prefix_token_count)];
        let mut h = FNV_OFFSET;
        for token in window {
            for byte in token.to_le_bytes() {
                h = fnv1a_step(h, byte);
            }
        }
        splitmix64(h)
    }

    /// Workers in rendezvous order for this prefix: the first entry is the
    /// hashed target, the rest are where the request goes if it is shed.
    fn ranked<'w>(&self, workers: &'w [Arc<Worker>], prefix_hash: u64) -> Vec<&'w Arc<Worker>> {
        let mut scored: Vec<(u64, &Arc<Worker>)> = workers
            .iter()
            .map(|worker| (rendezvous_score(prefix_hash, &worker.url), worker))
            .collect();
        // Descending score; the URL breaks the (astronomically unlikely) tie
        // so replicas still agree.
        scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.url.cmp(&b.1.url)));
        scored.into_iter().map(|(_, worker)| worker).collect()
    }

    fn record(&self, outcome: PrefixHashOutcome) {
        if let Some(metrics) = self.metrics.get() {
            metrics.record_prefix_hash(outcome);
        }
    }

    fn select_impl(
        &self,
        workers: &[Arc<Worker>],
        ctx: &SelectionContext<'_>,
    ) -> Option<(Arc<Worker>, PrefixHashOutcome)> {
        if workers.is_empty() {
            return None;
        }
        let tokens = ctx.request_tokens().filter(|tokens| !tokens.is_empty());
        let Some(tokens) = tokens else {
            let picked = select_k_with_snapshot(
                workers,
                ctx.load_snapshot(),
                DEFAULT_MIN_LOAD_CHOICES,
                None,
            )?;
            return Some((picked, PrefixHashOutcome::NoTokens));
        };
        let ranked = self.ranked(workers, self.prefix_hash(tokens));
        let Some(snapshot) = ctx.load_snapshot() else {
            return Some((Arc::clone(ranked[0]), PrefixHashOutcome::Unbounded));
        };
        let Some(threshold) = load_threshold(snapshot, workers, self.config.load_factor) else {
            // Nothing fresh to bound on: same as having no snapshot.
            return Some((Arc::clone(ranked[0]), PrefixHashOutcome::Unbounded));
        };
        for (rank, worker) in ranked.iter().enumerate() {
            // A worker with no fresh reading is not known to be busy; let it take
            // the request rather than skipping the whole hashed order.
            let within = load_of(snapshot, &worker.url).is_none_or(|load| load as f64 <= threshold);
            if within {
                let outcome = if rank == 0 {
                    PrefixHashOutcome::HashHit
                } else {
                    PrefixHashOutcome::LoadShed
                };
                return Some((Arc::clone(worker), outcome));
            }
        }
        // Unreachable when load_factor >= 1 (the minimum is always <= mean),
        // kept so a bad factor degrades to affinity instead of a 502.
        Some((Arc::clone(ranked[0]), PrefixHashOutcome::Unbounded))
    }
}

impl Policy for PrefixHashPolicy {
    fn select(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Option<Arc<Worker>> {
        let (worker, outcome) = self.select_impl(workers, ctx)?;
        self.record(outcome);
        Some(worker)
    }

    /// The hash decides; the router's own tokenizer must run for it.
    fn needs_request_tokens(&self) -> bool {
        true
    }

    /// The load bound reads the engine-reported snapshot.
    fn needs_load_snapshot(&self) -> bool {
        true
    }

    fn attach_metrics(&self, metrics: Arc<MetricsRegistry>) {
        let _ = self.metrics.set(metrics);
    }
}

/// Running plus waiting requests, from whichever load table has a fresh row.
fn load_of(snapshot: &EngineReportedLoadSnapshot, url: &str) -> Option<u64> {
    if let Some(load) = snapshot.fresh_native_cache_load_for_url(url) {
        return Some(load.num_running_reqs + load.num_waiting_reqs);
    }
    snapshot
        .fresh_load_for_url(url)
        .map(|load| load.num_running_reqs + load.num_waiting_reqs)
}

/// `mean load * load_factor` over the workers with a fresh reading, or None
/// when no worker has one.
fn load_threshold(
    snapshot: &EngineReportedLoadSnapshot,
    workers: &[Arc<Worker>],
    load_factor: f64,
) -> Option<f64> {
    let loads: Vec<u64> = workers
        .iter()
        .filter_map(|worker| load_of(snapshot, &worker.url))
        .collect();
    if loads.is_empty() {
        return None;
    }
    let mean = loads.iter().sum::<u64>() as f64 / loads.len() as f64;
    Some(mean * load_factor)
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

#[inline]
fn fnv1a_step(h: u64, byte: u8) -> u64 {
    (h ^ byte as u64).wrapping_mul(FNV_PRIME)
}

/// FNV-1a over the bytes, finalized with splitmix64 so nearby inputs land far
/// apart. Stable across builds and platforms, which a fleet of router
/// replicas needs; std's `DefaultHasher` is not.
fn hash_bytes(bytes: &[u8]) -> u64 {
    splitmix64(bytes.iter().fold(FNV_OFFSET, |h, b| fnv1a_step(h, *b)))
}

#[inline]
fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Highest-random-weight score of `worker_url` for this prefix.
fn rendezvous_score(prefix_hash: u64, worker_url: &str) -> u64 {
    splitmix64(prefix_hash ^ hash_bytes(worker_url.as_bytes()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
    use crate::state::load_monitor::engine_reported_load::EngineReportedSchedulingLoad;
    use std::collections::HashMap;
    use std::time::Instant;

    fn worker(id: &str) -> Arc<Worker> {
        Arc::new(Worker::new(WorkerSpec {
            id: WorkerId(id.into()),
            url: format!("http://{id}:30000"),
            mode: WorkerMode::Prefill,
            model_ids: vec![ModelId("model".into())],
            bootstrap_port: Some(8998),
        }))
    }

    fn fleet(n: usize) -> Vec<Arc<Worker>> {
        (0..n).map(|i| worker(&format!("prefill-{i}"))).collect()
    }

    fn snapshot(entries: &[(&Arc<Worker>, u64)]) -> EngineReportedLoadSnapshot {
        EngineReportedLoadSnapshot::from_native_cache_workers(
            1,
            entries
                .iter()
                .map(|(worker, running)| {
                    (
                        worker.url.clone(),
                        EngineReportedSchedulingLoad {
                            num_running_reqs: *running,
                            num_waiting_reqs: 0,
                            num_waiting_uncached_tokens: 0,
                            num_used_tokens: 0,
                            num_total_tokens: 0,
                            max_total_num_tokens: 100_000,
                            max_running_requests: 256,
                            prefill_throughput_tokens_per_s: None,
                            estimated_prefill_queue_ms: None,
                            captured_at: Instant::now(),
                        },
                    )
                })
                .collect::<HashMap<_, _>>(),
        )
    }

    fn policy() -> PrefixHashPolicy {
        PrefixHashPolicy::new(PrefixHashConfig::default())
    }

    fn tokens(seed: u32, len: usize) -> Vec<u32> {
        (0..len as u32)
            .map(|i| seed.wrapping_mul(7919) ^ i)
            .collect()
    }

    fn pick(policy: &PrefixHashPolicy, workers: &[Arc<Worker>], toks: &[u32]) -> Arc<Worker> {
        let model = ModelId("model".into());
        let ctx = SelectionContext::new(&model, None).with_request_tokens(Some(toks));
        policy.select(workers, &ctx).expect("a worker")
    }

    #[test]
    fn same_prefix_routes_to_the_same_worker_every_time() {
        let (policy, workers) = (policy(), fleet(4));
        let toks = tokens(1, 300);
        let first = pick(&policy, &workers, &toks);
        for _ in 0..50 {
            assert_eq!(pick(&policy, &workers, &toks).url, first.url);
        }
    }

    #[test]
    fn divergence_past_the_window_still_routes_together() {
        let policy = PrefixHashPolicy::new(PrefixHashConfig {
            prefix_token_count: 8,
            load_factor: 1.25,
        });
        let workers = fleet(4);
        let mut a = tokens(3, 64);
        let mut b = a.clone();
        b[40] ^= 0xffff; // differs after the window
        a[63] ^= 1;
        assert_eq!(
            pick(&policy, &workers, &a).url,
            pick(&policy, &workers, &b).url
        );
    }

    #[test]
    fn divergence_inside_the_window_may_route_apart() {
        let policy = PrefixHashPolicy::new(PrefixHashConfig {
            prefix_token_count: 8,
            load_factor: 1.25,
        });
        let workers = fleet(4);
        let moved = (0..200u32)
            .filter(|seed| {
                let a = tokens(*seed, 16);
                let mut b = a.clone();
                b[2] ^= 0x5555;
                pick(&policy, &workers, &a).url != pick(&policy, &workers, &b).url
            })
            .count();
        // A change inside the window re-hashes; 3/4 of keys move on 4 workers.
        assert!(moved > 100, "only {moved}/200 re-hashed prefixes moved");
    }

    #[test]
    fn distinct_prefixes_spread_across_the_fleet() {
        let (policy, workers) = (policy(), fleet(4));
        let mut counts: HashMap<String, usize> = HashMap::new();
        for seed in 0..400u32 {
            let url = pick(&policy, &workers, &tokens(seed, 32)).url.clone();
            *counts.entry(url).or_default() += 1;
        }
        assert_eq!(counts.len(), 4, "every worker owns some keys: {counts:?}");
        for (url, n) in &counts {
            assert!((50..=150).contains(n), "{url} got {n} of 400");
        }
    }

    #[test]
    fn removing_a_worker_moves_only_its_own_keys() {
        let (policy, full) = (policy(), fleet(5));
        let reduced: Vec<Arc<Worker>> = full[..4].to_vec();
        let gone = &full[4].url;
        let mut moved_from_survivor = 0;
        for seed in 0..300u32 {
            let toks = tokens(seed, 32);
            let before = pick(&policy, &full, &toks);
            let after = pick(&policy, &reduced, &toks);
            if &before.url != gone {
                assert_eq!(before.url, after.url, "key on a surviving worker moved");
            } else {
                moved_from_survivor += 1;
            }
        }
        assert!(moved_from_survivor > 0, "the removed worker owned no keys");
    }

    #[test]
    fn overloaded_target_is_shed_to_the_next_in_rank() {
        let (policy, workers) = (policy(), fleet(3));
        let toks = tokens(11, 32);
        let target = pick(&policy, &workers, &toks);
        // Fleet mean is (30+0+0)/3 = 10; threshold 12.5; the target at 30 is over.
        let loads: Vec<(&Arc<Worker>, u64)> = workers
            .iter()
            .map(|w| (w, if w.url == target.url { 30 } else { 0 }))
            .collect();
        let snap = snapshot(&loads);
        let model = ModelId("model".into());
        let ctx = SelectionContext::new(&model, None)
            .with_request_tokens(Some(&toks))
            .with_load_snapshot(&snap);
        let (picked, outcome) = policy.select_impl(&workers, &ctx).unwrap();
        assert_ne!(picked.url, target.url);
        assert!(matches!(outcome, PrefixHashOutcome::LoadShed));
        // The shed destination is deterministic too: second in rendezvous order.
        let ranked = policy.ranked(&workers, policy.prefix_hash(&toks));
        assert_eq!(picked.url, ranked[1].url);
    }

    #[test]
    fn target_within_the_bound_keeps_the_request() {
        let (policy, workers) = (policy(), fleet(3));
        let toks = tokens(12, 32);
        let target = pick(&policy, &workers, &toks);
        // Everyone at 10: mean 10, threshold 12.5, target passes.
        let loads: Vec<(&Arc<Worker>, u64)> = workers.iter().map(|w| (w, 10)).collect();
        let snap = snapshot(&loads);
        let model = ModelId("model".into());
        let ctx = SelectionContext::new(&model, None)
            .with_request_tokens(Some(&toks))
            .with_load_snapshot(&snap);
        let (picked, outcome) = policy.select_impl(&workers, &ctx).unwrap();
        assert_eq!(picked.url, target.url);
        assert!(matches!(outcome, PrefixHashOutcome::HashHit));
    }

    #[test]
    fn no_tokens_falls_back_to_power_of_two() {
        let (policy, workers) = (policy(), fleet(3));
        let model = ModelId("model".into());
        let ctx = SelectionContext::new(&model, None);
        let (picked, outcome) = policy.select_impl(&workers, &ctx).unwrap();
        assert!(workers.iter().any(|w| w.url == picked.url));
        assert!(matches!(outcome, PrefixHashOutcome::NoTokens));
        let empty: Vec<u32> = Vec::new();
        let ctx = SelectionContext::new(&model, None).with_request_tokens(Some(&empty));
        assert!(matches!(
            policy.select_impl(&workers, &ctx).unwrap().1,
            PrefixHashOutcome::NoTokens
        ));
    }

    #[test]
    fn no_snapshot_is_unbounded_affinity() {
        let (policy, workers) = (policy(), fleet(2));
        let toks = tokens(5, 32);
        let model = ModelId("model".into());
        let ctx = SelectionContext::new(&model, None).with_request_tokens(Some(&toks));
        assert!(matches!(
            policy.select_impl(&workers, &ctx).unwrap().1,
            PrefixHashOutcome::Unbounded
        ));
        assert!(policy.select_impl(&[], &ctx).is_none());
    }

    #[test]
    fn hash_is_stable_and_window_bounded() {
        let policy = PrefixHashPolicy::new(PrefixHashConfig {
            prefix_token_count: 4,
            load_factor: 1.0,
        });
        assert_eq!(
            policy.prefix_hash(&[1, 2, 3, 4, 5]),
            policy.prefix_hash(&[1, 2, 3, 4, 99])
        );
        assert_ne!(
            policy.prefix_hash(&[1, 2, 3, 4]),
            policy.prefix_hash(&[1, 2, 3, 5])
        );
        // Pinned (FNV-1a over little-endian token bytes, then splitmix64) so a
        // future refactor cannot silently re-shard every fleet.
        assert_eq!(policy.prefix_hash(&[1, 2, 3, 4]), 0xcc916d157f93752b_u64);
    }

    #[test]
    fn declares_its_needs() {
        let policy = policy();
        assert!(policy.needs_request_tokens());
        assert!(policy.needs_load_snapshot());
        assert!(!policy.uses_shared_prefill_admission());
    }
}
