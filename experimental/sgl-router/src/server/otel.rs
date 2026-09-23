// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! OpenTelemetry export and W3C trace-context propagation.
//!
//! Off unless `--otlp-endpoint` is set; every function here is a no-op then,
//! so the request path pays an atomic load and nothing else. When on:
//!
//! * `request_span` opens one server span per chat request and adopts the
//!   caller's `traceparent`/`tracestate` as its parent, so a client trace
//!   continues through the router.
//! * `inject_current_context` writes the router span's context onto the
//!   outgoing worker request. The engine reads `traceparent` off HTTP
//!   headers (`extract_trace_headers`), so its spans become children of the
//!   router's rather than a second, disconnected trace. This replaces the
//!   whitelisted pass-through of the caller's header: with a router span in
//!   between, forwarding the caller's context unchanged would make the
//!   engine a sibling of the router instead of its child.
//! * Spans export over OTLP/gRPC from a batch processor on its own thread;
//!   `SdkTracerProvider::shutdown` at exit flushes what is queued.

use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result};
use axum::http::{HeaderMap, HeaderName, HeaderValue};
use opentelemetry::global;
use opentelemetry::propagation::{Extractor, Injector};
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::trace::{SdkTracerProvider, Tracer};
use opentelemetry_sdk::Resource;
use tracing_opentelemetry::{OpenTelemetryLayer, OpenTelemetrySpanExt};

/// Owner of the exporter; `shutdown` flushes queued spans.
pub type SdkTracerProviderHandle = SdkTracerProvider;

/// Default `service.name` resource attribute.
pub const DEFAULT_SERVICE_NAME: &str = "sgl-router";

static ENABLED: AtomicBool = AtomicBool::new(false);

/// Whether an OTLP exporter was installed at startup.
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Install the W3C propagator and mark propagation on. Called by
/// [`build_layer`]; exposed so tests can exercise propagation with an
/// in-process tracer instead of an exporter.
pub fn enable_propagation() {
    global::set_text_map_propagator(TraceContextPropagator::new());
    ENABLED.store(true, Ordering::Relaxed);
}

/// The `tracing` layer that turns spans into OTLP exports, plus the provider
/// that owns the exporter (keep it; call `shutdown` on exit to flush).
pub fn build_layer<S>(
    endpoint: &str,
    service_name: &str,
) -> Result<(OpenTelemetryLayer<S, Tracer>, SdkTracerProvider)>
where
    S: tracing::Subscriber + for<'span> tracing_subscriber::registry::LookupSpan<'span>,
{
    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_endpoint(endpoint)
        .build()
        .with_context(|| format!("build OTLP span exporter for {endpoint}"))?;
    let provider = SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .with_resource(
            Resource::builder()
                .with_service_name(service_name.to_string())
                .build(),
        )
        .build();
    let tracer = provider.tracer("sgl-router");
    global::set_tracer_provider(provider.clone());
    enable_propagation();
    Ok((tracing_opentelemetry::layer().with_tracer(tracer), provider))
}

struct HeaderExtractor<'a>(&'a HeaderMap);

impl Extractor for HeaderExtractor<'_> {
    fn get(&self, key: &str) -> Option<&str> {
        self.0.get(key).and_then(|v| v.to_str().ok())
    }

    fn keys(&self) -> Vec<&str> {
        self.0.keys().map(|k| k.as_str()).collect()
    }
}

struct HeaderInjector<'a>(&'a mut HeaderMap);

impl Injector for HeaderInjector<'_> {
    fn set(&mut self, key: &str, value: String) {
        if let (Ok(name), Ok(value)) = (
            HeaderName::from_bytes(key.as_bytes()),
            HeaderValue::from_str(&value),
        ) {
            self.0.insert(name, value);
        }
    }
}

/// One server span per request, parented on the caller's trace context when
/// there is one. Fields the handler fills in later are declared empty here;
/// `tracing` drops records for undeclared fields.
pub fn request_span(operation: &'static str, headers: &HeaderMap) -> tracing::Span {
    let span = tracing::info_span!(
        "http.request",
        otel.name = operation,
        otel.kind = "server",
        worker.prefill = tracing::field::Empty,
        worker.decode = tracing::field::Empty,
        http.response.status_code = tracing::field::Empty,
    );
    if enabled() {
        let parent = global::get_text_map_propagator(|p| p.extract(&HeaderExtractor(headers)));
        // Err only when no OpenTelemetry layer tracks this span (disabled
        // subscriber); the span still works as a plain tracing span.
        if let Err(e) = span.set_parent(parent) {
            tracing::debug!(error = %e, "span has no OpenTelemetry context to parent");
        }
    }
    span
}

/// Write the current span's context as `traceparent`/`tracestate` onto an
/// outgoing header map, replacing any forwarded copy of the caller's.
pub fn inject_current_context(headers: &mut HeaderMap) {
    if !enabled() {
        return;
    }
    let cx = tracing::Span::current().context();
    global::get_text_map_propagator(|p| p.inject_context(&cx, &mut HeaderInjector(headers)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use tracing_subscriber::layer::SubscriberExt;

    const INCOMING: &str = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";

    fn with_test_tracer(f: impl FnOnce()) {
        enable_propagation();
        let provider = SdkTracerProvider::builder().build();
        let layer = tracing_opentelemetry::layer().with_tracer(provider.tracer("test"));
        let subscriber = tracing_subscriber::registry().with(layer);
        tracing::subscriber::with_default(subscriber, f);
    }

    fn traceparent(headers: &HeaderMap) -> String {
        headers
            .get("traceparent")
            .expect("traceparent injected")
            .to_str()
            .unwrap()
            .to_string()
    }

    #[test]
    fn continues_the_callers_trace_and_parents_the_worker_on_the_router_span() {
        with_test_tracer(|| {
            let mut incoming = HeaderMap::new();
            incoming.insert("traceparent", HeaderValue::from_static(INCOMING));
            let span = request_span("chat_completions", &incoming);
            let _guard = span.enter();
            let mut outgoing = HeaderMap::new();
            inject_current_context(&mut outgoing);
            let tp = traceparent(&outgoing);
            let parts: Vec<&str> = tp.split('-').collect();
            assert_eq!(parts.len(), 4, "{tp}");
            assert_eq!(
                parts[1], "4bf92f3577b34da6a3ce929d0e0e4736",
                "trace id continues"
            );
            assert_ne!(
                parts[2], "00f067aa0ba902b7",
                "parent span is the router's, not the caller's"
            );
            assert_eq!(parts[3], "01", "sampled");
        });
    }

    #[test]
    fn starts_a_trace_when_the_caller_sent_none() {
        with_test_tracer(|| {
            let span = request_span("chat_completions", &HeaderMap::new());
            let _guard = span.enter();
            let mut outgoing = HeaderMap::new();
            inject_current_context(&mut outgoing);
            let tp = traceparent(&outgoing);
            assert_eq!(tp.len(), 55, "{tp}");
            assert!(!tp.contains("00000000000000000000000000000000"), "{tp}");
        });
    }

    #[test]
    fn overrides_a_forwarded_caller_header() {
        with_test_tracer(|| {
            let mut incoming = HeaderMap::new();
            incoming.insert("traceparent", HeaderValue::from_static(INCOMING));
            let span = request_span("chat_completions", &incoming);
            let _guard = span.enter();
            // What the whitelist would have copied verbatim.
            let mut outgoing = incoming.clone();
            inject_current_context(&mut outgoing);
            assert_ne!(traceparent(&outgoing), INCOMING);
        });
    }
}
