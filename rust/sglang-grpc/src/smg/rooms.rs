//! `bootstrap_room` on the SMG transport.
//!
//! SMG's proto declares `int32`, and it draws the value from
//! `rand::rng().random_range(0..i32::MAX)` (`routers/grpc/common/stages/helpers.rs`)
//! — 31 bits, `[0, 2^31-2]`. The HTTP transport uses 63 bits. No server-side
//! transform can add entropy the client never generated, so the lever here is
//! detection, not arithmetic.
//!
//! Observed over 4,602 recorded PD requests: no zero, no repeat, uniform over
//! the 31-bit range. The zero case below is therefore constructed by its test,
//! not waited for.

/// `bootstrap_room == 0` is a reserved sentinel: `_apply_metadata_gate`
/// (`python/sglang/srt/disaggregation/utils.py`) reads a zero room as "metadata
/// has not landed" and downgrades `Success` to `Transferring`, after which
/// `_commit_transfer_to_req` aborts the request with a readiness message that
/// points nowhere near the router.
const RESERVED_ROOM_REPLACEMENT: i64 = 1;

/// Widen SMG's 32-bit room to the `int64` the Python dict carries.
///
/// Zero-extension, not sign-extension. SMG's generator never emits a negative
/// today, but the room lands in a `torch.uint64` metadata buffer
/// (`disaggregation/utils.py`), where a negative wraps and trips the
/// pre-existing corruption detector in `decode.py`. Both PD legs apply this
/// same pure function to the same `room32`, so they still rendezvous; the
/// safety property is determinism, not entropy.
pub fn widen_room(room32: i32) -> i64 {
    match (room32 as u32) as i64 {
        0 => RESERVED_ROOM_REPLACEMENT,
        room => room,
    }
}

/// A collision only steals KV when the two requests' lifetimes overlap; the
/// Python side treats reuse after completion as normal and pops the room in
/// `KVManager.clear_status`. So only concurrent reuse is tracked, and it is
/// turned into a loud, attributable, retryable rejection instead of one
/// request's decode consuming another's KV.
///
/// Expected rejection rate is roughly `concurrency / 2.1e9` per request. Lifting
/// it means widening `DisaggregatedParams.bootstrap_room` in `smg-grpc-client`,
/// which is upstream.
#[derive(Debug, Default)]
pub struct RoomRegistry {
    in_flight: std::sync::Mutex<std::collections::HashMap<i64, String>>,
}

/// A leaked entry poisons its room for the process lifetime, so the size is
/// reported rather than left to grow quietly. Chosen to sit far above any
/// plausible real concurrency; it is a leak detector, not a limit.
const ROOM_REGISTRY_WARN_SIZE: usize = 100_000;

impl RoomRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Reserve `room` for `rid` until the returned guard drops.
    pub fn claim(
        self: &std::sync::Arc<Self>,
        room: i64,
        rid: &str,
    ) -> Result<RoomGuard, tonic::Status> {
        let mut in_flight = self
            .in_flight
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        if let Some(holder) = in_flight.get(&room) {
            return Err(tonic::Status::already_exists(format!(
                "bootstrap_room {} is already in flight for request {}; SMG draws \
                 it from a 31-bit space on this transport, so concurrent \
                 collisions are expected and retryable",
                room, holder
            )));
        }
        in_flight.insert(room, rid.to_string());
        if in_flight.len() > ROOM_REGISTRY_WARN_SIZE {
            tracing::warn!(
                in_flight = in_flight.len(),
                "in-flight bootstrap_room registry is larger than any plausible \
                 concurrency; rooms are probably not being released"
            );
        }

        Ok(RoomGuard {
            registry: std::sync::Arc::clone(self),
            room,
        })
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.in_flight
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len()
    }
}

/// Releases the room on every exit path — completion, error, abort, panic. The
/// release must never live on the happy path: a missed removal poisons that
/// room for the process lifetime and every retry then fails.
#[derive(Debug)]
pub struct RoomGuard {
    registry: std::sync::Arc<RoomRegistry>,
    room: i64,
}

impl Drop for RoomGuard {
    fn drop(&mut self) {
        self.registry
            .in_flight
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(&self.room);
    }
}

#[cfg(test)]
mod tests;
