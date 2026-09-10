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

#[cfg(test)]
mod tests;
