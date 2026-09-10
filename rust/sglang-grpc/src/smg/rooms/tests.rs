use std::sync::Arc;

use tonic::Code;

use super::{RoomRegistry, widen_room};

/// T11.
#[test]
fn bootstrap_room_is_zero_extended() {
    assert_eq!(widen_room(1), 1);
    assert_eq!(widen_room(i32::MAX), 2_147_483_647);
    // A recorded value, to pin that the common case is untouched.
    assert_eq!(widen_room(507_686_309), 507_686_309);

    assert_eq!(
        widen_room(-1),
        4_294_967_295,
        "sign-extension would make this -1, which wraps in the torch.uint64 \
         metadata buffer and trips the decode-side corruption detector"
    );
    assert_eq!(widen_room(i32::MIN), 2_147_483_648);
}

/// T12. Never observed on the wire — SMG's generator has a closed lower bound
/// at 0, so the case is constructed here rather than waited for.
#[test]
fn bootstrap_room_zero_is_remapped() {
    assert_ne!(
        widen_room(0),
        0,
        "a zero room reads as 'metadata has not landed' and aborts the request \
         with a message that points nowhere near the router"
    );
}

/// The transform must be a pure function of `room32`, or the prefill and decode
/// legs — which each receive the same room from SMG independently — stop
/// rendezvousing.
#[test]
fn widening_is_deterministic_across_legs() {
    for room in [0, 1, 7, 507_686_309, i32::MAX, -1, i32::MIN] {
        assert_eq!(widen_room(room), widen_room(room));
    }
}

/// T13. Without the registry a concurrent collision means one request's decode
/// consumes another's KV, with no error anywhere.
#[test]
fn concurrent_room_reuse_is_rejected() {
    let registry = Arc::new(RoomRegistry::new());

    let first = registry.claim(507_686_309, "chatcmpl-a").unwrap();

    let status = registry.claim(507_686_309, "chatcmpl-b").unwrap_err();
    assert_eq!(status.code(), Code::AlreadyExists);
    assert!(
        status.message().contains("chatcmpl-a"),
        "the rejection must name the holder: {}",
        status.message()
    );

    // A different room is unaffected.
    let _other = registry.claim(507_686_310, "chatcmpl-c").unwrap();

    drop(first);
    registry
        .claim(507_686_309, "chatcmpl-b")
        .expect("reuse after completion is the normal case, not a collision");
}

/// R5: a missed release poisons the room for the process lifetime, so the guard
/// must free it on every exit path, not just the happy one.
#[test]
fn the_guard_releases_the_room_on_every_path() {
    let registry = Arc::new(RoomRegistry::new());

    // Normal scope exit.
    {
        let _guard = registry.claim(1, "a").unwrap();
        assert_eq!(registry.len(), 1);
    }
    assert_eq!(registry.len(), 0);

    // Unwind.
    let unwound = std::panic::catch_unwind({
        let registry = Arc::clone(&registry);
        move || {
            let _guard = registry.claim(2, "b").unwrap();
            panic!("simulated stream failure");
        }
    });
    assert!(unwound.is_err());
    assert_eq!(registry.len(), 0);
}
