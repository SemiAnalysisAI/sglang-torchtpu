use super::widen_room;

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
