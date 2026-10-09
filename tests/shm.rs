//! Shared-memory tests: two separate mappings of the same object (at
//! different virtual addresses) must see each other's writes.
//! Only built on Unix with the "shm" feature, and never under Miri
//! (Miri can't make real system calls).
#![cfg(all(unix, feature = "shm", not(miri)))]

use mrmw_kvcache::shm::SharedRing;
use mrmw_kvcache::{ReadError, RingBuffer};
use std::io::ErrorKind;

/// Per-process unique name so parallel test runs don't collide.
/// (macOS limits shm names to 31 characters, so keep these short.)
fn unique(tag: &str) -> String {
    format!("mrmw-{}-{}", tag, std::process::id())
}

#[test]
fn two_mappings_share_data() {
    let name = unique("share");
    let a = SharedRing::<[u64; 4], 16>::create(&name).unwrap();
    let b = SharedRing::<[u64; 4], 16>::open(&name).unwrap();

    // Same physical pages, different virtual addresses -- exactly the
    // situation between two processes.
    let addr_a = a.ring() as *const RingBuffer<[u64; 4], 16>;
    let addr_b = b.ring() as *const RingBuffer<[u64; 4], 16>;
    assert_ne!(addr_a, addr_b);

    assert_eq!(b.ring().read_at(0), Err(ReadError::NotYetWritten));

    // 20 pushes through `a` into 16 slots...
    for i in 0..20u64 {
        assert_eq!(a.ring().push([i, i + 1, i + 2, i + 3]), i);
    }
    // ...are all visible through `b`.
    assert_eq!(b.ring().write_cursor(), 20);
    assert_eq!(b.ring().read_at(3), Err(ReadError::Evicted));
    for i in 4..20u64 {
        assert_eq!(b.ring().read_at(i), Ok([i, i + 1, i + 2, i + 3]));
    }

    // And the other direction.
    let pos = b.ring().push([7; 4]);
    assert_eq!(a.ring().read_at(pos), Ok([7; 4]));
}

#[test]
fn create_is_exclusive_and_open_checks_size() {
    let name = unique("excl");
    let _first = SharedRing::<u64, 8>::create(&name).unwrap();

    // A second create with the same name fails, like O_EXCL -> EEXIST.
    let second = SharedRing::<u64, 8>::create(&name);
    assert_eq!(second.err().unwrap().kind(), ErrorKind::AlreadyExists);

    // Opening with a much bigger N means a different size: refused.
    let wrong = SharedRing::<u64, 1024>::open(&name);
    assert_eq!(wrong.err().unwrap().kind(), ErrorKind::InvalidData);
}

#[test]
fn owner_drop_unlinks() {
    let name = unique("unl");
    let ring = SharedRing::<u64, 8>::create(&name).unwrap();
    drop(ring); // C: shared_ring_close(&ring);  -> munmap + shm_unlink

    let reopened = SharedRing::<u64, 8>::open(&name);
    assert_eq!(reopened.err().unwrap().kind(), ErrorKind::NotFound);
}
