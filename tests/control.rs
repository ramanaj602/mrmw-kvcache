//! Control-plane tests: concurrent publish/get with heavy block reuse, and
//! a control plane shared between two shared-memory mappings.

use mrmw_kvcache::control::{FetchError, GetError, KvControlPlane, KvMeta, LookupError};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;

fn meta(prefix_hash: u64, request_id: u64) -> KvMeta {
    KvMeta {
        request_id,
        prefix_hash,
        layer_start: 0,
        layer_end: 32,
        device: 0,
        dtype_layout: 1,
    }
}

/// Deterministic payload for a prefix: its length and every byte depend on
/// the hash, so a torn read or a block shared by two prefixes can't pass.
fn payload(hash: u64, buf: &mut [u8]) -> usize {
    let len = 8 + (hash % 200) as usize;
    buf[..8].copy_from_slice(&hash.to_le_bytes());
    for (i, b) in buf.iter_mut().enumerate().take(len).skip(8) {
        *b = (hash.wrapping_mul(31).wrapping_add(i as u64)) as u8;
    }
    len
}

/// Run writers and readers against one plane. The pool is smaller than the
/// descriptor ring on purpose, so readers constantly race block reuse.
fn stress<const N: usize, const M: usize, const B: usize>(seconds_ms: u64) -> (u64, u64) {
    let plane: Arc<Box<KvControlPlane<N, M, B, 256>>> = Arc::new(KvControlPlane::new_boxed());
    let stop = Arc::new(AtomicBool::new(false));
    let hits = Arc::new(AtomicU64::new(0));
    let misses = Arc::new(AtomicU64::new(0));
    const PREFIXES: u64 = 300;

    let mut handles = Vec::new();
    for w in 0..4u64 {
        let (plane, stop) = (plane.clone(), stop.clone());
        handles.push(thread::spawn(move || {
            let mut buf = [0u8; 256];
            let mut i: u64 = w;
            while !stop.load(Ordering::Relaxed) {
                let hash = 1 + (i % PREFIXES);
                let len = payload(hash, &mut buf);
                plane.publish(meta(hash, i), &buf[..len]).unwrap();
                i += 4;
            }
        }));
    }
    for r in 0..4u64 {
        let (plane, stop, hits, misses) =
            (plane.clone(), stop.clone(), hits.clone(), misses.clone());
        handles.push(thread::spawn(move || {
            let mut out = [0u8; 256];
            let mut expect = [0u8; 256];
            let mut i: u64 = r * 7919;
            while !stop.load(Ordering::Relaxed) {
                let hash = 1 + (i % PREFIXES);
                match plane.get(hash, &mut out) {
                    Ok((desc, n)) => {
                        assert_eq!(desc.prefix_hash, hash, "descriptor for the wrong prefix");
                        let want = payload(hash, &mut expect);
                        assert_eq!(n, want, "wrong length for prefix {hash}");
                        assert_eq!(
                            &out[..n],
                            &expect[..n],
                            "wrong or torn bytes for prefix {hash}"
                        );
                        hits.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(GetError::Lookup(
                        LookupError::Miss | LookupError::Evicted | LookupError::Contended,
                    ))
                    | Err(GetError::Fetch(FetchError::Stale)) => {
                        misses.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(e) => panic!("unexpected error {e:?}"),
                }
                i += 13;
            }
        }));
    }
    thread::sleep(std::time::Duration::from_millis(seconds_ms));
    stop.store(true, Ordering::Relaxed);
    for h in handles {
        h.join().unwrap();
    }
    (hits.load(Ordering::Relaxed), misses.load(Ordering::Relaxed))
}

#[test]
fn concurrent_publish_get_never_returns_wrong_bytes() {
    // Roomy pool: most lookups should hit.
    let (hits, misses) = stress::<256, 512, 256>(500);
    assert!(hits > 0, "no hits at all (misses = {misses})");
}

#[test]
fn heavy_block_reuse_degrades_to_misses_only() {
    // 16 blocks for 300 prefixes and a 64-slot ring: blocks are reused
    // constantly. Correctness is the asserts inside `stress`; here we only
    // check both outcomes actually happened.
    let (hits, misses) = stress::<64, 128, 16>(500);
    assert!(hits > 0 && misses > 0, "hits = {hits}, misses = {misses}");
}

#[cfg(all(unix, feature = "shm", not(miri)))]
#[test]
fn control_plane_in_shared_memory() {
    use mrmw_kvcache::shm::SharedControlPlane;
    type Shm = SharedControlPlane<64, 128, 64, 512>;

    let name = format!("mrmw-cp-{}", std::process::id());
    let prefill = Shm::create(&name).unwrap();
    let decode = Shm::open(&name).unwrap();

    let kv: Vec<u8> = (0..500u32).map(|i| (i * 7) as u8).collect();
    prefill.get().publish(meta(0xABCD, 1), &kv).unwrap();

    let mut out = [0u8; 512];
    let (desc, n) = decode.get().get(0xABCD, &mut out).unwrap();
    assert_eq!((desc.request_id, n), (1, 500));
    assert_eq!(&out[..n], &kv[..]);
    assert_eq!(decode.get().lookup(0x1234), Err(LookupError::Miss));
}
