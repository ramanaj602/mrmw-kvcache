//! Multi-threaded stress test: 8 writers x 200,000 pushes while 4 readers
//! hammer positions right behind the write cursor, checking every
//! successful read for tearing (bytes mixed from two different writes).
//!
//! This is evidence, not proof: it only covers the interleavings the OS
//! happened to produce. Loom is what checks all of them (see README).
//! Far too big for Miri's interpreter, so it's skipped there.
#![cfg(not(miri))]

use mrmw_kvcache::{ReadError, RingBuffer};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;

/// A 64-byte entry whose words are all computed from `key`, so a torn read
/// (half old entry, half new) fails `is_coherent()`.
///
/// C: typedef struct { uint64_t key; uint64_t words[6]; uint64_t check; } kv_entry_t;
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct KvEntry {
    key: u64,
    words: [u64; 6],
    check: u64,
}

impl KvEntry {
    fn new(key: u64) -> KvEntry {
        let mut words: [u64; 6] = [0; 6];
        let mut i: usize = 0;
        while i < 6 {
            // wrapping_mul = C's unsigned overflow behaviour (Rust panics on
            // overflow in debug builds unless you ask for wrapping).
            words[i] = key.wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(i as u32 * 7);
            i += 1;
        }
        KvEntry { key, words, check: !key }
    }

    fn is_coherent(&self) -> bool {
        *self == KvEntry::new(self.key)
    }
}

const WRITERS: u64 = 8;
const PUSHES_PER_WRITER: u64 = 200_000;
const READERS: u64 = 4;
const N: usize = 4096;

/// Tiny xorshift PRNG so readers pick varied positions. C: same 3 lines.
fn xorshift(x: &mut u64) -> u64 {
    *x ^= *x << 13;
    *x ^= *x >> 7;
    *x ^= *x << 17;
    *x
}

#[test]
fn stress_no_torn_reads() {
    let ring: Box<RingBuffer<KvEntry, N>> = RingBuffer::new_boxed();
    let done = AtomicBool::new(false);
    let good_reads = AtomicU64::new(0);
    let misses = AtomicU64::new(0);

    // `thread::scope` = start some pthreads, and pthread_join all of them
    // automatically at the closing brace. Because of that guarantee the
    // threads may use the local variables above directly.
    // `s.spawn(move || { ... })` = pthread_create with the `{ ... }` body as
    // the thread function; `move` copies the loop variable (r / w) into it.
    thread::scope(|s| {
        for r in 0..READERS {
            let ring = &ring;
            let done = &done;
            let good_reads = &good_reads;
            let misses = &misses;
            s.spawn(move || {
                let mut seed: u64 = 0x2545_F491_4F6C_DD1D ^ r;
                while !done.load(Ordering::Relaxed) {
                    // Aim at the last 64 positions: that's where reads race
                    // against in-flight writes.
                    let head: u64 = ring.write_cursor();
                    let back: u64 = xorshift(&mut seed) % 64;
                    if back > head {
                        continue;
                    }
                    let pos: u64 = head - back;

                    // `match` = a switch on the Result's tag.
                    match ring.read_at(pos) {
                        Ok(entry) => {
                            assert!(entry.is_coherent(), "TORN READ at pos {pos}");
                            good_reads.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(_reason) => {
                            misses.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
            });
        }

        // Writers. The readers above are already running.
        let mut writers = Vec::new(); // C: pthread_t writers[WRITERS];
        for w in 0..WRITERS {
            let ring = &ring;
            writers.push(s.spawn(move || {
                for i in 0..PUSHES_PER_WRITER {
                    ring.push(KvEntry::new((w << 32) | i));
                }
            }));
        }
        for handle in writers {
            handle.join().unwrap(); // C: pthread_join(writers[i], NULL);
        }
        done.store(true, Ordering::Relaxed); // tell readers to stop
    }); // <- readers are joined here

    let total: u64 = WRITERS * PUSHES_PER_WRITER;
    assert_eq!(ring.write_cursor(), total);

    // Now that nothing is running, the newest N positions must all be
    // readable and coherent, and every key must be a real (writer, seq) pair.
    let mut pos: u64 = total - N as u64;
    while pos < total {
        let entry: KvEntry = ring.read_at(pos).expect("quiescent read must succeed");
        assert!(entry.is_coherent());
        assert!((entry.key >> 32) < WRITERS);
        assert!((entry.key & 0xFFFF_FFFF) < PUSHES_PER_WRITER);
        pos += 1;
    }
    assert_eq!(ring.read_at(total - N as u64 - 1), Err(ReadError::Evicted));
    assert_eq!(ring.read_at(total), Err(ReadError::NotYetWritten));

    println!(
        "{} pushes, {} coherent concurrent reads, {} misses, 0 torn",
        total,
        good_reads.load(Ordering::Relaxed),
        misses.load(Ordering::Relaxed)
    );
}
