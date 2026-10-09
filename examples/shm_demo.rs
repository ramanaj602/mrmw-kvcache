//! A real cross-process run: one writer process and three reader processes,
//! all mapping the same shared-memory ring, each at its own virtual address.
//!
//!     cargo run --release --example shm_demo
//!
//! The parent creates the ring, launches this same program three more times
//! as readers (like fork+exec), then pushes TOTAL entries. Each reader
//! follows the write cursor and checks every entry it reads for tearing.

use mrmw_kvcache::shm::SharedRing;
use mrmw_kvcache::{ReadError, RingBuffer};
use std::process::{Child, Command, ExitCode};
use std::time::{Duration, Instant};

/// Same 64-byte self-checking entry as the stress test.
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
            words[i] = key.wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(i as u32 * 7);
            i += 1;
        }
        KvEntry { key, words, check: !key }
    }
}

const N: usize = 1 << 14; // 16384 slots
const TOTAL: u64 = 2_000_000;
const READERS: usize = 3;

// `type` = typedef.
type Ring = SharedRing<KvEntry, N>;

fn main() -> ExitCode {
    // C: int main(int argc, char **argv)
    let args: Vec<String> = std::env::args().collect();
    if args.len() >= 4 && args[1] == "reader" {
        return reader(&args[2], &args[3]);
    }
    writer()
}

fn writer() -> ExitCode {
    let name: String = format!("mrmw-demo-{}", std::process::id());
    let shared: Ring = Ring::create(&name).expect("create shared ring");
    let ring = shared.ring();
    println!(
        "[writer pid {}] created /{}: {} slots, {} bytes, mapped at {:p}",
        std::process::id(),
        name,
        N,
        RingBuffer::<KvEntry, N>::SIZE,
        ring
    );

    // Launch the readers: this same executable with "reader <name> <id>".
    let exe = std::env::current_exe().unwrap();
    let mut children: Vec<Child> = Vec::new();
    for id in 0..READERS {
        let child = Command::new(&exe)
            .arg("reader")
            .arg(&name)
            .arg(id.to_string())
            .spawn()
            .expect("spawn reader");
        children.push(child);
    }

    // Give the readers a moment to map the segment before the flood starts.
    std::thread::sleep(Duration::from_millis(200));

    let start = Instant::now();
    for i in 0..TOTAL {
        ring.push(KvEntry::new(i));
    }
    println!("[writer] pushed {} entries in {:?}", TOTAL, start.elapsed());

    // C: waitpid() each child and check its exit status.
    let mut all_ok = true;
    for mut child in children {
        let status = child.wait().expect("wait for reader");
        if !status.success() {
            all_ok = false;
        }
    }

    // `shared` goes out of scope at the end of this function: munmap + shm_unlink.
    if all_ok {
        println!("[writer] all readers clean: 0 torn reads across processes");
        ExitCode::SUCCESS
    } else {
        eprintln!("[writer] a reader FAILED");
        ExitCode::FAILURE
    }
}

fn reader(name: &str, id: &str) -> ExitCode {
    let shared: Ring = Ring::open(name).expect("open shared ring");
    let ring = shared.ring();

    let mut verified: u64 = 0;
    let mut skipped: u64 = 0;
    let mut contended: u64 = 0;
    let mut pos: u64 = 0;

    while pos < TOTAL {
        match ring.read_at(pos) {
            Ok(entry) => {
                // One writer pushes key i at position i, so we can check the
                // entry's identity as well as its coherence.
                if entry != KvEntry::new(pos) {
                    eprintln!("[reader {}] TORN/WRONG entry at {}: {:?}", id, pos, entry);
                    return ExitCode::FAILURE;
                }
                verified += 1;
                pos += 1;
            }
            Err(ReadError::NotYetWritten) => {
                std::hint::spin_loop(); // caught up with the writer; wait
            }
            Err(ReadError::Evicted) => {
                // Fell more than N behind: jump to the middle of the live window.
                let oldest: u64 = ring.write_cursor().saturating_sub(N as u64 / 2);
                if oldest > pos {
                    skipped += oldest - pos;
                    pos = oldest;
                }
            }
            Err(ReadError::Contended) => {
                contended += 1;
            }
        }
    }

    println!(
        "[reader {} pid {}] mapped at {:p}: {} verified, {} skipped (evicted), {} contended",
        id,
        std::process::id(),
        ring,
        verified,
        skipped,
        contended
    );
    ExitCode::SUCCESS
}
