# mrmw-kvcache

A lock-free, **multi-reader / multi-writer** ring buffer that lives in
**shared memory across processes**. Writers claim a position with one atomic
add and publish seqlock-style; readers fetch a specific position with an
optimistic, never-blocking read.

The code is written for a C programmer who is new to Rust: every Rust-only idea
has a `C:` comment showing what you'd write in C. Pair it with the
*Ring Buffer Internals* companion doc for diagrams.

## Layout

```
src/lib.rs           RingBuffer<T, N>: init_in_place, new_boxed, push, read_at
src/shm.rs           SharedRing: shm_open + ftruncate + mmap wrapper (Unix)
tests/stress.rs      8 writers x 200k pushes, 4 readers checking for torn reads
tests/shm.rs         two mappings of one shm object see each other's writes
examples/shm_demo.rs 1 writer process + 3 reader processes over real shm
```

In C terms the shared struct is:

```c
typedef struct __attribute__((aligned(64))) {
    _Atomic uint64_t tag;     /* pos*2 = stable, pos*2+1 = busy, ~1 = empty */
    T                value;
} slot_t;

typedef struct {
    _Atomic uint64_t write_cursor;
    char             _pad0[56];   /* cursor gets its own cache line */
    slot_t           slots[N];    /* N is a power of two */
} ring_t;                         /* sizeof = 64 + N*align_up(8+sizeof(T), 64) */
```

## Build and run

You need a Rust toolchain (`curl https://sh.rustup.rs -sSf | sh`). Then:

```sh
cargo test --release                       # unit + shm + stress tests
cargo run --release --example shm_demo     # real multi-process run
cargo build --no-default-features          # pure-Rust build, no libc (for Miri)
```

`cargo` is Rust's `make` + package manager; `Cargo.toml` is the Makefile.

## Using it

```rust
use mrmw_kvcache::shm::SharedRing;

#[derive(Clone, Copy)]
#[repr(C)]
struct Entry { key: u64, val: [u8; 48] }    // plain data, no pointers

// process A
let shared = SharedRing::<Entry, 4096>::create("my-cache")?;
let pos = shared.ring().push(Entry { key: 1, val: [0; 48] });

// process B (after A's create returned)
let shared = SharedRing::<Entry, 4096>::open("my-cache")?;
match shared.ring().read_at(pos) {
    Ok(entry) => { /* coherent copy */ }
    Err(e)    => { /* NotYetWritten | Evicted | Contended */ }
}
```

## Changes from the original design doc

The structure, tag encoding, and API follow the doc. Testing turned up four
problems with the doc's version, and each is fixed here:

1. **Writer lapping (real bug, found by the stress test).** The doc's guard
   ("spin while busy, then store busy") is two separate steps. Two writers N
   positions apart could both pass it and write the payload at once. A writer
   descheduled between `fetch_add` and publishing could also overwrite a
   newer entry with an older one. The stress test lost entries in about 1 of
   6 runs on a 2-core machine. `push` now claims the slot with one
   compare-and-swap that only succeeds if the slot is stable *and* holds an
   older position. A writer that was lapped drops its (already-evicted)
   write. After the fix: 0 failures in 60 runs.
2. **Release fence after "mark busy".** A release *store* doesn't stop the
   payload write after it from becoming visible first. On ARM (Apple Silicon)
   that allows a torn read that passes the tag re-check. `push` now has
   `fence(Release)` after claiming the slot.
3. **Acquire fence before the re-check.** For the same reason, `read_at` has
   `fence(Acquire)` between the optimistic copy and the second tag load.
4. **`EMPTY_TAG`.** A zeroed tag means "position 0, stable", so `read_at(0)`
   on a fresh ring would have returned uninitialized bytes. Slots now start at
   `EMPTY_TAG` (`0xFFFF_FFFF_FFFF_FFFE`), which reads as `NotYetWritten`.

## Known limitations / next steps

- **Miri will flag the optimistic copy as a data race.** That's inherent to
  seqlocks written with plain loads and stores. The race is benign in practice
  because the copy is discarded when the tag moved, but it is UB in Rust's
  (and C11's) formal model. The standard fix is to copy the payload as
  relaxed atomic words. Next step: a Miri run (`cargo +nightly miri test
  --no-default-features`) plus that change.
- **Loom test not written yet.** It would exhaustively check the
  push/read_at/CAS protocol with 2 writers and 1 reader.
- `T: Copy` doesn't forbid raw pointers inside `T`. For cross-process use,
  keep pointers out of `T` yourself.
- `open()` must happen after `create()` returns. There's no
  "ready" handshake in the shared struct.
