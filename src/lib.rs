//! mrmw-kvcache: a lock-free, multi-reader / multi-writer ring buffer that
//! lives in memory shared between processes.
//!
//! Written for someone who knows C well and is new to Rust: every Rust-only
//! idea has a `C:` comment next to it showing what you'd write in C.
//! See also the "Ring Buffer Internals" companion doc for diagrams.
//!
//! The big picture:
//!   * Writers claim a unique absolute position `pos` with one atomic add,
//!     then publish into slot `pos % N` seqlock-style (busy -> write -> stable).
//!   * Readers ask for a specific `pos` and copy the slot optimistically,
//!     re-checking afterwards that nobody wrote over it mid-copy.
//!   * Nothing inside the struct is a pointer -- only atomics and plain data --
//!     because each process maps the shared memory at a different address.
//!
//! Memory layout (identical to the C struct you'd write):
//!
//! ```text
//!   offset 0   write_cursor   8 bytes   (_Atomic uint64_t)
//!   offset 8   _pad0         56 bytes   (keeps the cursor alone on its cache line)
//!   offset 64  slots[N]      N * align_up(8 + sizeof(T), 64) bytes
//! ```
//!
//! The tag word (one per slot):
//!
//! ```text
//!   tag == pos*2       stable: slot holds position `pos`, safe to read
//!   tag == pos*2 + 1   busy:   a writer is mid-publish, readers back off
//!   tag == EMPTY_TAG   never written since init
//! ```

// Every `unsafe` operation must sit inside an explicit `unsafe { }` block,
// even inside an `unsafe fn`. Makes the dangerous lines easy to find.
#![deny(unsafe_op_in_unsafe_fn)]

// `use` = `#include`, except it imports specific names.
use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::sync::atomic::{fence, AtomicU64, Ordering};

// The shm_open/mmap wrapper lives in src/shm.rs. It is only compiled on
// Unix and only when the "shm" feature is on (it is by default).
// C: #if defined(__unix__) && defined(FEATURE_SHM)  #include "shm.h"  #endif
#[cfg(all(unix, feature = "shm"))]
pub mod shm;

/// Tag value of a slot that has never been written: 0xFFFF_FFFF_FFFF_FFFE.
///
/// It is even (bit 0 clear), so writers don't mistake it for "busy".
/// Without it, a fresh all-zero tag would read as "position 0, stable" and
/// `read_at(0)` would return garbage before anything was pushed.
/// C: #define EMPTY_TAG (~(uint64_t)1)
pub const EMPTY_TAG: u64 = !1;

/// How many optimistic read attempts before giving up with `Contended`.
pub const READ_RETRIES: u32 = 64;

/// Why `read_at` could not return a value.
///
/// C: typedef enum { NOT_YET_WRITTEN, EVICTED, CONTENDED } read_error_t;
/// (`derive` asks the compiler to auto-generate printing (`Debug`),
/// copying (`Clone, Copy`) and `==` (`PartialEq, Eq`) for this type.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadError {
    /// Nobody has finished writing this position yet.
    NotYetWritten,
    /// The ring wrapped past this position; its slot now holds a newer one.
    Evicted,
    /// The slot kept changing under us for READ_RETRIES attempts.
    Contended,
}

/// One slot, exactly one cache line (or a multiple of one) in size.
///
/// C:
/// ```c
/// typedef struct __attribute__((aligned(64))) {
///     _Atomic uint64_t tag;
///     T                value;   /* may be garbage until tag says otherwise */
/// } slot_t;
/// ```
#[repr(C, align(64))]
pub struct Slot<T> {
    tag: AtomicU64,
    // UnsafeCell: "this field may be written through a shared pointer".
    //   C has no equivalent because C always allows that.
    // MaybeUninit<T>: "these bytes may not hold a valid T yet".
    //   C: union { T value; char raw[sizeof(T)]; }
    value: UnsafeCell<MaybeUninit<T>>,
}

// Promise the compiler that a Slot may be used from many threads at once.
// It can't prove that itself because of the UnsafeCell; the tag protocol is
// what actually makes it safe. In C you'd never write this line -- C assumes
// everything is shareable and leaves the correctness to you.
//
// `T: Copy + Send` limits T to plain-old-data that can move between threads:
// no destructors, no owned heap memory. (It does NOT forbid raw pointers
// inside T -- for cross-process use, keep pointers out of T yourself.)
unsafe impl<T: Copy + Send> Sync for Slot<T> {}

/// The shared ring. `N` (number of slots) must be a power of two.
///
/// `RingBuffer<T, N>` is a generic: the compiler stamps out a separate
/// struct + functions for each (T, N) you actually use, like a C++ template
/// or a C macro that generates `ring_kv_1024_t`.
///
/// C:
/// ```c
/// typedef struct {
///     _Atomic uint64_t write_cursor;
///     char             _pad0[56];
///     slot_t           slots[N];
/// } ring_t;
/// ```
#[repr(C)]
pub struct RingBuffer<T, const N: usize> {
    write_cursor: AtomicU64,
    _pad0: [u8; 56],
    slots: [Slot<T>; N],
}

// `impl` block = the functions that operate on RingBuffer.
// `&self` is the `ring_t *r` first argument C would take explicitly.
impl<T: Copy + Send, const N: usize> RingBuffer<T, N> {
    /// N - 1, used as `pos & MASK` instead of `pos % N`.
    /// The assert runs at COMPILE time: a non-power-of-two N won't build.
    /// C: _Static_assert((N & (N - 1)) == 0, "N must be a power of two");
    pub const MASK: u64 = {
        assert!(N.is_power_of_two(), "RingBuffer N must be a power of two");
        N as u64 - 1
    };

    /// Total size in bytes. This is what you pass to ftruncate() and mmap().
    /// C: sizeof(ring_t)
    pub const SIZE: usize = std::mem::size_of::<Self>();

    /// Initialize a ring in raw memory (e.g. fresh from mmap).
    ///
    /// C equivalent:
    /// ```c
    /// void init_in_place(ring_t *p) {
    ///     atomic_init(&p->write_cursor, 0);
    ///     memset(p->_pad0, 0, sizeof p->_pad0);
    ///     for (size_t i = 0; i < N; i++)
    ///         atomic_init(&p->slots[i].tag, EMPTY_TAG);
    ///     /* slot values intentionally left uninitialized */
    /// }
    /// ```
    ///
    /// # Safety
    /// `p` must point to at least `SIZE` writable bytes, aligned to 64,
    /// and nobody else may touch that memory until this returns.
    /// (An `unsafe fn` is one whose caller must uphold rules the compiler
    /// can't check -- i.e. a normal C function.)
    pub unsafe fn init_in_place(p: *mut Self) {
        let _ = Self::MASK; // forces the power-of-two check above

        // `std::ptr::addr_of_mut!((*p).field)` = `&p->field` in C.
        // We use it (instead of a normal reference) because the struct isn't
        // valid yet, and Rust forbids references to invalid data.
        // `.write(x)` = plain store through the pointer: `*ptr = x;`
        unsafe {
            std::ptr::addr_of_mut!((*p).write_cursor).write(AtomicU64::new(0));
            std::ptr::addr_of_mut!((*p)._pad0).write([0u8; 56]);

            // C: slot_t *slots = p->slots;
            let slots: *mut Slot<T> = std::ptr::addr_of_mut!((*p).slots) as *mut Slot<T>;
            let mut i: usize = 0;
            while i < N {
                let s: *mut Slot<T> = slots.add(i); // C: &slots[i]
                std::ptr::addr_of_mut!((*s).tag).write(AtomicU64::new(EMPTY_TAG));
                i += 1;
            }
        }
    }

    /// Allocate and initialize a ring on the heap (no shared memory).
    /// Same struct, same code paths -- used by the tests and for Miri.
    ///
    /// `Box<T>` is an owning pointer: C's `malloc` + automatic `free` when
    /// the Box goes out of scope.
    pub fn new_boxed() -> Box<Self> {
        // C: ring_t *p = aligned_alloc(64, sizeof(ring_t));
        let layout = std::alloc::Layout::new::<Self>();
        unsafe {
            let p: *mut Self = std::alloc::alloc(layout) as *mut Self;
            if p.is_null() {
                std::alloc::handle_alloc_error(layout); // aborts, like a failed malloc check
            }
            Self::init_in_place(p);
            Box::from_raw(p) // hand ownership of `p` to a Box
        }
    }

    /// Publish `value`. Returns the absolute position it was written to.
    ///
    /// C equivalent:
    /// ```c
    /// uint64_t push(ring_t *r, T value) {
    ///     uint64_t pos = atomic_fetch_add_explicit(&r->write_cursor, 1,
    ///                                              memory_order_relaxed);
    ///     slot_t *slot = &r->slots[pos & MASK];
    ///     /* CAS loop: stable-and-older -> busy(pos); see step (2) below */
    ///     if (!claim_slot(slot, pos)) return pos;   /* lapped: already evicted */
    ///     atomic_thread_fence(memory_order_release);
    ///     slot->value = value;
    ///     atomic_store_explicit(&slot->tag, pos*2, memory_order_release);
    ///     return pos;
    /// }
    /// ```
    pub fn push(&self, value: T) -> u64 {
        // (1) CLAIM: one atomic add gives every writer a unique `pos`.
        let pos: u64 = self.write_cursor.fetch_add(1, Ordering::Relaxed);
        // `self.slots[i]` is bounds-checked (it would panic, not corrupt
        // memory). `& MASK` keeps it in range, so the check never fires.
        let slot: &Slot<T> = &self.slots[(pos & Self::MASK) as usize];

        // (2) CLAIM THE SLOT: atomically flip its tag from "stable, older
        //     than us" to "busy, ours" with one compare-and-swap.
        //
        //     Why a CAS and not "wait until not busy, then store busy" (the
        //     original design)? Two reasons, both seen in the stress test:
        //       a) two writers that are N apart can both see "not busy" and
        //          both write the payload at once -> torn data;
        //       b) a writer descheduled between fetch_add and here can wake
        //          up after a NEWER writer (pos + N) published into this
        //          slot, and overwrite the newer entry with an older one.
        //     The CAS makes the check-and-mark one indivisible step, and the
        //     "older than us" test makes a slot's position only ever go up.
        //
        // C:
        //   for (;;) {
        //       uint64_t t = atomic_load_explicit(&slot->tag, memory_order_acquire);
        //       if (t & 1) { pause(); continue; }              /* busy */
        //       if (t != EMPTY_TAG && (t >> 1) > pos) return pos; /* lapped */
        //       if (atomic_compare_exchange_weak_explicit(&slot->tag, &t, pos*2 + 1,
        //               memory_order_acquire, memory_order_relaxed)) break;
        //   }
        loop {
            // `loop` = `for (;;)`
            let current: u64 = slot.tag.load(Ordering::Acquire);
            if current & 1 == 1 {
                std::hint::spin_loop(); // C: _mm_pause() / ARM `yield`
                continue; // another writer is mid-publish on this slot
            }
            if current != EMPTY_TAG && (current >> 1) > pos {
                // A newer position already lives here: ours was overwritten
                // before we even wrote it, i.e. it's already evicted. Drop
                // the write; readers asking for `pos` will get Evicted.
                return pos;
            }
            // compare_exchange_weak(expected, new, ...) returns Ok if the tag
            // still equalled `current` and is now `pos*2 + 1` (busy, ours).
            // "weak" may fail spuriously; we just loop, exactly as in C.
            let swapped = slot.tag.compare_exchange_weak(
                current,
                pos * 2 + 1,
                Ordering::Acquire,
                Ordering::Relaxed,
            );
            if swapped.is_ok() {
                break; // the slot is ours and marked busy
            }
        }

        // (3) PUBLISH, seqlock-style: (busy, done above) -> write -> stable.

        // A release STORE only orders the writes *before* it. This fence
        // stops the CPU from making the value write below visible before the
        // busy mark above. Needed on ARM (your Mac); free on x86.
        fence(Ordering::Release);

        // C: slot->value = value;
        // `.get()` turns the UnsafeCell into a raw `MaybeUninit<T> *`;
        // `.write()` stores the bytes without reading the old ones.
        unsafe {
            (*slot.value.get()).write(value);
        }

        slot.tag.store(pos * 2, Ordering::Release); // mark stable
        pos // no `return` needed: the last expression is the return value
    }

    /// Read the value at absolute position `pos` without ever blocking.
    ///
    /// `Result<T, ReadError>` is a tagged union: either `Ok(value)` or
    /// `Err(reason)`. The compiler forces the caller to check which one it
    /// got before using the value.
    /// C: `read_error_t read_at(ring_t *r, uint64_t pos, T *out);`
    ///
    /// C equivalent:
    /// ```c
    /// for (int i = 0; i < 64; i++) {
    ///     uint64_t t1 = atomic_load_explicit(&slot->tag, memory_order_acquire);
    ///     if (t1 & 1) continue;                         /* busy */
    ///     if (t1 == EMPTY_TAG) return NOT_YET_WRITTEN;
    ///     if ((t1 >> 1) != pos)
    ///         return (t1 >> 1) < pos ? NOT_YET_WRITTEN : EVICTED;
    ///     T tmp = slot->value;                          /* optimistic copy */
    ///     atomic_thread_fence(memory_order_acquire);
    ///     uint64_t t2 = atomic_load_explicit(&slot->tag, memory_order_acquire);
    ///     if (t1 == t2) { *out = tmp; return OK; }
    /// }
    /// return CONTENDED;
    /// ```
    pub fn read_at(&self, pos: u64) -> Result<T, ReadError> {
        let slot: &Slot<T> = &self.slots[(pos & Self::MASK) as usize];

        // Bounded retry: lock-free, not wait-free. One reader can be forced
        // to retry, but some writer is always completing, so the system as
        // a whole always makes progress.
        let mut attempt: u32 = 0;
        while attempt < READ_RETRIES {
            attempt += 1;

            let t1: u64 = slot.tag.load(Ordering::Acquire);
            if t1 & 1 == 1 {
                std::hint::spin_loop();
                continue; // writer mid-publish, try again
            }
            if t1 == EMPTY_TAG {
                return Err(ReadError::NotYetWritten);
            }

            let slot_pos: u64 = t1 >> 1;
            if slot_pos != pos {
                // The tag says which position the slot holds, so we know
                // which kind of miss this is.
                if slot_pos < pos {
                    return Err(ReadError::NotYetWritten);
                } else {
                    return Err(ReadError::Evicted);
                }
            }

            // Optimistic copy, no lock held.
            // C: T value = slot->value;
            let value: T = unsafe { (*slot.value.get()).assume_init_read() };

            // An acquire LOAD only orders what comes *after* it. This fence
            // keeps the copy above from being moved below the re-check.
            fence(Ordering::Acquire);

            // If the tag didn't move, nothing wrote over us mid-copy, so
            // `value` is coherent. If it moved, throw the copy away.
            let t2: u64 = slot.tag.load(Ordering::Acquire);
            if t1 == t2 {
                return Ok(value);
            }
        }
        Err(ReadError::Contended)
    }

    /// The next position a writer will claim (= number of pushes started).
    pub fn write_cursor(&self) -> u64 {
        self.write_cursor.load(Ordering::Acquire)
    }
}

// Unit tests. `#[cfg(test)]` = only compiled by `cargo test`.
// C: #ifdef UNIT_TEST ... #endif
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_matches_doc() {
        type R = RingBuffer<u64, 8>;
        // C: offsetof(ring_t, write_cursor) == 0, etc.
        assert_eq!(std::mem::offset_of!(R, write_cursor), 0);
        assert_eq!(std::mem::offset_of!(R, slots), 64);
        assert_eq!(std::mem::size_of::<Slot<u64>>(), 64);
        assert_eq!(std::mem::align_of::<R>(), 64);
        assert_eq!(R::SIZE, 64 + 8 * 64);

        // A 100-byte value: align_up(8 + 100, 64) = 128 bytes per slot.
        assert_eq!(std::mem::size_of::<Slot<[u8; 100]>>(), 128);
        assert_eq!(RingBuffer::<[u8; 100], 4>::SIZE, 64 + 4 * 128);
    }

    #[test]
    fn fresh_ring_reports_not_yet_written() {
        let r = RingBuffer::<u64, 4>::new_boxed();
        for pos in 0..8 {
            assert_eq!(r.read_at(pos), Err(ReadError::NotYetWritten));
        }
    }

    #[test]
    fn push_then_read() {
        let r = RingBuffer::<u64, 4>::new_boxed();
        for i in 0..4 {
            assert_eq!(r.push(i * 10), i);
        }
        for i in 0..4 {
            assert_eq!(r.read_at(i), Ok(i * 10));
        }
        assert_eq!(r.read_at(4), Err(ReadError::NotYetWritten));
        assert_eq!(r.write_cursor(), 4);
    }

    #[test]
    fn wraparound_evicts() {
        let r = RingBuffer::<u64, 4>::new_boxed();
        for i in 0..6 {
            r.push(i);
        }
        // 6 pushes into 4 slots: positions 0 and 1 were overwritten.
        assert_eq!(r.read_at(0), Err(ReadError::Evicted));
        assert_eq!(r.read_at(1), Err(ReadError::Evicted));
        for i in 2..6 {
            assert_eq!(r.read_at(i), Ok(i));
        }
        assert_eq!(r.read_at(6), Err(ReadError::NotYetWritten));
    }
}
