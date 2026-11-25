//! Control plane for KV-cache handoff (prefill -> decode).
//!
//! The ring in `lib.rs` is a *data plane*: it moves whole values. Real KV
//! blocks are big (megabytes per request per layer) and usually live in GPU
//! memory, so copying them through a ring is the wrong shape. This module
//! splits the job in two:
//!
//! ```text
//!   prefill worker                                     decode worker
//!   --------------                                     -------------
//!   1. write bytes into a pool block ----+        +--> 4. index.lookup(prefix_hash) -> pos
//!   2. descs.push(KvDesc{block, gen..}) -+--> ring +-> 5. descs.read_at(pos)  -> KvDesc
//!   3. index.insert(prefix_hash, pos)              +-> 6. pool.read(block, gen) -> bytes
//! ```
//!
//!   * `RingBuffer<KvDesc, N>`  -- the control plane. Small (40-byte)
//!     descriptors, published and read with the existing lock-free protocol.
//!   * `BlockPool`              -- the data plane. Fixed-size blocks, each
//!     with a generation counter so a reader can tell if the block it was
//!     pointed at has since been reused for something else.
//!   * `PrefixIndex`            -- prefix_hash -> latest descriptor position,
//!     so a decode worker can find KV for a prompt prefix it has seen before.
//!
//! Everything here is plain data + atomics (no pointers), so the whole
//! `KvControlPlane` can live in shared memory exactly like `RingBuffer`.
//!
//! In a real deployment the `BlockPool` is the part you swap out: `device`
//! and `block` in `KvDesc` would name a GPU/pinned-memory allocation or an
//! RDMA region instead of bytes in this struct. The descriptor ring, the
//! generation check and the index stay the same.
//!
//! Correctness rule for the whole module: every race degrades to a *miss*
//! (recompute the KV) and never to *wrong data*. Each step re-validates the
//! previous one: the descriptor must carry the prefix hash we asked for, and
//! the block must still carry the generation the descriptor recorded.

use crate::{ReadError, RingBuffer, SharedLayout};
use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::sync::atomic::{fence, AtomicU64, Ordering};

/// How many index entries `insert` / `lookup` probe before giving up.
pub const MAX_PROBE: usize = 16;

// ---------------------------------------------------------------------------
// Descriptor
// ---------------------------------------------------------------------------

/// Metadata a prefill worker supplies when it publishes a KV block.
///
/// C:
/// ```c
/// typedef struct {
///     uint64_t request_id, prefix_hash;
///     uint16_t layer_start, layer_end, device, dtype_layout;
/// } kv_meta_t;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvMeta {
    pub request_id: u64,
    /// Hash of the token prefix this KV covers. Decode workers look KV up by it.
    pub prefix_hash: u64,
    /// Layers covered: `layer_start..layer_end` (end exclusive).
    pub layer_start: u16,
    pub layer_end: u16,
    /// Which device / memory tier holds the block (0 = this host's pool).
    pub device: u16,
    /// Caller-defined encoding of dtype + layout (e.g. bf16 head-major).
    pub dtype_layout: u16,
}

/// One entry in the control-plane ring: where a KV block is and which
/// generation of that block it refers to. 40 bytes, so one descriptor fits
/// in one 64-byte ring slot together with its tag.
///
/// C:
/// ```c
/// typedef struct {
///     uint64_t request_id, prefix_hash;
///     uint64_t generation;      /* block generation when published */
///     uint32_t block, len;      /* pool block index, bytes used */
///     uint16_t layer_start, layer_end, device, dtype_layout;
/// } kv_desc_t;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct KvDesc {
    pub request_id: u64,
    pub prefix_hash: u64,
    pub generation: u64,
    pub block: u32,
    pub len: u32,
    pub layer_start: u16,
    pub layer_end: u16,
    pub device: u16,
    pub dtype_layout: u16,
}

// ---------------------------------------------------------------------------
// Block pool (the data plane)
// ---------------------------------------------------------------------------

/// One pool block. `gen` works like a slot tag in the ring:
///   gen even  -> stable; the block's bytes belong to generation `gen`
///   gen odd   -> a writer is filling it right now
///   gen == 0  -> never written (descriptors always carry gen >= 2)
///
/// C:
/// ```c
/// typedef struct __attribute__((aligned(64))) {
///     _Atomic uint64_t gen;
///     uint8_t          data[BLOCK_BYTES];
/// } block_t;
/// ```
#[repr(C, align(64))]
pub struct Block<const BLOCK_BYTES: usize> {
    gen: AtomicU64,
    data: UnsafeCell<[MaybeUninit<u8>; BLOCK_BYTES]>,
}

// Same promise as `Slot`: the `gen` protocol is what makes sharing safe.
unsafe impl<const BLOCK_BYTES: usize> Sync for Block<BLOCK_BYTES> {}

/// Why reading a block failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FetchError {
    /// The block was reused (or is being rewritten) since the descriptor was
    /// published. The KV is gone; recompute it.
    Stale,
    /// The descriptor points outside the pool or claims more bytes than a
    /// block holds. Only possible with a corrupt or mismatched descriptor.
    Invalid,
    /// `out` is shorter than `desc.len`.
    BufferTooSmall,
}

/// A fixed pool of `BLOCKS` blocks of `BLOCK_BYTES` each, handed out
/// round-robin. Round-robin means the oldest block is reused first, the same
/// eviction order as the ring; size the pool at least as big as the
/// descriptor ring so blocks don't go stale before their descriptors do.
///
/// C:
/// ```c
/// typedef struct {
///     _Atomic uint64_t alloc_cursor;
///     char             _pad0[56];
///     block_t          blocks[BLOCKS];
/// } block_pool_t;
/// ```
#[repr(C)]
pub struct BlockPool<const BLOCKS: usize, const BLOCK_BYTES: usize> {
    alloc_cursor: AtomicU64,
    _pad0: [u8; 56],
    blocks: [Block<BLOCK_BYTES>; BLOCKS],
}

impl<const BLOCKS: usize, const BLOCK_BYTES: usize> BlockPool<BLOCKS, BLOCK_BYTES> {
    pub const MASK: u64 = {
        assert!(
            BLOCKS.is_power_of_two(),
            "BlockPool BLOCKS must be a power of two"
        );
        assert!(
            BLOCKS <= u32::MAX as usize,
            "block index must fit in KvDesc::block (u32)"
        );
        assert!(
            BLOCK_BYTES <= u32::MAX as usize,
            "block size must fit in KvDesc::len (u32)"
        );
        BLOCKS as u64 - 1
    };

    /// # Safety
    /// Same contract as `RingBuffer::init_in_place`.
    pub unsafe fn init_in_place(p: *mut Self) {
        let _ = Self::MASK;
        unsafe {
            std::ptr::addr_of_mut!((*p).alloc_cursor).write(AtomicU64::new(0));
            std::ptr::addr_of_mut!((*p)._pad0).write([0u8; 56]);
            let blocks = std::ptr::addr_of_mut!((*p).blocks) as *mut Block<BLOCK_BYTES>;
            let mut i: usize = 0;
            while i < BLOCKS {
                std::ptr::addr_of_mut!((*blocks.add(i)).gen).write(AtomicU64::new(0));
                i += 1;
            }
        }
    }

    /// Take the next block round-robin, fill it with `data`, and return
    /// `(block index, generation)`. The caller records both in a `KvDesc`.
    ///
    /// C:
    /// ```c
    /// block_t *b = &pool->blocks[atomic_fetch_add(&pool->alloc_cursor, 1) & MASK];
    /// for (;;) {                                  /* claim: even g -> g+1 */
    ///     uint64_t g = atomic_load_explicit(&b->gen, memory_order_acquire);
    ///     if (g & 1) { pause(); continue; }
    ///     if (atomic_compare_exchange_weak(&b->gen, &g, g + 1)) break;
    /// }
    /// atomic_thread_fence(memory_order_release);
    /// memcpy(b->data, data, len);
    /// atomic_store_explicit(&b->gen, g + 2, memory_order_release);
    /// ```
    ///
    /// Precondition (checked by the caller, `KvControlPlane::publish`):
    /// `data.len() <= BLOCK_BYTES`.
    fn write(&self, data: &[u8]) -> (u32, u64) {
        debug_assert!(data.len() <= BLOCK_BYTES);
        let idx: u64 = self.alloc_cursor.fetch_add(1, Ordering::Relaxed) & Self::MASK;
        let block: &Block<BLOCK_BYTES> = &self.blocks[idx as usize];

        // Claim the block: even -> odd with one CAS. Unlike ring slots there
        // is no "older/newer" check: whoever claims last simply wins, and the
        // generation bump tells any older descriptor its data is gone.
        let mut g: u64;
        loop {
            g = block.gen.load(Ordering::Acquire);
            if g & 1 == 1 {
                std::hint::spin_loop(); // another writer is filling it
                continue;
            }
            if block
                .gen
                .compare_exchange_weak(g, g + 1, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }

        // Keep the bytes below from becoming visible before the odd gen.
        fence(Ordering::Release);

        // C: memcpy(b->data, data, len);
        unsafe {
            let dst = (*block.data.get()).as_mut_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        }

        let new_gen: u64 = g + 2;
        block.gen.store(new_gen, Ordering::Release); // stable again
        (idx as u32, new_gen)
    }

    /// Copy block `desc.block` into `out` if it still holds generation
    /// `desc.generation`. Returns the number of bytes copied.
    ///
    /// Same optimistic pattern as `RingBuffer::read_at`: check, copy, fence,
    /// re-check. A generation that moved is permanent (the block was reused),
    /// so there is no retry loop: that's `Stale`, not `Contended`.
    fn read(&self, desc: &KvDesc, out: &mut [u8]) -> Result<usize, FetchError> {
        let len: usize = desc.len as usize;
        if desc.block as usize >= BLOCKS || len > BLOCK_BYTES {
            return Err(FetchError::Invalid);
        }
        if out.len() < len {
            return Err(FetchError::BufferTooSmall);
        }
        let block: &Block<BLOCK_BYTES> = &self.blocks[desc.block as usize];

        if block.gen.load(Ordering::Acquire) != desc.generation {
            return Err(FetchError::Stale);
        }
        // Optimistic copy. C: memcpy(out, b->data, len);
        unsafe {
            let src = (*block.data.get()).as_ptr() as *const u8;
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), len);
        }
        fence(Ordering::Acquire); // keep the copy above the re-check
        if block.gen.load(Ordering::Acquire) != desc.generation {
            return Err(FetchError::Stale); // rewritten mid-copy: discard
        }
        Ok(len)
    }
}

// ---------------------------------------------------------------------------
// Prefix index
// ---------------------------------------------------------------------------

/// One index entry: 16 bytes, four per cache line.
/// `key == 0` means empty; `pos_plus_one == 0` means "no position yet".
#[repr(C)]
struct IndexEntry {
    key: AtomicU64,
    pos_plus_one: AtomicU64,
}

/// A fixed-size, lock-free hash table from prefix hash to the newest
/// descriptor position published for it. Open addressing, linear probing,
/// no deletes.
///
/// It is deliberately *lossy*: when all `MAX_PROBE` entries for a hash are
/// taken, `insert` overwrites the home entry. Readers never trust the index
/// alone -- `KvControlPlane::lookup` checks that the descriptor it finds
/// carries the requested hash -- so a lost or mismatched entry costs a cache
/// miss, never wrong KV.
///
/// C:
/// ```c
/// typedef struct { _Atomic uint64_t key, pos_plus_one; } index_entry_t;
/// typedef struct { index_entry_t entries[M]; } prefix_index_t;
/// ```
#[repr(C)]
pub struct PrefixIndex<const M: usize> {
    entries: [IndexEntry; M],
}

impl<const M: usize> PrefixIndex<M> {
    pub const MASK: u64 = {
        assert!(M.is_power_of_two(), "PrefixIndex M must be a power of two");
        M as u64 - 1
    };

    /// # Safety
    /// Same contract as `RingBuffer::init_in_place`.
    pub unsafe fn init_in_place(p: *mut Self) {
        let _ = Self::MASK;
        unsafe {
            let e = std::ptr::addr_of_mut!((*p).entries) as *mut IndexEntry;
            let mut i: usize = 0;
            while i < M {
                std::ptr::addr_of_mut!((*e.add(i)).key).write(AtomicU64::new(0));
                std::ptr::addr_of_mut!((*e.add(i)).pos_plus_one).write(AtomicU64::new(0));
                i += 1;
            }
        }
    }

    /// 0 marks an empty entry, so a real hash of 0 is stored as this instead.
    /// (A collision with a real hash equal to this value is harmless: the
    /// descriptor check in `lookup` rejects it.)
    const ZERO_KEY: u64 = 0x9E37_79B9_7F4A_7C15;

    fn key_of(hash: u64) -> u64 {
        if hash == 0 {
            Self::ZERO_KEY
        } else {
            hash
        }
    }

    /// Home entry for a key. Mixes the bits first (splitmix64 finalizer) so
    /// hashes that differ only in high bits don't all land on one entry.
    fn home(key: u64) -> u64 {
        let mut z: u64 = key;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (z ^ (z >> 31)) & Self::MASK
    }

    /// Record that `pos` is the newest descriptor for `hash`.
    ///
    /// C:
    /// ```c
    /// for (int i = 0; i < MAX_PROBE; i++) {
    ///     index_entry_t *e = &idx->entries[(home + i) & MASK];
    ///     uint64_t k = atomic_load(&e->key);
    ///     if (k == 0 && atomic_compare_exchange(&e->key, &k, key)) k = key;
    ///     if (k == key) { atomic_fetch_max(&e->pos_plus_one, pos + 1); return; }
    /// }
    /// /* table neighborhood full: overwrite the home entry (lossy) */
    /// ```
    fn insert(&self, hash: u64, pos: u64) {
        let key: u64 = Self::key_of(hash);
        let home: u64 = Self::home(key);
        let mut i: usize = 0;
        while i < MAX_PROBE.min(M) {
            let e: &IndexEntry = &self.entries[((home + i as u64) & Self::MASK) as usize];
            let mut k: u64 = e.key.load(Ordering::Acquire);
            if k == 0 {
                // Try to take the empty entry. If someone beat us to it, `Err`
                // carries the key they wrote -- maybe the same key as ours.
                k = match e
                    .key
                    .compare_exchange(0, key, Ordering::AcqRel, Ordering::Acquire)
                {
                    Ok(_) => key,
                    Err(actual) => actual,
                };
            }
            if k == key {
                // Newest position wins, even if writers finish out of order.
                e.pos_plus_one.fetch_max(pos + 1, Ordering::Release);
                return;
            }
            i += 1;
        }
        // Every probed entry belongs to another key. Evict the home entry.
        // Two plain stores, not one atomic update: a reader that sees the new
        // key with the old position will fail the descriptor check -> miss.
        let e: &IndexEntry = &self.entries[home as usize];
        e.pos_plus_one.store(0, Ordering::Release);
        e.key.store(key, Ordering::Release);
        e.pos_plus_one.store(pos + 1, Ordering::Release);
    }

    /// Newest descriptor position recorded for `hash`, if any.
    fn lookup(&self, hash: u64) -> Option<u64> {
        let key: u64 = Self::key_of(hash);
        let home: u64 = Self::home(key);
        let mut i: usize = 0;
        while i < MAX_PROBE.min(M) {
            let e: &IndexEntry = &self.entries[((home + i as u64) & Self::MASK) as usize];
            let k: u64 = e.key.load(Ordering::Acquire);
            if k == 0 {
                return None; // probe chain ends at the first empty entry
            }
            if k == key {
                let p: u64 = e.pos_plus_one.load(Ordering::Acquire);
                return if p == 0 { None } else { Some(p - 1) };
            }
            i += 1;
        }
        None
    }
}

// ---------------------------------------------------------------------------
// The control plane
// ---------------------------------------------------------------------------

/// Why `publish` failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PublishError {
    /// `data` is larger than one pool block (`BLOCK_BYTES`).
    TooLarge,
}

/// Why `lookup` found nothing usable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LookupError {
    /// No descriptor is indexed for this prefix (never published, or its
    /// index entry was overwritten, or a racing update left a mismatch).
    Miss,
    /// The index pointed at a descriptor the ring has since overwritten.
    Evicted,
    /// The descriptor's slot kept changing while we read it.
    Contended,
}

/// Why `get` (lookup + fetch in one call) failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GetError {
    Lookup(LookupError),
    Fetch(FetchError),
}

/// Descriptor ring + prefix index + block pool, in one shareable struct.
///
/// * `N`: descriptor ring slots (power of two)
/// * `M`: prefix index entries (power of two; about 2x the number of live
///   prefixes you expect keeps probe chains short)
/// * `BLOCKS`: pool blocks (power of two; use `BLOCKS >= N`)
/// * `BLOCK_BYTES`: bytes per block
///
/// C:
/// ```c
/// typedef struct {
///     ring_t         descs;   /* RingBuffer<KvDesc, N> */
///     prefix_index_t index;
///     block_pool_t   pool;
/// } kv_control_plane_t;
/// ```
#[repr(C)]
pub struct KvControlPlane<
    const N: usize,
    const M: usize,
    const BLOCKS: usize,
    const BLOCK_BYTES: usize,
> {
    descs: RingBuffer<KvDesc, N>,
    index: PrefixIndex<M>,
    pool: BlockPool<BLOCKS, BLOCK_BYTES>,
}

impl<const N: usize, const M: usize, const BLOCKS: usize, const BLOCK_BYTES: usize>
    KvControlPlane<N, M, BLOCKS, BLOCK_BYTES>
{
    /// Total size in bytes; what `ftruncate` / `mmap` need.
    pub const SIZE: usize = std::mem::size_of::<Self>();

    /// # Safety
    /// `p` must point to at least `SIZE` writable bytes, aligned to 64, and
    /// nobody else may touch that memory until this returns.
    pub unsafe fn init_in_place(p: *mut Self) {
        unsafe {
            RingBuffer::init_in_place(std::ptr::addr_of_mut!((*p).descs));
            PrefixIndex::init_in_place(std::ptr::addr_of_mut!((*p).index));
            BlockPool::init_in_place(std::ptr::addr_of_mut!((*p).pool));
        }
    }

    /// Heap-allocated plane (no shared memory); used by tests.
    pub fn new_boxed() -> Box<Self> {
        let layout = std::alloc::Layout::new::<Self>();
        unsafe {
            let p = std::alloc::alloc(layout) as *mut Self;
            if p.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            Self::init_in_place(p);
            Box::from_raw(p)
        }
    }

    /// Prefill side: store `data` in a pool block, publish a descriptor for
    /// it, and index it under `meta.prefix_hash`. Returns the descriptor's
    /// ring position.
    ///
    /// The order matters: bytes first, then the descriptor that points at
    /// them, then the index entry that points at the descriptor. A reader
    /// following the chain can therefore never reach a block before it's
    /// written -- and if it reaches one *after* it was reused, the generation
    /// check catches it.
    pub fn publish(&self, meta: KvMeta, data: &[u8]) -> Result<u64, PublishError> {
        if data.len() > BLOCK_BYTES {
            return Err(PublishError::TooLarge);
        }
        let (block, generation) = self.pool.write(data);
        let desc = KvDesc {
            request_id: meta.request_id,
            prefix_hash: meta.prefix_hash,
            generation,
            block,
            len: data.len() as u32,
            layer_start: meta.layer_start,
            layer_end: meta.layer_end,
            device: meta.device,
            dtype_layout: meta.dtype_layout,
        };
        let pos: u64 = self.descs.push(desc);
        self.index.insert(meta.prefix_hash, pos);
        Ok(pos)
    }

    /// Decode side, step 1: find the newest descriptor for a prefix.
    pub fn lookup(&self, prefix_hash: u64) -> Result<KvDesc, LookupError> {
        let pos: u64 = match self.index.lookup(prefix_hash) {
            Some(p) => p,
            None => return Err(LookupError::Miss),
        };
        match self.descs.read_at(pos) {
            // The index can be stale or mismatched; the descriptor is the
            // source of truth for which prefix it describes.
            Ok(desc) if desc.prefix_hash == prefix_hash => Ok(desc),
            Ok(_) => Err(LookupError::Miss),
            Err(ReadError::Evicted) => Err(LookupError::Evicted),
            // The index is only written after the push completes, so the
            // descriptor exists. NotYetWritten here means the slot was claimed
            // by a newer writer that hasn't finished -- i.e. ours is gone.
            Err(ReadError::NotYetWritten) => Err(LookupError::Evicted),
            Err(ReadError::Contended) => Err(LookupError::Contended),
        }
    }

    /// Decode side, step 2: copy the block a descriptor points at.
    pub fn fetch(&self, desc: &KvDesc, out: &mut [u8]) -> Result<usize, FetchError> {
        self.pool.read(desc, out)
    }

    /// `lookup` then `fetch`. On success returns the descriptor and the
    /// number of bytes written to `out`.
    pub fn get(&self, prefix_hash: u64, out: &mut [u8]) -> Result<(KvDesc, usize), GetError> {
        let desc = self.lookup(prefix_hash).map_err(GetError::Lookup)?;
        let n = self.fetch(&desc, out).map_err(GetError::Fetch)?;
        Ok((desc, n))
    }

    /// Read a descriptor by ring position (e.g. one handed over out of band).
    pub fn descriptor_at(&self, pos: u64) -> Result<KvDesc, ReadError> {
        self.descs.read_at(pos)
    }
}

unsafe impl<const N: usize, const M: usize, const BLOCKS: usize, const BLOCK_BYTES: usize>
    SharedLayout for KvControlPlane<N, M, BLOCKS, BLOCK_BYTES>
{
    const SIZE: usize = std::mem::size_of::<Self>();
    unsafe fn init_in_place(p: *mut Self) {
        unsafe { Self::init_in_place(p) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Plane = KvControlPlane<8, 16, 8, 64>;

    fn meta(prefix_hash: u64) -> KvMeta {
        KvMeta {
            request_id: 7,
            prefix_hash,
            layer_start: 0,
            layer_end: 32,
            device: 0,
            dtype_layout: 1,
        }
    }

    #[test]
    fn desc_fits_one_ring_slot() {
        assert_eq!(std::mem::size_of::<KvDesc>(), 40);
        assert_eq!(std::mem::size_of::<crate::Slot<KvDesc>>(), 64);
        assert_eq!(std::mem::size_of::<Block<64>>(), 128); // 8-byte gen + 64 bytes, rounded to 64
    }

    #[test]
    fn publish_lookup_fetch_roundtrip() {
        let p = Plane::new_boxed();
        let pos = p.publish(meta(42), b"hello kv").unwrap();
        let d = p.lookup(42).unwrap();
        assert_eq!(p.descriptor_at(pos), Ok(d));
        assert_eq!(
            (d.prefix_hash, d.len, d.layer_end, d.request_id),
            (42, 8, 32, 7)
        );
        assert_eq!(d.generation % 2, 0);
        let mut out = [0u8; 64];
        assert_eq!(p.fetch(&d, &mut out), Ok(8));
        assert_eq!(&out[..8], b"hello kv");
    }

    #[test]
    fn unknown_prefix_misses() {
        let p = Plane::new_boxed();
        assert_eq!(p.lookup(1), Err(LookupError::Miss));
        p.publish(meta(2), b"x").unwrap();
        assert_eq!(p.lookup(1), Err(LookupError::Miss));
    }

    #[test]
    fn hash_zero_is_a_valid_prefix() {
        let p = Plane::new_boxed();
        p.publish(meta(0), b"zero").unwrap();
        let mut out = [0u8; 64];
        assert_eq!(p.get(0, &mut out).unwrap().1, 4);
    }

    #[test]
    fn newest_publish_wins() {
        let p = Plane::new_boxed();
        p.publish(meta(5), b"old").unwrap();
        p.publish(meta(5), b"new!").unwrap();
        let mut out = [0u8; 64];
        let (_, n) = p.get(5, &mut out).unwrap();
        assert_eq!(&out[..n], b"new!");
    }

    #[test]
    fn reused_block_is_stale() {
        let p = Plane::new_boxed();
        p.publish(meta(1), b"first").unwrap();
        let old = p.lookup(1).unwrap();
        // 8 more publishes wrap the 8-block pool, reusing old.block.
        for h in 100..108 {
            p.publish(meta(h), b"filler").unwrap();
        }
        let mut out = [0u8; 64];
        assert_eq!(p.fetch(&old, &mut out), Err(FetchError::Stale));
    }

    #[test]
    fn evicted_descriptor_is_reported() {
        let p = Plane::new_boxed();
        p.publish(meta(1), b"a").unwrap();
        for h in 100..108 {
            p.publish(meta(h), b"b").unwrap(); // wraps the 8-slot ring
        }
        assert_eq!(p.lookup(1), Err(LookupError::Evicted));
    }

    #[test]
    fn oversized_and_invalid_inputs() {
        let p = Plane::new_boxed();
        assert_eq!(p.publish(meta(1), &[0u8; 65]), Err(PublishError::TooLarge));
        p.publish(meta(1), &[9u8; 64]).unwrap();
        let d = p.lookup(1).unwrap();
        let mut small = [0u8; 10];
        assert_eq!(p.fetch(&d, &mut small), Err(FetchError::BufferTooSmall));
        let bad = KvDesc { block: 99, ..d };
        assert_eq!(p.fetch(&bad, &mut [0u8; 64]), Err(FetchError::Invalid));
    }

    #[test]
    fn full_index_neighborhood_is_lossy_not_wrong() {
        // Index of 16 entries, MAX_PROBE 16: publishing many distinct prefixes
        // forces overwrites. Every successful get must return the bytes that
        // were published for exactly that prefix.
        let p = KvControlPlane::<64, 16, 64, 16>::new_boxed();
        for h in 1..=40u64 {
            p.publish(meta(h), &h.to_le_bytes()).unwrap();
        }
        let mut hits = 0;
        for h in 1..=40u64 {
            let mut out = [0u8; 16];
            if let Ok((d, n)) = p.get(h, &mut out) {
                assert_eq!(d.prefix_hash, h);
                assert_eq!(&out[..n], &h.to_le_bytes());
                hits += 1;
            }
        }
        assert!(
            hits >= 16,
            "expected at least a full table of hits, got {hits}"
        );
    }
}
