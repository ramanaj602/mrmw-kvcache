//! Cross-process backing: a `RingBuffer` living in a POSIX shared-memory
//! object. These are the exact libc calls you'd make in C
//! (shm_open, ftruncate, mmap, munmap, shm_unlink), called through the
//! `libc` crate, which is just Rust declarations of the C functions.
//!
//! Usage: exactly ONE process calls `SharedRing::create(name)`. Other
//! processes call `SharedRing::open(name)` only after `create` has returned,
//! because `create` is what initializes the ring.

use crate::RingBuffer;
use std::ffi::CString;
use std::io;

/// A handle to a ring mapped from a named shared-memory object.
///
/// C:
/// ```c
/// typedef struct {
///     ring_t *ring;     /* where mmap put it in THIS process */
///     char   *name;     /* "/my-ring" */
///     bool    owner;    /* true in the process that created it */
/// } shared_ring_t;
/// ```
pub struct SharedRing<T: Copy + Send, const N: usize> {
    ring: *mut RingBuffer<T, N>,
    name: CString, // a NUL-terminated C string, freed automatically
    owner: bool,
}

// Raw pointers aren't shareable across threads by default in Rust; we
// promise it's fine here because the thing they point at (RingBuffer) is
// built for concurrent use.
unsafe impl<T: Copy + Send, const N: usize> Send for SharedRing<T, N> {}
unsafe impl<T: Copy + Send, const N: usize> Sync for SharedRing<T, N> {}

/// Turn "my-ring" into the C string "/my-ring" that shm_open wants.
fn make_shm_name(name: &str) -> io::Result<CString> {
    let mut full: String = String::new();
    if !name.starts_with('/') {
        full.push('/');
    }
    full.push_str(name);
    if full[1..].contains('/') {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "shm name may not contain '/'"));
    }
    match CString::new(full) {
        Ok(c) => Ok(c),
        Err(_) => Err(io::Error::new(io::ErrorKind::InvalidInput, "shm name contains a NUL byte")),
    }
}

// `io::Result<X>` = Result<X, io::Error>: either Ok(X) or Err(an errno-style
// error). `io::Error::last_os_error()` reads `errno` for you.

impl<T: Copy + Send, const N: usize> SharedRing<T, N> {
    /// Create, size, map and initialize a new shared ring.
    /// Fails with AlreadyExists (EEXIST) if the name is taken.
    pub fn create(name: &str) -> io::Result<Self> {
        let cname: CString = make_shm_name(name)?; // `?` = "if error, return it"
        let size: usize = RingBuffer::<T, N>::SIZE;

        unsafe {
            // C: int fd = shm_open(name, O_CREAT | O_EXCL | O_RDWR, 0600);
            let fd = libc::shm_open(
                cname.as_ptr(),
                libc::O_CREAT | libc::O_EXCL | libc::O_RDWR,
                0o600 as libc::mode_t,
            );
            if fd < 0 {
                return Err(io::Error::last_os_error());
            }

            // C: if (ftruncate(fd, size) != 0) { close; unlink; return -1; }
            if libc::ftruncate(fd, size as libc::off_t) != 0 {
                let err = io::Error::last_os_error();
                libc::close(fd);
                libc::shm_unlink(cname.as_ptr());
                return Err(err);
            }

            // C: void *p = mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
            let p = libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            );
            let mmap_err = io::Error::last_os_error();
            libc::close(fd); // the mapping stays valid after close, same as C
            if p == libc::MAP_FAILED {
                libc::shm_unlink(cname.as_ptr());
                return Err(mmap_err);
            }

            // mmap returns page-aligned memory, so align(64) is satisfied.
            let ring = p as *mut RingBuffer<T, N>;
            RingBuffer::init_in_place(ring);

            Ok(SharedRing { ring, name: cname, owner: true })
        }
    }

    /// Map an existing ring created by another process with the same T and N.
    /// Refuses (InvalidData) if the object's size doesn't match
    /// `RingBuffer::<T, N>::SIZE` -- catches most T/N mismatches.
    pub fn open(name: &str) -> io::Result<Self> {
        let cname: CString = make_shm_name(name)?;
        let size: usize = RingBuffer::<T, N>::SIZE;

        unsafe {
            // C: int fd = shm_open(name, O_RDWR, 0);
            let fd = libc::shm_open(cname.as_ptr(), libc::O_RDWR, 0);
            if fd < 0 {
                return Err(io::Error::last_os_error());
            }

            // C: struct stat st; fstat(fd, &st);
            let mut st: libc::stat = std::mem::zeroed();
            if libc::fstat(fd, &mut st) != 0 {
                let err = io::Error::last_os_error();
                libc::close(fd);
                return Err(err);
            }
            // macOS rounds shm objects up to a whole page, so accept anything
            // in [size, size rounded up to a page].
            let page: usize = libc::sysconf(libc::_SC_PAGESIZE) as usize;
            let rounded: usize = size.div_ceil(page) * page;
            let actual: usize = st.st_size as usize;
            if actual < size || actual > rounded {
                libc::close(fd);
                let msg = format!(
                    "shm object is {actual} bytes, expected {size} for RingBuffer<{}, {N}>",
                    std::any::type_name::<T>()
                );
                return Err(io::Error::new(io::ErrorKind::InvalidData, msg));
            }

            let p = libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            );
            let mmap_err = io::Error::last_os_error();
            libc::close(fd);
            if p == libc::MAP_FAILED {
                return Err(mmap_err);
            }

            Ok(SharedRing { ring: p as *mut RingBuffer<T, N>, name: cname, owner: false })
        }
    }

    /// Remove a leftover name, e.g. after a crash. C: shm_unlink(name);
    pub fn unlink(name: &str) -> io::Result<()> {
        let cname: CString = make_shm_name(name)?;
        let rc = unsafe { libc::shm_unlink(cname.as_ptr()) };
        if rc != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    /// Borrow the ring so you can call `push` / `read_at` on it.
    /// C: `ring_t *ring(shared_ring_t *s) { return s->ring; }`
    pub fn ring(&self) -> &RingBuffer<T, N> {
        // Safe because the mapping lives exactly as long as `self`.
        unsafe { &*self.ring }
    }
}

/// `Drop` is a destructor: it runs automatically when a SharedRing goes out
/// of scope, the way you'd call `shared_ring_close(&s)` by hand in C.
impl<T: Copy + Send, const N: usize> Drop for SharedRing<T, N> {
    fn drop(&mut self) {
        unsafe {
            // C: munmap(s->ring, sizeof(ring_t)); if (s->owner) shm_unlink(s->name);
            libc::munmap(self.ring as *mut libc::c_void, RingBuffer::<T, N>::SIZE);
            if self.owner {
                libc::shm_unlink(self.name.as_ptr());
            }
        }
    }
}
