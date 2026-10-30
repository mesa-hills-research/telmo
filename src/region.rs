//! Memory that weights live in: a heap allocation aligned to 64 bytes, or a read-only mapping of
//! a file. Views into a region keep it alive through an `Arc`, so a model's weights can point
//! into one mapped pack without copying.

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::fs::File;
use std::ops::Deref;
use std::ptr::NonNull;
use std::sync::Arc;

/// Alignment of every region, and of every entry inside a pack.
pub const ALIGN: usize = 64;

enum Owner {
    Heap(Layout),
    /// Held for its `Drop`, which unmaps the file.
    #[allow(dead_code)]
    Map(map::Mapping),
    Empty,
}

/// A type with the regions' alignment, for the dangling pointer of an empty region.
#[repr(align(64))]
struct Aligned;

/// A block of bytes that never moves.
pub struct Region {
    ptr: NonNull<u8>,
    len: usize,
    owner: Owner,
}

// SAFETY: a region is plain memory, written only through `&mut` before it is shared.
unsafe impl Send for Region {}
unsafe impl Sync for Region {}

impl Region {
    /// A zeroed heap region aligned to 64 bytes.
    pub fn zeroed(len: usize) -> Region {
        if len == 0 {
            return Region {
                ptr: NonNull::<Aligned>::dangling().cast(),
                len: 0,
                owner: Owner::Empty,
            };
        }
        let layout = Layout::from_size_align(len, ALIGN).expect("region too large");
        // SAFETY: the layout has a nonzero size.
        let ptr = unsafe { alloc_zeroed(layout) };
        let ptr = NonNull::new(ptr).unwrap_or_else(|| std::alloc::handle_alloc_error(layout));
        Region {
            ptr,
            len,
            owner: Owner::Heap(layout),
        }
    }

    /// Maps a whole file read-only, or reads it into memory where mapping isn't available.
    /// Returns whether the file was mapped.
    pub fn of_file(file: &mut File, len: usize) -> std::io::Result<(Region, bool)> {
        if len > 0
            && let Some(mapping) = map::Mapping::new(file, len)?
        {
            let ptr = NonNull::new(mapping.ptr()).expect("a mapping is never null");
            return Ok((
                Region {
                    ptr,
                    len,
                    owner: Owner::Map(mapping),
                },
                true,
            ));
        }
        use std::io::{Read, Seek, SeekFrom};
        let mut region = Region::zeroed(len);
        file.seek(SeekFrom::Start(0))?;
        file.read_exact(region.as_mut_slice())?;
        Ok((region, false))
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn is_mapped(&self) -> bool {
        matches!(self.owner, Owner::Map(_))
    }

    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: the pointer covers `len` bytes for the region's life.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// The bytes of a heap region, for filling it before it is shared.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        assert!(!self.is_mapped(), "mapped regions are read-only");
        // SAFETY: as in as_slice, and `&mut self` makes the access unique.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// The bytes of a heap region as floats.
    pub fn as_mut_f32(&mut self) -> &mut [f32] {
        assert!(!self.is_mapped(), "mapped regions are read-only");
        // SAFETY: heap regions are 64-byte aligned, and any bytes are a valid f32.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr() as *mut f32, self.len / 4) }
    }

    /// The whole region as shared bytes.
    pub fn share(self) -> Bytes {
        let len = self.len;
        Bytes {
            region: Arc::new(self),
            start: 0,
            len,
        }
    }
}

impl Drop for Region {
    fn drop(&mut self) {
        if let Owner::Heap(layout) = self.owner {
            // SAFETY: allocated in `zeroed` with this layout.
            unsafe { dealloc(self.ptr.as_ptr(), layout) }
        }
    }
}

/// A range of a shared region.
#[derive(Clone)]
pub struct Bytes {
    region: Arc<Region>,
    start: usize,
    len: usize,
}

impl Bytes {
    pub fn empty() -> Bytes {
        Region::zeroed(0).share()
    }

    /// A sub-range, `start..start + len` of this one.
    pub fn slice(&self, start: usize, len: usize) -> Bytes {
        assert!(start.checked_add(len).is_some_and(|end| end <= self.len));
        Bytes {
            region: self.region.clone(),
            start: self.start + start,
            len,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.as_slice().as_ptr()
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.region.as_slice()[self.start..self.start + self.len]
    }

    /// Whether the bytes come from a mapped file.
    pub fn is_mapped(&self) -> bool {
        self.region.is_mapped()
    }

    /// The bytes as floats. They must start at a multiple of 4 and hold whole floats.
    pub fn as_f32(&self) -> &[f32] {
        let bytes = self.as_slice();
        assert!(
            bytes.as_ptr().align_offset(4) == 0 && bytes.len().is_multiple_of(4),
            "float data must be aligned"
        );
        // SAFETY: aligned and in bounds, checked above, and any bytes are a valid f32.
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4) }
    }
}

/// An f32 vector held in shared bytes (a bias, a norm's weights, an embedding table).
#[derive(Clone)]
pub struct Vector(Bytes);

impl Vector {
    pub fn new(bytes: Bytes) -> Vector {
        let _ = bytes.as_f32();
        Vector(bytes)
    }

    pub fn from_slice(values: &[f32]) -> Vector {
        let mut region = Region::zeroed(values.len() * 4);
        region.as_mut_f32().copy_from_slice(values);
        Vector(region.share())
    }

    pub fn bytes(&self) -> &Bytes {
        &self.0
    }
}

impl Deref for Vector {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        self.0.as_f32()
    }
}

impl From<Vec<f32>> for Vector {
    fn from(v: Vec<f32>) -> Vector {
        Vector::from_slice(&v)
    }
}

impl std::fmt::Debug for Vector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Vector({} floats)", self.len())
    }
}

/// Read-only file mappings. The two system calls are declared here, from the C library on
/// Unix and kernel32 on Windows, which the standard library links already, so the crate needs no
/// dependencies.
mod map {
    use std::fs::File;

    pub struct Mapping {
        ptr: *mut u8,
        len: usize,
    }

    impl Mapping {
        pub fn ptr(&self) -> *mut u8 {
            self.ptr
        }
    }

    #[cfg(all(unix, target_pointer_width = "64"))]
    mod sys {
        use std::ffi::{c_int, c_void};
        use std::fs::File;
        use std::os::fd::AsRawFd;

        const PROT_READ: c_int = 1;
        const MAP_PRIVATE: c_int = 2;
        #[cfg(any(
            target_os = "linux",
            target_os = "android",
            target_os = "macos",
            target_os = "ios",
            target_os = "freebsd"
        ))]
        const MADV_WILLNEED: c_int = 3;

        unsafe extern "C" {
            fn mmap(
                addr: *mut c_void,
                len: usize,
                prot: c_int,
                flags: c_int,
                fd: c_int,
                offset: i64,
            ) -> *mut c_void;
            fn munmap(addr: *mut c_void, len: usize) -> c_int;
            #[cfg(any(
                target_os = "linux",
                target_os = "android",
                target_os = "macos",
                target_os = "ios",
                target_os = "freebsd"
            ))]
            fn madvise(addr: *mut c_void, len: usize, advice: c_int) -> c_int;
        }

        pub fn map(file: &File, len: usize) -> std::io::Result<Option<*mut u8>> {
            // SAFETY: a fresh read-only private mapping of a file we hold open. The kernel
            // checks the descriptor and the length.
            let ptr = unsafe {
                mmap(
                    std::ptr::null_mut(),
                    len,
                    PROT_READ,
                    MAP_PRIVATE,
                    file.as_raw_fd(),
                    0,
                )
            };
            if ptr as isize == -1 {
                return Err(std::io::Error::last_os_error());
            }
            // Ask the system to start reading the file in, which helps a cold load. It is only
            // advice, so a failure doesn't matter.
            #[cfg(any(
                target_os = "linux",
                target_os = "android",
                target_os = "macos",
                target_os = "ios",
                target_os = "freebsd"
            ))]
            // SAFETY: the range is the mapping just made.
            unsafe {
                madvise(ptr, len, MADV_WILLNEED);
            }
            Ok(Some(ptr as *mut u8))
        }

        pub fn unmap(ptr: *mut u8, len: usize) {
            // SAFETY: the mapping was made by `map` with this length and is dropped once.
            unsafe {
                munmap(ptr as *mut c_void, len);
            }
        }
    }

    #[cfg(windows)]
    mod sys {
        use std::ffi::c_void;
        use std::fs::File;
        use std::os::windows::io::AsRawHandle;

        const PAGE_READONLY: u32 = 0x02;
        const FILE_MAP_READ: u32 = 0x04;

        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn CreateFileMappingW(
                file: *mut c_void,
                attributes: *const c_void,
                protect: u32,
                max_size_high: u32,
                max_size_low: u32,
                name: *const u16,
            ) -> *mut c_void;
            fn MapViewOfFile(
                mapping: *mut c_void,
                access: u32,
                offset_high: u32,
                offset_low: u32,
                bytes: usize,
            ) -> *mut c_void;
            fn UnmapViewOfFile(base: *const c_void) -> i32;
            fn CloseHandle(handle: *mut c_void) -> i32;
        }

        pub fn map(file: &File, len: usize) -> std::io::Result<Option<*mut u8>> {
            // SAFETY: plain calls on a handle we hold open. The view keeps the mapping object
            // alive after its handle is closed.
            unsafe {
                let mapping = CreateFileMappingW(
                    file.as_raw_handle(),
                    std::ptr::null(),
                    PAGE_READONLY,
                    0,
                    0,
                    std::ptr::null(),
                );
                if mapping.is_null() {
                    return Err(std::io::Error::last_os_error());
                }
                let view = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, len);
                let error = std::io::Error::last_os_error();
                CloseHandle(mapping);
                if view.is_null() {
                    return Err(error);
                }
                Ok(Some(view as *mut u8))
            }
        }

        pub fn unmap(ptr: *mut u8, _len: usize) {
            // SAFETY: the view was made by `map` and is dropped once.
            unsafe {
                UnmapViewOfFile(ptr as *const c_void);
            }
        }
    }

    #[cfg(not(any(all(unix, target_pointer_width = "64"), windows)))]
    mod sys {
        use std::fs::File;

        pub fn map(_file: &File, _len: usize) -> std::io::Result<Option<*mut u8>> {
            Ok(None)
        }

        pub fn unmap(_ptr: *mut u8, _len: usize) {}
    }

    impl Mapping {
        /// Maps `len` bytes of a file read-only, or `None` where this system has no mapping.
        pub fn new(file: &File, len: usize) -> std::io::Result<Option<Mapping>> {
            Ok(sys::map(file, len)?.map(|ptr| Mapping { ptr, len }))
        }
    }

    impl Drop for Mapping {
        fn drop(&mut self) {
            sys::unmap(self.ptr, self.len);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn a_mapped_file_reads_the_same_as_its_bytes() {
        let dir = std::env::temp_dir().join(format!("notate-region-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("data.bin");
        let data: Vec<u8> = (0..10_000u32).map(|i| (i * 31 % 251) as u8).collect();
        std::fs::File::create(&path)
            .unwrap()
            .write_all(&data)
            .unwrap();
        let mut file = std::fs::File::open(&path).unwrap();
        let (region, mapped) = Region::of_file(&mut file, data.len()).unwrap();
        assert_eq!(
            mapped,
            cfg!(any(all(unix, target_pointer_width = "64"), windows))
        );
        let bytes = region.share();
        assert_eq!(bytes.as_slice(), &data[..]);
        assert_eq!(bytes.slice(100, 20).as_slice(), &data[100..120]);
        drop(file);
        drop(bytes);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn heap_regions_are_aligned_and_zeroed() {
        let mut region = Region::zeroed(1000);
        assert_eq!(region.as_slice().as_ptr() as usize % ALIGN, 0);
        assert!(region.as_slice().iter().all(|&b| b == 0));
        region.as_mut_f32()[3] = 2.5;
        let v = Vector::new(region.share().slice(0, 16));
        assert_eq!(&v[..], &[0.0, 0.0, 0.0, 2.5]);
    }
}
