use std::alloc::{GlobalAlloc, Layout};
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::LocalKey;

thread_local! {
    static ALLOC_BYTES: Cell<u64> = const { Cell::new(0) };
    static DEALLOC_BYTES: Cell<u64> = const { Cell::new(0) };
    static ALLOC_COUNT: Cell<u64> = const { Cell::new(0) };
    static DEALLOC_COUNT: Cell<u64> = const { Cell::new(0) };
}

static ALLOC_INSTALLED: AtomicBool = AtomicBool::new(false);

fn bump(counter: &'static LocalKey<Cell<u64>>, by: u64) {
    let _ = counter.try_with(|count| count.set(count.get().wrapping_add(by)));
}

/// Counts requested bytes on the calling thread; allocator overhead is excluded.
pub struct CountingAllocator<A: GlobalAlloc> {
    inner: A,
}

impl<A: GlobalAlloc> CountingAllocator<A> {
    pub const fn new(inner: A) -> Self {
        Self { inner }
    }
}

// SAFETY: Methods preserve the inner allocator contract; const TLS and wrapping Cell updates cannot unwind.
unsafe impl<A: GlobalAlloc> GlobalAlloc for CountingAllocator<A> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_INSTALLED.store(true, Ordering::Relaxed);
        bump(&ALLOC_COUNT, 1);
        bump(&ALLOC_BYTES, layout.size() as u64);
        // SAFETY: The caller supplies a valid nonzero allocation layout.
        unsafe { self.inner.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        bump(&DEALLOC_COUNT, 1);
        bump(&DEALLOC_BYTES, layout.size() as u64);
        // SAFETY: The caller supplies a live inner allocation and its original layout.
        unsafe { self.inner.dealloc(ptr, layout) };
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOC_INSTALLED.store(true, Ordering::Relaxed);
        bump(&ALLOC_COUNT, 1);
        bump(&ALLOC_BYTES, layout.size() as u64);
        // SAFETY: The caller supplies a valid nonzero allocation layout.
        unsafe { self.inner.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOC_INSTALLED.store(true, Ordering::Relaxed);
        bump(&DEALLOC_COUNT, 1);
        bump(&DEALLOC_BYTES, layout.size() as u64);
        bump(&ALLOC_COUNT, 1);
        bump(&ALLOC_BYTES, new_size as u64);
        // SAFETY: The caller supplies a live inner allocation, its layout, and a valid new size.
        unsafe { self.inner.realloc(ptr, layout, new_size) }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AllocSnapshot {
    pub alloc_bytes: u64,
    pub dealloc_bytes: u64,
    pub alloc_count: u64,
    pub dealloc_count: u64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AllocDelta {
    pub net_bytes: i64,
    pub alloc_bytes: u64,
    pub alloc_count: u64,
    pub dealloc_count: u64,
}

/// The calling thread's counters; `None` precedes the first allocation anywhere in the process.
pub fn current_snapshot() -> Option<AllocSnapshot> {
    if !ALLOC_INSTALLED.load(Ordering::Relaxed) {
        return None;
    }
    Some(AllocSnapshot {
        alloc_bytes: ALLOC_BYTES.with(Cell::get),
        dealloc_bytes: DEALLOC_BYTES.with(Cell::get),
        alloc_count: ALLOC_COUNT.with(Cell::get),
        dealloc_count: DEALLOC_COUNT.with(Cell::get),
    })
}

pub fn bytes_allocated_by(body: impl FnOnce()) -> Option<u64> {
    if !ALLOC_INSTALLED.load(Ordering::Relaxed) {
        body();
        return None;
    }
    let before = ALLOC_BYTES.with(Cell::get);
    body();
    Some(ALLOC_BYTES.with(Cell::get).wrapping_sub(before))
}

/// `start` must be the earlier snapshot.
pub fn delta(start: AllocSnapshot, end: AllocSnapshot) -> AllocDelta {
    let alloc_bytes = end.alloc_bytes.saturating_sub(start.alloc_bytes);
    let dealloc_bytes = end.dealloc_bytes.saturating_sub(start.dealloc_bytes);
    AllocDelta {
        net_bytes: (alloc_bytes as i64).saturating_sub(dealloc_bytes as i64),
        alloc_bytes,
        alloc_count: end.alloc_count.saturating_sub(start.alloc_count),
        dealloc_count: end.dealloc_count.saturating_sub(start.dealloc_count),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delta_handles_dealloc_dominant() {
        let start = AllocSnapshot {
            alloc_bytes: 1_000,
            dealloc_bytes: 200,
            alloc_count: 10,
            dealloc_count: 3,
        };
        let end = AllocSnapshot {
            alloc_bytes: 1_100,
            dealloc_bytes: 1_000,
            alloc_count: 12,
            dealloc_count: 20,
        };
        let d = delta(start, end);
        assert_eq!(d.alloc_bytes, 100);
        assert_eq!(d.net_bytes, -700);
    }
}
