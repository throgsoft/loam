use std::sync::atomic::{AtomicUsize, Ordering};

use loam_time::par::{self, ChunkSource, Executor, Sequential};

static DRAINS: AtomicUsize = AtomicUsize::new(0);

struct Counting;

impl Executor for Counting {
    fn parallelism(&self) -> usize {
        3
    }

    fn for_each_chunk(&self, _workers: usize, chunks: &(dyn ChunkSource + Sync)) {
        DRAINS.fetch_add(1, Ordering::Relaxed);
        while chunks.run_next() {}
    }
}

#[test]
fn sequential_and_handed_in() {
    let mut rows = [1u32, 2, 3, 4];
    par::for_each_chunk(&Sequential, &mut rows, 1, |part| part[0] += 10);
    assert_eq!(DRAINS.load(Ordering::Relaxed), 0);

    par::for_each_chunk(&Counting, &mut rows, 1, |part| part[0] += 10);
    assert_eq!(DRAINS.load(Ordering::Relaxed), 1);
    assert_eq!(rows, [21, 22, 23, 24]);
}
