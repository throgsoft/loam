use std::sync::atomic::{AtomicUsize, Ordering};

use loam_time::par::{self, ChunkSource, Executor};

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

    fn join(&self, a: &mut (dyn FnMut() + Send), b: &mut (dyn FnMut() + Send)) {
        a();
        b();
    }
}

#[test]
fn an_empty_slot_stays_sequential_and_a_second_install_is_refused() {
    assert_eq!(par::executor().parallelism(), 1);
    let mut rows = [1u32, 2, 3, 4];
    par::for_each_chunk(&mut rows, 1, |part| part[0] += 10);
    assert_eq!(DRAINS.load(Ordering::Relaxed), 0);

    assert!(par::install(Box::new(Counting)).is_ok());
    assert_eq!(par::executor().parallelism(), 3);
    par::for_each_chunk(&mut rows, 1, |part| part[0] += 10);
    assert_eq!(DRAINS.load(Ordering::Relaxed), 1);
    assert_eq!(rows, [21, 22, 23, 24]);

    assert!(par::install(Box::new(Counting)).is_err());
    assert_eq!(par::executor().parallelism(), 3);
}
