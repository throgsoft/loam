use std::num::NonZeroUsize;
use std::thread;

use loam_time::par::{ChunkSource, Executor};

pub struct ScopedThreads {
    threads: usize,
}

impl Default for ScopedThreads {
    fn default() -> Self {
        Self {
            threads: thread::available_parallelism().map_or(1, NonZeroUsize::get),
        }
    }
}

impl Executor for ScopedThreads {
    fn parallelism(&self) -> usize {
        self.threads
    }

    fn for_each_chunk(&self, workers: usize, chunks: &(dyn ChunkSource + Sync)) {
        let workers = workers.clamp(1, self.threads);
        thread::scope(|scope| {
            for _ in 1..workers {
                scope.spawn(|| while chunks.run_next() {});
            }
            while chunks.run_next() {}
        });
    }
}

pub fn install() {
    if loam_time::par::install(Box::new(ScopedThreads::default())).is_err() {
        tracing::debug!("a parallel executor was already installed");
    }
}
