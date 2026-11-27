use std::sync::Mutex;

pub trait ChunkSource {
    fn run_next(&self) -> bool;
}

pub trait Executor: Send + Sync {
    fn parallelism(&self) -> usize;

    /// `workers` is a request; each worker drains `chunks` until `run_next` is false.
    fn for_each_chunk(&self, workers: usize, chunks: &(dyn ChunkSource + Sync));
}

pub struct Sequential;

impl Executor for Sequential {
    fn parallelism(&self) -> usize {
        1
    }

    fn for_each_chunk(&self, _workers: usize, chunks: &(dyn ChunkSource + Sync)) {
        while chunks.run_next() {}
    }
}

struct Chunks<'a, T, F> {
    rest: Mutex<&'a mut [T]>,
    chunk: usize,
    task: F,
}

impl<T: Send, F: Fn(&mut [T]) + Sync> ChunkSource for Chunks<'_, T, F> {
    fn run_next(&self) -> bool {
        let part = {
            let Ok(mut rest) = self.rest.lock() else {
                return false;
            };
            if rest.is_empty() {
                return false;
            }
            let take = rest.len().min(self.chunk);
            let (head, tail) = std::mem::take(&mut *rest).split_at_mut(take);
            *rest = tail;
            head
        };
        (self.task)(part);
        true
    }
}

/// Runs `task` on disjoint chunks of `data` in no fixed order.
pub fn for_each_chunk<T: Send>(
    executor: &dyn Executor,
    data: &mut [T],
    chunk: usize,
    task: impl Fn(&mut [T]) + Sync,
) {
    let chunk = chunk.max(1);
    let workers = executor
        .parallelism()
        .min(data.len().div_ceil(chunk))
        .max(1);
    if workers == 1 {
        for part in data.chunks_mut(chunk) {
            task(part);
        }
        return;
    }
    let chunks = Chunks {
        rest: Mutex::new(data),
        chunk,
        task,
    };
    executor.for_each_chunk(workers, &chunks);
}
