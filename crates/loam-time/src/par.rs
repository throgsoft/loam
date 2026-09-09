use std::sync::{Mutex, OnceLock};

use thiserror::Error;

pub trait ChunkSource {
    fn run_next(&self) -> bool;
}

pub trait Executor: Send + Sync {
    fn parallelism(&self) -> usize;

    fn for_each_chunk(&self, workers: usize, chunks: &(dyn ChunkSource + Sync));

    fn join(&self, a: &mut (dyn FnMut() + Send), b: &mut (dyn FnMut() + Send));
}

pub struct Sequential;

impl Executor for Sequential {
    fn parallelism(&self) -> usize {
        1
    }

    fn for_each_chunk(&self, _workers: usize, chunks: &(dyn ChunkSource + Sync)) {
        while chunks.run_next() {}
    }

    fn join(&self, a: &mut (dyn FnMut() + Send), b: &mut (dyn FnMut() + Send)) {
        a();
        b();
    }
}

#[derive(Debug, Error)]
#[error("a parallel executor is already installed")]
pub struct AlreadyInstalled;

static INSTALLED: OnceLock<Box<dyn Executor>> = OnceLock::new();

static SEQUENTIAL: Sequential = Sequential;

pub fn install(executor: Box<dyn Executor>) -> Result<(), AlreadyInstalled> {
    INSTALLED.set(executor).map_err(|_| AlreadyInstalled)
}

pub fn executor() -> &'static dyn Executor {
    match INSTALLED.get() {
        Some(installed) => installed.as_ref(),
        None => &SEQUENTIAL,
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

pub fn for_each_chunk<T: Send>(data: &mut [T], chunk: usize, task: impl Fn(&mut [T]) + Sync) {
    let chunk = chunk.max(1);
    let executor = executor();
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_join_runs_both_sides() {
        let (mut left, mut right) = (0u32, 0u32);
        let mut a = || left = 1;
        let mut b = || right = 2;
        executor().join(&mut a, &mut b);
        assert_eq!((left, right), (1, 2));
    }
}
