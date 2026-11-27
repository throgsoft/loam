//! Frame histories belong to the recording thread.

#[cfg(feature = "frame-trace")]
use std::cell::RefCell;
#[cfg(feature = "frame-trace")]
use std::collections::VecDeque;
use std::time::Duration;
#[cfg(feature = "frame-trace")]
use web_time::Instant;

#[derive(Clone, Debug)]
pub struct Section {
    pub name: &'static str,
    pub elapsed: Duration,
}

#[derive(Clone, Debug, Default)]
pub struct FrameTrace {
    pub sections: Vec<Section>,
}

impl FrameTrace {
    /// Sum of section durations, including nested scopes and synthetic sections.
    pub fn total(&self) -> Duration {
        self.sections.iter().map(|s| s.elapsed).sum()
    }
}

pub const DEFAULT_CAPACITY: usize = 120;

#[cfg(feature = "frame-trace")]
struct Tracer {
    history: VecDeque<FrameTrace>,
    capacity: usize,
}

#[cfg(feature = "frame-trace")]
impl Tracer {
    fn new(capacity: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(capacity),
            capacity,
        }
    }
}

#[cfg(feature = "frame-trace")]
thread_local! {
    static TRACER: RefCell<Tracer> = RefCell::new(Tracer::new(DEFAULT_CAPACITY));
    static CURRENT_SECTIONS: RefCell<Vec<Section>> = const { RefCell::new(Vec::new()) };
    static LAST_FRAME_END: std::cell::Cell<Option<Instant>> = const { std::cell::Cell::new(None) };
    static CURRENT_FRAME_START: std::cell::Cell<Option<Instant>> = const { std::cell::Cell::new(None) };
    static CURRENT_FRAME_ALLOC_START: std::cell::Cell<Option<crate::alloc::AllocSnapshot>> = const { std::cell::Cell::new(None) };
    static SPIKE_THRESHOLD: std::cell::Cell<Duration> =
        const { std::cell::Cell::new(Duration::from_millis(250)) };
    static FRAME_COUNTER: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

#[cfg(feature = "frame-trace")]
#[must_use = "Scope records on drop; binding it to `_` would record immediately"]
pub struct Scope {
    name: &'static str,
    start: Instant,
    recording_thread: std::marker::PhantomData<std::rc::Rc<()>>,
}

#[cfg(feature = "frame-trace")]
impl Drop for Scope {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed();
        let name = self.name;
        // Only `end_frame` drains the sections, so one recorded outside a frame would pile up.
        if CURRENT_FRAME_START.with(|c| c.get().is_none()) {
            return;
        }
        CURRENT_SECTIONS.with(|s| {
            if let Ok(mut s) = s.try_borrow_mut() {
                s.push(Section { name, elapsed });
            }
        });
    }
}

/// Records nothing unless it drops between `begin_frame` and `end_frame`.
#[cfg(feature = "frame-trace")]
#[inline]
pub fn scope(name: &'static str) -> Scope {
    Scope {
        name,
        start: Instant::now(),
        recording_thread: std::marker::PhantomData,
    }
}

#[cfg(feature = "frame-trace")]
pub fn begin_frame() {
    CURRENT_FRAME_START.with(|c| c.set(Some(Instant::now())));
    CURRENT_FRAME_ALLOC_START.with(|c| c.set(crate::alloc::current_snapshot()));
}

/// Adds the synthetic `between-frames` and `idle` sections after the first frame.
#[cfg(feature = "frame-trace")]
pub fn end_frame() {
    let now = Instant::now();
    let last_end = LAST_FRAME_END.with(|cell| {
        let prev = cell.get();
        cell.set(Some(now));
        prev
    });
    let frame_start = CURRENT_FRAME_START.with(|c| c.take());

    let alloc_start = CURRENT_FRAME_ALLOC_START.with(|c| c.take());
    let alloc_end = crate::alloc::current_snapshot();
    let alloc_delta: Option<crate::alloc::AllocDelta> = match (alloc_start, alloc_end) {
        (Some(a), Some(b)) => Some(crate::alloc::delta(a, b)),
        _ => None,
    };

    let frame_index = FRAME_COUNTER.with(|c| {
        let n = c.get();
        c.set(n.wrapping_add(1));
        n
    });

    let threshold = SPIKE_THRESHOLD.with(|c| c.get());
    let mut sections = CURRENT_SECTIONS.with(|s| std::mem::take(&mut *s.borrow_mut()));

    if let Some(last_end) = last_end {
        let between_frames = now.saturating_duration_since(last_end);
        sections.push(Section {
            name: "between-frames",
            elapsed: between_frames,
        });
        if let Some(frame_start) = frame_start {
            let idle = frame_start.saturating_duration_since(last_end);
            sections.push(Section {
                name: "idle",
                elapsed: idle,
            });
        }
    }

    // Release tracer borrows before invoking a tracing subscriber.
    let over_threshold: Vec<(&'static str, Duration)> = sections
        .iter()
        .filter(|section| section.elapsed > threshold)
        .map(|section| (section.name, section.elapsed))
        .collect();

    let mut reusable = TRACER.with(|t| {
        let mut t = t.borrow_mut();
        let reusable = if t.history.len() >= t.capacity {
            t.history
                .pop_front()
                .map(|frame| frame.sections)
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        t.history.push_back(FrameTrace { sections });
        reusable
    });
    reusable.clear();
    CURRENT_SECTIONS.with(|current| *current.borrow_mut() = reusable);

    for (name, elapsed) in over_threshold {
        let alloc_suffix = alloc_delta
            .map(|d| {
                format!(
                    " allocs={} ({:+.2}MB net)",
                    d.alloc_count,
                    d.net_bytes as f64 / (1024.0 * 1024.0),
                )
            })
            .unwrap_or_default();
        tracing::warn!(
            "frame_trace spike: section='{name}' elapsed={:.1}ms frame={frame_index}{alloc_suffix}",
            elapsed.as_secs_f32() * 1000.0,
        );
    }
}

#[cfg(feature = "frame-trace")]
pub fn set_capacity(capacity: usize) {
    TRACER.with(|t| {
        let mut t = t.borrow_mut();
        t.capacity = capacity.max(1);
        while t.history.len() > t.capacity {
            t.history.pop_front();
        }
    });
}

/// Oldest to newest; allocates.
#[cfg(feature = "frame-trace")]
pub fn history() -> Vec<FrameTrace> {
    TRACER.with(|t| t.borrow().history.iter().cloned().collect())
}

#[cfg(feature = "frame-trace")]
pub fn clear_history() {
    TRACER.with(|t| t.borrow_mut().history.clear());
}

#[cfg(not(feature = "frame-trace"))]
pub fn clear_history() {}

/// `f` must not call [`end_frame`], [`set_capacity`], or [`clear_history`]; the borrow is held.
#[cfg(feature = "frame-trace")]
pub fn with_history<R>(f: impl FnOnce(&std::collections::VecDeque<FrameTrace>) -> R) -> R {
    TRACER.with(|t| f(&t.borrow().history))
}

#[cfg(feature = "frame-trace")]
pub fn last_frame() -> Option<FrameTrace> {
    TRACER.with(|t| t.borrow().history.back().cloned())
}

/// `Duration::MAX` disables the spike warning.
#[cfg(feature = "frame-trace")]
pub fn set_spike_threshold(threshold: Duration) {
    SPIKE_THRESHOLD.with(|c| c.set(threshold));
}

/// Attributes the sample to the current frame, even if it arrives late.
#[cfg(feature = "frame-trace")]
pub fn record_external(name: &'static str, elapsed: Duration) {
    CURRENT_SECTIONS.with(|s| {
        if let Ok(mut s) = s.try_borrow_mut() {
            s.push(Section { name, elapsed });
        }
    });
}

#[cfg(not(feature = "frame-trace"))]
pub fn record_external(_name: &'static str, _elapsed: Duration) {}

#[cfg(not(feature = "frame-trace"))]
pub fn set_spike_threshold(_threshold: Duration) {}

#[derive(Clone, Debug)]
pub struct SectionStats {
    pub name: &'static str,
    pub samples: usize,
    pub mean: Duration,
    pub p50: Duration,
    pub p95: Duration,
    pub p99: Duration,
    pub max: Duration,
}

/// Nearest-rank percentile of ascending samples; empty input returns zero.
pub fn percentile(sorted: &[Duration], percent: u8) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let rank = (sorted.len() * usize::from(percent)).div_ceil(100);
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

/// Keyed by name, in descending p95 order.
#[cfg(feature = "frame-trace")]
pub fn aggregate() -> Vec<SectionStats> {
    use std::collections::HashMap;
    let mut buckets: HashMap<&'static str, Vec<Duration>> = HashMap::new();
    with_history(|frames| {
        for frame in frames {
            for section in &frame.sections {
                buckets
                    .entry(section.name)
                    .or_default()
                    .push(section.elapsed);
            }
        }
    });

    let mut stats: Vec<SectionStats> = buckets
        .into_iter()
        .map(|(name, mut samples)| {
            samples.sort();
            let n = samples.len();
            let mean = samples.iter().sum::<Duration>() / (n as u32).max(1);
            SectionStats {
                name,
                samples: n,
                mean,
                p50: percentile(&samples, 50),
                p95: percentile(&samples, 95),
                p99: percentile(&samples, 99),
                max: *samples.last().unwrap_or(&Duration::ZERO),
            }
        })
        .collect();

    stats.sort_by_key(|s| std::cmp::Reverse(s.p95));
    stats
}

#[cfg(not(feature = "frame-trace"))]
#[must_use]
pub struct Scope;

#[cfg(not(feature = "frame-trace"))]
#[inline]
pub fn scope(_name: &'static str) -> Scope {
    Scope
}

#[cfg(not(feature = "frame-trace"))]
pub fn end_frame() {}

#[cfg(not(feature = "frame-trace"))]
pub fn begin_frame() {}

#[cfg(not(feature = "frame-trace"))]
pub fn set_capacity(_capacity: usize) {}

#[cfg(not(feature = "frame-trace"))]
pub fn history() -> Vec<FrameTrace> {
    Vec::new()
}

#[cfg(not(feature = "frame-trace"))]
pub fn with_history<R>(f: impl FnOnce(&std::collections::VecDeque<FrameTrace>) -> R) -> R {
    let empty = std::collections::VecDeque::new();
    f(&empty)
}

#[cfg(not(feature = "frame-trace"))]
pub fn last_frame() -> Option<FrameTrace> {
    None
}

#[cfg(not(feature = "frame-trace"))]
pub fn aggregate() -> Vec<SectionStats> {
    Vec::new()
}

#[cfg(all(test, feature = "frame-trace"))]
mod tests {
    use super::*;

    #[test]
    fn retained_percentiles() {
        set_capacity(2);
        record_external("discarded", Duration::from_secs(10));
        end_frame();
        for elapsed in [10, 20] {
            record_external("slow", Duration::from_millis(elapsed));
            record_external("fast", Duration::from_millis(1));
            end_frame();
        }
        let stats = aggregate();
        assert!(!stats.iter().any(|row| row.name == "discarded"));
        let slow = stats.iter().position(|row| row.name == "slow").unwrap();
        let fast = stats.iter().position(|row| row.name == "fast").unwrap();
        assert!(slow < fast);
        assert_eq!(stats[slow].samples, 2);
        assert_eq!(stats[slow].mean, Duration::from_millis(15));
        assert_eq!(stats[slow].p50, Duration::from_millis(10));
        assert_eq!(stats[slow].p95, Duration::from_millis(20));
    }
}
