use crate::args::Args;
use crate::wasm::input_queue::InputMessage;
use loam_runtime::PhaseError;
use web_time::Instant;

use crate::session::surface::Attempt;

const MAX_SECONDS: u32 = 600;
const SAMPLE_CAPACITY: usize = 60 * MAX_SECONDS as usize + 1;

#[derive(Clone, Copy)]
enum Phase {
    Waiting,
    Warmup(f64),
    Ready,
    Sample(f64),
}

#[derive(Clone, Copy)]
struct Sample {
    cpu_ms: f64,
    interval_ms: f64,
}

#[derive(Default)]
struct Skips {
    paced: u64,
    empty: u64,
    reconfigured: u64,
    timeout: u64,
    other: u64,
}

#[derive(Default)]
struct Excluded {
    hidden: u64,
    focus: u64,
    resize: u64,
    recovery: u64,
    partial_windows: u64,
    partial_presents: u64,
}

pub(super) struct Probe {
    origin: Instant,
    sample_seconds: u32,
    warmup_seconds: u32,
    pixel_budget: Option<u32>,
    resolution: (u32, u32, f32),
    phase: Phase,
    warmup: (u64, f64),
    successful: u64,
    last_started_ms: f64,
    samples: Vec<Sample>,
    dropped_samples: u64,
    cpu_total_ms: f64,
    cpu_max_ms: f64,
    skips: Skips,
    excluded: Excluded,
}

impl Probe {
    pub(super) fn from_args(
        args: &Args,
        width: u32,
        height: u32,
        scale: f32,
        pixel_budget: Option<u32>,
    ) -> Option<Self> {
        Some(Self {
            origin: Instant::now(),
            sample_seconds: args.parse::<u32>("measure-seconds")?.clamp(1, MAX_SECONDS),
            warmup_seconds: args
                .parse::<u32>("warmup-seconds")
                .unwrap_or(30)
                .min(MAX_SECONDS),
            pixel_budget,
            resolution: (width, height, scale),
            phase: Phase::Waiting,
            warmup: (0, 0.0),
            successful: 0,
            last_started_ms: 0.0,
            samples: Vec::with_capacity(SAMPLE_CAPACITY),
            dropped_samples: 0,
            cpu_total_ms: 0.0,
            cpu_max_ms: 0.0,
            skips: Skips::default(),
            excluded: Excluded::default(),
        })
    }

    pub(super) fn observe_message(&mut self, message: &InputMessage) {
        match message {
            InputMessage::Resize { width, height, dpr }
                if (*width, *height, *dpr) != self.resolution =>
            {
                self.excluded.resize += 1;
                self.interrupt();
                self.resolution = (*width, *height, *dpr);
            }
            InputMessage::Visibility(false) => {
                self.excluded.hidden += 1;
                self.interrupt();
            }
            InputMessage::Focus(false) => {
                self.excluded.focus += 1;
                self.interrupt();
            }
            _ => {}
        }
    }

    pub(super) fn recover(&mut self) {
        self.excluded.recovery += 1;
        self.interrupt();
    }

    pub(super) fn invalid(self, error: PhaseError) -> String {
        format!(
            "Loam browser measurement\n\nmeasurement: invalid\nreason: session phase fault\nphase: {:?}\nsystem: {}\ncause: {:?}",
            error.phase,
            error.system.unwrap_or("<unnamed>"),
            error.cause,
        )
    }

    fn interrupt(&mut self) {
        if matches!(self.phase, Phase::Sample(_)) {
            self.excluded.partial_windows += 1;
            self.excluded.partial_presents += self.successful;
        }
        self.phase = Phase::Waiting;
        self.warmup = (0, 0.0);
        self.successful = 0;
        self.last_started_ms = 0.0;
        self.samples.clear();
        self.dropped_samples = 0;
        self.cpu_total_ms = 0.0;
        self.cpu_max_ms = 0.0;
    }

    pub(super) fn observe(
        &mut self,
        attempt: Attempt,
        started: Instant,
        cpu_ms: f64,
    ) -> Option<String> {
        match attempt {
            Attempt::Paced => self.skips.paced += 1,
            Attempt::EmptySurface => self.skips.empty += 1,
            Attempt::ReconfiguredSurface => self.skips.reconfigured += 1,
            Attempt::TimedOutSurface => self.skips.timeout += 1,
            Attempt::FailedSurface => self.skips.other += 1,
            Attempt::Presented => {
                let started_ms = started.duration_since(self.origin).as_secs_f64() * 1000.0;
                return self.presented(started_ms, cpu_ms);
            }
        }
        None
    }

    fn presented(&mut self, started_ms: f64, cpu_ms: f64) -> Option<String> {
        let finished_ms = started_ms + cpu_ms;
        match self.phase {
            Phase::Waiting if self.warmup_seconds == 0 => self.phase = Phase::Sample(started_ms),
            Phase::Waiting => {
                self.warmup.0 = 1;
                self.phase = Phase::Warmup(started_ms);
                return None;
            }
            Phase::Warmup(warmup_started_ms) => {
                self.warmup.0 += 1;
                self.warmup.1 = finished_ms - warmup_started_ms;
                if self.warmup.1 >= f64::from(self.warmup_seconds) * 1000.0 {
                    self.phase = Phase::Ready;
                }
                return None;
            }
            Phase::Ready => self.phase = Phase::Sample(started_ms),
            Phase::Sample(_) => {}
        }

        let interval_ms = if self.successful == 0 {
            0.0
        } else {
            started_ms - self.last_started_ms
        };
        self.successful += 1;
        self.last_started_ms = started_ms;
        self.cpu_total_ms += cpu_ms;
        self.cpu_max_ms = self.cpu_max_ms.max(cpu_ms);
        if self.samples.len() < SAMPLE_CAPACITY {
            self.samples.push(Sample {
                cpu_ms,
                interval_ms,
            });
        } else {
            self.dropped_samples += 1;
        }

        let Phase::Sample(sample_started_ms) = self.phase else {
            return None;
        };
        (finished_ms - sample_started_ms >= f64::from(self.sample_seconds) * 1000.0)
            .then(|| self.report(sample_started_ms, finished_ms))
    }

    fn report(&mut self, started_ms: f64, finished_ms: f64) -> String {
        self.samples
            .sort_unstable_by(|left, right| left.cpu_ms.total_cmp(&right.cpu_ms));
        let cpu_p50 = percentile(&self.samples, 50, |sample| sample.cpu_ms);
        let cpu_p95 = percentile(&self.samples, 95, |sample| sample.cpu_ms);
        self.samples
            .sort_unstable_by(|left, right| left.interval_ms.total_cmp(&right.interval_ms));
        let intervals = &self.samples[usize::from(!self.samples.is_empty())..];
        let interval_p95 = percentile(intervals, 95, |sample| sample.interval_ms);
        let interval_p99 = percentile(intervals, 99, |sample| sample.interval_ms);
        let Phase::Sample(sample_started_ms) = self.phase else {
            return String::new();
        };
        let cadence_span_ms = self.last_started_ms - sample_started_ms;
        let cadence_hz = if self.successful > 1 && cadence_span_ms > 0.0 {
            (self.successful - 1) as f64 * 1000.0 / cadence_span_ms
        } else {
            0.0
        };
        let percentile_scope = if self.dropped_samples == 0 {
            "complete"
        } else {
            "incomplete"
        };
        let (width, height, scale) = self.resolution;
        format!(
            "Loam browser measurement\nLong-press this text to copy.\n\nrequested sample: {} s\nrequested warmup: {} s\ncompleted warmup: {:.3} ms, {} successful presents\nmonotonic start: {:.3} ms\nmonotonic end: {:.3} ms\nactual sample window: {:.3} ms\nbuffer: {} x {} = {} pixels\neffective pixel scale: {:.6}\npixel budget: {}\nsuccessful presents: {}\nsuccessful-present cadence: {:.3} Hz across {} intervals / {:.3} ms\nframe interval: p95 {:.3} ms, p99 {:.3} ms\nCPU host wall duration: total {:.3} ms, mean {:.3} ms, p50 {:.3} ms, p95 {:.3} ms, max {:.3} ms\npercentile sample: {}, {} pairs, {} dropped after fixed capacity\nrun-total skipped attempts: paced {}, empty surface {}, reconfigured surface {}, timeout {}, other surface {}\nrun-total excluded periods: hidden {}, focus/pause {}, resize {}, device recovery {}\nrun-total discarded partial samples: {} windows, {} successful presents",
            self.sample_seconds,
            self.warmup_seconds,
            self.warmup.1,
            self.warmup.0,
            started_ms,
            finished_ms,
            finished_ms - started_ms,
            width,
            height,
            u64::from(width) * u64::from(height),
            scale,
            self.pixel_budget
                .map_or_else(|| "none".to_owned(), |pixels| pixels.to_string()),
            self.successful,
            cadence_hz,
            self.successful.saturating_sub(1),
            cadence_span_ms,
            interval_p95,
            interval_p99,
            self.cpu_total_ms,
            self.cpu_total_ms / self.successful as f64,
            cpu_p50,
            cpu_p95,
            self.cpu_max_ms,
            percentile_scope,
            self.samples.len(),
            self.dropped_samples,
            self.skips.paced,
            self.skips.empty,
            self.skips.reconfigured,
            self.skips.timeout,
            self.skips.other,
            self.excluded.hidden,
            self.excluded.focus,
            self.excluded.resize,
            self.excluded.recovery,
            self.excluded.partial_windows,
            self.excluded.partial_presents,
        )
    }
}

fn percentile<T>(sorted: &[T], percent: usize, value: impl Fn(&T) -> f64) -> f64 {
    let rank = (sorted.len() * percent).div_ceil(100);
    let index = rank.saturating_sub(1);
    sorted.get(index).map(value).unwrap_or(0.0)
}
