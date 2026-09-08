use std::ops::Range;
use std::time::Duration;
// `std::time::Instant::now` panics on wasm32.
use web_time::Instant;

/// Per-frame catch-up cap; excess ticks are dropped.
pub const DEFAULT_MAX_CATCH_UP: u32 = 10;

#[derive(Debug, Clone)]
pub struct FixedTimestep {
    dt: Duration,
    accumulator: Duration,
    last_instant: Option<Instant>,
    tick: u64,
    max_catch_up: u32,
}

impl FixedTimestep {
    /// Panics outside `1..=1_000_000_000` Hz, the range supported by nanosecond ticks.
    pub fn new(hz: u32) -> Self {
        assert!(
            (1..=1_000_000_000).contains(&hz),
            "tick rate must be between 1 and 1000000000 Hz"
        );
        Self {
            dt: Duration::from_nanos(1_000_000_000 / u64::from(hz)),
            accumulator: Duration::ZERO,
            last_instant: None,
            tick: 0,
            max_catch_up: DEFAULT_MAX_CATCH_UP,
        }
    }

    pub fn with_max_catch_up(mut self, n: u32) -> Self {
        self.max_catch_up = n;
        self
    }

    /// Discards elapsed and fractional time while preserving the next tick index.
    pub fn reset_clock(&mut self, now: Instant) {
        self.last_instant = Some(now);
        self.accumulator = Duration::ZERO;
    }

    pub fn tick(&self) -> u64 {
        self.tick
    }

    pub fn dt(&self) -> Duration {
        self.dt
    }

    pub fn dt_seconds(&self) -> f32 {
        self.dt.as_secs_f32()
    }

    /// Fraction of the pending tick elapsed, rounded to `[0, 1]`.
    pub fn alpha(&self) -> f32 {
        let a = self.accumulator.as_secs_f64() / self.dt.as_secs_f64();
        (a as f32).clamp(0.0, 1.0)
    }

    /// The first call only primes the clock; ticks past `max_catch_up` are dropped.
    pub fn advance(&mut self, now: Instant) -> Range<u64> {
        let last = match self.last_instant.replace(now) {
            Some(t) => t,
            None => return self.tick..self.tick,
        };

        self.accumulator += now.saturating_duration_since(last);

        let start = self.tick;
        let pending = self.accumulator.as_nanos() / self.dt.as_nanos();
        self.tick += pending.min(u128::from(self.max_catch_up)) as u64;

        self.accumulator =
            Duration::from_nanos((self.accumulator.as_nanos() % self.dt.as_nanos()) as u64);

        start..self.tick
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reset_clock_drops_pause_time_without_rewinding_ticks() {
        let mut clock = FixedTimestep::new(100);
        let start = Instant::now();
        clock.advance(start);
        clock.advance(start + Duration::from_millis(15));
        let resumed = start + Duration::from_secs(60);
        clock.reset_clock(resumed);
        assert_eq!(clock.advance(resumed), 1..1);
        assert_eq!(clock.advance(resumed + clock.dt()), 1..2);
    }

    #[test]
    fn tick_boundaries_carry_fractional_time() {
        let mut clock = FixedTimestep::new(100);
        let start = Instant::now();
        let dt = clock.dt();
        assert_eq!(clock.advance(start), 0..0);
        assert_eq!(clock.advance(start + dt / 2), 0..0);
        assert_eq!(clock.alpha(), 0.5);
        assert_eq!(clock.advance(start + dt), 0..1);
        assert_eq!(clock.alpha(), 0.0);
        assert_eq!(clock.advance(start + dt * 3 + dt / 2), 1..3);
        assert_eq!(clock.alpha(), 0.5);
        assert_eq!(clock.tick(), 3);
    }

    #[test]
    fn catch_up_drops_backlog_but_keeps_remainder() {
        let mut clock = FixedTimestep::new(100).with_max_catch_up(5);
        let start = Instant::now();
        clock.advance(start);
        let resumed = start + Duration::from_secs(86_400) + clock.dt() / 2;
        assert_eq!(clock.advance(resumed), 0..5);
        assert_eq!(clock.alpha(), 0.5);
        assert_eq!(clock.advance(resumed), 5..5);
        assert_eq!(clock.advance(resumed + clock.dt() / 2), 5..6);
    }

    #[test]
    fn zero_catch_up_discards_whole_ticks() {
        let mut clock = FixedTimestep::new(100).with_max_catch_up(0);
        let start = Instant::now();
        clock.advance(start);
        assert_eq!(clock.advance(start + Duration::from_millis(15)), 0..0);
        assert_eq!(clock.alpha(), 0.5);
    }

    #[test]
    fn nanosecond_ticks_advance() {
        let mut clock = FixedTimestep::new(1_000_000_000);
        let start = Instant::now();
        clock.advance(start);
        assert_eq!(clock.advance(start + Duration::from_nanos(1)), 0..1);
    }

    #[test]
    #[should_panic(expected = "tick rate must be between")]
    fn zero_tick_rate_is_rejected() {
        FixedTimestep::new(0);
    }

    #[test]
    #[should_panic(expected = "tick rate must be between")]
    fn subnanosecond_ticks_are_rejected() {
        FixedTimestep::new(1_000_000_001);
    }
}
