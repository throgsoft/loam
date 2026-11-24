use std::time::Duration;

use web_time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pace {
    Run,
    Wait(Instant),
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Pacer {
    period: Option<Duration>,
    anchor: Option<Instant>,
}

impl Pacer {
    pub fn set_target_fps(&mut self, fps: f32) {
        self.period = (fps.is_finite() && fps > 0.0)
            .then(|| Duration::from_nanos(((1_000_000_000.0 / f64::from(fps)) as u64).max(1)));
    }

    pub fn reset(&mut self) {
        self.anchor = None;
    }

    pub fn deadline(&self) -> Option<Instant> {
        Some(self.anchor? + self.period?)
    }

    /// `Wait` until the deadline; a late wake anchors the next period on the deadline, not on itself.
    pub fn decide(&mut self, now: Instant) -> Pace {
        let Some(period) = self.period else {
            self.anchor = None;
            return Pace::Run;
        };
        let Some(anchor) = self.anchor else {
            self.anchor = Some(now);
            return Pace::Run;
        };
        let deadline = anchor + period;
        if now < deadline {
            return Pace::Wait(deadline);
        }
        let floor = now.checked_sub(period).unwrap_or(now);
        self.anchor = Some(deadline.max(floor));
        Pace::Run
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn late_wake_up() {
        let period = Duration::from_millis(10);
        let mut pacer = Pacer::default();
        pacer.set_target_fps(100.0);
        let start = Instant::now();

        assert_eq!(pacer.decide(start), Pace::Run);
        assert_eq!(
            pacer.decide(start + Duration::from_millis(3)),
            Pace::Wait(start + period)
        );
        assert_eq!(pacer.decide(start + Duration::from_millis(12)), Pace::Run);
        assert_eq!(
            pacer.decide(start + Duration::from_millis(18)),
            Pace::Wait(start + period * 2)
        );

        pacer.set_target_fps(0.0);
        assert_eq!(pacer.decide(start + Duration::from_millis(18)), Pace::Run);
    }
}
