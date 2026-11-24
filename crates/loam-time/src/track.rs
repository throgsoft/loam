use std::num::NonZeroU32;

use glam::Vec4;

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum Ease {
    #[default]
    Linear,
    InOutCubic,
    OutCubic,
}

impl Ease {
    pub fn apply(self, u: f32) -> f32 {
        let u = u.clamp(0.0, 1.0);
        match self {
            Self::Linear => u,
            Self::InOutCubic => {
                if u < 0.5 {
                    4.0 * u * u * u
                } else {
                    let v = -2.0 * u + 2.0;
                    1.0 - v * v * v / 2.0
                }
            }
            Self::OutCubic => {
                let v = 1.0 - u;
                1.0 - v * v * v
            }
        }
    }
}

pub trait Interpolate: Copy {
    fn mix(from: Self, to: Self, u: f32) -> Self;
}

impl Interpolate for f32 {
    fn mix(from: Self, to: Self, u: f32) -> Self {
        from + (to - from) * u
    }
}

impl Interpolate for Vec4 {
    fn mix(from: Self, to: Self, u: f32) -> Self {
        from + (to - from) * u
    }
}

// `t` in timeline seconds; `ease` applies to the span reaching this key.
#[derive(Copy, Clone, Debug, PartialEq)]
struct Key<T> {
    t: f32,
    value: T,
    ease: Ease,
}

/// Sampling outside the key range holds the nearest endpoint.
#[derive(Clone, Debug, PartialEq)]
pub struct Track<T> {
    keys: Vec<Key<T>>,
}

impl<T> Default for Track<T> {
    fn default() -> Self {
        Self { keys: Vec::new() }
    }
}

impl<T> Track<T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn key(mut self, t: f32, value: T, ease: Ease) -> Self {
        self.keys.push(Key { t, value, ease });
        self
    }
}

impl<T: Interpolate> Track<T> {
    /// Requires ordered keys; returns `None` for an empty track.
    pub fn sample(&self, frame: u32, fps: NonZeroU32) -> Option<T> {
        let first = self.keys.first()?;
        let t = frame as f32 / fps.get() as f32;
        if t <= first.t {
            return Some(first.value);
        }
        let last = self.keys.last()?;
        if t >= last.t {
            return Some(last.value);
        }
        let i = self.keys.partition_point(|k| k.t <= t) - 1;
        let (a, b) = (&self.keys[i], &self.keys[i + 1]);
        let span = b.t - a.t;
        let u = if span > 0.0 { (t - a.t) / span } else { 1.0 };
        Some(T::mix(a.value, b.value, b.ease.apply(u)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FPS: NonZeroU32 = NonZeroU32::new(60).unwrap();

    #[test]
    fn a_track_samples_in_timeline_seconds_eases_into_its_keys_and_holds_its_ends() {
        let track = Track::new()
            .key(0.0, 0.0, Ease::Linear)
            .key(1.0, 10.0, Ease::Linear);
        let thirty = NonZeroU32::new(30).unwrap();
        assert_eq!(track.sample(0, thirty), Some(0.0));
        assert_eq!(track.sample(15, thirty), Some(5.0));
        assert_eq!(track.sample(30, thirty), Some(10.0));
        assert_eq!(track.sample(30, FPS), Some(5.0));
        assert_eq!(track.sample(60, FPS), Some(10.0));

        let eased = Track::new()
            .key(0.0, 0.0, Ease::OutCubic)
            .key(1.0, 10.0, Ease::InOutCubic);
        let quarter = eased.sample(15, FPS).unwrap();
        assert!(
            (quarter - 0.625).abs() < 1e-6,
            "the destination key's ease was not used: sampled {quarter}"
        );

        let late = Track::new()
            .key(1.0, 4.0, Ease::Linear)
            .key(2.0, 9.0, Ease::Linear);
        assert_eq!(late.sample(0, FPS), Some(4.0));
        assert_eq!(late.sample(30, FPS), Some(4.0));
        assert_eq!(late.sample(60, FPS), Some(4.0));
        assert_eq!(late.sample(120, FPS), Some(9.0));
        assert_eq!(late.sample(100_000, FPS), Some(9.0));
    }
}
