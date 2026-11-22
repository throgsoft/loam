use std::num::NonZeroU32;

use glam::Vec4;
use loam_math::{Bivector, Rotor, Rotor4};
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
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

impl Interpolate for Rotor4 {
    // Geodesic R₀·exp(u·log(R₀⁻¹·R₁)), the SO(4) slerp (Shoemake 1985, §3).
    fn mix(from: Self, to: Self, u: f32) -> Self {
        let relative = from.inverse() * to;
        from * (relative.log() * u).exp()
    }
}

/// `t` in timeline seconds; `ease` applies to the span reaching this key.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Key<T> {
    pub t: f32,
    pub value: T,
    #[serde(default)]
    pub ease: Ease,
}

/// Sampling outside the key range holds the nearest endpoint.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
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

    pub fn keys(&self) -> &[Key<T>] {
        &self.keys
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BodyTrack {
    pub name: String,
    #[serde(default)]
    pub position: Option<Track<Vec4>>,
    #[serde(default)]
    pub orientation: Option<Track<Rotor4>>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Timeline {
    pub fps: u32,
    pub frames: u32,
    #[serde(default)]
    pub w_slice: Option<Track<f32>>,
    #[serde(default)]
    pub bodies: Vec<BodyTrack>,
}

const UNIT_ROTOR_TOLERANCE: f32 = 1e-3;

impl Timeline {
    pub fn validate(&self) -> Result<NonZeroU32, TimelineError> {
        let fps = NonZeroU32::new(self.fps).ok_or(TimelineError::ZeroFps)?;
        if self.frames == 0 {
            return Err(TimelineError::ZeroFrames);
        }
        if let Some(track) = &self.w_slice {
            validate_times(track, "w_slice")?;
        }
        for (index, body) in self.bodies.iter().enumerate() {
            if self.bodies[..index].iter().any(|b| b.name == body.name) {
                return Err(TimelineError::DuplicateBody {
                    name: body.name.clone(),
                });
            }
            if let Some(track) = &body.position {
                validate_times(track, &format!("{}.position", body.name))?;
            }
            if let Some(track) = &body.orientation {
                validate_rotors(track, &format!("{}.orientation", body.name))?;
            }
        }
        Ok(fps)
    }
}

fn validate_times<T>(track: &Track<T>, channel: &str) -> Result<(), TimelineError> {
    let keys = track.keys();
    if keys.is_empty() {
        return Err(TimelineError::EmptyTrack {
            channel: channel.to_owned(),
        });
    }
    for (index, key) in keys.iter().enumerate() {
        if !key.t.is_finite() || key.t < 0.0 {
            return Err(TimelineError::KeyTime {
                channel: channel.to_owned(),
                index,
            });
        }
        if index > 0 && key.t <= keys[index - 1].t {
            return Err(TimelineError::KeyOrder {
                channel: channel.to_owned(),
                index,
            });
        }
    }
    Ok(())
}

fn validate_rotors(track: &Track<Rotor4>, channel: &str) -> Result<(), TimelineError> {
    validate_times(track, channel)?;
    for (index, key) in track.keys().iter().enumerate() {
        let norm_squared = key.value.norm_squared();
        let pseudoscalar = (key.value * key.value.inverse()).xyzw;
        if !norm_squared.is_finite()
            || (norm_squared - 1.0).abs() > UNIT_ROTOR_TOLERANCE
            || !pseudoscalar.is_finite()
            || pseudoscalar.abs() > UNIT_ROTOR_TOLERANCE
        {
            return Err(TimelineError::NonUnitRotor {
                channel: channel.to_owned(),
                index,
                norm_squared,
                pseudoscalar,
            });
        }
    }
    Ok(())
}

#[derive(Debug, thiserror::Error)]
pub enum TimelineError {
    #[error("fps must be nonzero")]
    ZeroFps,
    #[error("frame count must be nonzero")]
    ZeroFrames,
    #[error("channel '{channel}' has no keys")]
    EmptyTrack { channel: String },
    #[error("channel '{channel}' key {index} has a negative or non-finite time")]
    KeyTime { channel: String, index: usize },
    #[error("channel '{channel}' key {index} is not strictly later than its predecessor")]
    KeyOrder { channel: String, index: usize },
    #[error("channel '{channel}' key {index} is not a unit rotor (norm² = {norm_squared}, R reverse(R) pseudoscalar = {pseudoscalar})")]
    NonUnitRotor {
        channel: String,
        index: usize,
        norm_squared: f32,
        pseudoscalar: f32,
    },
    #[error("duplicate body name '{name}'")]
    DuplicateBody { name: String },
    #[error(transparent)]
    Ron(#[from] ron::error::SpannedError),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Playhead {
    frame: u32,
    frames: u32,
    playing: bool,
}

impl Playhead {
    /// Zero `frames` reads as one.
    pub fn new(frames: u32) -> Self {
        Self {
            frame: 0,
            frames: frames.max(1),
            playing: true,
        }
    }

    pub fn advance(&mut self) {
        if self.playing {
            self.frame = (self.frame + 1).min(self.last());
        }
    }

    pub fn seek(&mut self, frame: u32) {
        self.frame = frame.min(self.last());
    }

    pub fn frame(&self) -> u32 {
        self.frame
    }

    pub fn frames(&self) -> u32 {
        self.frames
    }

    pub fn playing(&self) -> bool {
        self.playing
    }

    pub fn set_playing(&mut self, playing: bool) {
        self.playing = playing;
    }

    pub fn finished(&self) -> bool {
        self.frame >= self.last()
    }

    fn last(&self) -> u32 {
        self.frames - 1
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
#[must_use]
pub enum Drive<T> {
    Host,
    Directed(T),
}

/// Paused and finished channels retain their sampled values.
#[derive(Clone, Debug)]
pub struct Director {
    timeline: Timeline,
    fps: NonZeroU32,
    playhead: Playhead,
}

impl Director {
    pub fn new(timeline: Timeline) -> Result<Self, TimelineError> {
        let fps = timeline.validate()?;
        let playhead = Playhead::new(timeline.frames);
        Ok(Self {
            timeline,
            fps,
            playhead,
        })
    }

    pub fn from_ron(text: &str) -> Result<Self, TimelineError> {
        Self::new(ron::from_str::<Timeline>(text)?)
    }

    pub fn timeline(&self) -> &Timeline {
        &self.timeline
    }

    pub fn advance(&mut self) {
        self.playhead.advance();
    }

    pub fn seek(&mut self, frame: u32) {
        self.playhead.seek(frame);
    }

    pub fn playhead(&self) -> Playhead {
        self.playhead
    }

    pub fn set_playing(&mut self, playing: bool) {
        self.playhead.set_playing(playing);
    }

    pub fn frame(&self) -> u32 {
        self.playhead.frame()
    }

    pub fn finished(&self) -> bool {
        self.playhead.finished()
    }

    pub fn w_slice(&self) -> Drive<f32> {
        self.drive(self.timeline.w_slice.as_ref())
    }

    pub fn position(&self, body: &str) -> Drive<Vec4> {
        self.drive(self.body(body).and_then(|b| b.position.as_ref()))
    }

    pub fn orientation(&self, body: &str) -> Drive<Rotor4> {
        self.drive(self.body(body).and_then(|b| b.orientation.as_ref()))
    }

    /// Preserves timeline order.
    pub fn bodies(&self) -> impl Iterator<Item = &str> {
        self.timeline.bodies.iter().map(|b| b.name.as_str())
    }

    fn body(&self, name: &str) -> Option<&BodyTrack> {
        self.timeline.bodies.iter().find(|b| b.name == name)
    }

    fn drive<T: Interpolate>(&self, track: Option<&Track<T>>) -> Drive<T> {
        match track.and_then(|t| t.sample(self.playhead.frame(), self.fps)) {
            Some(value) => Drive::Directed(value),
            None => Drive::Host,
        }
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
