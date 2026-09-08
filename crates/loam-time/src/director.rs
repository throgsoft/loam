//! Frame-indexed animation samples; callers decide how to apply them.

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
    /// Overrides the host animation for this channel.
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
    use loam_math::Bivector4;
    use std::f32::consts::PI;

    const FPS: NonZeroU32 = NonZeroU32::new(60).unwrap();

    fn rotor(b: Bivector4) -> Rotor4 {
        b.exp()
    }

    fn spin_timeline() -> Timeline {
        Timeline {
            fps: 60,
            frames: 121,
            w_slice: Some(Track::new().key(0.0, -1.0, Ease::Linear).key(
                2.0,
                1.0,
                Ease::InOutCubic,
            )),
            bodies: vec![BodyTrack {
                name: "row".to_owned(),
                position: Some(
                    Track::new()
                        .key(0.0, Vec4::new(0.0, 0.0, 0.0, -3.0), Ease::Linear)
                        .key(2.0, Vec4::new(1.0, 2.0, 3.0, 0.0), Ease::Linear),
                ),
                orientation: Some(Track::new().key(0.0, Rotor4::IDENTITY, Ease::Linear).key(
                    2.0,
                    rotor(Bivector4::new(0.7, 0.0, 0.0, 0.0, 0.0, 0.3)),
                    Ease::Linear,
                )),
            }],
        }
    }

    #[test]
    fn easing_clamps_endpoints() {
        for ease in [Ease::Linear, Ease::InOutCubic, Ease::OutCubic] {
            assert_eq!(ease.apply(0.0), 0.0);
            assert!((ease.apply(1.0) - 1.0).abs() < 1e-6);
            assert_eq!(ease.apply(-4.0), 0.0);
            assert!((ease.apply(9.0) - 1.0).abs() < 1e-6);
        }
        assert!((Ease::InOutCubic.apply(0.5) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn sampling_uses_timeline_rate() {
        let track = Track::new()
            .key(0.0, 0.0, Ease::Linear)
            .key(1.0, 10.0, Ease::Linear);
        let thirty = NonZeroU32::new(30).unwrap();
        assert_eq!(track.sample(0, thirty), Some(0.0));
        assert_eq!(track.sample(15, thirty), Some(5.0));
        assert_eq!(track.sample(30, thirty), Some(10.0));
        assert_eq!(track.sample(30, FPS), Some(5.0));
        assert_eq!(track.sample(60, FPS), Some(10.0));
    }

    #[test]
    fn destination_key_selects_easing() {
        let track = Track::new()
            .key(0.0, 0.0, Ease::OutCubic)
            .key(1.0, 10.0, Ease::InOutCubic);
        let quarter = track.sample(15, FPS).unwrap();
        assert!((quarter - 0.625).abs() < 1e-6, "sampled {quarter}");
    }

    #[test]
    fn position_track_interpolates_w() {
        let track = Track::new()
            .key(0.0, Vec4::new(0.0, 0.0, 0.0, -4.0), Ease::Linear)
            .key(1.0, Vec4::new(2.0, -6.0, 0.0, 4.0), Ease::Linear);
        let quarter = track.sample(15, FPS).unwrap();
        assert!(
            (quarter - Vec4::new(0.5, -1.5, 0.0, -2.0)).length() < 1e-6,
            "sampled {quarter}"
        );
    }

    #[test]
    fn sampling_holds_endpoints() {
        let track = Track::new()
            .key(1.0, 4.0, Ease::Linear)
            .key(2.0, 9.0, Ease::Linear);
        assert_eq!(track.sample(0, FPS), Some(4.0));
        assert_eq!(track.sample(30, FPS), Some(4.0));
        assert_eq!(track.sample(60, FPS), Some(4.0));
        assert_eq!(track.sample(120, FPS), Some(9.0));
        assert_eq!(track.sample(100_000, FPS), Some(9.0));
    }

    #[test]
    fn seeking_matches_sequential_playback() {
        let mut walked = Director::new(spin_timeline()).unwrap();
        for _ in 0..37 {
            walked.advance();
        }
        let mut sought = Director::new(spin_timeline()).unwrap();
        sought.seek(37);
        let mut wandered = Director::new(spin_timeline()).unwrap();
        wandered.seek(500);
        wandered.seek(3);
        wandered.seek(37);

        assert_eq!(walked.frame(), 37);
        for other in [&sought, &wandered] {
            assert_eq!(walked.w_slice(), other.w_slice());
            assert_eq!(walked.position("row"), other.position("row"));
            assert_eq!(walked.orientation("row"), other.orientation("row"));
        }
    }

    #[test]
    fn playback_stops_at_last_frame() {
        let mut director = Director::new(spin_timeline()).unwrap();
        for expected in 1..=10 {
            director.advance();
            assert_eq!(director.frame(), expected);
        }
        assert!(!director.finished());
        for _ in 0..1000 {
            director.advance();
        }
        assert_eq!(director.frame(), 120);
        assert!(director.finished());
        let final_value = director.orientation("row");
        assert!(matches!(final_value, Drive::Directed(_)));
        director.set_playing(false);
        director.advance();
        assert_eq!(director.orientation("row"), final_value);
    }

    #[test]
    fn paused_seek_clamps_to_frame_range() {
        let mut playhead = Playhead::new(10);
        playhead.advance();
        playhead.set_playing(false);
        playhead.advance();
        playhead.advance();
        assert_eq!(playhead.frame(), 1);
        playhead.seek(99);
        assert_eq!(playhead.frame(), 9);
        assert!(playhead.finished());
        playhead.seek(0);
        assert_eq!(playhead.frame(), 0);
    }

    #[test]
    fn rotor_interpolation_preserves_norm_and_endpoints() {
        let from = rotor(Bivector4::new(0.31, -0.62, 0.14, 0.83, -0.27, 0.45));
        let to = rotor(Bivector4::new(-0.71, 0.22, 0.96, -0.18, 0.53, -0.34));
        let probes = [
            Vec4::new(1.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 1.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 1.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 1.0),
            Vec4::new(0.5, -0.5, 0.5, -0.5),
        ];

        for step in 0..=100 {
            let u = step as f32 / 100.0;
            let r = Rotor4::mix(from, to, u);
            assert!(
                (r.norm_squared() - 1.0).abs() < 1e-6,
                "u = {u}, norm² = {}",
                r.norm_squared()
            );
        }

        for probe in probes {
            let start = Rotor4::mix(from, to, 0.0).apply(probe);
            let end = Rotor4::mix(from, to, 1.0).apply(probe);
            assert!((start - from.apply(probe)).length() < 1e-5);
            assert!((end - to.apply(probe)).length() < 1e-5);
        }
    }

    #[test]
    fn rotor_track_uses_short_arc() {
        let track = Track::new().key(0.0, Rotor4::IDENTITY, Ease::Linear).key(
            1.0,
            rotor(Bivector4::new(1.9 * PI, 0.0, 0.0, 0.0, 0.0, 0.0)),
            Ease::Linear,
        );
        let midpoint = track.sample(30, FPS).unwrap();
        let travelled = midpoint.log().magnitude();
        assert!(
            (travelled - 0.05 * PI).abs() < 1e-3,
            "midpoint travelled {travelled} rad"
        );
    }

    #[test]
    fn isoclinic_half_turn_interpolates_rotation_action() {
        for sign in [-1.0, 1.0] {
            let to = Rotor4 {
                s: 0.0,
                xyzw: sign,
                ..Rotor4::IDENTITY
            };
            let timeline = Timeline {
                fps: 60,
                frames: 61,
                w_slice: None,
                bodies: vec![BodyTrack {
                    name: "row".to_owned(),
                    position: None,
                    orientation: Some(Track::new().key(0.0, Rotor4::IDENTITY, Ease::Linear).key(
                        1.0,
                        to,
                        Ease::Linear,
                    )),
                }],
            };
            let mut director = Director::new(timeline).unwrap();
            director.seek(30);
            let Drive::Directed(midpoint) = director.orientation("row") else {
                panic!("missing orientation")
            };
            for probe in [Vec4::X, Vec4::Y, Vec4::Z, Vec4::W] {
                let rotated = midpoint.apply(probe);
                assert!((rotated.length() - 1.0).abs() < 1e-5);
                assert!((rotated - probe).length() > 0.5);
                assert!((midpoint.apply(rotated) + probe).length() < 1e-5);
            }
            director.seek(60);
            let Drive::Directed(endpoint) = director.orientation("row") else {
                panic!("missing orientation")
            };
            for probe in [Vec4::X, Vec4::Y, Vec4::Z, Vec4::W] {
                assert!((endpoint.apply(probe) + probe).length() < 1e-5);
            }
        }
    }

    #[test]
    fn invalid_rotor_keys_are_rejected() {
        let half = std::f32::consts::FRAC_1_SQRT_2;
        for rotor in [
            Rotor4 {
                xy: 0.5,
                ..Rotor4::IDENTITY
            },
            Rotor4 {
                xy: f32::NAN,
                ..Rotor4::IDENTITY
            },
            Rotor4 {
                xy: f32::INFINITY,
                ..Rotor4::IDENTITY
            },
            Rotor4 {
                s: half,
                xyzw: half,
                ..Rotor4::IDENTITY
            },
            Rotor4 {
                s: 0.0,
                xy: half,
                zw: half,
                ..Rotor4::IDENTITY
            },
        ] {
            let timeline = Timeline {
                fps: 60,
                frames: 2,
                w_slice: None,
                bodies: vec![BodyTrack {
                    name: "row".to_owned(),
                    position: None,
                    orientation: Some(Track::new().key(0.0, rotor, Ease::Linear)),
                }],
            };
            assert!(matches!(
                Director::new(timeline).unwrap_err(),
                TimelineError::NonUnitRotor { index: 0, .. }
            ));
        }
    }

    #[test]
    fn invalid_timelines_are_rejected() {
        let base = || Timeline {
            fps: 60,
            frames: 60,
            w_slice: None,
            bodies: Vec::new(),
        };

        let mut zero_fps = base();
        zero_fps.fps = 0;
        assert!(matches!(
            Director::new(zero_fps).unwrap_err(),
            TimelineError::ZeroFps
        ));

        let mut zero_frames = base();
        zero_frames.frames = 0;
        assert!(matches!(
            Director::new(zero_frames).unwrap_err(),
            TimelineError::ZeroFrames
        ));

        let mut empty = base();
        empty.w_slice = Some(Track::new());
        assert!(matches!(
            Director::new(empty).unwrap_err(),
            TimelineError::EmptyTrack { .. }
        ));

        for time in [0.5, 1.0] {
            let mut unordered = base();
            unordered.w_slice = Some(Track::new().key(1.0, 0.0, Ease::Linear).key(
                time,
                1.0,
                Ease::Linear,
            ));
            assert!(matches!(
                Director::new(unordered).unwrap_err(),
                TimelineError::KeyOrder { index: 1, .. }
            ));
        }

        for time in [-1.0, f32::NAN, f32::INFINITY] {
            let mut invalid = base();
            invalid.w_slice = Some(Track::new().key(time, 0.0, Ease::Linear));
            assert!(matches!(
                Director::new(invalid).unwrap_err(),
                TimelineError::KeyTime { index: 0, .. }
            ));
        }

        let mut duplicate = base();
        duplicate.bodies = vec![
            BodyTrack {
                name: "row".to_owned(),
                position: None,
                orientation: None,
            },
            BodyTrack {
                name: "row".to_owned(),
                position: None,
                orientation: None,
            },
        ];
        assert!(matches!(
            Director::new(duplicate).unwrap_err(),
            TimelineError::DuplicateBody { .. }
        ));
    }

    #[test]
    fn ron_defaults_load_into_directed_channels() {
        let text = r#"(
            fps: 30,
            frames: 90,
            w_slice: Some([
                (t: 0.0, value: -2.0),
                (t: 3.0, value: 2.0, ease: InOutCubic),
            ]),
            bodies: [
                (
                    name: "letter_l",
                    position: Some([
                        (t: 0.0, value: (0.0, 0.0, 0.0, -4.0)),
                        (t: 2.0, value: (0.0, 0.0, 0.0, 0.0), ease: OutCubic),
                    ]),
                    orientation: Some([
                        (t: 0.0, value: (s: 1.0, xy: 0.0, xz: 0.0, xw: 0.0,
                                         yz: 0.0, yw: 0.0, zw: 0.0, xyzw: 0.0)),
                    ]),
                ),
            ],
        )"#;
        let director = Director::from_ron(text).unwrap();
        assert_eq!(director.bodies().collect::<Vec<_>>(), ["letter_l"]);
        assert_eq!(director.w_slice(), Drive::Directed(-2.0));
        assert_eq!(
            director.position("letter_l"),
            Drive::Directed(Vec4::new(0.0, 0.0, 0.0, -4.0))
        );
        assert_eq!(
            director.orientation("letter_l"),
            Drive::Directed(Rotor4::IDENTITY)
        );
    }

    #[test]
    fn missing_channels_remain_host_driven() {
        let timeline = Timeline {
            fps: 60,
            frames: 60,
            w_slice: None,
            bodies: vec![BodyTrack {
                name: "row".to_owned(),
                position: None,
                orientation: Some(Track::new().key(0.0, Rotor4::IDENTITY, Ease::Linear)),
            }],
        };
        let director = Director::new(timeline).unwrap();
        assert_eq!(director.w_slice(), Drive::Host);
        assert_eq!(director.position("row"), Drive::Host);
        assert_eq!(director.orientation("unnamed"), Drive::Host);
        assert!(matches!(director.orientation("row"), Drive::Directed(_)));
    }
}
