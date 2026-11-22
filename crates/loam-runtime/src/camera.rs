use loam_math::{EuclideanR3, Space};

use crate::input::{ActionId, Input, PointerButton};
use crate::view::{Eye, Views};

type Vec3 = <EuclideanR3 as Space>::Vector;

/// The dispatch entry that `Session::new` registers ahead of every program system to steer the root rig.
pub const CAMERA_RIG: &str = "camera rig";
/// The simulation entry `Session::new` registers to move a free rig once per tick.
pub const CAMERA_TRAVEL: &str = "camera travel";

pub const MOVE_FORWARD: ActionId = ActionId("move forward");
pub const MOVE_BACKWARD: ActionId = ActionId("move backward");
pub const MOVE_LEFT: ActionId = ActionId("move left");
pub const MOVE_RIGHT: ActionId = ActionId("move right");
pub const MOVE_UP: ActionId = ActionId("move up");
pub const MOVE_DOWN: ActionId = ActionId("move down");

/// How the engine steers the root image's eye from input.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Rig {
    /// The program owns the eye and the engine leaves it alone.
    #[default]
    Manual,
    Orbit(Orbit),
    /// Moves only when the program binds keys to the `MOVE_*` actions and only while the cursor is captured.
    Free(FreeCamera),
}

impl Rig {
    /// `None` while the rig leaves the cursor free.
    pub fn cursor(&self) -> Option<CursorPolicy> {
        match self {
            Rig::Free(free) => Some(free.cursor),
            Rig::Manual | Rig::Orbit(_) => None,
        }
    }

    pub(crate) fn eye(&self) -> Option<Eye> {
        match self {
            Rig::Manual => None,
            Rig::Orbit(orbit) => Some(orbit.eye()),
            Rig::Free(free) => Some(free.eye),
        }
    }

    fn steer(&mut self, input: &Input) -> Option<Eye> {
        match self {
            Rig::Manual => {}
            Rig::Orbit(orbit) => orbit.apply(input),
            Rig::Free(free) => {
                if input.cursor_locked {
                    free.look(input.look);
                }
            }
        }
        self.eye()
    }

    fn travel(&mut self, input: &Input, dt: f32) -> Option<Eye> {
        let Rig::Free(free) = self else {
            return None;
        };
        if !input.cursor_locked {
            return None;
        }
        let axes = [
            input.axis(MOVE_LEFT, MOVE_RIGHT),
            input.axis(MOVE_DOWN, MOVE_UP),
            input.axis(MOVE_BACKWARD, MOVE_FORWARD),
        ];
        free.travel(axes, free.speed * dt);
        Some(free.eye)
    }
}

pub(crate) fn steer(views: &mut Views, input: &Input) {
    let root = views.root_mut();
    if let Some(eye) = root.rig.steer(input) {
        root.look(eye);
    }
}

pub(crate) fn travel(views: &mut Views, input: &Input, dt: f32) {
    let root = views.root_mut();
    if let Some(eye) = root.rig.travel(input, dt) {
        root.look(eye);
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CursorPolicy {
    /// The cursor stays captured and is free only while Alt is held down.
    Hold,
    /// The cursor starts captured and each Alt press frees or recaptures it.
    #[default]
    Toggle,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FreeCamera {
    pub eye: Eye,
    /// Image-space units per second.
    pub speed: f32,
    /// How the cursor is captured while this camera holds the eye.
    pub cursor: CursorPolicy,
}

impl Default for FreeCamera {
    fn default() -> Self {
        Self {
            eye: Eye::default(),
            speed: FREE_SPEED,
            cursor: CursorPolicy::default(),
        }
    }
}

impl FreeCamera {
    pub fn look(&mut self, delta: [f32; 2]) {
        let forward = Vec3::from(self.eye.forward);
        let yaw = (-forward.x).atan2(-forward.z) - delta[0] * 0.002;
        let pitch = (forward.y.clamp(-1.0, 1.0).asin() + delta[1] * 0.002).clamp(-1.5, 1.5);
        let (yaw_sin, yaw_cos) = yaw.sin_cos();
        let (pitch_sin, pitch_cos) = pitch.sin_cos();
        let forward = Vec3::new(-yaw_sin * pitch_cos, pitch_sin, -yaw_cos * pitch_cos);
        let right = Vec3::new(yaw_cos, 0.0, -yaw_sin);
        self.eye.forward = forward.to_array();
        self.eye.right = right.to_array();
        self.eye.up = right.cross(forward).to_array();
    }

    pub fn travel(&mut self, axes: [f32; 3], distance: f32) {
        let direction = Vec3::from(self.eye.right) * axes[0]
            + Vec3::Y * axes[1]
            + Vec3::from(self.eye.forward) * axes[2];
        if let Some(direction) = direction.try_normalize() {
            self.eye.position = (Vec3::from(self.eye.position) + direction * distance).to_array();
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Orbit {
    pub target: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
}

impl Default for Orbit {
    fn default() -> Self {
        Self::around([0.0; 3], DEFAULT_DISTANCE)
    }
}

impl Orbit {
    pub fn around(target: [f32; 3], distance: f32) -> Self {
        Self {
            target,
            yaw: 0.0,
            pitch: 0.0,
            distance,
        }
    }

    pub fn apply(&mut self, input: &Input) {
        self.drag(input.drag(PointerButton::Secondary));
        self.zoom(input.scroll[1]);
    }

    pub fn drag(&mut self, delta: [f32; 2]) {
        self.yaw -= delta[0] * ORBIT_GAIN;
        self.pitch = (self.pitch + delta[1] * ORBIT_GAIN).clamp(-PITCH_LIMIT, PITCH_LIMIT);
    }

    pub fn zoom(&mut self, lines: f32) {
        self.distance =
            (self.distance * (-lines * ZOOM_GAIN).exp()).clamp(MIN_DISTANCE, MAX_DISTANCE);
    }

    pub fn eye(&self) -> Eye {
        let (yaw_sin, yaw_cos) = self.yaw.sin_cos();
        let (pitch_sin, pitch_cos) = self.pitch.sin_cos();
        let offset = Vec3::new(yaw_sin * pitch_cos, -pitch_sin, yaw_cos * pitch_cos);
        Eye::looking_at(
            (Vec3::from(self.target) + offset * self.distance).to_array(),
            self.target,
            [0.0, 1.0, 0.0],
        )
    }
}

const DEFAULT_DISTANCE: f32 = 6.0;
const ORBIT_GAIN: f32 = 0.006;
const PITCH_LIMIT: f32 = 1.5;
const ZOOM_GAIN: f32 = 0.12;
const MIN_DISTANCE: f32 = 1.5;
const MAX_DISTANCE: f32 = 20.0;
// A walking pace that crosses a scene a few units wide in about a second.
const FREE_SPEED: f32 = 4.5;

#[cfg(test)]
mod tests {
    use std::f32::consts::FRAC_PI_2;

    use super::*;

    #[test]
    fn freecam_keeps_the_activation_heading_and_does_not_accelerate_diagonally() {
        let mut camera = FreeCamera {
            eye: Eye::looking_at([2.0, 1.0, 3.0], [1.0, 1.5, 2.0], [0.0, 1.0, 0.0]),
            ..FreeCamera::default()
        };
        let before = camera.eye;
        camera.look([0.0; 2]);
        assert!((Vec3::from(camera.eye.forward) - Vec3::from(before.forward)).length() < 1e-6);
        camera.look([0.1, 0.1]);
        assert!(camera.eye.forward[0] > before.forward[0]);
        assert!(camera.eye.forward[1] > before.forward[1]);
        camera.travel([1.0, 0.0, 1.0], 2.0);
        assert!(
            ((Vec3::from(camera.eye.position) - Vec3::from(before.position)).length() - 2.0).abs()
                < 1e-6
        );
    }

    #[test]
    fn looking_at_mirrors_the_eye_basis_or_the_orbit_sits_on_the_wrong_side_of_its_target() {
        let close = |got: [f32; 3], want: [f32; 3]| {
            assert!(
                got.iter().zip(want).all(|(g, w)| (g - w).abs() <= 1e-6),
                "{got:?} is not {want:?}"
            );
        };

        let eye = Eye::looking_at([0.0, 0.0, 5.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        close(eye.forward, [0.0, 0.0, -1.0]);
        close(eye.right, [1.0, 0.0, 0.0]);
        close(eye.up, [0.0, 1.0, 0.0]);

        let orbit = Orbit {
            target: [1.0, 2.0, 3.0],
            yaw: FRAC_PI_2,
            pitch: 0.0,
            distance: 4.0,
        };
        let turned = orbit.eye();
        close(turned.position, [5.0, 2.0, 3.0]);
        close(turned.forward, [-1.0, 0.0, 0.0]);
        close(turned.right, [0.0, 0.0, -1.0]);
        close(turned.up, [0.0, 1.0, 0.0]);

        let pitched = Orbit {
            target: [1.0, 2.0, 3.0],
            yaw: 0.0,
            pitch: -FRAC_PI_2,
            distance: 4.0,
        }
        .eye();
        close(pitched.position, [1.0, 6.0, 3.0]);
    }
}
