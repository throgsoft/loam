use glam::Vec3;
use loam_camera::{CameraController, FirstPersonController};
use loam_input::FrameInput;
use loam_math::EuclideanR3;

use crate::Camera;

const DEFAULT_SPEED: f32 = 4.5;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CursorMode {
    /// MMO-style: cursor released while Alt is held, re-grabbed on release.
    Hold,
    /// FPS sticky-modifier: Alt press flips the grab, release is ignored.
    #[default]
    Toggle,
}

impl CursorMode {
    pub fn from_token(s: &str) -> Option<Self> {
        match s {
            "hold" => Some(Self::Hold),
            "toggle" => Some(Self::Toggle),
            _ => None,
        }
    }

    pub fn token(self) -> &'static str {
        match self {
            Self::Hold => "hold",
            Self::Toggle => "toggle",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Freecam {
    pub controller: FirstPersonController,
    pub position: Vec3,
    /// Units per second; look sensitivity lives on `controller`.
    pub speed: f32,
    active: bool,
    cursor_grabbed: bool,
    cursor_mode: CursorMode,
}

impl Default for Freecam {
    fn default() -> Self {
        Self::new()
    }
}

impl Freecam {
    pub fn new() -> Self {
        Self {
            controller: FirstPersonController::new(0.0, 0.0),
            position: Vec3::ZERO,
            speed: DEFAULT_SPEED,
            active: false,
            cursor_grabbed: false,
            cursor_mode: CursorMode::default(),
        }
    }

    pub fn with_speed(mut self, speed: f32) -> Self {
        self.speed = speed;
        self
    }

    pub fn with_cursor_mode(mut self, mode: CursorMode) -> Self {
        self.cursor_mode = mode;
        self
    }

    pub fn cursor_mode(&self) -> CursorMode {
        self.cursor_mode
    }

    /// Does not touch the current grab, only how future Alt events are read.
    pub fn set_cursor_mode(&mut self, mode: CursorMode) {
        self.cursor_mode = mode;
    }

    pub fn active(&self) -> bool {
        self.active
    }

    /// Always false when inactive.
    pub fn cursor_grabbed(&self) -> bool {
        self.cursor_grabbed
    }

    pub fn set_active(
        &mut self,
        active: bool,
        camera: &Camera<EuclideanR3>,
        runtime: &crate::Runtime,
    ) {
        if active == self.active {
            return;
        }
        self.active = active;
        if active {
            self.position = camera.position;
            self.controller.yaw = (-camera.forward.x).atan2(-camera.forward.z);
            self.controller.pitch = camera.forward.y.clamp(-1.0, 1.0).asin();
            self.cursor_grabbed = true;
            self.controller.use_raw_delta = true;
            runtime.request_grab();
        } else {
            self.cursor_grabbed = false;
            self.controller.use_raw_delta = false;
            runtime.request_release();
        }
    }

    pub fn on_alt(&mut self, pressed: bool, runtime: &crate::Runtime) {
        if !self.active {
            return;
        }
        let target_grabbed = match self.cursor_mode {
            CursorMode::Hold => !pressed,
            CursorMode::Toggle => {
                if !pressed {
                    return;
                }
                !self.cursor_grabbed
            }
        };
        if target_grabbed == self.cursor_grabbed {
            return;
        }
        self.cursor_grabbed = target_grabbed;
        self.controller.use_raw_delta = target_grabbed;
        if target_grabbed {
            runtime.request_grab();
        } else {
            runtime.request_release();
            runtime.request_warp_to_center();
        }
    }

    /// No-op when inactive; look freezes when the cursor is released.
    pub fn advance(
        &mut self,
        input: FrameInput,
        camera: &mut Camera<EuclideanR3>,
        dt: f32,
        runtime: &crate::Runtime,
    ) {
        if !self.active {
            return;
        }
        if self.cursor_grabbed && runtime.cursor_state().grab != crate::cursor::GrabMode::None {
            self.controller.advance(input, camera, &EuclideanR3, dt);
        }
        let mut delta = camera.forward * input.move_forward
            + camera.right * input.move_right
            + Vec3::Y * input.move_up;
        if delta.length_squared() > 1e-6 {
            delta = delta.normalize();
            self.position += delta * self.speed * dt;
            camera.position = self.position;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec2;

    #[test]
    fn activation_preserves_view_and_does_not_reseed_motion() {
        let runtime = crate::Runtime::default();
        let mut camera =
            Camera::looking_at(Vec3::new(3.0, 2.0, 7.0), Vec3::ZERO, Vec3::Y, &EuclideanR3);
        let before = camera;
        let mut freecam = Freecam::new().with_speed(2.0);
        freecam.set_active(true, &camera, &runtime);
        freecam.advance(FrameInput::default(), &mut camera, 0.5, &runtime);
        assert!((camera.forward - before.forward).length() < 1e-6);
        assert_eq!(camera.position, before.position);
        freecam.advance(
            FrameInput {
                move_forward: 1.0,
                ..Default::default()
            },
            &mut camera,
            0.5,
            &runtime,
        );
        assert!((camera.position - before.position - before.forward).length() < 1e-6);
        freecam.set_active(true, &before, &runtime);
        assert_eq!(freecam.position, camera.position);
    }

    #[test]
    fn platform_grab_controls_look_without_blocking_movement() {
        let runtime = crate::Runtime::default();
        let mut camera = Camera::at_origin();
        let mut freecam = Freecam::new().with_speed(2.0);
        freecam.set_active(true, &camera, &runtime);
        let look = FrameInput {
            mouse_raw_delta: Vec2::new(100.0, 50.0),
            ..Default::default()
        };
        freecam.advance(look, &mut camera, 0.5, &runtime);
        assert_eq!(camera.forward, -Vec3::Z);
        runtime.mark_cursor_applied(crate::cursor::GrabMode::Locked, false);
        freecam.advance(look, &mut camera, 0.5, &runtime);
        assert!((camera.forward + Vec3::Z).length() > 0.01);
        let before = camera;
        runtime.mark_cursor_applied(crate::cursor::GrabMode::None, true);
        freecam.advance(
            FrameInput {
                move_forward: 1.0,
                ..look
            },
            &mut camera,
            0.5,
            &runtime,
        );
        assert_eq!(camera.forward, before.forward);
        assert!((camera.position - before.position - before.forward).length() < 1e-6);
        freecam.set_active(false, &camera, &runtime);
        let before = camera;
        freecam.advance(
            FrameInput {
                move_forward: 1.0,
                ..look
            },
            &mut camera,
            0.5,
            &runtime,
        );
        assert_eq!(camera.position, before.position);
    }

    #[test]
    fn alt_release_only_regrabs_in_hold_mode() {
        let runtime = crate::Runtime::default();
        for mode in [CursorMode::Hold, CursorMode::Toggle] {
            let mut freecam = Freecam::new().with_cursor_mode(mode);
            freecam.set_active(true, &Camera::at_origin(), &runtime);
            freecam.on_alt(true, &runtime);
            assert!(!freecam.cursor_grabbed());
            freecam.on_alt(false, &runtime);
            assert_eq!(freecam.cursor_grabbed(), mode == CursorMode::Hold);
            freecam.set_active(false, &Camera::at_origin(), &runtime);
            freecam.on_alt(false, &runtime);
            assert!(!freecam.cursor_grabbed());
        }
    }
}
