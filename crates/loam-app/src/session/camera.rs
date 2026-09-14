use loam_math::{EuclideanR3, Space};
use loam_runtime::{Ctx, Eye, Input, Phase, PointerButton, Session, Stores};

type Vec3 = <EuclideanR3 as Space>::Vector;

/// A free camera in the root image space.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FreeCamera {
    pub eye: Eye,
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

/// Orbits a target using the delta convention in `Pointer`.
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

pub fn orbit<A: Stores>(
    session: &mut Session<A>,
    camera: impl Fn(&mut A) -> &mut Orbit + Send + 'static,
) {
    session.system(Phase::Dispatch, "orbit", move |ctx: Ctx<'_, A>| {
        let orbit = camera(ctx.app);
        orbit.apply(ctx.input);
        let mut eye = orbit.eye();
        let root = ctx.views.root_mut();
        eye.aspect = root.eye.aspect;
        root.eye = eye;
        Ok(())
    });
}

#[cfg(test)]
mod tests {
    use std::f32::consts::FRAC_PI_2;

    use super::*;

    loam_runtime::stores! {
        #[derive(Default)]
        pub struct Viewer {
            camera: Value<Orbit>,
        }
    }

    #[test]
    fn a_reset_brings_the_orbit_camera_back_with_the_scene() {
        use loam_runtime::{Input, Pointer, PointerButton, PointerPhase, Session, SimConfig};

        let mut session = Session::new(Viewer::default(), SimConfig::default());
        session.app.camera.set(Orbit::around([0.0; 3], 5.0));
        orbit(&mut session, |app| app.camera.get_mut());
        session.set_initial().unwrap();
        session.boundary(Input::default()).unwrap();
        let root = session.views().root();
        let boot = session.views().get(root).unwrap().eye.position;

        let drag = Input {
            pointers: vec![Pointer {
                id: 0,
                button: Some(PointerButton::Secondary),
                ndc: [0.0; 2],
                delta: [40.0, 0.0],
                phase: PointerPhase::Moved,
                time: 0.0,
            }],
            ..Input::default()
        };
        session.boundary(drag).unwrap();
        let turned = session.views().get(root).unwrap().eye.position;
        assert_ne!(turned, boot, "a secondary drag left the orbit where it was");

        session.reset().unwrap();
        session.boundary(Input::default()).unwrap();
        assert_eq!(
            session.views().get(root).unwrap().eye.position,
            boot,
            "the reset did not bring the camera back"
        );
    }

    #[test]
    fn freecam_keeps_the_activation_heading_and_does_not_accelerate_diagonally() {
        let mut camera = FreeCamera {
            eye: Eye::looking_at([2.0, 1.0, 3.0], [1.0, 1.5, 2.0], [0.0, 1.0, 0.0]),
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

    #[test]
    fn orbit_zoom_runs_past_its_distance_limits() {
        let mut orbit = Orbit::around([0.0; 3], 5.0);
        orbit.zoom(100.0);
        assert_eq!(orbit.distance, 1.5);
        orbit.zoom(-100.0);
        assert_eq!(orbit.distance, 20.0);
    }
}
