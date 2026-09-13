use loam_math::{EuclideanR3, Space};

use crate::Eye;

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
