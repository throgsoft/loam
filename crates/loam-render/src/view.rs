//! Root image space: the eye at the origin looking down -Z. Depth is
//! `near / -z`, reversed-Z in `Depth32Float`, cleared to zero and compared
//! `GreaterEqual`; every pass that writes depth writes that one quantity.

use glam::{Mat4, Vec3};
use loam_math::hyperbolic::{hyperboloid_to_klein, poincare_to_hyperboloid, poincare_to_klein};
use loam_math::{Iso3, Iso3H};
use wgpu::{CompareFunction, TextureFormat};

pub const DEPTH_FORMAT: TextureFormat = TextureFormat::Depth32Float;
pub const DEPTH_COMPARE: CompareFunction = CompareFunction::GreaterEqual;
pub const DEPTH_CLEAR: f32 = 0.0;

/// H³ hits at least `H3_DEPTH_SEPARATION` apart stay ordered within this hyperbolic distance of an eye at most `H3_EYE_CHART_REACH` from the chart origin.
pub const H3_DEPTH_ENVELOPE: f32 = 6.0;
pub const H3_DEPTH_SEPARATION: f32 = 0.05;
pub const H3_EYE_CHART_REACH: f32 = 1.0;

pub fn root_projection(fov_y: f32, aspect: f32, near: f32) -> Mat4 {
    Mat4::perspective_infinite_reverse_rh(fov_y, aspect, near)
}

pub fn projective_depth(image: Vec3, near: f32) -> f32 {
    near / -image.z
}

/// Places an R³ image space so that `eye` sits at the root origin.
pub fn eye_relative(eye: Iso3) -> Mat4 {
    Mat4::from_rotation_translation(eye.rotation, eye.translation).inverse()
}

/// Klein image of a Poincaré point, with `eye_inverse` composed first on the Lorentz embedding.
pub fn h3_image_of(eye_inverse: &Iso3H, p: Vec3) -> Vec3 {
    hyperboloid_to_klein(eye_inverse.matrix * poincare_to_hyperboloid(p))
}

/// `samples` chords of a Poincaré-chart segment in the Klein image; [`klein_tessellation_error`] bounds the gap.
pub fn klein_tessellate_segment(p0: Vec3, p1: Vec3, samples: usize, mut emit: impl FnMut(Vec3)) {
    let samples = samples.max(1);
    for i in 0..=samples {
        emit(poincare_to_klein(p0.lerp(p1, i as f32 / samples as f32)));
    }
}

// Chord remainder (1/n)²/8 · max|k''|, with |k''| ≤ 20 |p1 - p0|² max(|p0|, |p1|) for k = 2p / (1 + |p|²).
pub fn klein_tessellation_error(p0: Vec3, p1: Vec3, samples: usize) -> f32 {
    let reach = p0.length().max(p1.length());
    2.5 * (p1 - p0).length_squared() * reach / (samples.max(1) as f32).powi(2)
}
