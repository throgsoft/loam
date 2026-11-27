use std::hash::Hasher;

use glam::Vec3;
use loam_math::Space;

pub(super) fn hash<H: Hasher>(hasher: &mut H, center: Vec3, radius: f32) {
    super::hash_f32s(hasher, center.as_ref());
    hasher.write_u32(radius.to_bits());
}

pub(super) fn bounding_radius(radius: f32) -> f32 {
    radius
}

pub(super) fn is_well_formed(center: Vec3, radius: f32) -> bool {
    center.is_finite() && radius.is_finite() && radius >= 0.0
}

pub fn support(center: Vec3, radius: f32, direction: Vec3) -> Vec3 {
    let dir = direction.normalize_or(Vec3::Y);
    center + dir * radius
}

/// Geodesic, so it keeps its distance contract in a curved space.
pub fn distance<S: Space<Point = Vec3, Vector = Vec3>>(
    space: &S,
    p: Vec3,
    center: Vec3,
    radius: f32,
) -> f32 {
    space.distance(p, center) - radius
}
