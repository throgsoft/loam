use std::hash::Hasher;

use glam::Vec3;

pub(super) fn hash<H: Hasher>(hasher: &mut H, normal: Vec3, offset: f32) {
    super::hash_f32s(hasher, normal.as_ref());
    hasher.write_u32(offset.to_bits());
}

pub(super) fn bounding_radius() -> f32 {
    f32::INFINITY
}

pub(super) fn is_well_formed(normal: Vec3, offset: f32) -> bool {
    normal.is_finite() && offset.is_finite() && (normal.length_squared() - 1.0).abs() < 1e-5
}

/// Chart coordinates, so it is a true distance only in a flat chart.
pub fn distance(p: Vec3, normal: Vec3, offset: f32) -> f32 {
    p.dot(normal) - offset
}
