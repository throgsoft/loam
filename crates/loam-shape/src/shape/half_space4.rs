use std::hash::Hasher;

use glam::Vec4;

pub(super) fn hash<H: Hasher>(hasher: &mut H, normal: Vec4, offset: f32) {
    super::hash_f32s(hasher, normal.as_ref());
    hasher.write_u32(offset.to_bits());
}

pub(super) fn bounding_radius() -> f32 {
    f32::INFINITY
}

pub(super) fn is_well_formed(normal: Vec4, offset: f32) -> bool {
    normal.is_finite() && offset.is_finite() && (normal.length_squared() - 1.0).abs() < 1e-5
}

pub fn distance(p: Vec4, normal: Vec4, offset: f32) -> f32 {
    p.dot(normal) - offset
}
