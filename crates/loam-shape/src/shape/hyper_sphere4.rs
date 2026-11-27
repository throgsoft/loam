use std::hash::Hasher;

use glam::Vec4;

pub(super) fn hash<H: Hasher>(hasher: &mut H, center: Vec4, radius: f32) {
    super::hash_f32s(hasher, center.as_ref());
    hasher.write_u32(radius.to_bits());
}

pub(super) fn bounding_radius(radius: f32) -> f32 {
    radius
}

pub(super) fn is_well_formed(center: Vec4, radius: f32) -> bool {
    center.is_finite() && radius.is_finite() && radius >= 0.0
}

pub fn support(center: Vec4, radius: f32, direction: Vec4) -> Vec4 {
    let d = direction.length_squared();
    let dir = if d > 1e-12 {
        direction / d.sqrt()
    } else {
        Vec4::Y
    };
    center + dir * radius
}

pub fn distance(p: Vec4, center: Vec4, radius: f32) -> f32 {
    (p - center).length() - radius
}
