use std::hash::Hasher;

use glam::Vec3;

pub(super) fn hash<H: Hasher>(hasher: &mut H, half_extents: Vec3) {
    super::hash_f32s(hasher, half_extents.as_ref());
}

pub(super) fn bounding_radius(half_extents: Vec3) -> f32 {
    half_extents.length()
}

pub(super) fn is_well_formed(half_extents: Vec3) -> bool {
    half_extents.is_finite() && half_extents.cmpge(Vec3::ZERO).all()
}

/// Chart coordinates, exact only in a flat chart.
pub fn distance(p: Vec3, half_extents: Vec3) -> f32 {
    let q = p.abs() - half_extents;
    q.max(Vec3::ZERO).length() + q.x.max(q.y.max(q.z)).min(0.0)
}
