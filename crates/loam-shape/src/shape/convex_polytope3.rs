use std::hash::Hasher;

use glam::Vec3;

pub(super) fn hash<H: Hasher>(hasher: &mut H, vertices: &[Vec3]) {
    for v in vertices {
        super::hash_f32s(hasher, v.as_ref());
    }
}

pub fn bounding_radius(vertices: &[Vec3]) -> f32 {
    vertices
        .iter()
        .map(|v| v.length_squared())
        .fold(0.0_f32, f32::max)
        .sqrt()
}

pub(super) fn is_well_formed(vertices: &[Vec3]) -> bool {
    !vertices.is_empty() && vertices.iter().all(|v| v.is_finite())
}

/// Panics on an empty vertex list, which [`super::Shape::is_well_formed`] rejects.
pub fn support(vertices: &[Vec3], direction: Vec3) -> Vec3 {
    let mut best = vertices[0];
    let mut best_d = best.dot(direction);
    for &v in &vertices[1..] {
        let d = v.dot(direction);
        if d > best_d {
            best_d = d;
            best = v;
        }
    }
    best
}
