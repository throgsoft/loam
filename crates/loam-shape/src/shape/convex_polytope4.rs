use std::hash::Hasher;

use glam::Vec4;

pub(super) fn hash<H: Hasher>(hasher: &mut H, vertices: &[Vec4]) {
    for v in vertices {
        super::hash_f32s(hasher, v.as_ref());
    }
}

pub fn bounding_radius(vertices: &[Vec4]) -> f32 {
    vertices
        .iter()
        .map(|v| v.length_squared())
        .fold(0.0_f32, f32::max)
        .sqrt()
}

pub(super) fn is_well_formed(vertices: &[Vec4]) -> bool {
    !vertices.is_empty() && vertices.iter().all(|v| v.is_finite())
}

/// Panics on an empty vertex list, which [`super::Shape::is_well_formed`] rejects.
pub fn support(vertices: &[Vec4], direction: Vec4) -> Vec4 {
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
