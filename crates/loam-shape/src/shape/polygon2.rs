use std::hash::Hasher;

use glam::Vec2;

pub(super) fn hash<H: Hasher>(hasher: &mut H, vertices: &[Vec2]) {
    for v in vertices {
        super::hash_f32s(hasher, v.as_ref());
    }
}

pub(super) fn bounding_radius(vertices: &[Vec2]) -> f32 {
    vertices
        .iter()
        .map(|v| v.length_squared())
        .fold(0.0_f32, f32::max)
        .sqrt()
}

pub(super) fn is_well_formed(vertices: &[Vec2]) -> bool {
    if vertices.len() < 3 || !vertices.iter().all(|v| v.is_finite()) {
        return false;
    }
    let mut has_area = false;
    for i in 0..vertices.len() {
        let origin = vertices[i];
        let edge = vertices[(i + 1) % vertices.len()] - origin;
        for &vertex in vertices {
            let offset = vertex - origin;
            let cross = edge.perp_dot(offset);
            if !cross.is_finite() || cross < 0.0 {
                return false;
            }
            has_area |= cross > 0.0;
        }
    }
    has_area
}
