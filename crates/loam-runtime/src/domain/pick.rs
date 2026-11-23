use loam_shape::polytope::Polytope4;

use super::{DomainSpace, Pose};
use crate::view::{DomainRay, Vec4};

// Haines, "Fast Ray-Convex Polyhedron Intersection", Graphics Gems II (1991).
pub(super) fn polytope_entry<S: DomainSpace>(
    space: &S,
    ray: &DomainRay<S>,
    pose: &Pose<S>,
    polytope: Polytope4,
    scale: f32,
) -> Option<f32> {
    let canonical = |point| {
        let local = space.local(pose, point).ok()?;
        Some(Vec4::from_array(space.chart_point(local).coordinates) / scale)
    };
    let origin = canonical(ray.origin)?;
    let direction = canonical(space.exp(ray.origin, ray.direction))? - origin;
    let topology = polytope.topology();
    let mut entry = 0.0f32;
    let mut exit = f32::INFINITY;
    for cell in topology.cells {
        let centroid = cell
            .iter()
            .map(|&index| topology.vertices[index as usize])
            .sum::<Vec4>()
            / cell.len() as f32;
        let offset = centroid.length();
        let normal = centroid / offset;
        let along = normal.dot(direction);
        let gap = offset - normal.dot(origin);
        if along == 0.0 {
            if gap < 0.0 {
                return None;
            }
            continue;
        }
        let t = gap / along;
        if along < 0.0 {
            entry = entry.max(t);
        } else {
            exit = exit.min(t);
        }
        if entry > exit {
            return None;
        }
    }
    Some(entry)
}
