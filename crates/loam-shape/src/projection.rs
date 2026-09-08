use glam::Vec4;

use crate::polytope::Polytope4;

/// Cell frame and eye in the polytope's unit-circumradius coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SchlegelParams {
    pub cell_index: u32,
    pub cell_normal: Vec4,
    pub cell_basis: [Vec4; 3],
    pub cell_offset: f32,
    pub viewpoint_distance: f32,
}

impl SchlegelParams {
    /// Returns `None` for an invalid cell or clearance that cannot place a finite eye outside the polytope.
    pub fn new(polytope: Polytope4, cell_index: u32, eye_margin: f32) -> Option<Self> {
        if !eye_margin.is_finite() || eye_margin <= 0.0 {
            return None;
        }
        let topology = polytope.topology();
        let cell = topology.cells.get(cell_index as usize)?;
        let anchor = topology.vertices[*cell.first()? as usize];
        let centroid = cell
            .iter()
            .map(|&index| topology.vertices[index as usize])
            .sum::<Vec4>()
            / cell.len() as f32;
        let cell_offset = centroid.length();
        let cell_normal = centroid.try_normalize()?;
        let mut cell_basis = [Vec4::ZERO; 3];
        let mut count = 0;

        for &index in cell.iter().skip(1) {
            let delta = topology.vertices[index as usize] - anchor;
            let mut axis = delta - delta.dot(cell_normal) * cell_normal;
            for &previous in cell_basis.iter().take(count) {
                axis -= axis.dot(previous) * previous;
            }
            let length = axis.length();
            if length > 1e-6 {
                cell_basis[count] = axis / length;
                count += 1;
                if count == cell_basis.len() {
                    break;
                }
            }
        }
        if count != cell_basis.len() {
            return None;
        }

        // Coxeter, Regular Polytopes, 3rd ed., ch. 13.
        let support = topology
            .vertices
            .iter()
            .map(|vertex| cell_normal.dot(*vertex))
            .fold(f32::NEG_INFINITY, f32::max);
        let viewpoint_distance = support + eye_margin;
        if !viewpoint_distance.is_finite() || viewpoint_distance <= support {
            return None;
        }
        Some(Self {
            cell_index,
            cell_normal,
            cell_basis,
            cell_offset,
            viewpoint_distance,
        })
    }
}
