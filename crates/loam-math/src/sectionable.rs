use glam::{Vec3, Vec4};

/// Edges below this absolute w separation count as parallel.
pub const EDGE_PARALLEL_EPSILON: f32 = 1e-6;

/// Slices within this distance of a vertex shift by this amount before cap assembly.
pub const SLICE_PERTURBATION_EPSILON: f32 = 1e-5;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WPlane {
    pub w_slice: f32,
}

impl WPlane {
    pub const fn new(w_slice: f32) -> Self {
        Self { w_slice }
    }
}

impl WPlane {
    /// Returns the segment parameter and xyz intersection; near-parallel edges return `None`.
    pub fn intersect_edge(&self, p0: Vec4, p1: Vec4) -> Option<(f32, Vec3)> {
        let dw = p1.w - p0.w;
        if dw.abs() < EDGE_PARALLEL_EPSILON {
            return None;
        }
        let t = (self.w_slice - p0.w) / dw;
        if !(0.0..=1.0).contains(&t) {
            return None;
        }

        let p = p0 + t * (p1 - p0);
        Some((t, Vec3::new(p.x, p.y, p.z)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edge_section_returns_the_crossing_and_refuses_edges_that_miss_the_slice() {
        let slice = WPlane::new(0.0);

        let (t, p3) = slice
            .intersect_edge(
                Vec4::new(1.0, 2.0, 3.0, -1.0),
                Vec4::new(5.0, 6.0, 7.0, 1.0),
            )
            .expect("the edge straddles the slice");
        assert!((t - 0.5).abs() < 1e-6);
        assert_eq!(p3, Vec3::new(3.0, 4.0, 5.0));

        let (t, p3) = slice
            .intersect_edge(Vec4::new(2.0, 2.0, 2.0, 0.0), Vec4::new(5.0, 5.0, 5.0, 1.0))
            .expect("an endpoint on the slice counts as a crossing");
        assert!(t.abs() < 1e-6);
        assert_eq!(p3, Vec3::new(2.0, 2.0, 2.0));

        let one_sided =
            slice.intersect_edge(Vec4::new(0.0, 0.0, 0.0, 0.5), Vec4::new(0.0, 0.0, 0.0, 1.5));
        assert!(one_sided.is_none());

        let parallel = slice.intersect_edge(Vec4::ZERO, Vec4::new(1.0, 1.0, 1.0, 1e-7));
        assert!(parallel.is_none());
    }
}
