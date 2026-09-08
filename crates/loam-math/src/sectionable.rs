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
    fn r4_edge_section_midpoint() {
        let slice = WPlane::new(0.0);
        let p0 = Vec4::new(1.0, 2.0, 3.0, -1.0);
        let p1 = Vec4::new(5.0, 6.0, 7.0, 1.0);
        let (t, p3) = slice.intersect_edge(p0, p1).unwrap();
        assert!((t - 0.5).abs() < 1e-6);
        assert_eq!(p3, Vec3::new(3.0, 4.0, 5.0));
    }

    #[test]
    fn r4_edge_section_no_crossing_returns_none() {
        let slice = WPlane::new(0.0);
        let p0 = Vec4::new(0.0, 0.0, 0.0, 0.5);
        let p1 = Vec4::new(0.0, 0.0, 0.0, 1.5);
        assert!(slice.intersect_edge(p0, p1).is_none());
    }

    #[test]
    fn r4_edge_section_parallel_edge_returns_none() {
        let slice = WPlane::new(0.0);
        let p0 = Vec4::new(0.0, 0.0, 0.0, 0.0);
        let p1 = Vec4::new(1.0, 1.0, 1.0, 1e-7);
        assert!(slice.intersect_edge(p0, p1).is_none());
    }

    #[test]
    fn r4_edge_section_endpoint_on_slice_returns_t_zero() {
        let slice = WPlane::new(0.0);
        let p0 = Vec4::new(2.0, 2.0, 2.0, 0.0);
        let p1 = Vec4::new(5.0, 5.0, 5.0, 1.0);
        let (t, p3) = slice.intersect_edge(p0, p1).unwrap();
        assert!(t.abs() < 1e-6);
        assert_eq!(p3, Vec3::new(2.0, 2.0, 2.0));
    }
}
