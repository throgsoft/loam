use glam::{Vec3, Vec4};
use loam_math::{Rotor, Rotor4, WPlane};
use loam_runtime::SegmentRecord;
use loam_shape::polytope::{polytope_section_perimeter_append, Polytope4, SectionScratch};
use loam_shape::LineMesh;

use crate::consts::BODY_SIZE;

#[derive(Default)]
pub(crate) struct Cutter {
    scratch: SectionScratch,
    mesh: LineMesh<3>,
    rotated: Vec<Vec4>,
    pub(crate) color: [f32; 4],
    pub(crate) width_px: f32,
}

impl Cutter {
    pub(crate) fn new(color: [f32; 4], width_px: f32) -> Self {
        Self {
            color,
            width_px,
            ..Self::default()
        }
    }

    /// The slice is a world hyperplane; the body's own w offsets it, and its rotor turns the vertices before the cut.
    pub(crate) fn cut(
        &mut self,
        polytope: Polytope4,
        rotor: Rotor4,
        position: Vec4,
        slice: f32,
        out: &mut Vec<SegmentRecord>,
    ) {
        let topology = polytope.topology();
        self.rotated.clear();
        self.rotated.extend(
            topology
                .vertices
                .iter()
                .map(|vertex| rotor.apply(*vertex * BODY_SIZE)),
        );
        self.mesh.segments.clear();
        self.mesh.colors.clear();
        self.mesh.widths.clear();
        polytope_section_perimeter_append(
            topology.edges,
            topology.cells,
            &self.rotated,
            WPlane::new(slice - position.w),
            &mut self.scratch,
            &mut self.mesh,
        );
        let offset = position.truncate();
        for (start, end) in &self.mesh.segments {
            out.push(SegmentRecord {
                start: (Vec3::from_array(*start) + offset).to_array(),
                _pad0: 0.0,
                end: (Vec3::from_array(*end) + offset).to_array(),
                _pad1: 0.0,
                start_color: self.color,
                end_color: self.color,
                width_px: self.width_px,
                _pad2: [0.0; 3],
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_tesseract_cut_at_w_zero_is_the_cube_of_its_own_edge_length() {
        let mut cutter = Cutter::default();
        let mut out = Vec::new();
        cutter.cut(
            Polytope4::Tesseract,
            Rotor4::IDENTITY,
            Vec4::ZERO,
            0.0,
            &mut out,
        );

        let half = 0.5 * BODY_SIZE;
        assert_eq!(
            out.len(),
            24,
            "the six cells that straddle w = 0 each cut to a square, so each of the cube's twelve edges is emitted twice"
        );
        for record in &out {
            for point in [record.start, record.end] {
                for coordinate in point {
                    assert!(
                        (coordinate.abs() - half).abs() < 1e-5,
                        "a cut vertex left the cube of half extent {half}: {point:?}"
                    );
                }
            }
            let length = (Vec3::from_array(record.end) - Vec3::from_array(record.start)).length();
            assert!(
                (length - BODY_SIZE).abs() < 1e-5,
                "a cut edge is {length}, not the tesseract's edge length {BODY_SIZE}"
            );
        }
    }

    #[test]
    fn a_body_lifted_past_the_slice_cuts_nothing() {
        let mut cutter = Cutter::default();
        let mut out = Vec::new();
        cutter.cut(
            Polytope4::Tesseract,
            Rotor4::IDENTITY,
            Vec4::W * (BODY_SIZE + 1.0),
            0.0,
            &mut out,
        );
        assert!(
            out.is_empty(),
            "the cut ignored the body's own w and sliced it where it is not: {} segments",
            out.len()
        );
    }
}
