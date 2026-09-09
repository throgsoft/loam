use glam::Vec4;
use loam_math::{Rotor, Rotor4};
use loam_runtime::PointRecord;
use loam_shape::polytope::Polytope4;

use crate::color::{point_color, w_extent, ColorMode};
use crate::consts::BODY_SIZE;

pub(crate) const VERTEX_SIZE_PX: f32 = 4.0;

const CELL_CENTER_INSET: f32 = 0.5;

/// Cell centres per polytope; `Polytope4::cell_centers` allocates, so the row's are built once.
#[derive(Default)]
pub(crate) struct Cloud {
    centers: Vec<(Polytope4, Vec<Vec4>)>,
    pub(crate) vertices: bool,
    pub(crate) cell_centers: bool,
    records: Vec<PointRecord>,
}

impl Cloud {
    pub(crate) fn new(row: impl Iterator<Item = Polytope4>) -> Self {
        let mut centers: Vec<(Polytope4, Vec<Vec4>)> = Vec::new();
        for polytope in row {
            if !centers.iter().any(|(held, _)| *held == polytope) {
                centers.push((polytope, polytope.cell_centers()));
            }
        }
        Self {
            centers,
            vertices: true,
            cell_centers: true,
            records: Vec::new(),
        }
    }

    pub(crate) fn clear(&mut self) {
        self.records.clear();
    }

    pub(crate) fn records(&self) -> &[PointRecord] {
        &self.records
    }

    pub(crate) fn append(
        &mut self,
        polytope: Polytope4,
        rotor: Rotor4,
        position: Vec4,
        mode: ColorMode,
    ) {
        let topology = polytope.topology();
        let offset = position.truncate();
        let extent = w_extent(topology, BODY_SIZE);
        if self.vertices {
            for vertex in topology.vertices {
                let local = rotor.apply(*vertex * BODY_SIZE);
                self.records.push(PointRecord {
                    position: (local.truncate() + offset).to_array(),
                    radius_px: VERTEX_SIZE_PX,
                    color: point_color(mode, *vertex, local, extent),
                });
            }
        }
        if !self.cell_centers {
            return;
        }
        let Some((_, centers)) = self.centers.iter().find(|(held, _)| *held == polytope) else {
            return;
        };
        for center in centers {
            let local = rotor.apply(*center * BODY_SIZE * CELL_CENTER_INSET);
            self.records.push(PointRecord {
                position: (local.truncate() + offset).to_array(),
                radius_px: VERTEX_SIZE_PX * 0.5,
                color: point_color(mode, *center, local, extent),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::*;

    #[test]
    fn the_tesseract_publishes_a_point_for_every_vertex_and_every_cell_centre() {
        let mut cloud = Cloud::new([Polytope4::Tesseract].into_iter());
        cloud.clear();
        cloud.append(
            Polytope4::Tesseract,
            Rotor4::IDENTITY,
            Vec4::ZERO,
            ColorMode::VertexGradient,
        );
        assert_eq!(
            cloud.records().len(),
            24,
            "the tesseract's 16 vertices and 8 cell centres came to {}",
            cloud.records().len()
        );

        let half = 0.5 * BODY_SIZE;
        for record in &cloud.records()[..16] {
            for coordinate in record.position {
                assert!(
                    (coordinate.abs() - half).abs() < 1e-5,
                    "a vertex left the cube of half extent {half}: {:?}",
                    record.position
                );
            }
        }
        let inset = half * CELL_CENTER_INSET;
        for record in &cloud.records()[16..] {
            let distance = Vec3::from_array(record.position).length();
            assert!(
                distance < 1e-5 || (distance - inset).abs() < 1e-5,
                "a cell centre sits at {distance}, not on an axis at {inset} or on the slice"
            );
        }
    }
}
