use loam_math::{EuclideanR4, Projection, RasterizableSpace};
use loam_shape::{polytope::Polytope4, projection::SchlegelParams};

#[test]
fn cell_plane_and_projection_agree() {
    for polytope in Polytope4::ALL {
        let topology = polytope.topology();
        for cell_index in [0, topology.cells.len() / 2, topology.cells.len() - 1] {
            let params = SchlegelParams::new(polytope, cell_index as u32, 0.5).unwrap();
            let projection = Projection::schlegel_with_basis(
                params.cell_normal,
                params.cell_offset,
                params.viewpoint_distance,
                params.cell_basis,
            );
            assert!((params.cell_normal.length() - 1.0).abs() < 1e-5);
            for i in 0..3 {
                assert!((params.cell_basis[i].length() - 1.0).abs() < 1e-5);
                assert!(params.cell_basis[i].dot(params.cell_normal).abs() < 1e-5);
                for j in i + 1..3 {
                    assert!(params.cell_basis[i].dot(params.cell_basis[j]).abs() < 1e-5);
                }
            }
            for vertex in topology.vertices {
                assert!(vertex.dot(params.cell_normal) <= params.cell_offset + 5e-4);
            }
            for &index in topology.cells[cell_index] {
                let vertex = topology.vertices[index as usize];
                assert!((vertex.dot(params.cell_normal) - params.cell_offset).abs() < 5e-4);
                let projected = EuclideanR4::project_point(vertex, &projection);
                let reconstructed = params.cell_basis[0] * projected.x
                    + params.cell_basis[1] * projected.y
                    + params.cell_basis[2] * projected.z
                    + params.cell_normal * params.cell_offset;
                assert!((reconstructed - vertex).length() < 5e-4, "{polytope:?}");
            }
        }
    }
}

#[test]
fn invalid_cell_or_eye_clearance_is_rejected() {
    let polytope = Polytope4::Tesseract;
    assert!(SchlegelParams::new(polytope, polytope.cell_count() as u32, 0.5).is_none());
    for margin in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::MIN_POSITIVE] {
        assert!(SchlegelParams::new(polytope, 0, margin).is_none());
    }
}
