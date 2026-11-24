use glam::Vec3;
use loam_math::WPlane;
use loam_shape::polytope::{polytope_section_faces_append, Polytope4, SectionScratch};
use loam_shape::TriangleMesh;

#[test]
fn section_caps() {
    for polytope in Polytope4::ALL {
        let (normals, offset) = polytope.face_planes();
        for w in [-0.2, 0.0, 0.2] {
            let topology = polytope.topology();
            let mut mesh = TriangleMesh::<3>::default();
            let mut scratch = SectionScratch::default();
            polytope_section_faces_append(
                topology.edges,
                topology.cells,
                topology.vertices,
                WPlane::new(w),
                [1.0; 4],
                &mut scratch,
                &mut mesh,
            );
            assert!(!mesh.indices.is_empty(), "{polytope:?}, w={w}");
            for point in mesh.vertices {
                let p = Vec3::from_array(point).extend(w);
                let distance = normals
                    .iter()
                    .map(|n| n.dot(p) - offset)
                    .fold(f32::NEG_INFINITY, f32::max);
                assert!(
                    distance.abs() < 1e-3,
                    "{polytope:?}: {p:?}, distance={distance}"
                );
            }
        }
    }
}
