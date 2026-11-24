use glam::Vec3;

use loam_physics::GeometryStore;
use loam_shape::Shape;

fn tetrahedron(apex: Vec3) -> Shape {
    Shape::ConvexPolytope3D {
        vertices: vec![Vec3::ZERO, Vec3::X, Vec3::Y, apex],
    }
}

#[test]
fn equal_shapes_share_one_entry_and_a_single_moved_vertex_gets_its_own() {
    let mut store = GeometryStore::default();

    let first = store.prepare(tetrahedron(Vec3::Z));
    let twin = store.prepare(tetrahedron(Vec3::Z));
    let taller = store.prepare(tetrahedron(Vec3::Z * 2.0));

    assert_eq!(
        first.geometry(),
        twin.geometry(),
        "two equal hulls were stored twice"
    );
    assert_ne!(
        first.geometry(),
        taller.geometry(),
        "a hull with a different apex was drawn from the other's entry"
    );
    assert_eq!(store.prepared(), 2);
    assert_eq!(store.get(taller), Some(&tetrahedron(Vec3::Z * 2.0)));
}
