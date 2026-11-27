use glam::Vec4;
use loam_math::Rotor as _;
use loam_shape::polytope::Polytope4;

#[test]
fn first_cell_flat_on_the_floor() {
    const SIZE: f32 = 1.3;
    let tilted = Vec4::new(1.0, -2.0, 0.5, 3.0).normalize();
    for polytope in Polytope4::ALL {
        for down in [Vec4::NEG_Y, tilted] {
            let (frame, reach) = polytope.resting(SIZE, down);
            let topology = polytope.topology();
            let depth = |index: usize| frame.apply(topology.vertices[index] * SIZE).dot(down);
            for index in topology.cells[0] {
                let depth = depth(*index as usize);
                assert!(
                    (depth - reach).abs() < 1e-4,
                    "{polytope:?} toward {down}: a vertex of the resting cell is at {depth}, off the floor at {reach}"
                );
            }
            let deepest = (0..topology.vertices.len())
                .map(depth)
                .fold(f32::NEG_INFINITY, f32::max);
            assert!(
                deepest < reach + 1e-4,
                "{polytope:?} toward {down}: a vertex at {deepest} is through the floor at {reach}"
            );
            let (_, doubled) = polytope.resting(2.0 * SIZE, down);
            assert!(
                (doubled - 2.0 * reach).abs() < 1e-4,
                "{polytope:?}: the reach does not scale with the size"
            );
        }
    }
}
