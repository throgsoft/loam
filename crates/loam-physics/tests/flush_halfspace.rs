#![cfg(feature = "r4")]

use glam::Vec4;
use loam_math::EuclideanR4;
use loam_physics::euclidean_r4::{
    flush_halfspace4_body_r4, register_default_narrowphase, sphere_body_r4,
};
use loam_physics::World;
use loam_time::par::Sequential;

const OFFSET: f32 = 1.5;
const RADIUS: f32 = 0.5;
const DROP: f32 = 0.25;
// A 60 Hz tick in 4 substeps, the longest substep the builder states.
const SUBSTEPS_PER_SECOND: usize = 240;
const SECONDS: usize = 3;
// A sphere resting higher than this over the plane shows a gap.
const FLUSH: f32 = 0.01;

#[test]
fn resting_sphere() {
    let mut world = World::new(EuclideanR4);
    register_default_narrowphase(&mut world.narrowphase);
    world.set_gravity(Some(Vec4::NEG_Y * 9.8)).unwrap();
    world.push_body(flush_halfspace4_body_r4(Vec4::Y, OFFSET).unwrap());
    let start = Vec4::Y * (OFFSET + RADIUS + DROP);
    let ball = world.push_body(sphere_body_r4(start, Vec4::ZERO, RADIUS, 1.0).unwrap());

    for _ in 0..SECONDS * SUBSTEPS_PER_SECOND {
        world
            .step(1.0 / SUBSTEPS_PER_SECOND as f32, &Sequential)
            .unwrap();
    }

    let body = world.body(ball).unwrap();
    assert!(
        body.velocity.length() < 1e-2,
        "the sphere is still moving at {}",
        body.velocity
    );
    let lowest = body.position.y - RADIUS;
    assert!(
        (OFFSET..OFFSET + FLUSH).contains(&lowest),
        "the sphere rests with its lowest point at {lowest}, off the plane at {OFFSET}"
    );
}
