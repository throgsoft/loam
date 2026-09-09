#![cfg(not(target_arch = "wasm32"))]

use glam::Vec3;
use loam_math::EuclideanR3;
use loam_physics::euclidean_r3::{halfspace_body_r3, register_default_narrowphase, sphere_body_r3};
use loam_physics::world::ISLANDS_PER_SOLVE_WORKER;
use loam_physics::{RigidBody, World};

const RADIUS: f32 = 0.5;
const COLUMNS: usize = 2 * ISLANDS_PER_SOLVE_WORKER + 8;
const DT: f32 = 1.0 / 240.0;
const STEPS: usize = 30;

fn columns() -> World<EuclideanR3> {
    let mut world = World::new(EuclideanR3);
    register_default_narrowphase(&mut world.narrowphase);
    world.gravity = Some(Vec3::new(0.0, -9.8, 0.0));
    let floor = world.push_body(halfspace_body_r3(Vec3::Y, 0.0).unwrap());
    world.bodies[floor].restitution = 0.0;
    for column in 0..COLUMNS {
        let x = column as f32 * 4.0;
        for level in 0..2 {
            let y = RADIUS + level as f32 * 2.0 * RADIUS;
            let id = world
                .push_body(sphere_body_r3(Vec3::new(x, y, 0.0), Vec3::ZERO, RADIUS, 1.0).unwrap());
            world.bodies[id].restitution = 0.0;
        }
    }
    world
}

fn sample(body: &RigidBody<EuclideanR3>, words: &mut Vec<u32>) {
    for value in [
        body.position.x,
        body.position.y,
        body.position.z,
        body.velocity.x,
        body.velocity.y,
        body.velocity.z,
        body.orientation.rotation.x,
        body.orientation.rotation.y,
        body.orientation.rotation.z,
        body.orientation.rotation.w,
        body.orientation.translation.x,
        body.orientation.translation.y,
        body.orientation.translation.z,
        body.angular_velocity.xy,
        body.angular_velocity.yz,
        body.angular_velocity.zx,
    ] {
        words.push(value.to_bits());
    }
}

fn run() -> u64 {
    let mut world = columns();
    for _ in 0..STEPS {
        world.step(DT);
    }
    let islands = world.islands().len();
    assert!(
        islands >= 2 * ISLANDS_PER_SOLVE_WORKER,
        "the fixture settled into {islands} islands, too few for a second solve worker"
    );
    world.state_hash(sample)
}

#[test]
fn islands_solved_through_the_shim_land_bit_for_bit_on_the_serial_result() {
    let serial = run();
    loam_app::par_native::install();
    assert!(
        loam_time::par::executor().parallelism() > 1,
        "one core, so this run cannot tell the two paths apart"
    );
    assert_eq!(serial, run());
}
