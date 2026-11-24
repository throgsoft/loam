#![cfg(not(target_arch = "wasm32"))]

use glam::Vec3;
use loam_app::par_native::ScopedThreads;
use loam_math::EuclideanR3;
use loam_physics::euclidean_r3::{box_body, halfspace_body_r3, register_default_narrowphase};
use loam_physics::world::PAIRS_PER_NARROWPHASE_WORKER;
use loam_physics::{BodyId, RigidBody, World};
use loam_time::par::{Executor, Sequential};

const HALF: f32 = 0.5;
const GAP: f32 = 0.05;
const COLUMNS: usize = PAIRS_PER_NARROWPHASE_WORKER + 8;
const DT: f32 = 1.0 / 240.0;
const STEPS: usize = 30;

// Each column is a sleeping box on the floor and a box that falls onto it within the run.
fn columns() -> (World<EuclideanR3>, Vec<BodyId>) {
    let mut world = World::new(EuclideanR3);
    register_default_narrowphase(&mut world.narrowphase);
    world
        .set_gravity(Some(Vec3::new(0.0, -9.8, 0.0)))
        .expect("valid gravity");
    let floor = world.push_body(halfspace_body_r3(Vec3::Y, 0.0).unwrap());
    world.set_restitution(floor, 0.0).unwrap();
    let mut sleepers = Vec::new();
    for column in 0..COLUMNS {
        let x = column as f32 * 4.0;
        let resting = world.push_body(
            box_body(Vec3::new(x, HALF, 0.0), Vec3::ZERO, Vec3::splat(HALF), 1.0).unwrap(),
        );
        world.sleep_body(resting).unwrap();
        sleepers.push(resting);
        world.push_body(
            box_body(
                Vec3::new(x, 3.0 * HALF + GAP, 0.0),
                Vec3::NEG_Y,
                Vec3::splat(HALF),
                1.0,
            )
            .unwrap(),
        );
    }
    (world, sleepers)
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
        body.angular_velocity.xy,
        body.angular_velocity.yz,
        body.angular_velocity.zx,
    ] {
        words.push(value.to_bits());
    }
    words.push(u32::from(body.is_sleeping()));
}

fn run(executor: &dyn Executor) -> u64 {
    let (mut world, sleepers) = columns();
    for _ in 0..STEPS {
        world.step(DT, executor).expect("step");
    }
    let candidates = world.counters().candidates as usize;
    assert!(
        candidates >= 2 * PAIRS_PER_NARROWPHASE_WORKER,
        "the fixture has {candidates} candidate pairs, too few for a second narrowphase worker"
    );
    let woken = sleepers
        .iter()
        .filter(|&&id| !world.body(id).expect("body").is_sleeping())
        .count();
    assert_eq!(
        woken, COLUMNS,
        "the falling boxes did not wake every sleeper"
    );
    world.state_hash(sample)
}

#[test]
fn threaded_pairs() {
    let serial = run(&Sequential);
    let threads = ScopedThreads::default();
    assert!(
        threads.parallelism() > 1,
        "one core, so this run cannot tell the two paths apart"
    );
    assert_eq!(serial, run(&threads));
}
