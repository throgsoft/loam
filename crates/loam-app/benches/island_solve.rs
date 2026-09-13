use std::hint::black_box;
use std::time::Instant;

use glam::Vec3;
use loam_math::EuclideanR3;
use loam_physics::euclidean_r3::{halfspace_body_r3, register_default_narrowphase, sphere_body_r3};
use loam_physics::World;

const COLUMN_COUNTS: [usize; 5] = [64, 256, 512, 1024, 2048];
const SOLVER_ITERS: [usize; 2] = [8, 64];
const LEVELS: usize = 3;
const RADIUS: f32 = 0.5;
const DT: f32 = 1.0 / 240.0;
const SETTLE_STEPS: usize = 180;
const REPS: u32 = 20;
const BATCHES: usize = 15;

fn columns(count: usize) -> World<EuclideanR3> {
    let mut world = World::new(EuclideanR3);
    register_default_narrowphase(&mut world.narrowphase);
    world
        .set_gravity(Some(Vec3::new(0.0, -9.8, 0.0)))
        .expect("valid gravity");
    let floor = world.push_body(halfspace_body_r3(Vec3::Y, 0.0).unwrap());
    world.set_restitution(floor, 0.0).unwrap();
    for column in 0..count {
        let x = (column % 16) as f32 * 4.0;
        let z = (column / 16) as f32 * 4.0;
        for level in 0..LEVELS {
            let y = RADIUS + level as f32 * 2.0 * RADIUS;
            let id = world
                .push_body(sphere_body_r3(Vec3::new(x, y, z), Vec3::ZERO, RADIUS, 1.0).unwrap());
            world.set_restitution(id, 0.0).unwrap();
        }
    }
    for _ in 0..SETTLE_STEPS {
        world.step(DT).expect("step");
    }
    world
}

fn median_nanos(mut body: impl FnMut()) -> f64 {
    let mut batches = [0.0_f64; BATCHES];
    for batch in &mut batches {
        let start = Instant::now();
        for _ in 0..REPS {
            body();
        }
        *batch = start.elapsed().as_nanos() as f64 / f64::from(REPS);
    }
    batches.sort_unstable_by(f64::total_cmp);
    batches[BATCHES / 2]
}

fn main() {
    let parallel = std::env::var("LOAM_PAR").is_ok_and(|value| value == "1");
    if parallel {
        loam_app::par_native::install();
    }
    let mode = if parallel { "par" } else { "seq" };
    let threads = loam_time::par::executor().parallelism();
    println!("mode threads islands bodies iters step_ns");
    for count in COLUMN_COUNTS {
        let mut world = columns(count);
        let islands = world.islands().len();
        let bodies = world.bodies().len();
        for iters in SOLVER_ITERS {
            world.set_solver_iterations(iters);
            for _ in 0..8 {
                world.step(DT).expect("step");
            }
            let step_ns = median_nanos(|| {
                world.step(DT).expect("step");
                black_box(&world.time());
            });
            println!("{mode} {threads} {islands} {bodies} {iters} {step_ns:.0}");
        }
    }
}
