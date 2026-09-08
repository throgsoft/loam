use std::hint::black_box;
use std::time::Instant;

use glam::Vec3;
use loam_math::{EuclideanR3, Space};
use loam_physics::body::BodyId;
use loam_physics::collider::Collider;
use loam_physics::euclidean_r3::{
    box_body, halfspace_body_r3, register_default_narrowphase, sphere_body_r3,
};
use loam_physics::world::PairKey;
use loam_physics::World;

const BODY_COUNTS: [usize; 3] = [100, 200, 400];
const SPREAD_AT_100: f32 = 6.0;
const SETTLE_STEPS: usize = 240;
const DT: f32 = 1.0 / 240.0;
const GRAVITY_Y: f32 = -9.8;
const SEED: u64 = 0x9e37_79b9_7f4a_7c15;
const REPS: u32 = 200;
const BATCHES: usize = 9;

// Marsaglia 2003, "Xorshift RNGs", Journal of Statistical Software 8(14), 13/7/17.
struct Xorshift(u64);

impl Xorshift {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        let unit = (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32;
        lo + (hi - lo) * unit
    }
}

fn settled_scene(count: usize) -> World<EuclideanR3> {
    let spread = SPREAD_AT_100 * (count as f32 / 100.0).cbrt();
    let mut rng = Xorshift::new(SEED);
    let mut world = World::new(EuclideanR3);
    register_default_narrowphase(&mut world.narrowphase);
    world.gravity = Some(Vec3::new(0.0, GRAVITY_Y, 0.0));
    world.push_body(halfspace_body_r3(Vec3::Y, 0.0).unwrap());

    for _ in 0..count {
        let position = Vec3::new(
            rng.range(-spread, spread),
            rng.range(0.5, spread + 0.5),
            rng.range(-spread, spread),
        );
        let id = if rng.next_u64() & 1 == 0 {
            world.push_body(sphere_body_r3(position, Vec3::ZERO, rng.range(0.2, 0.8), 1.0).unwrap())
        } else {
            world.push_body(
                box_body(position, Vec3::ZERO, Vec3::splat(rng.range(0.2, 0.6)), 1.0).unwrap(),
            )
        };
        world.bodies[id].restitution = 0.0;
    }
    for _ in 0..SETTLE_STEPS {
        world.step(DT);
    }
    world
}

fn bounding_radius(collider: &Collider) -> f32 {
    match collider {
        Collider::Sphere { radius, .. } => *radius,
        Collider::ConvexPolytope3D { vertices } => vertices
            .iter()
            .map(|v| v.length_squared())
            .fold(0.0_f32, f32::max)
            .sqrt(),
        Collider::HalfSpace { .. } => f32::INFINITY,
        other => unreachable!("the scene builds spheres, boxes and one half-space, not {other:?}"),
    }
}

fn scan(world: &World<EuclideanR3>, radii: &mut Vec<f32>, pairs: &mut Vec<PairKey>) {
    radii.clear();
    radii.extend(
        world
            .bodies
            .iter()
            .map(|body| bounding_radius(body.collider())),
    );
    pairs.clear();
    let n = world.bodies.len();
    for i in 0..n {
        for j in (i + 1)..n {
            let (a, b) = (&world.bodies[i], &world.bodies[j]);
            if a.inv_mass() == 0.0 && b.inv_mass() == 0.0 {
                continue;
            }
            if world.space.distance(a.position, b.position) <= radii[i] + radii[j] {
                pairs.push(canonical(world.bodies.id_at(i), world.bodies.id_at(j)));
            }
        }
    }
    pairs.sort_unstable();
}

fn canonical(a: BodyId, b: BodyId) -> PairKey {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
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
    println!("bodies pairs sweep_ns scan_ns");
    let mut radii = Vec::new();
    let mut scan_out = Vec::new();
    let mut sweep_out = Vec::new();
    for count in BODY_COUNTS {
        let mut world = settled_scene(count);
        world.broadphase_into(&mut sweep_out);
        scan(&world, &mut radii, &mut scan_out);
        assert_eq!(
            sweep_out, scan_out,
            "candidate sets differ at {count} bodies"
        );
        let sweep_ns = median_nanos(|| {
            world.broadphase_into(&mut sweep_out);
            black_box(&sweep_out);
        });
        let scan_ns = median_nanos(|| {
            scan(&world, &mut radii, &mut scan_out);
            black_box(&scan_out);
        });
        println!(
            "{} {} {sweep_ns:.0} {scan_ns:.0}",
            world.bodies.len(),
            sweep_out.len()
        );
    }
}
