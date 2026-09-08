use crate::body::{BodyId, RigidBody};
use crate::euclidean_r3::{
    halfspace_body_r3, register_default_narrowphase as register_narrowphase_r3, sphere_body_r3,
};
use crate::world::World;
use glam::Vec3;
use loam_math::EuclideanR3;
use loam_time::{Checkpoint, Tape};
use std::ops::Range;
const GRAVITY_MAGNITUDE: f32 = 9.8;

pub fn sample_body_r3(body: &RigidBody<EuclideanR3>, words: &mut Vec<u32>) {
    let p = body.position;
    let v = body.velocity;
    let w = body.angular_velocity;
    words.extend_from_slice(&[
        p.x.to_bits(),
        p.y.to_bits(),
        p.z.to_bits(),
        v.x.to_bits(),
        v.y.to_bits(),
        v.z.to_bits(),
        w.xy.to_bits(),
        w.yz.to_bits(),
        w.zx.to_bits(),
    ]);
}

const ISLAND_RADIUS: f32 = 0.5;
const ISLAND_X: [f32; 3] = [-4.0, 0.0, 4.0];
const ISLAND_SIZES: [usize; 3] = [4, 2, 1];
const ISLAND_GAP: f32 = 0.05;
pub const MULTI_ISLAND_DT: f32 = 1.0 / 60.0;
pub const MULTI_ISLAND_STEPS: usize = 240;
pub fn multi_island_world() -> World<EuclideanR3> {
    let mut world = World::new(EuclideanR3);
    register_narrowphase_r3(&mut world.narrowphase);
    world.gravity = Some(Vec3::new(0.0, -GRAVITY_MAGNITUDE, 0.0));
    world.push_body(halfspace_body_r3(Vec3::Y, 0.0).unwrap());
    for (group, &size) in ISLAND_SIZES.iter().enumerate() {
        for level in 0..size {
            let y = ISLAND_RADIUS + ISLAND_GAP + level as f32 * (2.0 * ISLAND_RADIUS + ISLAND_GAP);
            world.push_body(
                sphere_body_r3(
                    Vec3::new(ISLAND_X[group], y, 0.0),
                    Vec3::ZERO,
                    ISLAND_RADIUS,
                    1.0,
                )
                .unwrap(),
            );
        }
    }
    world
}

const THROW_WORDS: u32 = 5;
const NO_THROW: u32 = u32::MAX;
const THROW_PERIOD: u64 = 20;
const THROW_IMPULSE: f32 = 2.0;
pub const REPLAY_TICKS: u64 = 180;
pub const REPLAY_SEED: u64 = 0x5eed_f11c_c0de_0001;
const CHECKPOINT_PERIOD: u64 = 30;
const REPLAY_TICK_HZ: u32 = 60;

// Vigna 2016, "An experimental exploration of Marsaglia's xorshift generators, scrambled", §4.
fn xorshift64star(state: &mut u64) -> u64 {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    state.wrapping_mul(0x2545_f491_4f6c_dd1d)
}

fn signed_unit(draw: u64) -> f32 {
    ((draw >> 40) as u32) as f32 * (1.0 / 8_388_608.0) - 1.0
}

fn generate_throws(
    seed: u64,
    ticks: u64,
    dynamic_slots: Range<u32>,
) -> Vec<[u32; THROW_WORDS as usize]> {
    let mut state = seed | 1;
    (0..ticks)
        .map(|tick| {
            if !tick.is_multiple_of(THROW_PERIOD) {
                return [NO_THROW, 0, 0, 0, 0];
            }
            let span = u64::from(dynamic_slots.end - dynamic_slots.start);
            let slot = dynamic_slots.start + (xorshift64star(&mut state) % span) as u32;
            let mut component =
                || (signed_unit(xorshift64star(&mut state)) * THROW_IMPULSE).to_bits();
            [slot, 0, component(), component(), component()]
        })
        .collect()
}

fn apply_throw(world: &mut World<EuclideanR3>, frame: [u32; THROW_WORDS as usize]) {
    let [slot, generation, x, y, z] = frame;
    if slot == NO_THROW {
        return;
    }
    if let Some(body) = world.bodies.get_mut(BodyId::forge(slot, generation)) {
        body.apply_impulse(Vec3::new(
            f32::from_bits(x),
            f32::from_bits(y),
            f32::from_bits(z),
        ));
    }
}

fn drive_flick_chamber(
    ticks: u64,
    input: impl Fn(u64) -> [u32; THROW_WORDS as usize],
) -> Vec<Checkpoint> {
    let mut world = multi_island_world();
    let mut checkpoints = Vec::new();
    for tick in 0..ticks {
        apply_throw(&mut world, input(tick));
        world.step(MULTI_ISLAND_DT);
        if (tick + 1).is_multiple_of(CHECKPOINT_PERIOD) {
            checkpoints.push(Checkpoint {
                tick,
                state_hash: world.state_hash(sample_body_r3),
            });
        }
    }
    checkpoints
}
pub fn record_flick_chamber_tape(seed: u64) -> Tape {
    let dynamic_slots = 1..1 + ISLAND_SIZES.iter().sum::<usize>() as u32;
    let frames = generate_throws(seed, REPLAY_TICKS, dynamic_slots);

    let mut tape = Tape::new(REPLAY_TICK_HZ, seed, THROW_WORDS);
    for frame in &frames {
        tape.push_tick(frame);
    }
    for checkpoint in drive_flick_chamber(REPLAY_TICKS, |tick| frames[tick as usize]) {
        tape.checkpoint(checkpoint.tick, checkpoint.state_hash);
    }
    tape
}
pub fn replay_flick_chamber_tape(tape: &Tape) -> Vec<Checkpoint> {
    assert_eq!(
        tape.words_per_tick(),
        THROW_WORDS,
        "tape frame width is not the flick chamber's: {tape}",
    );
    drive_flick_chamber(tape.ticks(), |tick| {
        let frame = tape.input(tick).expect("tick is inside the tape");
        <[u32; THROW_WORDS as usize]>::try_from(frame).expect("frame width checked above")
    })
}
