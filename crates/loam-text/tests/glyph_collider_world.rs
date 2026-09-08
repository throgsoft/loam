//! Both halves of a faithful silhouette are checked: a sphere over ink lands on
//! it, and a sphere over the counter of `O` falls through. A per-letter convex
//! hull, or a bounding volume, passes the first and fails the second.

use ab_glyph::FontRef;
use glam::{Vec2, Vec4};

use loam_math::EuclideanR4;
use loam_physics::euclidean_r4::{register_default_narrowphase, sphere_body_r4};
use loam_physics::{BodyId, RigidBody, World};
use loam_shape::Visualizable;
use loam_text::glyph::{layout_word, GlyphParams, GlyphSolid};

const WORD: &str = "LOAM";
const DT: f32 = 1.0 / 120.0;
const BALL_RADIUS: f32 = 0.03;
const DROP_Z: f32 = 0.25;
const IMPACT_STEPS: usize = 30;

fn word() -> Vec<GlyphSolid> {
    let bytes = include_bytes!("../../hero/fonts/lmroman10-bold.otf");
    let font = FontRef::try_from_slice(bytes).expect("parse font");
    layout_word(&font, WORD, &GlyphParams::default()).expect("layout")
}

fn word_world(letters: &[GlyphSolid]) -> World<EuclideanR4> {
    let mut world = World::new(EuclideanR4);
    register_default_narrowphase(&mut world.narrowphase);
    world.gravity = Some(Vec4::new(0.0, 0.0, -9.8, 0.0));
    for letter in letters {
        for (centre, hull) in letter.colliders_4d() {
            world.push_body(RigidBody::fixed(centre, hull, 1.0, &EuclideanR4).unwrap());
        }
    }
    world
}

fn drop_ball(world: &mut World<EuclideanR4>, at: Vec2) -> BodyId {
    world.push_body(
        sphere_body_r4(
            Vec4::new(at.x, at.y, DROP_Z, 0.0),
            Vec4::ZERO,
            BALL_RADIUS,
            1.0,
        )
        .unwrap(),
    )
}

fn ink_mid_y(letter: &GlyphSolid) -> f32 {
    let mesh = Visualizable::<3>::to_triangles(letter).expect("mesh");
    let lo = mesh.vertices.iter().fold(f32::INFINITY, |m, v| m.min(v[1]));
    let hi = mesh
        .vertices
        .iter()
        .fold(f32::NEG_INFINITY, |m, v| m.max(v[1]));
    0.5 * (lo + hi)
}

fn ink_runs(letter: &GlyphSolid, mid: f32) -> Vec<(f32, f32)> {
    const PROBES: usize = 2048;
    let x0 = letter.pen_origin().x - 0.25 * letter.advance();
    let width = 1.5 * letter.advance();
    let mut runs = Vec::new();
    let mut start = None;
    for k in 0..=PROBES {
        let x = x0 + width * k as f32 / PROBES as f32;
        let inside = letter.distance_2d(Vec2::new(x, mid)) <= 0.0;
        match (inside, start) {
            (true, None) => start = Some(x),
            (false, Some(from)) => {
                runs.push((from, x));
                start = None;
            }
            _ => {}
        }
    }
    runs
}

fn widest_stroke(letter: &GlyphSolid) -> Vec2 {
    let mid = ink_mid_y(letter);
    let run = ink_runs(letter, mid)
        .into_iter()
        .max_by(|a, b| (a.1 - a.0).total_cmp(&(b.1 - b.0)))
        .expect("letter has ink at mid height");
    Vec2::new(0.5 * (run.0 + run.1), mid)
}

#[test]
fn a_body_dropped_on_a_letter_is_held_up_by_its_front_face() {
    let letters = word();
    let front = 0.5 * GlyphParams::default().depth;

    let mut world = word_world(&letters);
    let balls: Vec<_> = letters
        .iter()
        .map(|letter| drop_ball(&mut world, widest_stroke(letter)))
        .collect();
    for step in 0..IMPACT_STEPS * 2 {
        world.step(DT);
        if step < IMPACT_STEPS {
            continue;
        }
        for (letter, &ball) in letters.iter().zip(&balls) {
            let z = world.bodies.get(ball).unwrap().position.z;
            assert!(
                z > front,
                "{:?}: ball sank to z = {z}, past the front face at {front}",
                letter.ch()
            );
            assert!(
                z < front + 2.0 * BALL_RADIUS,
                "{:?}: ball is floating at z = {z}",
                letter.ch()
            );
        }
    }
    for (letter, &ball) in letters.iter().zip(&balls) {
        let body = world.bodies.get(ball).unwrap();
        assert!(
            body.velocity.z.abs() < 0.5,
            "{:?}: ball is still moving vertically at {}",
            letter.ch(),
            body.velocity.z
        );
    }
}

#[test]
fn a_body_dropped_down_the_counter_of_o_falls_through() {
    let letters = word();
    let o = letters.iter().find(|l| l.ch() == 'O').expect("O");
    let mid = ink_mid_y(o);
    let runs = ink_runs(o, mid);
    assert_eq!(runs.len(), 2, "'O' is not two strokes at mid height");

    let counter = Vec2::new(0.5 * (runs[0].1 + runs[1].0), mid);
    assert!(
        o.distance_2d(counter) > o.collider_margin() + BALL_RADIUS,
        "counter probe {counter} is not clear of the ink"
    );

    let mut world = word_world(&letters);
    let ball = drop_ball(&mut world, counter);
    for _ in 0..120 {
        world.step(DT);
        if world.bodies[ball].position.z < -DROP_Z {
            break;
        }
    }
    let z = world.bodies.get(ball).unwrap().position.z;
    assert!(z < -DROP_Z, "ball stopped at z = {z} instead of falling");
}
