//! A rigid body in `loam-physics` has one collider and no per-collider offset,
//! which is what makes `GlyphSolid::colliders_4d`'s static-only contract a
//! measurement rather than a caution.

use ab_glyph::FontRef;
use glam::{Vec2, Vec4, Vec4Swizzles};

use loam_math::{EuclideanR4, Rotor};
use loam_physics::euclidean_r4::{
    halfspace4_body_r4, polytope_body_r4, register_default_narrowphase,
};
use loam_physics::manifold::PENETRATION_SLOP;
use loam_physics::{BodyId, World};
use loam_shape::{Shape, Visualizable};
use loam_text::glyph::{layout_word, GlyphParams, GlyphSolid};

const WORD: &str = "LOAM";
const GRAVITY: f32 = -9.8;

const DT: f32 = 1.0 / 240.0;

const DROP_CLEARANCE: f32 = 0.05;

const SETTLE_STEPS: usize = 1920;
const REST_WINDOW: usize = 240;
const REST_SPREAD: f32 = 0.02;
const LANDING_SLIDE: f32 = 0.15;

fn word() -> Vec<GlyphSolid> {
    let bytes = include_bytes!("../../hero/fonts/lmroman10-bold.otf");
    let font = FontRef::try_from_slice(bytes).expect("parse font");
    layout_word(&font, WORD, &GlyphParams::default()).expect("layout")
}

fn hull_of(letter: &GlyphSolid) -> (Vec4, Vec<Vec4>) {
    let (centre, shape) = letter.rigid_hull_4d().expect("letter has a hull");
    let Shape::ConvexPolytope4D { vertices } = shape else {
        panic!("dynamic letter collider is not 4D convex");
    };
    (centre, vertices)
}

fn floor_world() -> World<EuclideanR4> {
    let mut world = World::new(EuclideanR4);
    register_default_narrowphase(&mut world.narrowphase);
    world.gravity = Some(Vec4::new(0.0, GRAVITY, 0.0, 0.0));
    let floor = world.push_body(halfspace4_body_r4(Vec4::Y, 0.0).unwrap());
    world.bodies[floor].restitution = 0.0;
    world
}

fn drop_letter(world: &mut World<EuclideanR4>, letter: &GlyphSolid) -> (BodyId, Vec4) {
    let (centre, vertices) = hull_of(letter);
    let lowest = vertices.iter().fold(f32::INFINITY, |m, v| m.min(v.y));
    let spawn = Vec4::new(centre.x, DROP_CLEARANCE - lowest, centre.z, centre.w);
    let id = world.push_body(polytope_body_r4(spawn, Vec4::ZERO, vertices, 1.0).unwrap());
    world.bodies[id].restitution = 0.0;
    (id, spawn)
}

fn deepest_y(world: &World<EuclideanR4>, id: BodyId) -> f32 {
    let body = &world.bodies[id];
    let Shape::ConvexPolytope4D { vertices } = body.collider() else {
        unreachable!("spawned as a 4D polytope")
    };
    vertices
        .iter()
        .map(|v| body.orientation.rotation.apply(*v).y + body.position.y)
        .fold(f32::INFINITY, f32::min)
}

fn assert_resting_on_the_floor(ch: char, deepest: f32) {
    assert!(
        deepest <= 1.0e-4,
        "{ch:?} floats with its lowest point at {deepest}"
    );
    assert!(
        deepest >= -2.0 * PENETRATION_SLOP,
        "{ch:?} sank to {deepest} through the floor"
    );
}

#[test]
fn a_letters_hull_contains_both_its_render_mesh_and_its_cover() {
    let letters = word();
    let params = GlyphParams::default();
    for letter in &letters {
        let (centre, vertices) = hull_of(letter);
        let sides = letter.rigid_hull_sides();
        let ring: Vec<Vec2> = vertices[..sides]
            .iter()
            .map(|v| (*v + centre).xy())
            .collect();
        let outside = |p: Vec2| {
            (0..sides).any(|k| {
                let a = ring[k];
                let b = ring[(k + 1) % sides];
                (b - a).perp_dot(p - a) < -1.0e-5
            })
        };

        let mesh = Visualizable::<3>::to_triangles(letter).expect("mesh");
        for v in &mesh.vertices {
            assert!(
                !outside(Vec2::new(v[0], v[1])),
                "{:?} renders a vertex at ({}, {}) outside its dynamic hull",
                letter.ch(),
                v[0],
                v[1]
            );
        }

        let cover = letter.collider_cover().expect("cover");
        for index in 0..cover.piece_count() {
            let (lo, hi) = cover.piece_bounds(index);
            for y in [lo[1], hi[1]] {
                for x in [lo[0], hi[0]] {
                    assert!(
                        !outside(Vec2::new(x, y)),
                        "{:?} covers ({x}, {y}) outside its dynamic hull",
                        letter.ch()
                    );
                }
            }
        }

        for v in &vertices {
            let world = *v + centre;
            assert!((world.z.abs() - 0.5 * params.depth).abs() < 1.0e-6);
            assert!(world.w >= params.slab.0 - 1.0e-6 && world.w <= params.slab.1 + 1.0e-6);
        }
    }
}

#[test]
fn a_whole_word_dropped_together_settles_in_its_own_line() {
    let letters = word();
    let mut world = floor_world();
    let dropped: Vec<(BodyId, Vec4)> = letters
        .iter()
        .map(|letter| drop_letter(&mut world, letter))
        .collect();
    assert_eq!(world.bodies.iter().count(), letters.len() + 1);

    let mut rest: Vec<Vec<Vec4>> = vec![Vec::with_capacity(REST_WINDOW); letters.len()];
    for step in 0..SETTLE_STEPS {
        world.step(DT);
        if step >= SETTLE_STEPS - REST_WINDOW {
            for (samples, (id, _)) in rest.iter_mut().zip(&dropped) {
                samples.push(world.bodies[*id].position);
            }
        }
    }

    for ((samples, (id, spawn)), letter) in rest.iter().zip(&dropped).zip(&letters) {
        let spread = samples
            .iter()
            .map(|p| p.distance(samples[0]))
            .fold(0.0f32, f32::max);
        assert!(
            spread < REST_SPREAD,
            "{:?} still moves {spread} in the final second",
            letter.ch()
        );
        assert_resting_on_the_floor(letter.ch(), deepest_y(&world, *id));
        let settled = world.bodies[*id].position;
        assert!(settled.y < spawn.y - 0.5 * DROP_CLEARANCE);
        assert!(
            (settled.x - spawn.x).abs() < LANDING_SLIDE,
            "{:?} slid to x = {} from {}",
            letter.ch(),
            settled.x,
            spawn.x
        );
    }

    for pair in dropped.windows(2) {
        assert!(world.bodies[pair[1].0].position.x > world.bodies[pair[0].0].position.x);
    }
}
