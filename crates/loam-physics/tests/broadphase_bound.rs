use glam::Vec2;
use loam_math::{Bivector2, EuclideanR2, Iso2, IsometryGroup, Rotor2, Space};
use loam_physics::body::{BodyDef, BodyId, RigidBody};
use loam_physics::collider::{Collider, ColliderKind};
use loam_physics::integrator::{BroadphaseBound, PhysicsSpace};
use loam_physics::world::PairKey;
use loam_physics::World;

const CELL: f32 = 10.0;
const RADIUS: f32 = 0.5;

#[derive(Clone, Copy)]
struct SeamR2 {
    bound: BroadphaseBound,
}

fn wrap(x: f32) -> f32 {
    x - CELL * (x / CELL + 0.5).floor()
}

fn seam_distance(a: Vec2, b: Vec2) -> f32 {
    Vec2::new(wrap(a.x - b.x), a.y - b.y).length()
}

fn cross2d(u: Vec2, v: Vec2) -> f32 {
    u.x * v.y - u.y * v.x
}

fn inv_inertia(body: &RigidBody<SeamR2>) -> f32 {
    if body.inv_mass() > 0.0 && body.inertia > 0.0 {
        1.0 / body.inertia
    } else {
        0.0
    }
}

impl Space for SeamR2 {
    type Point = Vec2;
    type Vector = Vec2;
    type Frame = Rotor2;

    fn frame_at(&self, _at: Vec2) -> Rotor2 {
        Rotor2::IDENTITY
    }

    fn distance(&self, a: Vec2, b: Vec2) -> f32 {
        (a - b).length()
    }

    fn exp(&self, at: Vec2, v: Vec2) -> Vec2 {
        let moved = at + v;
        Vec2::new(wrap(moved.x), moved.y)
    }

    fn log(&self, from: Vec2, to: Vec2) -> Vec2 {
        to - from
    }

    fn parallel_transport(&self, _from: Vec2, _to: Vec2, v: Vec2) -> Vec2 {
        v
    }

    fn chart_envelope(&self) -> f32 {
        f32::INFINITY
    }

    fn valid_point(&self, p: Vec2) -> bool {
        p.is_finite()
    }
}

impl IsometryGroup for SeamR2 {
    type Iso = Iso2;

    fn iso_identity(&self) -> Iso2 {
        EuclideanR2.iso_identity()
    }

    fn iso_compose(&self, a: Iso2, b: Iso2) -> Iso2 {
        EuclideanR2.iso_compose(a, b)
    }

    fn iso_inverse(&self, a: Iso2) -> Iso2 {
        EuclideanR2.iso_inverse(a)
    }

    fn iso_apply(&self, iso: Iso2, p: Vec2) -> Vec2 {
        let moved = EuclideanR2.iso_apply(iso, p);
        Vec2::new(wrap(moved.x), moved.y)
    }

    fn iso_transport(&self, iso: Iso2, at: Vec2, v: Vec2) -> Vec2 {
        EuclideanR2.iso_transport(iso, at, v)
    }
}

impl PhysicsSpace for SeamR2 {
    type AngVel = Bivector2;
    type Inertia = f32;

    fn broadphase_bound(&self) -> BroadphaseBound {
        self.bound
    }

    fn supports_collider(&self, kind: ColliderKind) -> bool {
        matches!(kind, ColliderKind::Sphere)
    }

    fn valid_vector(&self, vector: Vec2) -> bool {
        vector.is_finite()
    }

    fn valid_orientation(&self, orientation: Iso2) -> bool {
        EuclideanR2.valid_orientation(orientation)
    }

    fn valid_angular_velocity(&self, angular_velocity: Bivector2) -> bool {
        angular_velocity.0.is_finite()
    }

    fn valid_inertia(&self, inertia: f32) -> bool {
        EuclideanR2.valid_inertia(inertia)
    }

    fn integrate_orientation(&self, iso: Iso2, omega: Bivector2, dt: f32) -> Iso2 {
        EuclideanR2.integrate_orientation(iso, omega, dt)
    }

    fn apply_inv_inertia(&self, inertia: f32, torque: Bivector2) -> Bivector2 {
        EuclideanR2.apply_inv_inertia(inertia, torque)
    }

    fn wedge(&self, a: Vec2, b: Vec2) -> Bivector2 {
        Bivector2(cross2d(a, b))
    }

    fn velocity_at_point(&self, body: &RigidBody<SeamR2>, p: Vec2) -> Vec2 {
        let r = self.log(body.position, p);
        let w = body.angular_velocity.0;
        body.velocity + Vec2::new(-w * r.y, w * r.x)
    }

    fn effective_mass_inv(
        &self,
        a: &RigidBody<SeamR2>,
        b: &RigidBody<SeamR2>,
        contact_point: Vec2,
        direction: Vec2,
    ) -> f32 {
        let ra = self.log(a.position, contact_point);
        let rb = self.log(b.position, contact_point);
        let ra_cross = cross2d(ra, direction);
        let rb_cross = cross2d(rb, direction);
        a.inv_mass()
            + b.inv_mass()
            + ra_cross * ra_cross * inv_inertia(a)
            + rb_cross * rb_cross * inv_inertia(b)
    }

    fn apply_contact_impulse(
        &self,
        a: &mut RigidBody<SeamR2>,
        b: &mut RigidBody<SeamR2>,
        contact_point: Vec2,
        direction: Vec2,
        magnitude: f32,
    ) {
        let ra = self.log(a.position, contact_point);
        let rb = self.log(b.position, contact_point);
        let lin = direction * magnitude;
        a.velocity -= lin * a.inv_mass();
        b.velocity += lin * b.inv_mass();

        let inv_i_a = inv_inertia(a);
        let inv_i_b = inv_inertia(b);
        a.angular_velocity = Bivector2(a.angular_velocity.0 - cross2d(ra, lin) * inv_i_a);
        b.angular_velocity = Bivector2(b.angular_velocity.0 + cross2d(rb, lin) * inv_i_b);
    }
}

const PLACES: [f32; 4] = [0.25, 9.75, 5.0, 5.6];

fn seam_world(bound: BroadphaseBound) -> (World<SeamR2>, Vec<BodyId>) {
    let mut world = World::new(SeamR2 { bound });
    let ids = PLACES
        .iter()
        .map(|&x| {
            world.push_body(
                BodyDef::new(
                    Vec2::new(x, 0.0),
                    Vec2::ZERO,
                    Collider::sphere_at_origin(RADIUS),
                    1.0,
                    1.0,
                    world.space(),
                )
                .unwrap(),
            )
        })
        .collect();
    (world, ids)
}

fn key(a: BodyId, b: BodyId) -> PairKey {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

fn every_pair(ids: &[BodyId]) -> Vec<PairKey> {
    let mut pairs = Vec::new();
    for i in 0..ids.len() {
        for j in (i + 1)..ids.len() {
            pairs.push(key(ids[i], ids[j]));
        }
    }
    pairs.sort_unstable();
    pairs
}

fn touching_under_the_quotient(world: &World<SeamR2>, ids: &[BodyId]) -> Vec<PairKey> {
    let mut pairs = Vec::new();
    for i in 0..ids.len() {
        for j in (i + 1)..ids.len() {
            let a = world.bodies()[ids[i]].position;
            let b = world.bodies()[ids[j]].position;
            if seam_distance(a, b) <= 2.0 * RADIUS {
                pairs.push(key(ids[i], ids[j]));
            }
        }
    }
    pairs.sort_unstable();
    pairs
}

#[test]
fn an_unknown_bound_keeps_the_pair_a_flat_sweep_prunes_across_the_seam() {
    let (unknown, ids) = seam_world(BroadphaseBound::Unknown);
    let seam = key(ids[0], ids[1]);
    let inside = key(ids[2], ids[3]);

    let mut overlapping = vec![seam, inside];
    overlapping.sort_unstable();
    assert_eq!(
        touching_under_the_quotient(&unknown, &ids),
        overlapping,
        "the fixture must have one seam-crossing overlap and one ordinary one"
    );

    let unpruned = unknown.broadphase();
    assert_eq!(
        unpruned,
        every_pair(&ids),
        "an unknown bound must leave every pair a candidate"
    );

    let (certified, _) = seam_world(BroadphaseBound::Certified);
    let swept = certified.broadphase();
    assert!(
        swept.binary_search(&inside).is_ok(),
        "the certified sweep dropped a pair that overlaps inside the chart"
    );
    assert!(
        swept.binary_search(&seam).is_err(),
        "the certified sweep kept the seam pair, so this scene cannot tell the bounds apart"
    );
}
