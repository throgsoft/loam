//! World angular velocity composes as `rotation_current * delta_rotor` under Rotor4's left-first convention.

use glam::Vec4;

use loam_math::{Bivector, Bivector4, EuclideanR4, Iso4Flat, Rotor};
use loam_shape::polytope::Polytope4;

use crate::body::RigidBody;
use crate::collider::{Collider, ColliderKind};
use crate::collision::{epa_r4, gjk_intersect_r4, GjkResult4, PosedHull4, Sphere4 as GjkSphere4};
use crate::integrator::PhysicsSpace;
use crate::narrowphase::Narrowphase;
use crate::response::Contact;

/// Point velocity uses the negative Clifford left contraction.
pub fn omega_cross_r(omega: Bivector4, r: glam::Vec4) -> glam::Vec4 {
    -omega.contract_vec(r)
}

fn inv_inertia(body: &RigidBody<EuclideanR4>) -> f32 {
    if body.inv_mass() > 0.0 && body.inertia > 0.0 {
        1.0 / body.inertia
    } else {
        0.0
    }
}

impl PhysicsSpace for EuclideanR4 {
    type AngVel = Bivector4;
    type Inertia = f32;

    fn supports_collider(&self, kind: ColliderKind) -> bool {
        matches!(
            kind,
            ColliderKind::Sphere | ColliderKind::ConvexPolytope4D | ColliderKind::HalfSpace4D
        )
    }

    fn valid_initial_state(&self, position: Vec4, velocity: Vec4, inertia: f32) -> bool {
        position.is_finite()
            && velocity.is_finite()
            && inertia.is_finite()
            && inertia >= 0.0
            && (inertia == 0.0 || inertia.recip().is_finite())
    }

    fn integrate_orientation(&self, iso: Iso4Flat, omega: Bivector4, dt: f32) -> Iso4Flat {
        debug_assert!(
            omega.xy.is_finite()
                && omega.xz.is_finite()
                && omega.xw.is_finite()
                && omega.yz.is_finite()
                && omega.yw.is_finite()
                && omega.zw.is_finite(),
            "non-finite Bivector4 angular velocity in integrate_orientation",
        );
        let delta = (omega * dt).exp();
        let composed = iso.rotation * delta;
        Iso4Flat {
            rotation: composed.normalize(),
            translation: iso.translation,
        }
    }

    fn apply_inv_inertia(&self, inertia: f32, torque: Bivector4) -> Bivector4 {
        if inertia > 0.0 {
            torque * (1.0 / inertia)
        } else {
            Bivector4::ZERO
        }
    }

    fn wedge(&self, a: Vec4, b: Vec4) -> Bivector4 {
        Bivector4::wedge(a, b)
    }

    fn velocity_at_point(&self, body: &RigidBody<EuclideanR4>, p: Vec4) -> Vec4 {
        let r = p - body.position;
        body.velocity + omega_cross_r(body.angular_velocity, r)
    }

    fn effective_mass_inv(
        &self,
        a: &RigidBody<EuclideanR4>,
        b: &RigidBody<EuclideanR4>,
        contact_point: Vec4,
        direction: Vec4,
    ) -> f32 {
        let ra = contact_point - a.position;
        let rb = contact_point - b.position;
        let ra_wedge = Bivector4::wedge(ra, direction);
        let rb_wedge = Bivector4::wedge(rb, direction);
        a.inv_mass()
            + b.inv_mass()
            + ra_wedge.magnitude_squared() * inv_inertia(a)
            + rb_wedge.magnitude_squared() * inv_inertia(b)
    }

    fn apply_contact_impulse(
        &self,
        a: &mut RigidBody<EuclideanR4>,
        b: &mut RigidBody<EuclideanR4>,
        contact_point: Vec4,
        direction: Vec4,
        magnitude: f32,
    ) {
        let ra = contact_point - a.position;
        let rb = contact_point - b.position;
        let lin = direction * magnitude;
        a.velocity -= lin * a.inv_mass();
        b.velocity += lin * b.inv_mass();

        let inv_i_a = inv_inertia(a);
        let inv_i_b = inv_inertia(b);
        a.angular_velocity = a.angular_velocity + Bivector4::wedge(ra, lin) * (-inv_i_a);
        b.angular_velocity = b.angular_velocity + Bivector4::wedge(rb, lin) * inv_i_b;
    }
}

fn sphere_sphere_r4(
    a: &RigidBody<EuclideanR4>,
    b: &RigidBody<EuclideanR4>,
    space: &EuclideanR4,
) -> Option<Contact<EuclideanR4>> {
    let Collider::Sphere { radius: ra, .. } = *a.collider() else {
        return None;
    };
    let Collider::Sphere { radius: rb, .. } = *b.collider() else {
        return None;
    };

    use loam_math::Space;
    let d = space.distance(a.position, b.position);
    let combined = ra + rb;
    if d >= combined {
        return None;
    }
    let log = space.log(a.position, b.position);
    let len = log.length();
    let normal = if len > 1e-8 { log / len } else { Vec4::Y };

    let surface_a = a.position + normal * ra;
    let surface_b = b.position - normal * rb;
    let point = (surface_a + surface_b) * 0.5;

    Some(Contact {
        normal,
        point,
        penetration: combined - d,
        restitution: (a.restitution + b.restitution) * 0.5,
    })
}

fn sphere_halfspace_r4(
    a: &RigidBody<EuclideanR4>,
    b: &RigidBody<EuclideanR4>,
    _space: &EuclideanR4,
) -> Option<Contact<EuclideanR4>> {
    let Collider::Sphere { radius, .. } = *a.collider() else {
        return None;
    };
    let Collider::HalfSpace4D { normal, offset } = *b.collider() else {
        return None;
    };
    let signed = a.position.dot(normal) - offset;
    let penetration = radius - signed;
    if penetration <= 0.0 {
        return None;
    }
    let contact_normal = -normal;
    let point = a.position - normal * radius;
    Some(Contact {
        normal: contact_normal,
        point,
        penetration,
        restitution: (a.restitution + b.restitution) * 0.5,
    })
}

fn polytope_halfspace_r4(
    a: &RigidBody<EuclideanR4>,
    b: &RigidBody<EuclideanR4>,
    _space: &EuclideanR4,
) -> Option<Contact<EuclideanR4>> {
    let Collider::ConvexPolytope4D { vertices: va_local } = a.collider() else {
        return None;
    };
    let Collider::HalfSpace4D {
        normal: plane_n,
        offset,
    } = *b.collider()
    else {
        return None;
    };

    let mut deepest = Vec4::ZERO;
    let mut deepest_depth = 0.0_f32;
    for &v_local in va_local {
        let v_world = a.orientation.rotation.apply(v_local) + a.position;
        let signed = v_world.dot(plane_n) - offset;
        let depth = -signed;
        if depth > deepest_depth {
            deepest_depth = depth;
            deepest = v_world;
        }
    }
    if deepest_depth <= 0.0 {
        return None;
    }
    Some(Contact {
        normal: -plane_n,
        point: deepest,
        penetration: deepest_depth,
        restitution: (a.restitution + b.restitution) * 0.5,
    })
}

fn polytope4_bounding_radius(local_vertices: &[Vec4]) -> f32 {
    local_vertices
        .iter()
        .map(|v| v.length_squared())
        .fold(0.0_f32, f32::max)
        .sqrt()
}

fn validate_contact4(
    info: &crate::collision::ContactInfo4,
    a: &RigidBody<EuclideanR4>,
    b: &RigidBody<EuclideanR4>,
) -> Option<Contact<EuclideanR4>> {
    if !info.penetration.is_finite()
        || info.penetration <= 0.0
        || !info.normal.is_finite()
        || !info.point.is_finite()
    {
        return None;
    }
    let n2 = info.normal.length_squared();
    if !(0.5..=1.5).contains(&n2) {
        return None;
    }
    Some(Contact {
        normal: info.normal,
        point: info.point,
        penetration: info.penetration,
        restitution: (a.restitution + b.restitution) * 0.5,
    })
}

fn polytope_polytope_r4(
    a: &RigidBody<EuclideanR4>,
    b: &RigidBody<EuclideanR4>,
    _space: &EuclideanR4,
) -> Option<Contact<EuclideanR4>> {
    let Collider::ConvexPolytope4D { vertices: va_local } = a.collider() else {
        return None;
    };
    let Collider::ConvexPolytope4D { vertices: vb_local } = b.collider() else {
        return None;
    };

    let ra = polytope4_bounding_radius(va_local);
    let rb = polytope4_bounding_radius(vb_local);
    let center_dist_sq = (b.position - a.position).length_squared();
    let combined = ra + rb;
    if center_dist_sq > combined * combined {
        return None;
    }

    let hull_a = PosedHull4 {
        local: va_local,
        position: a.position,
        rotation: a.orientation.rotation,
    };
    let hull_b = PosedHull4 {
        local: vb_local,
        position: b.position,
        rotation: b.orientation.rotation,
    };

    let initial_dir = b.position - a.position;
    let simplex = match gjk_intersect_r4(&hull_a, &hull_b, initial_dir) {
        GjkResult4::Intersecting { simplex } => simplex,
        GjkResult4::Separated => return None,
    };
    let info = epa_r4(&hull_a, &hull_b, simplex, combined)?;
    validate_contact4(&info, a, b)
}

fn sphere_polytope_r4(
    a: &RigidBody<EuclideanR4>,
    b: &RigidBody<EuclideanR4>,
    _space: &EuclideanR4,
) -> Option<Contact<EuclideanR4>> {
    let Collider::Sphere { radius, .. } = *a.collider() else {
        return None;
    };
    let Collider::ConvexPolytope4D { vertices: vb_local } = b.collider() else {
        return None;
    };

    let rb = polytope4_bounding_radius(vb_local);
    let center_dist_sq = (b.position - a.position).length_squared();
    let combined = radius + rb;
    if center_dist_sq > combined * combined {
        return None;
    }

    let support_a = GjkSphere4 {
        center: a.position,
        radius,
    };
    let support_b = PosedHull4 {
        local: vb_local,
        position: b.position,
        rotation: b.orientation.rotation,
    };
    let initial_dir = b.position - a.position;
    let simplex = match gjk_intersect_r4(&support_a, &support_b, initial_dir) {
        GjkResult4::Intersecting { simplex } => simplex,
        GjkResult4::Separated => return None,
    };
    let info = epa_r4(&support_a, &support_b, simplex, combined)?;
    validate_contact4(&info, a, b)
}

pub fn register_default_narrowphase(np: &mut Narrowphase<EuclideanR4>) {
    np.register(ColliderKind::Sphere, ColliderKind::Sphere, sphere_sphere_r4);
    np.register(
        ColliderKind::Sphere,
        ColliderKind::HalfSpace4D,
        sphere_halfspace_r4,
    );
    np.register(
        ColliderKind::ConvexPolytope4D,
        ColliderKind::ConvexPolytope4D,
        polytope_polytope_r4,
    );
    np.register(
        ColliderKind::Sphere,
        ColliderKind::ConvexPolytope4D,
        sphere_polytope_r4,
    );
    np.register(
        ColliderKind::ConvexPolytope4D,
        ColliderKind::HalfSpace4D,
        polytope_halfspace_r4,
    );
}

pub fn ball4_inertia(mass: f32, radius: f32) -> f32 {
    mass * radius * radius / 3.0
}

/// Uniform-density moment about any centroidal rotation plane.
pub fn regular_polytope4_inertia(shape: Polytope4, mass: f32, circumradius: f32) -> f32 {
    const SQRT_5: f32 = 2.236_068;
    // Lasserre and Avrachenkov, American Mathematical Monthly 108, 2001.
    let mean_radius_sq = match shape {
        Polytope4::Pentatope => 1.0 / 6.0,
        Polytope4::Tesseract => 1.0 / 3.0,
        Polytope4::Cell16 => 4.0 / 15.0,
        Polytope4::Cell24 => 13.0 / 30.0,
        Polytope4::Cell600 => (11.0 + 3.0 * SQRT_5) / 30.0,
        Polytope4::Cell120 => (215.0 + 69.0 * SQRT_5) / 600.0,
    };
    0.5 * mass * circumradius * circumradius * mean_radius_sq
}

pub fn sphere_body_r4(
    position: Vec4,
    velocity: Vec4,
    radius: f32,
    mass: f32,
) -> Option<RigidBody<EuclideanR4>> {
    RigidBody::new(
        position,
        velocity,
        Collider::sphere_at_origin(radius),
        mass,
        ball4_inertia(mass, radius),
        &EuclideanR4,
    )
}

/// The solid occupies `dot(p, normalize(normal)) <= offset`.
pub fn halfspace4_body_r4(normal: Vec4, offset: f32) -> Option<RigidBody<EuclideanR4>> {
    let n = normal.try_normalize()?;
    RigidBody::fixed(
        Vec4::ZERO,
        Collider::HalfSpace4D { normal: n, offset },
        1.0,
        &EuclideanR4,
    )
}

/// Uses a bounding-ball inertia approximation.
pub fn polytope_body_r4(
    position: Vec4,
    velocity: Vec4,
    vertices: Vec<Vec4>,
    mass: f32,
) -> Option<RigidBody<EuclideanR4>> {
    let bounding_r_sq = vertices
        .iter()
        .map(|v| v.length_squared())
        .fold(0.0, f32::max);
    let inertia = mass * bounding_r_sq / 3.0;
    RigidBody::new(
        position,
        velocity,
        Collider::ConvexPolytope4D { vertices },
        mass,
        inertia,
        &EuclideanR4,
    )
}

pub use loam_shape::polytope_geom::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::World;

    fn assert_close(a: f32, b: f32, tol: f32) {
        assert!(
            (a - b).abs() <= tol,
            "expected {a} close to {b} (tol {tol})"
        );
    }

    #[test]
    fn wall_contact_leaves_through_the_near_face_in_either_pair_order() {
        let mut np = Narrowphase::<EuclideanR4>::new();
        register_default_narrowphase(&mut np);

        let wall = RigidBody::fixed(
            Vec4::ZERO,
            Collider::ConvexPolytope4D {
                vertices: tesseract_vertices(0.2)
                    .into_iter()
                    .map(|v| v * Vec4::new(1.0, 20.0, 20.0, 20.0))
                    .collect(),
            },
            1.0,
            &EuclideanR4,
        )
        .unwrap();
        let ball = sphere_body_r4(Vec4::new(-0.05, 0.0, 0.0, 0.0), Vec4::ZERO, 0.2, 1.0).unwrap();

        let forward = np.test(&ball, &wall, &EuclideanR4).expect("overlapping");
        assert!(
            (-forward.normal).dot(Vec4::X) < -0.99,
            "ball leaves along {:?}, not back out of the near face",
            -forward.normal
        );
        let reversed = np.test(&wall, &ball, &EuclideanR4).expect("overlapping");
        assert!(
            reversed.normal.dot(Vec4::X) < -0.99,
            "flipped pair leaves the ball along {:?}",
            reversed.normal
        );
        assert_close(forward.penetration, reversed.penetration, 1e-6);
    }

    #[test]
    fn sphere_settles_on_4d_floor() {
        let mut world = World::new(EuclideanR4);
        register_default_narrowphase(&mut world.narrowphase);
        world.gravity = Some(Vec4::new(0.0, -9.8, 0.0, 0.0));
        let _floor = world.push_body(halfspace4_body_r4(Vec4::Y, 0.0).unwrap());
        let ball = world.push_body(
            sphere_body_r4(Vec4::new(0.0, 2.0, 0.0, 0.0), Vec4::ZERO, 0.5, 1.0).unwrap(),
        );
        for _ in 0..300 {
            world.step(1.0 / 60.0);
        }
        let body = &world.bodies[ball];
        let lowest = body.position.y - 0.5;
        assert!(
            (-0.05..=0.05).contains(&lowest),
            "ball tunneled through 4D floor: y_bottom = {lowest}"
        );
        assert!(
            body.velocity.length() < 0.5,
            "ball still moving: |v| = {}",
            body.velocity.length()
        );
    }

    #[test]
    fn pentatope_settles_on_4d_floor() {
        let mut world = World::new(EuclideanR4);
        register_default_narrowphase(&mut world.narrowphase);
        world.gravity = Some(Vec4::new(0.0, -9.8, 0.0, 0.0));
        let floor = world.push_body(halfspace4_body_r4(Vec4::Y, 0.0).unwrap());
        let body_id = world.push_body(
            polytope_body_r4(
                Vec4::new(0.0, 3.0, 0.0, 0.0),
                Vec4::ZERO,
                pentatope_vertices(0.5),
                1.0,
            )
            .unwrap(),
        );
        world.bodies[floor].restitution = 0.0;
        world.bodies[body_id].restitution = 0.0;

        for _ in 0..600 {
            world.step(1.0 / 60.0);
        }
        let body = &world.bodies[body_id];

        assert!(
            body.position.y.is_finite() && (-0.5..=1.0).contains(&body.position.y),
            "pentatope position out of expected resting band: y = {}",
            body.position.y
        );
        assert!(
            body.position.x.abs() < 5.0
                && body.position.z.abs() < 5.0
                && body.position.w.abs() < 5.0,
            "pentatope drifted too far horizontally: pos = {:?}",
            body.position
        );

        assert!(
            body.velocity.length() < 1.0,
            "pentatope still moving after 10 s: |v| = {}, v = {:?}",
            body.velocity.length(),
            body.velocity
        );

        let omega = body.angular_velocity;
        let omega_mag2 = omega.xy * omega.xy
            + omega.xz * omega.xz
            + omega.xw * omega.xw
            + omega.yz * omega.yz
            + omega.yw * omega.yw
            + omega.zw * omega.zw;
        assert!(
            omega_mag2.is_finite() && omega_mag2 < 4.0,
            "pentatope angular velocity blew up: |ω|² = {omega_mag2}, ω = {omega:?}"
        );
    }

    #[test]
    fn tesseract_settles_on_4d_floor() {
        let mut world = World::new(EuclideanR4);
        register_default_narrowphase(&mut world.narrowphase);
        world.gravity = Some(Vec4::new(0.0, -9.8, 0.0, 0.0));
        let floor = world.push_body(halfspace4_body_r4(Vec4::Y, 0.0).unwrap());
        let body_id = world.push_body(
            polytope_body_r4(
                Vec4::new(0.0, 3.0, 0.0, 0.0),
                Vec4::ZERO,
                tesseract_vertices(0.5),
                1.0,
            )
            .unwrap(),
        );
        world.bodies[floor].restitution = 0.0;
        world.bodies[body_id].restitution = 0.0;

        for _ in 0..600 {
            world.step(1.0 / 60.0);
        }
        let body = &world.bodies[body_id];

        assert!(
            body.position.y.is_finite() && (-0.3..=1.0).contains(&body.position.y),
            "tesseract position out of expected resting band: y = {}",
            body.position.y
        );
        assert!(
            body.velocity.length() < 1.5,
            "tesseract still moving after 10 s: |v| = {}, v = {:?}",
            body.velocity.length(),
            body.velocity
        );
        let omega = body.angular_velocity;
        let omega_mag2 = omega.xy * omega.xy
            + omega.xz * omega.xz
            + omega.xw * omega.xw
            + omega.yz * omega.yz
            + omega.yw * omega.yw
            + omega.zw * omega.zw;
        assert!(
            omega_mag2.is_finite() && omega_mag2 < 4.0,
            "tesseract angular velocity blew up: |ω|² = {omega_mag2}, ω = {omega:?}"
        );
    }

    const CORNER_DROP_DT: f32 = 1.0 / 240.0;
    const CORNER_DROP_CIRCUMRADIUS: f32 = 0.45;
    const CORNER_DROP_GRAVITY: f32 = -9.8;
    const CORNER_DROP_DRAG: f32 = 1.2;

    struct CornerDrop {
        world: World<EuclideanR4>,
        body: crate::body::BodyId,
        decay: f32,
    }

    impl CornerDrop {
        fn new() -> Self {
            let mut world = World::new(EuclideanR4);
            register_default_narrowphase(&mut world.narrowphase);
            world.gravity = Some(Vec4::new(0.0, CORNER_DROP_GRAVITY, 0.0, 0.0));
            let floor = world.push_body(halfspace4_body_r4(Vec4::Y, 0.0).unwrap());
            world.bodies[floor].restitution = 0.05;
            let body = world.push_body(
                polytope_body_r4(
                    Vec4::new(0.0, 1.6, 0.0, 0.0),
                    Vec4::ZERO,
                    cell24_vertices(CORNER_DROP_CIRCUMRADIUS),
                    1.0,
                )
                .unwrap(),
            );
            world.bodies[body].restitution = 0.05;
            world.bodies[body].orientation.rotation = Bivector4::new(0.0, 0.0, 0.0, 0.6, 0.4, 0.0)
                .exp()
                .normalize();
            world.bodies[body].inertia =
                regular_polytope4_inertia(Polytope4::Cell24, 1.0, CORNER_DROP_CIRCUMRADIUS);
            Self {
                world,
                body,
                decay: (-CORNER_DROP_DRAG * CORNER_DROP_DT).exp(),
            }
        }

        fn step(&mut self) {
            self.world.step(CORNER_DROP_DT);
            let decay = self.decay;
            let body = &mut self.world.bodies[self.body];
            body.angular_velocity = body.angular_velocity * decay;
        }

        fn height(&self) -> f32 {
            self.world.bodies[self.body].position.y
        }

        fn angular_speed(&self) -> f32 {
            self.world.bodies[self.body].angular_velocity.magnitude()
        }

        fn energy(&self) -> f32 {
            let body = &self.world.bodies[self.body];
            0.5 * body.velocity.length_squared()
                + 0.5 * body.inertia * body.angular_velocity.magnitude().powi(2)
                - CORNER_DROP_GRAVITY * body.position.y
        }
    }

    #[test]
    fn a_hull_dropped_on_a_corner_settles_without_climbing_its_own_contacts() {
        const LANDING_STEPS: usize = 400;
        const RESTING_STEPS: usize = 800;
        const CLIMB_TOLERANCE: f32 = 5e-4;

        let mut drop = CornerDrop::new();
        let start_energy = drop.energy();
        let mut landing_spin = 0.0_f32;
        for _ in 0..LANDING_STEPS {
            drop.step();
            landing_spin = landing_spin.max(drop.angular_speed());
        }
        assert!(
            landing_spin > 1.0,
            "the hull never rocked over, so a settle proves nothing: peak |ω| \
             was {landing_spin} rad/s"
        );

        let resting_height = drop.height();
        let mut peak_height = resting_height;
        for _ in 0..RESTING_STEPS {
            drop.step();
            peak_height = peak_height.max(drop.height());
        }

        assert!(
            peak_height - resting_height < CLIMB_TOLERANCE,
            "a resting hull climbed {} against gravity over {RESTING_STEPS} \
             steps",
            peak_height - resting_height,
        );
        assert!(
            drop.angular_speed() < 0.02,
            "a resting hull held {} rad/s against a {CORNER_DROP_DRAG}/s \
             damper, so the contact solve is re-injecting it",
            drop.angular_speed(),
        );
        assert!(
            drop.energy() < start_energy,
            "the drop ended with {} of energy against the {start_energy} it \
             started with",
            drop.energy(),
        );
    }

    #[test]
    fn falling_sphere_accelerates_in_r4() {
        let mut world = World::new(EuclideanR4);
        register_default_narrowphase(&mut world.narrowphase);
        world.gravity = Some(Vec4::new(0.0, -9.8, 0.0, 0.0));

        let id = world.push_body(
            sphere_body_r4(Vec4::new(0.0, 5.0, 0.0, 0.0), Vec4::ZERO, 0.5, 1.0).unwrap(),
        );
        world.step(1.0 / 60.0);
        let body = &world.bodies[id];
        assert!(body.velocity.y < -0.1 && body.velocity.y > -0.2);
        assert_close(body.velocity.x, 0.0, 1e-6);
        assert_close(body.velocity.z, 0.0, 1e-6);
        assert_close(body.velocity.w, 0.0, 1e-6);
    }

    #[test]
    fn head_on_sphere_collision_reverses_velocity() {
        let mut world = World::new(EuclideanR4);
        register_default_narrowphase(&mut world.narrowphase);

        world.push_body(
            sphere_body_r4(
                Vec4::new(-1.0, 0.0, 0.0, 0.0),
                Vec4::new(2.0, 0.0, 0.0, 0.0),
                0.5,
                1.0,
            )
            .unwrap(),
        );
        world.push_body(
            sphere_body_r4(
                Vec4::new(1.0, 0.0, 0.0, 0.0),
                Vec4::new(-2.0, 0.0, 0.0, 0.0),
                0.5,
                1.0,
            )
            .unwrap(),
        );

        for _ in 0..120 {
            world.step(1.0 / 120.0);
        }
        let a = &world.bodies[0];
        let b = &world.bodies[1];
        assert!(
            a.velocity.x < 0.0,
            "body 0 should bounce back: v.x = {}",
            a.velocity.x
        );
        assert!(
            b.velocity.x > 0.0,
            "body 1 should bounce back: v.x = {}",
            b.velocity.x
        );
        assert_close(a.velocity.y, 0.0, 1e-4);
        assert_close(a.velocity.z, 0.0, 1e-4);
        assert_close(a.velocity.w, 0.0, 1e-4);
    }

    #[test]
    fn sphere_sphere_off_plane_contact_resolves_along_line_of_centers() {
        let mut world = World::new(EuclideanR4);
        register_default_narrowphase(&mut world.narrowphase);
        let a_pos = Vec4::new(-0.8, -0.4, 0.3, 0.2);
        let b_pos = Vec4::new(0.8, 0.4, -0.3, -0.2);
        let a = world
            .push_body(sphere_body_r4(a_pos, (b_pos - a_pos).normalize() * 2.0, 0.5, 1.0).unwrap());
        let b = world
            .push_body(sphere_body_r4(b_pos, (a_pos - b_pos).normalize() * 2.0, 0.5, 1.0).unwrap());
        for _ in 0..120 {
            world.step(1.0 / 120.0);
        }
        let rel = world.bodies[b].velocity - world.bodies[a].velocity;
        let axis = (b_pos - a_pos).normalize();
        let v_along = rel.dot(axis);
        assert!((rel - axis * v_along).length() < 1e-4);
        assert!(
            v_along > 0.0,
            "relative velocity should now be separating: {v_along}"
        );
    }

    #[test]
    fn sphere_inside_tesseract_produces_contact() {
        let mut world = World::new(EuclideanR4);
        register_default_narrowphase(&mut world.narrowphase);
        let _a = world.push_body(sphere_body_r4(Vec4::ZERO, Vec4::ZERO, 0.3, 1.0).unwrap());
        let _b = world.push_body(
            polytope_body_r4(Vec4::ZERO, Vec4::ZERO, tesseract_vertices(0.8), 0.0).unwrap(),
        );
        let pair_found = {
            let (a, b) = world.bodies.dense_mut().split_at_mut(1);
            world.narrowphase.test(&a[0], &b[0], &EuclideanR4).is_some()
        };
        assert!(
            pair_found,
            "sphere inside tesseract should produce a contact"
        );
    }

    #[test]
    fn separated_pentatopes_produce_no_contact() {
        let mut world = World::new(EuclideanR4);
        register_default_narrowphase(&mut world.narrowphase);
        let _a = world.push_body(
            polytope_body_r4(Vec4::ZERO, Vec4::ZERO, pentatope_vertices(1.0), 1.0).unwrap(),
        );
        let _b = world.push_body(
            polytope_body_r4(
                Vec4::new(10.0, 0.0, 0.0, 0.0),
                Vec4::ZERO,
                pentatope_vertices(1.0),
                1.0,
            )
            .unwrap(),
        );
        let (a, b) = world.bodies.dense_mut().split_at_mut(1);
        assert!(world.narrowphase.test(&a[0], &b[0], &EuclideanR4).is_none());
    }

    #[test]
    fn integrated_orientation_advances_a_body_point_along_the_world_frame_omega() {
        let space = EuclideanR4;
        let start = Iso4Flat {
            rotation: Bivector4::new(0.8, 0.0, 0.0, std::f32::consts::FRAC_PI_2, 0.0, 0.9).exp(),
            translation: Vec4::ZERO,
        };
        let omega = Bivector4::new(0.7, 0.0, 0.0, 0.0, 0.5, 0.0);
        let local = Vec4::new(0.5, -0.5, 0.5, 0.5);
        let dt = 1e-3;

        let before = start.rotation.apply(local);
        let after = space
            .integrate_orientation(start, omega, dt)
            .rotation
            .apply(local);
        let residual = (after - (before + omega_cross_r(omega, before) * dt)).length();
        assert!(
            residual < 1e-5,
            "integrated orientation left the world-frame field ω⌋r: residual \
             {residual} over a step of {}",
            (after - before).length()
        );
    }

    #[test]
    fn orientation_integration_preserves_unit_rotor() {
        let space = EuclideanR4;
        let mut iso = Iso4Flat::IDENTITY;
        let omega = Bivector4::new(0.2, 0.0, 0.0, 0.0, 0.0, 0.15);
        for _ in 0..1000 {
            iso = space.integrate_orientation(iso, omega, 1.0 / 60.0);
        }
        let n = iso.rotation.norm_squared();
        assert!(
            (n - 1.0).abs() < 1e-3,
            "rotor drifted off the unit manifold: |R|² = {n}"
        );
    }
}
