use glam::{Quat, Vec3};

use loam_math::{Bivector, Bivector3, EuclideanR3, Iso3};
use loam_shape::shape::convex_polytope3;
use loam_shape::{Shape, ShapeKind};

use crate::body::{BodyDef, RigidBody};
use crate::collision::{epa, gjk_intersect, ContactInfo, GjkResult, PosedHull, Sphere, SupportFn};
use crate::error::EditError;
use crate::euclidean::{register_pairs, Euclidean};
use crate::narrowphase::Narrowphase;
use crate::space::{BroadphaseBound, PhysicsSpace};

// `(s, xy, yz, zx)` maps to `(w, z, x, y)`, matching `Rotor3::apply`.
fn rotor_to_quat(r: loam_math::Rotor3) -> Quat {
    Quat::from_xyzw(r.yz, r.zx, r.xy, r.s)
}

fn omega_cross_r(w: Bivector3, r: Vec3) -> Vec3 {
    Vec3::new(
        w.zx * r.z - w.xy * r.y,
        w.xy * r.x - w.yz * r.z,
        w.yz * r.y - w.zx * r.x,
    )
}

fn wedge(r: Vec3, f: Vec3) -> Bivector3 {
    Bivector3::new(
        r.x * f.y - r.y * f.x,
        r.y * f.z - r.z * f.y,
        r.z * f.x - r.x * f.z,
    )
}

fn bivec_mag_sq(b: Bivector3) -> f32 {
    b.xy * b.xy + b.yz * b.yz + b.zx * b.zx
}

fn inv_inertia(body: &RigidBody<EuclideanR3>) -> f32 {
    if body.inv_mass() > 0.0 && body.inertia > 0.0 {
        1.0 / body.inertia
    } else {
        0.0
    }
}

impl PhysicsSpace for EuclideanR3 {
    type AngVel = Bivector3;
    type Inertia = f32;

    fn broadphase_bound(&self) -> BroadphaseBound {
        BroadphaseBound::Certified
    }

    fn supports_collider(&self, kind: ShapeKind) -> bool {
        matches!(
            kind,
            ShapeKind::Sphere | ShapeKind::ConvexPolytope3D | ShapeKind::HalfSpace
        )
    }

    fn valid_vector(&self, vector: Vec3) -> bool {
        vector.is_finite()
    }

    fn valid_orientation(&self, orientation: Iso3) -> bool {
        orientation.rotation.is_finite()
            && (orientation.rotation.length_squared() - 1.0).abs() <= 1e-4
            && orientation.translation.is_finite()
    }

    fn valid_angular_velocity(&self, angular_velocity: Bivector3) -> bool {
        angular_velocity.xy.is_finite()
            && angular_velocity.yz.is_finite()
            && angular_velocity.zx.is_finite()
    }

    fn valid_inertia(&self, inertia: f32) -> bool {
        inertia.is_finite() && inertia >= 0.0 && (inertia == 0.0 || inertia.recip().is_finite())
    }

    fn integrate_orientation(&self, iso: Iso3, omega: Bivector3, dt: f32) -> Iso3 {
        debug_assert!(
            omega.xy.is_finite() && omega.yz.is_finite() && omega.zx.is_finite(),
            "non-finite Bivector3 angular velocity in integrate_orientation",
        );
        let delta_rotor = (omega * dt).exp();
        let delta_quat = rotor_to_quat(delta_rotor);
        let composed = delta_quat * iso.rotation;
        let rotation = composed.normalize();
        Iso3 {
            rotation,
            translation: iso.translation,
        }
    }

    fn apply_inv_inertia(&self, inertia: f32, torque: Bivector3) -> Bivector3 {
        if inertia > 0.0 {
            torque * (1.0 / inertia)
        } else {
            Bivector3::ZERO
        }
    }

    fn wedge(&self, a: Vec3, b: Vec3) -> Bivector3 {
        wedge(a, b)
    }

    fn velocity_at_point(&self, body: &RigidBody<EuclideanR3>, p: Vec3) -> Vec3 {
        let r = p - body.position;
        body.velocity + omega_cross_r(body.angular_velocity, r)
    }

    fn effective_mass_inv(
        &self,
        a: &RigidBody<EuclideanR3>,
        b: &RigidBody<EuclideanR3>,
        contact_point: Vec3,
        direction: Vec3,
    ) -> f32 {
        let ra = contact_point - a.position;
        let rb = contact_point - b.position;
        let ra_wedge = wedge(ra, direction);
        let rb_wedge = wedge(rb, direction);
        a.inv_mass()
            + b.inv_mass()
            + bivec_mag_sq(ra_wedge) * inv_inertia(a)
            + bivec_mag_sq(rb_wedge) * inv_inertia(b)
    }

    fn apply_contact_impulse(
        &self,
        a: &mut RigidBody<EuclideanR3>,
        b: &mut RigidBody<EuclideanR3>,
        contact_point: Vec3,
        direction: Vec3,
        magnitude: f32,
    ) {
        let ra = contact_point - a.position;
        let rb = contact_point - b.position;
        let lin = direction * magnitude;
        a.velocity -= lin * a.inv_mass();
        b.velocity += lin * b.inv_mass();
        let inv_i_a = inv_inertia(a);
        let inv_i_b = inv_inertia(b);
        a.angular_velocity = a.angular_velocity + wedge(ra, lin) * (-inv_i_a);
        b.angular_velocity = b.angular_velocity + wedge(rb, lin) * inv_i_b;
    }
}

impl Euclidean for EuclideanR3 {
    const POLYTOPE: ShapeKind = ShapeKind::ConvexPolytope3D;
    const HALFSPACE: ShapeKind = ShapeKind::HalfSpace;
    const COINCIDENT_NORMAL: Vec3 = Vec3::Y;

    type Hull<'a> = PosedHull<'a, Vec3, Quat>;
    type Ball = Sphere<Vec3>;

    fn polytope(shape: &Shape) -> Option<&[Vec3]> {
        match shape {
            Shape::ConvexPolytope3D { vertices } => Some(vertices),
            _ => None,
        }
    }

    fn halfspace(shape: &Shape) -> Option<(Vec3, f32)> {
        match *shape {
            Shape::HalfSpace { normal, offset } => Some((normal, offset)),
            _ => None,
        }
    }

    fn bounding_radius(vertices: &[Vec3]) -> f32 {
        convex_polytope3::bounding_radius(vertices)
    }

    fn rotate(orientation: Iso3, v: Vec3) -> Vec3 {
        orientation.rotation * v
    }

    fn hull(local: &[Vec3], position: Vec3, orientation: Iso3) -> PosedHull<'_, Vec3, Quat> {
        PosedHull {
            local,
            position,
            rotation: orientation.rotation,
        }
    }

    fn ball(center: Vec3, radius: f32) -> Sphere<Vec3> {
        Sphere { center, radius }
    }

    fn penetration<A: SupportFn<Vec3>, B: SupportFn<Vec3>>(
        a: &A,
        b: &B,
        initial_direction: Vec3,
        scale: f32,
    ) -> Option<ContactInfo<Vec3>> {
        let simplex = match gjk_intersect(a, b, initial_direction) {
            GjkResult::Intersecting { simplex } => simplex,
            GjkResult::Separated => return None,
        };
        epa(a, b, simplex, scale)
    }
}

pub fn register_default_narrowphase(np: &mut Narrowphase<EuclideanR3>) {
    register_pairs(np);
}

pub fn sphere_inertia(mass: f32, radius: f32) -> f32 {
    (2.0 / 5.0) * mass * radius * radius
}

pub fn sphere_body_r3(
    position: Vec3,
    velocity: Vec3,
    radius: f32,
    mass: f32,
) -> Result<BodyDef<EuclideanR3>, EditError> {
    BodyDef::new(
        position,
        velocity,
        Shape::sphere_at_origin(radius),
        mass,
        sphere_inertia(mass, radius),
        &EuclideanR3,
    )
}

/// `normal` points to the side the world is on; the plane is `dot(p, normal) = offset`.
pub fn halfspace_body_r3(normal: Vec3, offset: f32) -> Result<BodyDef<EuclideanR3>, EditError> {
    let n = normal.normalize_or_zero();
    BodyDef::fixed(
        Vec3::ZERO,
        Shape::HalfSpace { normal: n, offset },
        1.0,
        &EuclideanR3,
    )
}

/// Mean of the principal-axis tensor's diagonal; exact for a cube.
pub fn box_inertia(mass: f32, half_extents: Vec3) -> f32 {
    let w = half_extents.x * 2.0;
    let h = half_extents.y * 2.0;
    let d = half_extents.z * 2.0;
    mass * (w * w + h * h + d * d) / 18.0
}

/// Vertices are centred at the origin; support queries do not use winding.
pub fn box_vertices(half_extents: Vec3) -> Vec<Vec3> {
    let (hx, hy, hz) = (half_extents.x, half_extents.y, half_extents.z);
    vec![
        Vec3::new(-hx, -hy, -hz),
        Vec3::new(hx, -hy, -hz),
        Vec3::new(hx, hy, -hz),
        Vec3::new(-hx, hy, -hz),
        Vec3::new(-hx, -hy, hz),
        Vec3::new(hx, -hy, hz),
        Vec3::new(hx, hy, hz),
        Vec3::new(-hx, hy, hz),
    ]
}

pub fn box_body(
    position: Vec3,
    velocity: Vec3,
    half_extents: Vec3,
    mass: f32,
) -> Result<BodyDef<EuclideanR3>, EditError> {
    BodyDef::new(
        position,
        velocity,
        Shape::ConvexPolytope3D {
            vertices: box_vertices(half_extents),
        },
        mass,
        box_inertia(mass, half_extents),
        &EuclideanR3,
    )
}

/// Uses a bounding-sphere inertia approximation.
pub fn polytope_body(
    position: Vec3,
    velocity: Vec3,
    vertices: Vec<Vec3>,
    mass: f32,
) -> Result<BodyDef<EuclideanR3>, EditError> {
    let bounding_r_sq = vertices
        .iter()
        .map(|v| v.length_squared())
        .fold(0.0, f32::max);
    let inertia = (2.0 / 5.0) * mass * bounding_r_sq;
    BodyDef::new(
        position,
        velocity,
        Shape::ConvexPolytope3D { vertices },
        mass,
        inertia,
        &EuclideanR3,
    )
}

pub fn tetrahedron_vertices(r: f32) -> Vec<Vec3> {
    let k = r / 3.0_f32.sqrt();
    vec![
        Vec3::new(k, k, k),
        Vec3::new(k, -k, -k),
        Vec3::new(-k, k, -k),
        Vec3::new(-k, -k, k),
    ]
}

pub fn cube_vertices(r: f32) -> Vec<Vec3> {
    let h = r / 3.0_f32.sqrt();
    box_vertices(Vec3::splat(h))
}

pub fn octahedron_vertices(r: f32) -> Vec<Vec3> {
    vec![
        Vec3::new(r, 0.0, 0.0),
        Vec3::new(-r, 0.0, 0.0),
        Vec3::new(0.0, r, 0.0),
        Vec3::new(0.0, -r, 0.0),
        Vec3::new(0.0, 0.0, r),
        Vec3::new(0.0, 0.0, -r),
    ]
}

pub fn icosahedron_vertices(r: f32) -> Vec<Vec3> {
    let phi = (1.0 + 5.0_f32.sqrt()) * 0.5;
    let norm = (1.0 + phi * phi).sqrt();
    let s = r / norm;
    let p = phi * s;
    vec![
        Vec3::new(0.0, s, p),
        Vec3::new(0.0, s, -p),
        Vec3::new(0.0, -s, p),
        Vec3::new(0.0, -s, -p),
        Vec3::new(s, p, 0.0),
        Vec3::new(s, -p, 0.0),
        Vec3::new(-s, p, 0.0),
        Vec3::new(-s, -p, 0.0),
        Vec3::new(p, 0.0, s),
        Vec3::new(p, 0.0, -s),
        Vec3::new(-p, 0.0, s),
        Vec3::new(-p, 0.0, -s),
    ]
}

pub fn dodecahedron_vertices(r: f32) -> Vec<Vec3> {
    let phi = (1.0 + 5.0_f32.sqrt()) * 0.5;
    let inv_phi = 1.0 / phi;
    let norm = 3.0_f32.sqrt();
    let s = r / norm;
    let a = s * inv_phi;
    let b = s * phi;
    let mut v = Vec::with_capacity(20);
    for &x in &[-s, s] {
        for &y in &[-s, s] {
            for &z in &[-s, s] {
                v.push(Vec3::new(x, y, z));
            }
        }
    }
    for &y in &[-a, a] {
        for &z in &[-b, b] {
            v.push(Vec3::new(0.0, y, z));
        }
    }
    for &x in &[-a, a] {
        for &z in &[-b, b] {
            v.push(Vec3::new(z, 0.0, x));
        }
    }
    for &x in &[-a, a] {
        for &y in &[-b, b] {
            v.push(Vec3::new(y, x, 0.0));
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::world::World;
    use loam_time::par::Sequential;

    fn assert_close(a: f32, b: f32, tol: f32) {
        assert!(
            (a - b).abs() <= tol,
            "expected {a} close to {b} (tol {tol})"
        );
    }

    #[test]
    fn sphere_collision() {
        let mut world = World::new(EuclideanR3);
        register_default_narrowphase(&mut world.narrowphase);

        world.push_body(
            sphere_body_r3(
                Vec3::new(-1.0, 0.0, 0.0),
                Vec3::new(2.0, 0.0, 0.0),
                0.5,
                1.0,
            )
            .unwrap(),
        );
        world.push_body(
            sphere_body_r3(
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(-2.0, 0.0, 0.0),
                0.5,
                1.0,
            )
            .unwrap(),
        );

        for _ in 0..120 {
            world.step(1.0 / 120.0, &Sequential).unwrap();
        }
        let a = &world.bodies[0];
        let b = &world.bodies[1];
        assert_close(a.angular_velocity.magnitude(), 0.0, 1e-3);
        assert_close(b.angular_velocity.magnitude(), 0.0, 1e-3);
        assert!(a.velocity.x < 0.0, "a.velocity.x = {}", a.velocity.x);
        assert!(b.velocity.x > 0.0, "b.velocity.x = {}", b.velocity.x);

        let mut world = World::new(EuclideanR3);
        register_default_narrowphase(&mut world.narrowphase);

        let target_id = world.push_body(sphere_body_r3(Vec3::ZERO, Vec3::ZERO, 0.5, 10.0).unwrap());
        let projectile_id = world.push_body(
            sphere_body_r3(
                Vec3::new(2.0, 0.4, 0.0),
                Vec3::new(-5.0, 0.0, 0.0),
                0.3,
                1.0,
            )
            .unwrap(),
        );

        for _ in 0..120 {
            world.step(1.0 / 240.0, &Sequential).unwrap();
        }

        let target_omega = world.bodies[target_id].angular_velocity.magnitude();
        let proj_omega = world.bodies[projectile_id].angular_velocity.magnitude();
        assert!(
            target_omega > 1e-3,
            "target gained no angular velocity: omega mag = {target_omega}"
        );
        assert!(
            proj_omega > 1e-3,
            "projectile gained no angular velocity: omega mag = {proj_omega}"
        );
    }

    #[test]
    fn orientation_integration() {
        let space = EuclideanR3;
        let start = Iso3 {
            rotation: rotor_to_quat(Bivector3::new(0.8, 1.2, 0.9).exp()),
            translation: Vec3::ZERO,
        };
        let omega = Bivector3::new(0.7, 0.0, 0.5);
        let local = Vec3::new(0.5, -0.5, 0.5);
        let dt = 1e-3;

        let before = start.rotation * local;
        let after = space.integrate_orientation(start, omega, dt).rotation * local;
        let residual = (after - (before + omega_cross_r(omega, before) * dt)).length();
        assert!(
            residual < 1e-5,
            "integrated orientation left the world-frame field omega _| r: residual \
             {residual} over a step of {}",
            (after - before).length()
        );

        let mut iso = Iso3::IDENTITY;
        let omega = Bivector3::new(0.2, 0.3, -0.1);
        for _ in 0..1000 {
            iso = space.integrate_orientation(iso, omega, 1.0 / 60.0);
        }
        let len = iso.rotation.length();
        assert!(
            (len - 1.0).abs() < 1e-3,
            "orientation drifted off the unit manifold: |q| = {len}"
        );
    }
}
