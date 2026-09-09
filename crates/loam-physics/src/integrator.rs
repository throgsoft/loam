use std::ops::Mul;

use loam_math::{Bivector, IsometryGroup, Space};

use crate::body::RigidBody;

pub trait PhysicsSpace: Space + IsometryGroup {
    type AngVel: Bivector;

    type Inertia: Copy;

    fn supports_collider(&self, kind: crate::ColliderKind) -> bool;

    fn valid_point(&self, position: Self::Point) -> bool;

    fn valid_vector(&self, vector: Self::Vector) -> bool;

    fn valid_inertia(&self, inertia: Self::Inertia) -> bool;

    fn valid_orientation(&self, orientation: Self::Iso) -> bool;

    fn valid_angular_velocity(&self, angular_velocity: Self::AngVel) -> bool;

    fn valid_initial_state(
        &self,
        position: Self::Point,
        velocity: Self::Vector,
        inertia: Self::Inertia,
    ) -> bool {
        self.valid_point(position) && self.valid_vector(velocity) && self.valid_inertia(inertia)
    }

    fn integrate_orientation(&self, iso: Self::Iso, omega: Self::AngVel, dt: f32) -> Self::Iso;

    fn apply_inv_inertia(&self, inertia: Self::Inertia, torque: Self::AngVel) -> Self::AngVel;

    fn wedge(&self, a: Self::Vector, b: Self::Vector) -> Self::AngVel;

    fn velocity_at_point(&self, body: &RigidBody<Self>, p: Self::Point) -> Self::Vector
    where
        Self: Sized;

    /// Sleeping and static bodies contribute zero.
    fn effective_mass_inv(
        &self,
        a: &RigidBody<Self>,
        b: &RigidBody<Self>,
        contact_point: Self::Point,
        direction: Self::Vector,
    ) -> f32
    where
        Self: Sized;

    /// Subtracts from A, adds to B.
    fn apply_contact_impulse(
        &self,
        a: &mut RigidBody<Self>,
        b: &mut RigidBody<Self>,
        contact_point: Self::Point,
        direction: Self::Vector,
        magnitude: f32,
    ) where
        Self: Sized;
}

/// Velocity is transported to the new position before orientation advances.
pub fn integrate_body<S>(space: &S, body: &mut RigidBody<S>, dt: f32)
where
    S: PhysicsSpace,
    S::Vector: Mul<f32, Output = S::Vector>,
{
    if body.inv_mass() == 0.0 {
        return;
    }

    let p_old = body.position;
    let v_dt = body.velocity * dt;
    let p_new = space.exp(p_old, v_dt);
    body.velocity = space.parallel_transport(p_old, p_new, body.velocity);
    body.position = p_new;
    body.orientation = space.integrate_orientation(body.orientation, body.angular_velocity, dt);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::BodyDef;
    use crate::collider::Collider;
    use crate::geometry::GeometryStore;
    use glam::Vec3;
    use loam_math::EuclideanR3;

    fn row(def: BodyDef<EuclideanR3>) -> RigidBody<EuclideanR3> {
        def.into_row(&mut GeometryStore::default(), &EuclideanR3)
    }

    #[test]
    fn static_body_skips_integration() {
        let mut body = row(BodyDef::<EuclideanR3>::fixed(
            Vec3::ZERO,
            Collider::sphere_at_origin(0.5),
            1.0,
            &EuclideanR3,
        )
        .unwrap());
        body.velocity = Vec3::new(10.0, 0.0, 0.0);
        integrate_body(&EuclideanR3, &mut body, 1.0);
        assert_eq!(body.position, Vec3::ZERO);
    }

    #[test]
    fn dynamic_body_in_e3_moves_linearly() {
        let mut body = row(BodyDef::<EuclideanR3>::new(
            Vec3::ZERO,
            Vec3::new(1.0, 2.0, -3.0),
            Collider::sphere_at_origin(0.1),
            1.0,
            0.1,
            &EuclideanR3,
        )
        .unwrap());
        integrate_body(&EuclideanR3, &mut body, 0.5);
        assert_eq!(body.position, Vec3::new(0.5, 1.0, -1.5));
        assert_eq!(body.velocity, Vec3::new(1.0, 2.0, -3.0));
    }

    #[test]
    fn zero_dt_does_not_advance_state() {
        let mut body = row(BodyDef::<EuclideanR3>::new(
            Vec3::new(2.0, 3.0, 5.0),
            Vec3::new(7.0, 11.0, 13.0),
            Collider::sphere_at_origin(0.1),
            1.0,
            0.1,
            &EuclideanR3,
        )
        .unwrap());
        let before = (body.position, body.velocity);
        integrate_body(&EuclideanR3, &mut body, 0.0);
        assert_eq!((body.position, body.velocity), before);
    }
}
