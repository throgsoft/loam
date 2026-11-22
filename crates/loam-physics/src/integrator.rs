use std::ops::Mul;

use loam_math::{Bivector, IsometryGroup, Space};

use crate::body::RigidBody;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BroadphaseBound {
    /// Chart distance is a lower bound on separation everywhere, seams and identifications included.
    Certified,
    /// No distance pruning; every pair that passes the static and mask filters is a candidate.
    Unknown,
}

pub trait PhysicsSpace: Space + IsometryGroup + Send + Sync {
    type AngVel: Bivector + Send + Sync;

    type Inertia: Copy + Send + Sync;

    fn broadphase_bound(&self) -> BroadphaseBound;

    fn supports_collider(&self, kind: crate::ColliderKind) -> bool;

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
