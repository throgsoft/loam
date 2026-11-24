use std::ops::{Add, Mul, Neg, Sub};

use glam::{Vec2, Vec3, Vec4};
use loam_math::{Bivector, IsometryGroup, Space};

use crate::body::RigidBody;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BroadphaseBound {
    /// Chart distance is a lower bound on separation everywhere, seams and identifications included.
    Certified,
    /// No distance pruning; every pair that passes the static and mask filters is a candidate.
    Unknown,
}

pub trait VectorOps:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Neg<Output = Self>
    + Mul<f32, Output = Self>
    + PartialEq
{
    fn zero() -> Self;
    fn dot(self, rhs: Self) -> f32;
    fn is_finite(self) -> bool;

    fn length_squared(self) -> f32 {
        self.dot(self)
    }

    fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    fn normalize_or(self, fallback: Self) -> Self {
        let l2 = self.length_squared();
        if l2 > 1e-12 {
            self * (1.0 / l2.sqrt())
        } else {
            fallback
        }
    }
}

impl VectorOps for Vec2 {
    fn zero() -> Self {
        Vec2::ZERO
    }
    fn dot(self, rhs: Self) -> f32 {
        Vec2::dot(self, rhs)
    }
    fn is_finite(self) -> bool {
        Vec2::is_finite(self)
    }
}

impl VectorOps for Vec3 {
    fn zero() -> Self {
        Vec3::ZERO
    }
    fn dot(self, rhs: Self) -> f32 {
        Vec3::dot(self, rhs)
    }
    fn is_finite(self) -> bool {
        Vec3::is_finite(self)
    }
}

impl VectorOps for Vec4 {
    fn zero() -> Self {
        Vec4::ZERO
    }
    fn dot(self, rhs: Self) -> f32 {
        Vec4::dot(self, rhs)
    }
    fn is_finite(self) -> bool {
        Vec4::is_finite(self)
    }
}

pub trait PhysicsSpace: Space<Vector: VectorOps> + IsometryGroup + Send + Sync {
    type AngVel: Bivector + Send + Sync;

    type Inertia: Copy + Send + Sync;

    fn broadphase_bound(&self) -> BroadphaseBound;

    fn supports_collider(&self, kind: loam_shape::ShapeKind) -> bool;

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
pub fn integrate_body<S: PhysicsSpace>(space: &S, body: &mut RigidBody<S>, dt: f32) {
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
