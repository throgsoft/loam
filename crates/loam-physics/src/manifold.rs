//! Contact anchors and impulses persist across steps; manifold keys use generational handles.

use crate::body::{BodyId, RigidBody};
use crate::collision::VectorOps;
use crate::integrator::PhysicsSpace;
use crate::response::Contact;
use crate::response::FRICTION_COEFF;

pub const MAX_POINTS: usize = 4;

// Coumans, Bullet Physics SDK 3.x, gContactBreakingThreshold.
pub const CONTACT_BREAK_DISTANCE: f32 = 0.02;

const CONTACT_BREAK_DISTANCE_SQ: f32 = CONTACT_BREAK_DISTANCE * CONTACT_BREAK_DISTANCE;

const MERGE_RADIUS_SQ: f32 = CONTACT_BREAK_DISTANCE_SQ;

#[cfg_attr(feature = "persist", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "persist",
    serde(bound(
        serialize = "S::Point: serde::Serialize, S::Vector: serde::Serialize, S::Iso: serde::Serialize, S::AngVel: serde::Serialize, S::Inertia: serde::Serialize",
        deserialize = "S::Point: serde::Deserialize<'de>, S::Vector: serde::Deserialize<'de>, S::Iso: serde::Deserialize<'de>, S::AngVel: serde::Deserialize<'de>, S::Inertia: serde::Deserialize<'de>"
    ))
)]
pub struct ContactPoint<S: PhysicsSpace> {
    pub world_point: S::Point,
    /// Witness in A's local frame.
    pub anchor_a: S::Vector,
    /// Witness in B's local frame.
    pub anchor_b: S::Vector,
    /// Unit, from A toward B.
    pub normal: S::Vector,
    pub penetration: f32,
    pub normal_impulse: f32,
    /// Valid within one step only: the slide direction can flip.
    pub tangent_dir: S::Vector,
    /// Cleared before each step; stored as a magnitude.
    pub tangent_impulse: f32,
    /// Snapshot before the warm-start, constant across the PGS iterations.
    pub velocity_bias: f32,
}

impl<S: PhysicsSpace> Clone for ContactPoint<S> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<S: PhysicsSpace> Copy for ContactPoint<S> {}

#[cfg_attr(feature = "persist", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "persist",
    serde(bound(
        serialize = "S::Point: serde::Serialize, S::Vector: serde::Serialize, S::Iso: serde::Serialize, S::AngVel: serde::Serialize, S::Inertia: serde::Serialize",
        deserialize = "S::Point: serde::Deserialize<'de>, S::Vector: serde::Deserialize<'de>, S::Iso: serde::Deserialize<'de>, S::AngVel: serde::Deserialize<'de>, S::Inertia: serde::Deserialize<'de>"
    ))
)]
pub struct Manifold<S: PhysicsSpace> {
    /// Always `< body_b`.
    pub body_a: BodyId,
    pub body_b: BodyId,
    /// Set on first contact and kept.
    pub restitution: f32,
    pub points: Vec<ContactPoint<S>>,
}

impl<S: PhysicsSpace> Clone for Manifold<S> {
    fn clone(&self) -> Self {
        Self {
            body_a: self.body_a,
            body_b: self.body_b,
            restitution: self.restitution,
            points: self.points.clone(),
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.body_a = source.body_a;
        self.body_b = source.body_b;
        self.restitution = source.restitution;
        self.points.clone_from(&source.points);
    }
}

impl<S: PhysicsSpace> Manifold<S>
where
    S::Vector: VectorOps,
{
    pub fn new(body_a: BodyId, body_b: BodyId, restitution: f32) -> Self {
        debug_assert!(body_a < body_b);
        Self {
            body_a,
            body_b,
            restitution,
            points: Vec::with_capacity(MAX_POINTS),
        }
    }

    pub(crate) fn reset(&mut self, body_a: BodyId, body_b: BodyId, restitution: f32) {
        debug_assert!(body_a < body_b);
        self.body_a = body_a;
        self.body_b = body_b;
        self.restitution = restitution;
        self.points.clear();
    }

    pub(crate) fn is_valid(&self, space: &S, body_a: &RigidBody<S>, body_b: &RigidBody<S>) -> bool
    where
        S::Point: Copy + std::ops::Sub<Output = S::Vector>,
    {
        if !self.restitution.is_finite()
            || self.restitution < 0.0
            || self.points.is_empty()
            || self.points.len() > MAX_POINTS
        {
            return false;
        }
        self.points.iter().all(|point| {
            let normal_length = VectorOps::length_squared(point.normal);
            let tangent_length = VectorOps::length_squared(point.tangent_dir);
            if !space.valid_point(point.world_point)
                || !space.valid_vector(point.anchor_a)
                || !space.valid_vector(point.anchor_b)
                || !space.valid_vector(point.normal)
                || !space.valid_vector(point.tangent_dir)
                || (normal_length - 1.0).abs() > 1.0e-4
                || !point.penetration.is_finite()
                || !point.normal_impulse.is_finite()
                || point.normal_impulse < 0.0
                || !point.tangent_impulse.is_finite()
                || point.tangent_impulse < 0.0
                || !point.velocity_bias.is_finite()
            {
                return false;
            }
            if point.tangent_impulse == 0.0 {
                if tangent_length != 0.0 {
                    return false;
                }
            } else {
                let limit = point.normal_impulse * FRICTION_COEFF;
                let slack = f32::EPSILON * limit.abs().max(1.0);
                if (tangent_length - 1.0).abs() > 1.0e-4
                    || VectorOps::dot(point.normal, point.tangent_dir).abs() > 1.0e-4
                    || point.tangent_impulse > limit + slack
                {
                    return false;
                }
            }
            let anchor_a = anchor_world_point(space, body_a, point.anchor_a);
            let anchor_b = anchor_world_point(space, body_b, point.anchor_b);
            if !space.valid_point(anchor_a) || !space.valid_point(anchor_b) {
                return false;
            }
            let gap = space.log(anchor_a, anchor_b);
            if !space.valid_vector(gap) {
                return false;
            }
            let midpoint = space.exp(anchor_a, gap * 0.5);
            let midpoint_error = space.log(midpoint, point.world_point);
            let penetration_error = point.penetration + VectorOps::dot(gap, point.normal);
            space.valid_point(midpoint)
                && space.valid_vector(midpoint_error)
                && VectorOps::length(midpoint_error) <= CONTACT_BREAK_DISTANCE
                && penetration_error.is_finite()
                && penetration_error.abs() <= CONTACT_BREAK_DISTANCE
        })
    }

    /// Retains anchors within [`CONTACT_BREAK_DISTANCE`]; bodies must match the manifold key.
    pub fn refresh(&mut self, space: &S, a: &RigidBody<S>, b: &RigidBody<S>) {
        self.points.retain_mut(|cp| {
            let pa = anchor_world_point(space, a, cp.anchor_a);
            let pb = anchor_world_point(space, b, cp.anchor_b);
            let gap = space.log(pa, pb);
            let separation = VectorOps::dot(gap, cp.normal);
            if separation > CONTACT_BREAK_DISTANCE {
                return false;
            }
            let tangential = gap - cp.normal * separation;
            if VectorOps::length_squared(tangential) > CONTACT_BREAK_DISTANCE_SQ {
                return false;
            }
            cp.world_point = space.exp(pa, gap * 0.5);
            cp.penetration = -separation;
            true
        });
    }

    /// Nearby contacts retain impulses; a full manifold replaces its weakest contact.
    pub fn add_or_update(
        &mut self,
        space: &S,
        a: &RigidBody<S>,
        b: &RigidBody<S>,
        contact: Contact<S>,
    ) where
        S::Point: Copy + std::ops::Sub<Output = S::Vector>,
    {
        let new_point = contact.point;
        let half_gap = contact.normal * (0.5 * contact.penetration);
        let anchor_a = local_anchor(space, a, space.exp(new_point, half_gap));
        let anchor_b = local_anchor(space, b, space.exp(new_point, -half_gap));

        for cp in &mut self.points {
            let delta = new_point - cp.world_point;
            if VectorOps::length_squared(delta) < MERGE_RADIUS_SQ {
                cp.world_point = new_point;
                cp.anchor_a = anchor_a;
                cp.anchor_b = anchor_b;
                cp.normal_impulse *= VectorOps::dot(cp.normal, contact.normal).max(0.0);
                cp.tangent_impulse = 0.0;
                cp.tangent_dir = VectorOps::zero();
                cp.normal = contact.normal;
                cp.penetration = contact.penetration;
                return;
            }
        }

        let fresh = ContactPoint {
            world_point: new_point,
            anchor_a,
            anchor_b,
            normal: contact.normal,
            penetration: contact.penetration,
            normal_impulse: 0.0,
            tangent_dir: VectorOps::zero(),
            tangent_impulse: 0.0,
            velocity_bias: 0.0,
        };

        if self.points.len() < MAX_POINTS {
            self.points.push(fresh);
        } else {
            if let Some((worst, _)) = self.points.iter().enumerate().min_by(|(_, a), (_, b)| {
                let sa = a.normal_impulse + a.tangent_impulse.abs();
                let sb = b.normal_impulse + b.tangent_impulse.abs();
                sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
            }) {
                self.points[worst] = fresh;
            }
        }
    }
}

// Exact inverses only because every `PhysicsSpace` is flat.
fn local_anchor<S: PhysicsSpace>(space: &S, body: &RigidBody<S>, p: S::Point) -> S::Vector {
    let lever = space.log(body.position, p);
    space.iso_transport(space.iso_inverse(body.orientation), body.position, lever)
}

fn anchor_world_point<S: PhysicsSpace>(
    space: &S,
    body: &RigidBody<S>,
    anchor: S::Vector,
) -> S::Point {
    let lever = space.iso_transport(body.orientation, body.position, anchor);
    space.exp(body.position, lever)
}

pub const DEFAULT_PGS_ITERS: usize = 8;

pub const BAUMGARTE_BETA: f32 = 0.2;

/// Penetration tolerated without bias, in world units.
pub const PENETRATION_SLOP: f32 = 0.005;

pub const MAX_LINEAR_CORRECTION: f32 = 0.5;

/// Restitution is suppressed below this approach speed, in world units per second.
pub const RESTITUTION_THRESHOLD: f32 = 1.0;
