use std::collections::HashMap;

use loam_shape::field::{DistanceField, FieldKind};

use crate::body::RigidBody;
use crate::collider::ColliderKind;
use crate::geometry::GeometryStore;
use crate::integrator::PhysicsSpace;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldRefusal {
    /// No query is registered for the body's collider kind.
    Collider(ColliderKind),
    /// The field's value is not an exact distance, so it is not a separation.
    Kind(FieldKind),
    DegenerateGradient,
}

/// `separation` is negative in penetration, `normal` points from the field surface toward the body, `witness` lies on that surface, all within `error`.
pub struct FieldContact<S: PhysicsSpace> {
    pub separation: f32,
    pub normal: S::Vector,
    pub witness: S::Point,
    pub error: f32,
}

impl<S: PhysicsSpace> Clone for FieldContact<S> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<S: PhysicsSpace> Copy for FieldContact<S> {}

pub type FieldContactFn<S> = fn(
    body: &RigidBody<S>,
    geometry: &GeometryStore,
    field: &dyn DistanceField,
    space: &S,
) -> Result<FieldContact<S>, FieldRefusal>;

pub struct FieldNarrowphase<S: PhysicsSpace> {
    dispatch: HashMap<ColliderKind, FieldContactFn<S>>,
    order: Vec<ColliderKind>,
}

impl<S: PhysicsSpace> Default for FieldNarrowphase<S> {
    fn default() -> Self {
        Self {
            dispatch: HashMap::new(),
            order: Vec::new(),
        }
    }
}

impl<S: PhysicsSpace> FieldNarrowphase<S> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, body: ColliderKind, query: FieldContactFn<S>) {
        if self.dispatch.insert(body, query).is_none() {
            self.order.push(body);
        }
    }

    pub fn registrations(&self) -> &[ColliderKind] {
        &self.order
    }

    pub fn test(
        &self,
        body: &RigidBody<S>,
        geometry: &GeometryStore,
        field: &dyn DistanceField,
        space: &S,
    ) -> Result<FieldContact<S>, FieldRefusal> {
        let kind = body.collider().kind();
        match self.dispatch.get(&kind) {
            Some(&query) => query(body, geometry, field, space),
            None => Err(FieldRefusal::Collider(kind)),
        }
    }
}

#[cfg(feature = "r3")]
mod r3 {
    use glam::Vec3;
    use loam_math::EuclideanR3;
    use loam_shape::field::{DistanceField, FieldKind, MIN_GRADIENT_NORM};

    use super::{FieldContact, FieldNarrowphase, FieldRefusal};
    use crate::body::RigidBody;
    use crate::collider::{Collider, ColliderKind};
    use crate::geometry::GeometryStore;

    /// Separation is the field distance at the sphere's center minus its radius; the field must be an exact distance.
    pub fn sphere_against_field(
        body: &RigidBody<EuclideanR3>,
        geometry: &GeometryStore,
        field: &dyn DistanceField,
        _space: &EuclideanR3,
    ) -> Result<FieldContact<EuclideanR3>, FieldRefusal> {
        if field.field_kind() != FieldKind::ExactDistance {
            return Err(FieldRefusal::Kind(field.field_kind()));
        }
        let Some(&Collider::Sphere { radius, .. }) = geometry.get(body.collider()) else {
            return Err(FieldRefusal::Collider(body.collider().kind()));
        };
        let center = body.position;
        let point = [center.x, center.y, center.z, 0.0];
        let distance = field.distance(point);
        let gradient = field.gradient(point);
        let direction = Vec3::new(gradient[0], gradient[1], gradient[2]);
        if direction.length() < MIN_GRADIENT_NORM {
            return Err(FieldRefusal::DegenerateGradient);
        }
        let normal = direction.normalize();
        Ok(FieldContact {
            separation: distance - radius,
            normal,
            witness: center - normal * distance,
            error: field.error(),
        })
    }

    pub fn register_field_contacts(narrowphase: &mut FieldNarrowphase<EuclideanR3>) {
        narrowphase.register(ColliderKind::Sphere, sphere_against_field);
    }
}

#[cfg(feature = "r3")]
pub use r3::{register_field_contacts, sphere_against_field};

#[cfg(all(test, feature = "r3"))]
mod tests {
    use glam::Vec3;
    use loam_math::EuclideanR3;
    use loam_shape::field::{DistanceField, FieldKind};

    use super::*;
    use crate::body::BodyId;
    use crate::collider::Collider;
    use crate::edit::EditError;
    use crate::euclidean_r3::{halfspace_body_r3, sphere_body_r3};
    use crate::manifold::PENETRATION_SLOP;
    use crate::world::{FieldId, World};

    struct Ground;

    impl DistanceField for Ground {
        fn field_kind(&self) -> FieldKind {
            FieldKind::ExactDistance
        }

        fn distance(&self, point: [f32; 4]) -> f32 {
            point[1]
        }

        fn error(&self) -> f32 {
            0.0
        }
    }

    struct Bounded;

    impl DistanceField for Bounded {
        fn field_kind(&self) -> FieldKind {
            FieldKind::ConservativeBound
        }

        fn distance(&self, point: [f32; 4]) -> f32 {
            point[1] * 0.5
        }

        fn error(&self) -> f32 {
            1.0
        }
    }

    struct Shell;

    impl DistanceField for Shell {
        fn field_kind(&self) -> FieldKind {
            FieldKind::ExactDistance
        }

        fn distance(&self, point: [f32; 4]) -> f32 {
            (point[0] * point[0] + point[1] * point[1] + point[2] * point[2]).sqrt() - 1.0
        }

        fn error(&self) -> f32 {
            0.0
        }
    }

    fn narrowphase() -> FieldNarrowphase<EuclideanR3> {
        let mut np = FieldNarrowphase::new();
        register_field_contacts(&mut np);
        np
    }

    fn one_body(collider: Collider, at: Vec3) -> World<EuclideanR3> {
        let mut world = World::new(EuclideanR3);
        world.push_body(
            crate::body::BodyDef::new(at, Vec3::ZERO, collider, 1.0, 1.0, &EuclideanR3)
                .expect("body def"),
        );
        world
    }

    const RADIUS: f32 = 0.5;

    fn ground_world() -> (World<EuclideanR3>, BodyId, FieldId, BodyId) {
        let mut world = World::new(EuclideanR3);
        register_field_contacts(&mut world.field_narrowphase);
        world.gravity = Some(Vec3::new(0.0, -9.8, 0.0));
        let anchor = world.push_body(halfspace_body_r3(Vec3::Y, 0.0).expect("anchor"));
        world.bodies[anchor].restitution = 0.0;
        let field = world
            .insert_field(anchor, Box::new(Ground))
            .expect("field handle");
        let ball = world.push_body(
            sphere_body_r3(Vec3::new(0.0, 2.0, 0.0), Vec3::ZERO, RADIUS, 1.0).expect("ball"),
        );
        world.bodies[ball].restitution = 0.0;
        (world, anchor, field, ball)
    }

    #[test]
    fn a_sphere_dropped_onto_an_exact_half_space_field_rests_at_the_analytic_height() {
        let (mut world, anchor, field, ball) = ground_world();
        assert_eq!(
            world.bind_field(anchor, field),
            Err(EditError::AnchorBindsOwnField)
        );
        world.bind_field(ball, field).expect("binding");

        for _ in 0..600 {
            world.step(1.0 / 240.0);
        }
        let rest = world.bodies[ball].position.y;
        assert!(
            (rest - (RADIUS - PENETRATION_SLOP)).abs() < 1.0e-3,
            "the ball rests at {rest}, not at radius minus the solver slop"
        );
    }

    #[test]
    fn a_binding_made_after_a_snapshot_is_gone_after_the_restore() {
        let (mut world, _, field, ball) = ground_world();
        let saved = world.snapshot();
        world.bind_field(ball, field).expect("binding");
        assert_eq!(world.field_bindings().len(), 1);
        world.restore(&saved).expect("restore");
        assert!(
            world.field_bindings().is_empty(),
            "the binding survived a restore that predates it"
        );
    }

    #[test]
    fn a_restore_whose_field_list_differs_is_refused_and_leaves_the_world_alone() {
        let (mut world, _, field, ball) = ground_world();
        world.bind_field(ball, field).expect("binding");
        for _ in 0..60 {
            world.step(1.0 / 240.0);
        }
        let saved = world.snapshot();
        let second = world.push_body(halfspace_body_r3(Vec3::Y, -8.0).expect("anchor"));
        world
            .insert_field(second, Box::new(Ground))
            .expect("field handle");
        let height = world.bodies[ball].position.y;

        assert_eq!(world.restore(&saved), Err(EditError::FieldMismatch));
        assert_eq!(world.bodies[ball].position.y, height);
        assert_eq!(world.field_bindings().len(), 1);
    }

    #[test]
    fn a_conservative_bound_field_refuses_a_sphere_and_names_the_field_kind() {
        let world = one_body(Collider::sphere_at_origin(0.5), Vec3::new(0.0, 1.0, 0.0));
        let id = world.bodies.id_at(0);
        assert_eq!(
            narrowphase()
                .test(&world.bodies[id], world.geometry(), &Bounded, &EuclideanR3)
                .err(),
            Some(FieldRefusal::Kind(FieldKind::ConservativeBound))
        );
    }

    #[test]
    fn a_hull_body_refuses_an_exact_field_and_names_the_collider_kind() {
        let corners: Vec<Vec3> = (0..8)
            .map(|i| {
                Vec3::new(
                    if i & 1 == 0 { -0.5 } else { 0.5 },
                    if i & 2 == 0 { -0.5 } else { 0.5 },
                    if i & 4 == 0 { -0.5 } else { 0.5 },
                )
            })
            .collect();
        let world = one_body(
            Collider::ConvexPolytope3D { vertices: corners },
            Vec3::new(0.0, 1.0, 0.0),
        );
        let id = world.bodies.id_at(0);
        assert_eq!(
            narrowphase()
                .test(&world.bodies[id], world.geometry(), &Ground, &EuclideanR3)
                .err(),
            Some(FieldRefusal::Collider(ColliderKind::ConvexPolytope3D))
        );
    }

    #[test]
    fn a_degenerate_gradient_refuses_rather_than_inventing_a_normal() {
        let world = one_body(Collider::sphere_at_origin(0.5), Vec3::ZERO);
        let id = world.bodies.id_at(0);
        assert_eq!(
            narrowphase()
                .test(&world.bodies[id], world.geometry(), &Shell, &EuclideanR3)
                .err(),
            Some(FieldRefusal::DegenerateGradient)
        );
    }
}
