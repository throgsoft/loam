use std::collections::BTreeMap;

use crate::body::{BodyArena, BodyId};
use crate::collider::ColliderKind;
use crate::geometry::GeometryStore;
use crate::integrator::PhysicsSpace;
use crate::manifold::Manifold;
use crate::world::{FieldId, PairKey};

/// Physics state without configuration: gravity, solver iterations, narrowphase functions, and the field objects stay with the world, while field bindings and anchor ids travel.
#[cfg_attr(feature = "persist", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "persist",
    serde(bound(
        serialize = "S::Point: serde::Serialize, S::Vector: serde::Serialize, S::Iso: serde::Serialize, S::AngVel: serde::Serialize, S::Inertia: serde::Serialize",
        deserialize = "S::Point: serde::Deserialize<'de>, S::Vector: serde::Deserialize<'de>, S::Iso: serde::Deserialize<'de>, S::AngVel: serde::Deserialize<'de>, S::Inertia: serde::Deserialize<'de>"
    ))
)]
pub struct WorldState<S: PhysicsSpace> {
    pub bodies: BodyArena<S>,
    pub geometry: GeometryStore,
    pub manifolds: BTreeMap<PairKey, Manifold<S>>,
    pub time: f32,
    pub field_bindings: Vec<(BodyId, FieldId)>,
    pub field_anchors: Vec<BodyId>,
    #[cfg_attr(feature = "persist", serde(skip))]
    pub(crate) registrations: Vec<(ColliderKind, ColliderKind)>,
}
