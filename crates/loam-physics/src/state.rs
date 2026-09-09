use std::collections::BTreeMap;

use crate::body::{BodyArena, BodyId};
use crate::collider::ColliderKind;
use crate::geometry::GeometryStore;
use crate::integrator::PhysicsSpace;
use crate::manifold::Manifold;
use crate::world::PairKey;

/// Physics state without configuration: gravity, solver iterations, and narrowphase functions stay with the world.
pub struct WorldState<S: PhysicsSpace> {
    pub bodies: BodyArena<S>,
    pub geometry: GeometryStore,
    pub manifolds: BTreeMap<PairKey, Manifold<S>>,
    pub dirty: Vec<BodyId>,
    pub time: f32,
    pub(crate) registrations: Vec<(ColliderKind, ColliderKind)>,
}
