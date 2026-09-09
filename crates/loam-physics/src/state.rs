use std::collections::BTreeMap;

use crate::body::{BodyArena, BodyId};
use crate::collider::ColliderKind;
use crate::integrator::PhysicsSpace;
use crate::manifold::Manifold;
use crate::world::PairKey;

pub struct WorldState<S: PhysicsSpace> {
    pub bodies: BodyArena<S>,
    pub manifolds: BTreeMap<PairKey, Manifold<S>>,
    pub dirty: Vec<BodyId>,
    pub time: f32,
    pub(crate) registrations: Vec<(ColliderKind, ColliderKind)>,
}
