use std::collections::BTreeMap;

use loam_shape::ShapeKind;

use crate::body::{BodyArena, BodyId};
use crate::geometry::GeometryStore;
use crate::integrator::PhysicsSpace;
use crate::manifold::Manifold;
use crate::world::{FieldId, PairKey};

pub struct WorldState<S: PhysicsSpace> {
    pub(crate) bodies: BodyArena<S>,
    pub(crate) geometry: GeometryStore,
    pub(crate) manifolds: BTreeMap<PairKey, Manifold<S>>,
    pub(crate) time: f32,
    pub(crate) field_bindings: Vec<(BodyId, FieldId)>,
    pub(crate) field_anchors: Vec<BodyId>,
    pub(crate) registrations: Vec<(ShapeKind, ShapeKind)>,
    pub(crate) field_registrations: Vec<ShapeKind>,
}
