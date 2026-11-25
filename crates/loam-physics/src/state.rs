use std::collections::BTreeMap;
use std::fmt;

use loam_shape::ShapeKind;

use crate::body::{BodyArena, BodyId};
use crate::geometry::GeometryStore;
use crate::manifold::Manifold;
use crate::space::PhysicsSpace;
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

/// Why a world refuses a snapshot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateError {
    RegistrationMismatch,
    FieldMismatch,
    InvalidTime,
    InvalidBodyArena,
    InvalidBody,
    InvalidGeometry,
    InvalidFieldBinding,
    InvalidManifold,
}

impl fmt::Display for StateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            Self::RegistrationMismatch => "the narrowphase registrations are not the snapshot's",
            Self::FieldMismatch => "the world's field anchors are not the snapshot's",
            Self::InvalidTime => "the world time is negative or not finite",
            Self::InvalidBodyArena => "the body arena is inconsistent",
            Self::InvalidBody => "a body has invalid state",
            Self::InvalidGeometry => "the geometry store is inconsistent",
            Self::InvalidFieldBinding => "a field binding is invalid",
            Self::InvalidManifold => "a contact manifold is invalid",
        };
        f.write_str(text)
    }
}

impl std::error::Error for StateError {}
