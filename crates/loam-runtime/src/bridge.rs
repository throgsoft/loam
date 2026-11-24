use crate::domain::{DomainError, DomainId};
use crate::entity::Entity;
use crate::store::StoreError;
use crate::view::{ImageSpaceId, Placement, ViewId};

/// Linked from its anchor to the source view's eye; releasing either entity unlinks it and unplaces `image` inside that despawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bridge {
    pub source: DomainId,
    pub view: ViewId,
    pub image: ImageSpaceId,
}

pub struct BridgeSpec {
    pub anchor: Entity,
    pub into: ImageSpaceId,
    pub source: DomainId,
    pub view: ViewId,
    pub placement: Placement,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BridgeError {
    UnknownDomain(DomainId),
    UnknownView(ViewId),
    UnknownImage(ImageSpaceId),
    Nonlinear(&'static str),
    Domain(DomainError),
    Link(StoreError),
}

impl From<DomainError> for BridgeError {
    fn from(error: DomainError) -> Self {
        Self::Domain(error)
    }
}

impl From<StoreError> for BridgeError {
    fn from(error: StoreError) -> Self {
        Self::Link(error)
    }
}
