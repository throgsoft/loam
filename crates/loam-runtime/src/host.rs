use crate::command::Rejection;
use crate::domain::DomainError;
use crate::input::Bindings;
use crate::session::{RestoreError, Session};
use crate::stores::Stores;

pub struct HostConfig {
    pub title: &'static str,
    pub bindings: Bindings,
}

impl HostConfig {
    pub fn new(title: &'static str, bindings: Bindings) -> Self {
        Self { title, bindings }
    }
}

#[derive(Debug)]
pub enum HostError {
    MissingCapability(&'static str),
    Setup(Rejection),
}

impl From<Rejection> for HostError {
    fn from(rejection: Rejection) -> Self {
        Self::Setup(rejection)
    }
}

impl From<DomainError> for HostError {
    fn from(error: DomainError) -> Self {
        Self::Setup(Rejection::Domain(error))
    }
}

impl From<RestoreError> for HostError {
    fn from(error: RestoreError) -> Self {
        Self::Setup(Rejection::Restore(error))
    }
}

/// The host owns the window, device, and loop; the session stays a CPU value.
pub fn run<A: Stores>(_session: Session<A>, _config: HostConfig) -> Result<(), HostError> {
    todo!()
}
