use std::fmt;

use loam_runtime::{ActionId, DomainError, PhaseError, Rejection, RestoreError};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Key {
    Space,
    Escape,
    Letter(char),
    Digit(u8),
}

#[derive(Clone, Debug, Default)]
pub struct Bindings {
    keys: Vec<(Key, ActionId)>,
}

impl Bindings {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn key(mut self, key: Key, action: ActionId) -> Self {
        self.keys.push((key, action));
        self
    }

    pub fn action(&self, key: Key) -> Option<ActionId> {
        self.keys
            .iter()
            .find(|(bound, _)| *bound == key)
            .map(|(_, action)| *action)
    }
}

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
    Setup(Rejection),
    Phase(PhaseError),
    Host(String),
}

impl fmt::Display for HostError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Setup(rejection) => fmt::Display::fmt(rejection, f),
            Self::Phase(error) => fmt::Display::fmt(error, f),
            Self::Host(message) => f.write_str(message),
        }
    }
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

impl From<PhaseError> for HostError {
    fn from(error: PhaseError) -> Self {
        Self::Phase(error)
    }
}
