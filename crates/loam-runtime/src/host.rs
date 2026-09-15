use std::fmt;
use std::ops::Range;

use crate::command::Rejection;
use crate::domain::DomainError;
use crate::input::{ActionEvent, ActionId, Bindings, Input, Key};
use crate::phase::PhaseError;
use crate::session::{Publication, RestoreError, Session};
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
    Phase(PhaseError),
    Host(String),
}

impl fmt::Display for HostError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingCapability(capability) => write!(f, "the host is missing {capability}"),
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

pub fn run_headless<A: Stores>(
    session: &mut Session<A>,
    config: &HostConfig,
    steps: u32,
    holds: &[(Key, Range<u32>)],
) -> Result<Publication, HostError> {
    let mut publication = Publication::default();
    let mut held: Vec<ActionId> = Vec::new();
    for step in 0..steps {
        let now: Vec<ActionId> = holds
            .iter()
            .filter(|(_, range)| range.contains(&step))
            .filter_map(|(key, _)| config.bindings.action(*key))
            .collect();
        let edges = |from: &[ActionId], to: &[ActionId], pressed: bool| {
            from.iter()
                .filter(|action| !to.contains(action))
                .map(|&action| ActionEvent { action, pressed })
                .collect::<Vec<_>>()
        };
        let mut actions = edges(&now, &held, true);
        actions.extend(edges(&held, &now, false));
        held = now;
        session.boundary(Input {
            actions,
            held: held.clone(),
            ..Input::default()
        })?;
        session.tick()?;
        session.publish(&mut publication)?;
    }
    Ok(publication)
}
