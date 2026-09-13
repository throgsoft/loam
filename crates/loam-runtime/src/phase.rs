use std::fmt;

use crate::bridge::{Drag, DragError, DragRelease};
use crate::command::{CommandResult, Commands};
use crate::domain::{ChartPoint, DomainError, Domains};
use crate::input::Input;
use crate::session::{Manipulation, PreparedGeometry};
use crate::stores::Stores;
use crate::view::{Pick, Views};

/// Fixed order; dispatch runs while paused and is the only phase where capacity grows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Phase {
    Dispatch,
    Simulation,
    Publication,
}

impl Phase {
    pub const ALL: [Phase; 3] = [Phase::Dispatch, Phase::Simulation, Phase::Publication];

    fn index(self) -> usize {
        self as usize
    }
}

/// Where a new entry goes relative to a named entry of the same phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Order {
    Before(&'static str),
    After(&'static str),
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Tick(pub u64);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Step {
    pub tick: Tick,
    pub dt: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PhaseError {
    pub phase: Phase,
    pub system: Option<&'static str>,
    pub cause: DomainError,
}

impl PhaseError {
    pub(crate) fn system(phase: Phase, system: &'static str, cause: DomainError) -> Self {
        Self {
            phase,
            system: Some(system),
            cause,
        }
    }

    pub(crate) fn unnamed(phase: Phase, cause: DomainError) -> Self {
        Self {
            phase,
            system: None,
            cause,
        }
    }
}

impl fmt::Display for PhaseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.system {
            Some(system) => write!(
                formatter,
                "system `{system}` failed in {:?}: {:?}",
                self.phase, self.cause
            ),
            None => write!(formatter, "{:?} phase failed: {:?}", self.phase, self.cause),
        }
    }
}

impl std::error::Error for PhaseError {}

pub struct Ctx<'a, A> {
    pub app: &'a mut A,
    pub domains: &'a mut Domains,
    pub views: &'a mut Views,
    pub commands: &'a mut Commands<A>,
    pub results: &'a [CommandResult],
    pub input: &'a Input,
    pub prepared: &'a [PreparedGeometry],
    pub step: Step,
    pub(crate) manipulation: &'a mut Manipulation,
}

impl<A: Stores> Ctx<'_, A> {
    pub fn pick(&self, ndc: [f32; 2]) -> Option<Pick> {
        self.domains.pick(self.views, self.prepared, ndc)
    }

    pub fn dragging(&self) -> Option<Drag> {
        self.manipulation.dragging()
    }

    pub fn grab(&mut self, ndc: [f32; 2], time: f64) -> Result<Pick, DragError> {
        self.manipulation.grab(
            self.domains,
            self.views,
            self.prepared,
            self.commands,
            ndc,
            time,
        )
    }

    pub fn drag(&mut self, ndc: [f32; 2], time: f64) -> Result<ChartPoint, DragError> {
        self.manipulation
            .drag(self.domains, self.views, self.commands, ndc, time)
    }

    pub fn release(&mut self) -> Option<DragRelease> {
        self.manipulation.release(self.commands)
    }

    pub fn release_at(&mut self, time: f64) -> Option<DragRelease> {
        self.manipulation.release_at(self.commands, time)
    }

    pub fn cancel_drag(&mut self) -> Option<DragRelease> {
        self.manipulation.cancel(self.commands)
    }
}

pub trait System<A>: Send + 'static {
    fn run(&mut self, ctx: Ctx<'_, A>) -> Result<(), DomainError>;
}

impl<A: 'static, F> System<A> for F
where
    F: FnMut(Ctx<'_, A>) -> Result<(), DomainError> + Send + 'static,
{
    fn run(&mut self, ctx: Ctx<'_, A>) -> Result<(), DomainError> {
        self(ctx)
    }
}

type Runner<A> = Box<dyn FnMut(Ctx<'_, A>) -> Result<(), DomainError> + Send>;

pub struct SystemEntry<A> {
    name: &'static str,
    run: Runner<A>,
}

impl<A: 'static> SystemEntry<A> {
    pub(crate) fn new(name: &'static str, system: impl System<A>) -> Self {
        let mut system = system;
        Self {
            name,
            run: Box::new(move |ctx| system.run(ctx)),
        }
    }
}

impl<A> SystemEntry<A> {
    pub fn name(&self) -> &'static str {
        self.name
    }

    pub(crate) fn run(&mut self, ctx: Ctx<'_, A>) -> Result<(), DomainError> {
        (self.run)(ctx)
    }
}

pub(crate) struct Phases<A> {
    entries: [Vec<SystemEntry<A>>; 3],
}

impl<A> Phases<A> {
    pub(crate) fn new() -> Self {
        Self {
            entries: Default::default(),
        }
    }

    pub(crate) fn push(&mut self, phase: Phase, entry: SystemEntry<A>) {
        self.entries[phase.index()].push(entry);
    }

    pub(crate) fn insert(
        &mut self,
        phase: Phase,
        order: Order,
        entry: SystemEntry<A>,
    ) -> Option<()> {
        let list = &mut self.entries[phase.index()];
        let index = match order {
            Order::Before(name) => list.iter().position(|entry| entry.name() == name)?,
            Order::After(name) => list.iter().position(|entry| entry.name() == name)? + 1,
        };
        list.insert(index, entry);
        Some(())
    }

    pub(crate) fn entries(&self, phase: Phase) -> &[SystemEntry<A>] {
        &self.entries[phase.index()]
    }

    pub(crate) fn entries_mut(&mut self, phase: Phase) -> &mut [SystemEntry<A>] {
        &mut self.entries[phase.index()]
    }
}
