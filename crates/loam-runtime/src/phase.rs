use std::any::TypeId;

use crate::command::{CommandResult, Commands};
use crate::domain::{DomainId, Domains};
use crate::input::Input;
use crate::view::{Pick, Views};

/// Fixed order; dispatch runs while paused and is the only phase where capacity grows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Phase {
    Dispatch,
    Simulation,
    Publication,
    Presentation,
}

impl Phase {
    pub const ALL: [Phase; 4] = [
        Phase::Dispatch,
        Phase::Simulation,
        Phase::Publication,
        Phase::Presentation,
    ];

    fn index(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Tick(pub u64);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Step {
    pub tick: Tick,
    pub dt: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StoreId(TypeId);

impl StoreId {
    pub fn of<T: 'static>() -> Self {
        Self(TypeId::of::<T>())
    }

    pub fn type_id(self) -> TypeId {
        self.0
    }
}

/// Declared at registration; the runtime infers nothing from a function pointer.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Access {
    reads: Vec<StoreId>,
    writes: Vec<StoreId>,
    domains: Vec<DomainId>,
    views: bool,
    commands: bool,
}

impl Access {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reads<T: 'static>(mut self) -> Self {
        self.reads.push(StoreId::of::<T>());
        self
    }

    pub fn writes<T: 'static>(mut self) -> Self {
        self.writes.push(StoreId::of::<T>());
        self
    }

    pub fn domain(mut self, id: DomainId) -> Self {
        self.domains.push(id);
        self
    }

    pub fn views(mut self) -> Self {
        self.views = true;
        self
    }

    pub fn commands(mut self) -> Self {
        self.commands = true;
        self
    }

    pub fn read_set(&self) -> &[StoreId] {
        &self.reads
    }

    pub fn write_set(&self) -> &[StoreId] {
        &self.writes
    }

    pub fn domain_set(&self) -> &[DomainId] {
        &self.domains
    }

    pub fn touches_views(&self) -> bool {
        self.views
    }

    pub fn submits_commands(&self) -> bool {
        self.commands
    }
}

/// The common callback every adapter targets.
pub struct Ctx<'a, A> {
    pub app: &'a mut A,
    pub domains: &'a mut Domains,
    pub views: &'a mut Views,
    pub commands: &'a mut Commands<A>,
    pub results: &'a [CommandResult],
    pub input: &'a Input,
    pub step: Step,
}

impl<A> Ctx<'_, A> {
    pub fn pick(&self, ndc: [f32; 2]) -> Option<Pick> {
        self.domains.pick(self.views, ndc)
    }
}

pub trait System<A, Marker>: Send + 'static {
    fn run(&mut self, ctx: Ctx<'_, A>);
}

/// Marker types, one per adapted signature.
pub mod signature {
    use super::{Commands, Ctx, Domains, Input, Step, System, Views};

    macro_rules! adapt {
        ($marker:ident, $( $param:ty => $pick:ident ),+) => {
            pub struct $marker;

            impl<A: 'static, F> System<A, $marker> for F
            where
                F: FnMut($($param),+) + Send + 'static,
            {
                fn run(&mut self, ctx: Ctx<'_, A>) {
                    self($(ctx.$pick),+)
                }
            }
        };
    }

    pub struct Full;

    impl<A: 'static, F> System<A, Full> for F
    where
        F: FnMut(Ctx<'_, A>) + Send + 'static,
    {
        fn run(&mut self, ctx: Ctx<'_, A>) {
            self(ctx)
        }
    }

    adapt!(App, &mut A => app);
    adapt!(AppStep, &mut A => app, Step => step);
    adapt!(AppDomains, &mut A => app, &mut Domains => domains);
    adapt!(AppDomainsStep, &mut A => app, &mut Domains => domains, Step => step);
    adapt!(AppInputCommands, &mut A => app, &Input => input, &mut Commands<A> => commands);
    adapt!(AppInputViews, &mut A => app, &Input => input, &mut Views => views);
    adapt!(
        AppDomainsInputStep,
        &mut A => app,
        &mut Domains => domains,
        &Input => input,
        Step => step
    );
}

type Runner<A> = Box<dyn FnMut(Ctx<'_, A>) + Send>;

pub struct SystemEntry<A> {
    name: &'static str,
    access: Access,
    run: Runner<A>,
}

impl<A: 'static> SystemEntry<A> {
    pub(crate) fn new<M>(name: &'static str, access: Access, system: impl System<A, M>) -> Self {
        let mut system = system;
        Self {
            name,
            access,
            run: Box::new(move |ctx: Ctx<'_, A>| system.run(ctx)),
        }
    }

    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn access(&self) -> &Access {
        &self.access
    }

    pub(crate) fn run(&mut self, ctx: Ctx<'_, A>) {
        (self.run)(ctx)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Schedule {
    InStep,
    Ahead,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Readback {
    None,
    Required,
    Optional,
}

/// Ordered by the session, executed by the host's GPU context.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkItem {
    pub name: &'static str,
    pub schedule: Schedule,
    pub readback: Readback,
}

pub enum Entry<A> {
    System(SystemEntry<A>),
    Work(WorkItem),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EntryId {
    pub phase: Phase,
    pub index: u32,
}

pub(crate) struct Phases<A> {
    entries: [Vec<Entry<A>>; 4],
}

impl<A> Phases<A> {
    pub(crate) fn new() -> Self {
        Self {
            entries: Default::default(),
        }
    }

    pub(crate) fn push(&mut self, phase: Phase, entry: Entry<A>) -> EntryId {
        let list = &mut self.entries[phase.index()];
        list.push(entry);
        EntryId {
            phase,
            index: (list.len() - 1) as u32,
        }
    }

    pub(crate) fn entries(&self, phase: Phase) -> &[Entry<A>] {
        &self.entries[phase.index()]
    }

    pub(crate) fn entries_mut(&mut self, phase: Phase) -> &mut [Entry<A>] {
        &mut self.entries[phase.index()]
    }
}
