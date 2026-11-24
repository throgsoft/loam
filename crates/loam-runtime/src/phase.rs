use std::fmt;

use loam_time::par::Executor;

use crate::assets::PreparedGeometry;
use crate::command::{CommandResult, Commands};
use crate::domain::{ChartPoint, DomainError, Domains};
use crate::drag::{Drag, DragError, DragRelease, Manipulation};
use crate::entity::{Entity, SceneId};
use crate::image::Views;
use crate::input::{Input, PointerButton};
use crate::publication::Overlay;
use crate::stores::Stores;
use crate::view::Pick;

/// Fixed order; dispatch runs while paused and is the only phase where capacity grows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Phase {
    Dispatch,
    Simulation,
    Publication,
}

impl Phase {
    fn index(self) -> usize {
        self as usize
    }
}

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
                "system `{system}` failed in {:?}: {}",
                self.phase, self.cause
            ),
            None => write!(formatter, "{:?} phase failed: {}", self.phase, self.cause),
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
    /// The publication's clamped alpha in the publication phase and 1 in the others.
    pub alpha: f32,
    /// Cleared before the publication phase, so only a publication system's lines and points reach the screen.
    pub overlay: &'a mut Overlay,
    pub(crate) executor: &'a dyn Executor,
    pub(crate) manipulation: &'a mut Manipulation,
}

impl<A> Ctx<'_, A> {
    fn reborrow(&mut self) -> Ctx<'_, A> {
        Ctx {
            app: self.app,
            domains: self.domains,
            views: self.views,
            commands: self.commands,
            results: self.results,
            input: self.input,
            prepared: self.prepared,
            step: self.step,
            alpha: self.alpha,
            overlay: self.overlay,
            executor: self.executor,
            manipulation: self.manipulation,
        }
    }
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
        self.manipulation.release(self.domains, self.commands)
    }

    pub fn release_at(&mut self, time: f64) -> Option<DragRelease> {
        self.manipulation
            .release_at(self.domains, self.commands, time)
    }

    pub fn cancel_drag(&mut self) -> Option<DragRelease> {
        self.manipulation.cancel(self.commands)
    }

    /// Carries entities that `allow` accepts with `button` and returns this boundary's last release.
    pub fn carry(
        &mut self,
        button: PointerButton,
        mut allow: impl FnMut(&A, Entity) -> bool,
    ) -> Option<DragRelease> {
        let app = &*self.app;
        if self
            .manipulation
            .dragging()
            .is_some_and(|drag| !allow(app, drag.entity))
        {
            self.manipulation.cancel(self.commands);
        }
        let (domains, views, prepared) = (&*self.domains, &*self.views, self.prepared);
        self.manipulation
            .carry(domains, views, self.commands, self.input, button, |ndc| {
                domains
                    .pick_lifted(views, prepared, ndc)
                    .filter(|pick| allow(app, pick.entity))
            })
    }
}

type Runner<A> = Box<dyn FnMut(Ctx<'_, A>) -> Result<(), DomainError> + Send>;

pub struct SystemEntry<A> {
    name: &'static str,
    run: Runner<A>,
}

impl<A: 'static> SystemEntry<A> {
    pub(crate) fn new(
        name: &'static str,
        system: impl FnMut(Ctx<'_, A>) -> Result<(), DomainError> + Send + 'static,
    ) -> Self {
        Self {
            name,
            run: Box::new(system),
        }
    }
}

impl<A: Stores> SystemEntry<A> {
    pub(crate) fn on_change<K: PartialEq + Send + 'static>(
        name: &'static str,
        watch: impl Fn(&A) -> K + Send + 'static,
        mut system: impl FnMut(Ctx<'_, A>) -> Result<(), DomainError> + Send + 'static,
    ) -> Self {
        // A restore advances the scene's epoch, so keying on the scene runs the system again after one.
        let mut seen: Option<(SceneId, K)> = None;
        Self::new(name, move |mut ctx: Ctx<'_, A>| {
            let scene = ctx.commands.entities().scene();
            let key = watch(ctx.app);
            if seen
                .as_ref()
                .is_some_and(|(held_scene, held)| *held_scene == scene && *held == key)
            {
                return Ok(());
            }
            system(ctx.reborrow())?;
            seen = Some((scene, watch(ctx.app)));
            Ok(())
        })
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
