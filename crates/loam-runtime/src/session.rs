use loam_shape::polytope::Polytope4Topology;

use crate::command::{CommandResult, Commands, Dispatch};
use crate::domain::{
    DomainBuilder, DomainError, DomainHandle, DomainId, DomainSnapshot, DomainSpace, Domains,
};
use crate::entity::{Entities, EntitiesSnapshot, Epoch, RuntimeId, SceneId};
use crate::input::Input;
use crate::phase::{
    Access, Ctx, Entry, EntryId, Order, Phase, Phases, Step, System, SystemEntry, Tick, WorkItem,
};
use crate::store::SchemaId;
use crate::stores::Stores;
use crate::view::{Pick, ViewRecords, ViewTarget, Views};

/// Fixed steps on both hosts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SimConfig {
    pub fixed_hz: u32,
    pub max_ticks_per_frame: u32,
    pub overlap: bool,
    pub seed: u64,
}

impl SimConfig {
    /// `None` when `fixed_hz` is zero, which stops the simulation.
    pub fn dt(&self) -> Option<f32> {
        (self.fixed_hz > 0).then(|| 1.0 / self.fixed_hz as f32)
    }
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            fixed_hz: 60,
            max_ticks_per_frame: 4,
            overlap: false,
            seed: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stamp {
    pub tick: Tick,
    pub sequence: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PreparedId(u32);

impl PreparedId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MaterialId(u32);

impl MaterialId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum PreparedGeometry {
    Lines4 {
        segments: Vec<[[f32; 4]; 2]>,
    },
    Lines3 {
        segments: Vec<[[f32; 3]; 2]>,
    },
    Mesh3 {
        positions: Vec<[f32; 3]>,
        indices: Vec<u32>,
    },
}

impl PreparedGeometry {
    pub fn edges_of(topology: &Polytope4Topology, scale: f32) -> Self {
        let segments = topology
            .edges
            .iter()
            .map(|&[i, j]| {
                [
                    (topology.vertices[i as usize] * scale).to_array(),
                    (topology.vertices[j as usize] * scale).to_array(),
                ]
            })
            .collect();
        Self::Lines4 { segments }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Material {
    Lines { color: [f32; 4], width_px: f32 },
    Flat { color: [f32; 4] },
}

impl Material {
    pub fn lines(color: [f32; 4], width_px: f32) -> Self {
        Self::Lines { color, width_px }
    }

    pub fn flat(color: [f32; 4]) -> Self {
        Self::Flat { color }
    }
}

pub struct PublishedView {
    pub domain: DomainId,
    pub target: ViewTarget,
    pub records: ViewRecords,
}

/// What a frame presents; publication writes it and the host reads it.
pub struct Publication<A: Stores> {
    pub app: A::Records,
    pub views: Vec<PublishedView>,
    pub stamp: Stamp,
}

impl<A: Stores> Default for Publication<A> {
    fn default() -> Self {
        Self {
            app: A::Records::default(),
            views: Vec::new(),
            stamp: Stamp::default(),
        }
    }
}

pub struct SessionSnapshot<A: Stores> {
    pub app: A::Snapshot,
    pub entities: EntitiesSnapshot,
    pub domains: Vec<DomainSnapshot>,
    pub tick: Tick,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RestoreError {
    Pending,
    NoInitial,
    Schema(SchemaId),
    Domain(DomainId),
}

/// The simulation entry `Session::new` registers; `system_at` places app entries around it.
pub const DOMAIN_STEP: &str = "domain step";

/// A CPU value with no GPU or window object; `Send`, and it builds for wasm32.
pub struct Session<A: Stores> {
    pub app: A,
    entities: Entities,
    domains: Domains,
    views: Views,
    phases: Phases<A>,
    commands: Commands<A>,
    results: Vec<CommandResult>,
    input: Input,
    prepared: Vec<PreparedGeometry>,
    materials: Vec<Material>,
    config: SimConfig,
    tick: Tick,
    initial: Option<SessionSnapshot<A>>,
}

impl<A: Stores> Session<A> {
    pub fn new(mut app: A, config: SimConfig) -> Self {
        let scene = SceneId {
            runtime: RuntimeId::allocate(),
            epoch: Epoch::default(),
        };
        app.bind(scene);
        let mut phases = Phases::new();
        phases.push(
            Phase::Simulation,
            Entry::System(SystemEntry::fallible(
                DOMAIN_STEP,
                Access::new().every_domain(),
                |ctx: Ctx<'_, A>| {
                    for domain in ctx.domains.iter_mut() {
                        domain.step(ctx.step)?;
                    }
                    Ok(())
                },
            )),
        );
        Self {
            app,
            entities: Entities::new(scene),
            domains: Domains::new(scene.runtime),
            views: Views::new(),
            phases,
            commands: Commands::new(),
            results: Vec::new(),
            input: Input::default(),
            prepared: Vec::new(),
            materials: Vec::new(),
            config,
            tick: Tick::default(),
            initial: None,
        }
    }

    pub fn config(&self) -> SimConfig {
        self.config
    }

    pub fn scene(&self) -> SceneId {
        self.entities.scene()
    }

    pub fn current_tick(&self) -> Tick {
        self.tick
    }

    pub fn register_domain<S: DomainSpace>(
        &mut self,
        builder: DomainBuilder<S>,
    ) -> DomainHandle<S> {
        let domain = builder.build(self.domains.next_id(), self.scene());
        let handle = domain.handle();
        self.domains.push(Box::new(domain));
        handle
    }

    pub fn domains(&self) -> &Domains {
        &self.domains
    }

    pub fn domains_mut(&mut self) -> &mut Domains {
        &mut self.domains
    }

    pub fn views(&self) -> &Views {
        &self.views
    }

    pub fn views_mut(&mut self) -> &mut Views {
        &mut self.views
    }

    pub fn entities(&self) -> &Entities {
        &self.entities
    }

    pub fn prepare(&mut self, geometry: PreparedGeometry) -> PreparedId {
        self.prepared.push(geometry);
        PreparedId((self.prepared.len() - 1) as u32)
    }

    pub fn prepared(&self, id: PreparedId) -> Option<&PreparedGeometry> {
        self.prepared.get(id.index())
    }

    pub fn add_material(&mut self, material: Material) -> MaterialId {
        self.materials.push(material);
        MaterialId((self.materials.len() - 1) as u32)
    }

    pub fn material(&self, id: MaterialId) -> Option<&Material> {
        self.materials.get(id.index())
    }

    pub fn system<M>(
        &mut self,
        phase: Phase,
        name: &'static str,
        access: Access,
        system: impl System<A, M>,
    ) -> EntryId {
        self.phases
            .push(phase, Entry::System(SystemEntry::new(name, access, system)))
    }

    /// `None` when no entry of that phase has the named anchor.
    pub fn system_at<M>(
        &mut self,
        phase: Phase,
        order: Order,
        name: &'static str,
        access: Access,
        system: impl System<A, M>,
    ) -> Option<EntryId> {
        self.phases.insert(
            phase,
            order,
            Entry::System(SystemEntry::new(name, access, system)),
        )
    }

    pub fn work(&mut self, phase: Phase, item: WorkItem) -> EntryId {
        self.phases.push(phase, Entry::Work(item))
    }

    pub fn entries(&self, phase: Phase) -> &[Entry<A>] {
        self.phases.entries(phase)
    }

    pub fn work_items(&self, phase: Phase) -> impl Iterator<Item = &WorkItem> {
        self.entries(phase).iter().filter_map(|entry| match entry {
            Entry::Work(item) => Some(item),
            Entry::System(_) => None,
        })
    }

    pub fn dispatch<R>(&mut self, f: impl FnOnce(&mut Dispatch<'_, A>) -> R) -> R {
        let mut dispatch = Dispatch::new(
            &mut self.app,
            &mut self.domains,
            &mut self.views,
            &mut self.entities,
        );
        f(&mut dispatch)
    }

    /// Runs dispatch entries, commits deferred commands, and grows capacity; runs while paused.
    pub fn boundary(&mut self, _input: Input) {
        todo!()
    }

    /// One fixed step: the simulation phase's entries in their order, the domain step among them.
    pub fn tick(&mut self) -> Result<(), DomainError> {
        let Some(dt) = self.config.dt() else {
            return Ok(());
        };
        let step = Step {
            tick: self.tick,
            dt,
        };
        self.run_phase(Phase::Simulation, step)?;
        self.tick = Tick(self.tick.0 + 1);
        Ok(())
    }

    pub fn publish(&mut self, _into: &mut Publication<A>) {
        todo!()
    }

    pub fn results(&self) -> &[CommandResult] {
        &self.results
    }

    pub fn pick(&self, ndc: [f32; 2]) -> Option<Pick> {
        self.domains.pick(&self.views, ndc)
    }

    /// Needs a quiescent boundary: no pending command or reservation.
    pub fn snapshot(&self) -> SessionSnapshot<A> {
        todo!()
    }

    /// Advances the epoch; every external handle from before the restore fails.
    pub fn restore(&mut self, _from: &SessionSnapshot<A>) -> Result<(), RestoreError> {
        todo!()
    }

    pub fn set_initial(&mut self) {
        self.initial = Some(self.snapshot());
    }

    pub fn reset(&mut self) -> Result<(), RestoreError> {
        let Some(initial) = self.initial.take() else {
            return Err(RestoreError::NoInitial);
        };
        let result = self.restore(&initial);
        self.initial = Some(initial);
        result
    }

    fn run_phase(&mut self, phase: Phase, step: Step) -> Result<(), DomainError> {
        let Session {
            app,
            domains,
            views,
            phases,
            commands,
            results,
            input,
            ..
        } = self;
        for entry in phases.entries_mut(phase) {
            if let Entry::System(system) = entry {
                system.run(Ctx {
                    app: &mut *app,
                    domains: &mut *domains,
                    views: &mut *views,
                    commands: &mut *commands,
                    results: results.as_slice(),
                    input: &*input,
                    step,
                })?;
            }
        }
        Ok(())
    }
}
