use loam_shape::polytope::Polytope4Topology;

use crate::command::{Command, CommandResult, Commands, Dispatch, Outcome, Rejection, Request};
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

/// What one boundary applied; the session grows its tables here and nowhere else.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Growth {
    pub commands: usize,
    pub spawned: usize,
    pub despawned: usize,
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
    domains: Domains,
    views: Views,
    phases: Phases<A>,
    commands: Commands<A>,
    batch: Vec<Request<A>>,
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
            domains: Domains::new(scene.runtime),
            views: Views::new(),
            phases,
            commands: Commands::new(scene),
            batch: Vec::new(),
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
        self.commands.entities().scene()
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
        self.commands.entities()
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
            self.commands.entities_mut(),
        );
        f(&mut dispatch)
    }

    /// Commits deferred commands, runs each dispatch entry and then its commands, and counts what grew; runs while paused.
    pub fn boundary(&mut self, input: Input) -> Result<Growth, DomainError> {
        self.input = input;
        self.results.clear();
        let mut growth = Growth::default();
        self.commit(&mut growth);
        let step = Step {
            tick: self.tick,
            dt: self.config.dt().unwrap_or(0.0),
        };
        for index in 0..self.phases.entries(Phase::Dispatch).len() {
            self.run_entry(Phase::Dispatch, index, step)?;
            self.commit(&mut growth);
        }
        self.app.boundary();
        for domain in self.domains.iter_mut() {
            domain.boundary();
        }
        Ok(growth)
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
        for index in 0..self.phases.entries(Phase::Simulation).len() {
            self.run_entry(Phase::Simulation, index, step)?;
        }
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

    fn commit(&mut self, growth: &mut Growth) {
        let mut batch = std::mem::take(&mut self.batch);
        self.commands.drain_into(&mut batch);
        for request in batch.drain(..) {
            let despawn = matches!(request.command, Command::Despawn(_));
            let outcome = match request.command {
                Command::Reset => self
                    .reset()
                    .map(|()| Outcome::Done)
                    .map_err(Rejection::Restore),
                command => self.dispatch(|dispatch| dispatch.apply(command)),
            };
            growth.commands += 1;
            match outcome {
                Ok(Outcome::Spawned(_)) => growth.spawned += 1,
                Ok(Outcome::Done) if despawn => growth.despawned += 1,
                _ => {}
            }
            self.results.push(CommandResult {
                request: request.id,
                outcome,
            });
        }
        self.batch = batch;
    }

    fn run_entry(&mut self, phase: Phase, index: usize, step: Step) -> Result<(), DomainError> {
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
        let Some(Entry::System(system)) = phases.entries_mut(phase).get_mut(index) else {
            return Ok(());
        };
        system.run(Ctx {
            app,
            domains,
            views,
            commands,
            results: results.as_slice(),
            input,
            step,
        })
    }
}

#[cfg(test)]
mod tests {
    use loam_math::{EuclideanR4, Iso4Flat};

    use super::*;
    use crate::command::SpawnBundle;
    use crate::domain::Pose;
    use crate::entity::Entity;
    use crate::store::tests::alloc_probe::bytes_allocated_by;
    use crate::store::LogCapacity;

    crate::stores! {
        #[derive(Default)]
        pub struct Churn {
            counters: Store<u32>,
            pool: Value<Vec<Entity>>,
        }
    }

    #[test]
    fn warmed_tick_and_boundary_allocate_nothing_outside_reported_growth() {
        let mut session = Session::new(Churn::default(), SimConfig::default());
        let r4 = session
            .register_domain(DomainBuilder::new("r4", EuclideanR4).tracked(LogCapacity::default()));
        let pool = session.dispatch(|d| {
            (0..8)
                .map(|value| {
                    d.spawn(
                        SpawnBundle::new()
                            .at(r4, Pose(Iso4Flat::IDENTITY))
                            .row(value),
                    )
                    .unwrap()
                })
                .collect()
        });
        session.app.pool.set(pool);
        session.system(
            Phase::Simulation,
            "churn",
            Access::new().writes::<u32>().commands(),
            |ctx: Ctx<'_, Churn>| {
                for (_, counter) in ctx.app.counters.iter_mut() {
                    *counter += 1;
                }
                let pool = ctx.app.pool.get_mut();
                let retired = pool.swap_remove(0);
                ctx.commands.submit(Command::Despawn(retired));
                let fresh = ctx.commands.spawn(SpawnBundle::new()).unwrap();
                pool.push(fresh.entity);
            },
        );
        let cycle = |session: &mut Session<Churn>| {
            session.tick().unwrap();
            session.boundary(Input::default()).unwrap()
        };
        for _ in 0..16 {
            cycle(&mut session);
        }

        let bytes = bytes_allocated_by(|| {
            for _ in 0..16 {
                let growth = cycle(&mut session);
                assert_eq!(
                    growth,
                    Growth {
                        commands: 2,
                        spawned: 1,
                        despawned: 1,
                    }
                );
            }
            let quiet = session.boundary(Input::default()).unwrap();
            assert_eq!(quiet, Growth::default());
        });
        assert_eq!(
            bytes, 0,
            "16 warmed cycles asked the allocator for {bytes} bytes"
        );
        assert_eq!(session.entities().len(), 8);
    }
}
