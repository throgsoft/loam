use loam_shape::polytope::Polytope4Topology;

use crate::command::{
    Command, CommandResult, Commands, Dispatch, Outcome, Rejection, Request, RequestId,
};
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
    pub config: SimConfig,
    pub next_request: RequestId,
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
    sequence: u64,
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
            sequence: 0,
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

    /// Stamps every record buffer with the tick and a sequence that advances while paused.
    pub fn publish(&mut self, into: &mut Publication<A>) -> Result<(), DomainError> {
        self.sequence += 1;
        let stamp = Stamp {
            tick: self.tick,
            sequence: self.sequence,
        };
        self.app.publish(&mut into.app, stamp);
        let mut count = 0;
        for domain in self.domains.iter() {
            for &target in domain.views() {
                let current = into
                    .views
                    .get(count)
                    .is_some_and(|view| view.domain == domain.id() && view.target == target);
                if !current {
                    into.views.truncate(count);
                    into.views.push(PublishedView {
                        domain: domain.id(),
                        target,
                        records: ViewRecords::default(),
                    });
                }
                domain.publish(target.view, &mut into.views[count].records, stamp)?;
                count += 1;
            }
        }
        into.views.truncate(count);
        into.stamp = stamp;
        Ok(())
    }

    pub fn results(&self) -> &[CommandResult] {
        &self.results
    }

    pub fn pick(&self, ndc: [f32; 2]) -> Option<Pick> {
        self.domains.pick(&self.views, ndc)
    }

    /// `Pending` while a deferred command or a reservation is outstanding.
    pub fn snapshot(&self) -> Result<SessionSnapshot<A>, RestoreError> {
        if !self.commands.is_empty() || self.entities().has_reservations() {
            return Err(RestoreError::Pending);
        }
        Ok(SessionSnapshot {
            app: self.app.snapshot(),
            entities: self.entities().snapshot(),
            domains: self
                .domains
                .iter()
                .map(|domain| domain.snapshot())
                .collect(),
            tick: self.tick,
            config: self.config,
            next_request: self.commands.next_request(),
        })
    }

    /// Cancels pending commands and reservations, then advances the epoch; every earlier external handle fails.
    pub fn restore(&mut self, from: &SessionSnapshot<A>) -> Result<(), RestoreError> {
        if from.domains.len() != self.domains.len() {
            let first = from.domains.len().min(self.domains.len());
            return Err(RestoreError::Domain(DomainId::new(first)));
        }
        self.commands.cancel_into(&mut self.results);
        self.commands.restore(&from.entities, from.next_request);
        let scene = self.scene();
        for (domain, snapshot) in self.domains.iter_mut().zip(&from.domains) {
            domain.restore(snapshot, scene)?;
        }
        self.app.restore(&from.app, scene);
        self.tick = from.tick;
        self.config = from.config;
        Ok(())
    }

    /// Refused like `snapshot`; `reset` restores what it captures.
    pub fn set_initial(&mut self) -> Result<(), RestoreError> {
        self.initial = Some(self.snapshot()?);
        Ok(())
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
        let mut cancelled = false;
        for request in batch.drain(..) {
            let despawn = matches!(request.command, Command::Despawn(_));
            let outcome = if cancelled {
                Err(Rejection::Cancelled)
            } else {
                match request.command {
                    Command::Reset => {
                        let outcome = self
                            .reset()
                            .map(|()| Outcome::Done)
                            .map_err(Rejection::Restore);
                        cancelled = outcome.is_ok();
                        outcome
                    }
                    command => self.dispatch(|dispatch| dispatch.apply(command)),
                }
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
    use loam_math::{EuclideanR4, Iso4Flat, Space};

    use super::*;
    use crate::command::SpawnBundle;
    use crate::domain::{Instance, Pose};
    use crate::entity::Entity;
    use crate::store::tests::alloc_probe::bytes_allocated_by;
    use crate::store::{LogCapacity, Store};
    use crate::view::{DepthEnvelope, DomainRay, ImageRay, ViewMapping, ViewSpec};

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

    crate::stores! {
        pub struct Shown {
            scores: Published<u32>,
        }
    }

    struct Flat;

    impl ViewMapping<EuclideanR4> for Flat {
        fn name(&self) -> &'static str {
            "flat"
        }

        fn image_point(
            &self,
            eye: &Pose<EuclideanR4>,
            point: <EuclideanR4 as Space>::Point,
        ) -> Option<[f32; 3]> {
            let relative = point - eye.0.translation;
            Some([relative.x, relative.y, relative.z])
        }

        fn lift(
            &self,
            _eye: &Pose<EuclideanR4>,
            _ray: &ImageRay,
        ) -> Option<DomainRay<EuclideanR4>> {
            None
        }

        fn depth_envelope(&self) -> DepthEnvelope {
            DepthEnvelope {
                near: 0.0,
                far: 1.0,
            }
        }
    }

    #[test]
    fn warmed_publish_allocates() {
        let shown = Shown {
            scores: Store::tracked(LogCapacity::default()),
        };
        let mut session = Session::new(shown, SimConfig::default());
        let r4 = session
            .register_domain(DomainBuilder::new("r4", EuclideanR4).tracked(LogCapacity::default()));
        let geometry = session.prepare(PreparedGeometry::Lines4 {
            segments: Vec::new(),
        });
        let material = session.add_material(Material::flat([1.0; 4]));
        let root = session.views().root();
        session.dispatch(|d| {
            let eye = d
                .spawn(SpawnBundle::new().at(r4, Pose(Iso4Flat::IDENTITY)))
                .unwrap();
            d.domains
                .typed(r4)
                .unwrap()
                .add_view(ViewSpec::new(root, eye, Flat));
            for value in 0..8 {
                d.spawn(
                    SpawnBundle::new()
                        .at(r4, Pose(Iso4Flat::IDENTITY))
                        .instance(Instance::new(geometry, material))
                        .row(value),
                )
                .unwrap();
            }
        });
        session.system(
            Phase::Simulation,
            "churn",
            Access::new().writes::<u32>().domain(r4.id()),
            move |app: &mut Shown, domains: &mut Domains, step: Step| {
                for (_, score) in app.scores.iter_mut() {
                    *score += 1;
                }
                for (_, pose) in domains.typed(r4).unwrap().poses.iter_mut() {
                    pose.0.translation.x += step.dt;
                }
            },
        );
        let mut publication = Publication::default();
        let cycle = |session: &mut Session<Shown>, publication: &mut Publication<Shown>| {
            session.tick().unwrap();
            session.boundary(Input::default()).unwrap();
            session.publish(publication).unwrap();
        };
        for _ in 0..4 {
            cycle(&mut session, &mut publication);
        }

        let bytes = bytes_allocated_by(|| {
            for _ in 0..16 {
                cycle(&mut session, &mut publication);
            }
        });
        assert_eq!(
            bytes, 0,
            "16 warmed publishes asked the allocator for {bytes} bytes"
        );
        assert_eq!(publication.app.scores.rows().len(), 8);
        assert_eq!(publication.views[0].records.instances.rows().len(), 8);
    }
}
