use loam_shape::polytope::Polytope4Topology;

use crate::bulk::{
    Bulk, BulkAction, BulkCheckpoint, BulkError, BulkId, BulkSnapshot, BulkSpec, InFlight, Landed,
    Landing, SnapshotPolicy, Wait, WorkOrder, WorkStats,
};
use crate::command::{
    Command, CommandResult, Commands, Dispatch, Outcome, Rejection, Request, RequestId,
};
use crate::domain::{
    DomainBuilder, DomainError, DomainHandle, DomainId, DomainSnapshot, DomainSpace, Domains,
};
use crate::entity::{Entities, EntitiesSnapshot, Epoch, RuntimeId, SceneId};
use crate::input::Input;
use crate::phase::{
    Access, Ctx, Entry, EntryId, Order, Phase, Phases, Schedule, Step, System, SystemEntry, Tick,
    WorkItem,
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
    /// In-flight work orders before `issue_work` delays the rest.
    pub work_queue: u32,
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
            work_queue: 4,
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
    pub bulk_elements: usize,
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

    /// Chart distance from the origin to the farthest vertex.
    pub fn bounding_radius(&self) -> f32 {
        fn farthest<'a>(points: impl Iterator<Item = &'a [f32]>) -> f32 {
            points
                .map(|point| point.iter().map(|c| c * c).sum::<f32>().sqrt())
                .fold(0.0, f32::max)
        }
        match self {
            Self::Lines4 { segments } => {
                farthest(segments.iter().flatten().map(|point| point.as_slice()))
            }
            Self::Lines3 { segments } => {
                farthest(segments.iter().flatten().map(|point| point.as_slice()))
            }
            Self::Mesh3 { positions, .. } => {
                farthest(positions.iter().map(|point| point.as_slice()))
            }
        }
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

#[derive(Clone, Copy)]
pub struct Library<'a> {
    pub geometry: &'a [PreparedGeometry],
    pub materials: &'a [Material],
}

impl Library<'_> {
    pub(crate) fn line_style(&self, material: MaterialId) -> ([f32; 4], f32) {
        match self.materials.get(material.index()) {
            Some(Material::Lines { color, width_px }) => (*color, *width_px),
            Some(Material::Flat { color }) => (*color, 1.0),
            None => ([1.0; 4], 1.0),
        }
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PublishError {
    Borrowed,
    Domain(DomainError),
}

impl From<DomainError> for PublishError {
    fn from(error: DomainError) -> Self {
        Self::Domain(error)
    }
}

/// The one record buffer a session publishes into.
pub struct Records<A: Stores> {
    idle: Option<Publication<A>>,
}

impl<A: Stores> Default for Records<A> {
    fn default() -> Self {
        Self {
            idle: Some(Publication::default()),
        }
    }
}

impl<A: Stores> Records<A> {
    pub fn publish(&mut self, session: &mut Session<A>) -> Result<Stamp, PublishError> {
        let buffer = self.idle.as_mut().ok_or(PublishError::Borrowed)?;
        session.publish(buffer)?;
        Ok(buffer.stamp)
    }

    /// Until the buffer is released, [`Self::publish`] returns [`PublishError::Borrowed`].
    pub fn lend(&mut self) -> Option<Publication<A>> {
        self.idle.take()
    }

    pub fn release(&mut self, publication: Publication<A>) {
        self.idle = Some(publication);
    }
}

pub struct SessionSnapshot<A: Stores> {
    pub app: A::Snapshot,
    pub entities: EntitiesSnapshot,
    pub domains: Vec<DomainSnapshot>,
    pub tick: Tick,
    pub config: SimConfig,
    pub next_request: RequestId,
    pub bulk: BulkSnapshot,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RestoreError {
    Pending,
    NoInitial,
    Schema(SchemaId),
    Domain(DomainId),
    Readback(&'static str),
    NoCheckpoint(&'static str),
    CheckpointTick(&'static str),
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
    resume: Option<EntryId>,
    bulk: Bulk,
    checkpoints: Vec<Option<BulkCheckpoint>>,
    restore_plan: Vec<(BulkId, BulkAction)>,
    flight: InFlight,
    work: Vec<WorkOrder>,
    work_head: usize,
    ahead_for: Vec<EntryId>,
    ahead_tick: Tick,
    stats: WorkStats,
    wait: Option<Wait>,
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
            resume: None,
            bulk: Bulk::default(),
            checkpoints: Vec::new(),
            restore_plan: Vec::new(),
            flight: InFlight::new(config.work_queue),
            work: Vec::new(),
            work_head: 0,
            ahead_for: Vec::new(),
            ahead_tick: Tick::default(),
            stats: WorkStats::default(),
            wait: None,
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

    pub fn register_bulk(&mut self, spec: BulkSpec) -> BulkId {
        let id = self.bulk.register(spec);
        self.checkpoints.resize(self.bulk.len(), None);
        id
    }

    pub fn remove_bulk(&mut self, id: BulkId) -> bool {
        if !self.bulk.remove(id) {
            return false;
        }
        self.flight.cancel_bulk(id);
        if let Some(slot) = self.checkpoints.get_mut(id.index()) {
            *slot = None;
        }
        true
    }

    pub fn bulk(&self) -> &Bulk {
        &self.bulk
    }

    pub fn work_list(&self) -> &[WorkOrder] {
        &self.work[self.work_head..]
    }

    pub fn work_stats(&self) -> WorkStats {
        WorkStats {
            discarded: self.flight.discarded(),
            ..self.stats
        }
    }

    /// Drains the plan in order into `execute`; a full in-flight queue stops the drain and counts a delay until a result is released.
    pub fn issue_work(&mut self, mut execute: impl FnMut(&WorkOrder)) -> usize {
        let mut issued = 0;
        while let Some(order) = self.work.get(self.work_head).copied() {
            let full = {
                let Session { phases, flight, .. } = self;
                let writes = match phases
                    .entries(order.entry.phase)
                    .get(order.entry.index as usize)
                {
                    Some(Entry::Work(item)) => item.write_set(),
                    _ => &[][..],
                };
                !flight.submit(order, writes)
            };
            if full {
                self.stats.delayed += 1;
                return issued;
            }
            self.work_head += 1;
            self.stats.issued += 1;
            execute(&order);
            issued += 1;
        }
        self.work.clear();
        self.work_head = 0;
        issued
    }

    pub fn submitted(&mut self, request: RequestId) -> bool {
        self.flight.submitted(request)
    }

    pub fn land_readback(&mut self, request: RequestId, rows: Option<&[u8]>) -> Landing {
        self.flight.land(request, rows)
    }

    /// Landed results not yet released.
    pub fn readbacks(&self) -> impl Iterator<Item = Landed<'_>> {
        self.flight.landed()
    }

    /// Frees the queue slot a landed result holds; until then the slot counts against `work_queue`.
    pub fn release_readback(&mut self, request: RequestId) -> bool {
        self.flight.release(request)
    }

    /// The entry stopped for a required readback; `boundary` and `tick` resume there once it lands.
    pub fn waiting(&self) -> Option<Wait> {
        self.wait
    }

    pub fn cancel_work(&mut self) {
        self.flight.cancel_all();
        self.work.clear();
        self.work_head = 0;
        self.ahead_for.clear();
        self.resume = self.wait.take().map(|wait| wait.entry);
    }

    pub fn checkpoint(&mut self, id: BulkId, tick: Tick, rows: &[u8]) -> Result<(), BulkError> {
        if !self.bulk.is_live(id) {
            return Err(if id.index() < self.bulk.len() {
                BulkError::Removed(id)
            } else {
                BulkError::Unknown(id)
            });
        }
        self.checkpoints.resize(self.bulk.len(), None);
        match &mut self.checkpoints[id.index()] {
            Some(existing) => {
                existing.tick = tick;
                existing.rows.clear();
                existing.rows.extend_from_slice(rows);
            }
            slot => {
                *slot = Some(BulkCheckpoint {
                    tick,
                    rows: rows.to_vec(),
                })
            }
        }
        Ok(())
    }

    pub fn checkpoint_rows(&self, id: BulkId) -> Option<&[u8]> {
        Some(self.checkpoints.get(id.index())?.as_ref()?.rows.as_slice())
    }

    /// Hands `apply` each store `restore` planned with its checkpoint rows, empty for a reinitialization.
    pub fn apply_restore(&mut self, mut apply: impl FnMut(BulkId, BulkAction, &[u8])) -> usize {
        let mut plan = std::mem::take(&mut self.restore_plan);
        let applied = plan.len();
        for (id, action) in plan.drain(..) {
            let rows = self
                .checkpoints
                .get(id.index())
                .and_then(|slot| slot.as_ref())
                .map_or(&[][..], |checkpoint| checkpoint.rows.as_slice());
            apply(id, action, rows);
        }
        self.restore_plan = plan;
        applied
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

    /// Commits deferred commands, runs each dispatch entry and then its commands, and counts what grew; runs while paused, and a call that resumes a suspended entry keeps the input it started with.
    pub fn boundary(&mut self, input: Input) -> Result<Growth, DomainError> {
        let mut growth = Growth::default();
        let suspended = self.wait.map(|wait| wait.entry).or(self.resume);
        let mut index = match suspended {
            Some(entry) if entry.phase == Phase::Dispatch => {
                self.wait = None;
                self.resume = None;
                entry.index as usize
            }
            Some(_) => {
                return Ok(growth);
            }
            None => {
                self.input = input;
                self.results.clear();
                self.commit(&mut growth);
                self.plan(Phase::Dispatch, self.tick);
                0
            }
        };
        let step = Step {
            tick: self.tick,
            dt: self.config.dt().unwrap_or(0.0),
        };
        while index < self.phases.entries(Phase::Dispatch).len() {
            if !self.run_entry(Phase::Dispatch, index, step)? {
                return Ok(growth);
            }
            self.commit(&mut growth);
            index += 1;
        }
        growth.bulk_elements = self.bulk.take_growth();
        self.app.boundary();
        for domain in self.domains.iter_mut() {
            domain.boundary();
        }
        Ok(growth)
    }

    /// One fixed step: the simulation phase's entries in their order, the domain step among them; a call that resumes a suspended entry finishes that same step.
    pub fn tick(&mut self) -> Result<(), DomainError> {
        let Some(dt) = self.config.dt() else {
            return Ok(());
        };
        let step = Step {
            tick: self.tick,
            dt,
        };
        let suspended = self.wait.map(|wait| wait.entry).or(self.resume);
        let mut index = match suspended {
            Some(entry) if entry.phase == Phase::Simulation => {
                self.wait = None;
                self.resume = None;
                entry.index as usize
            }
            Some(_) => {
                return Ok(());
            }
            None => {
                self.plan(Phase::Simulation, step.tick);
                0
            }
        };
        while index < self.phases.entries(Phase::Simulation).len() {
            if !self.run_entry(Phase::Simulation, index, step)? {
                return Ok(());
            }
            index += 1;
        }
        self.tick = Tick(self.tick.0 + 1);
        Ok(())
    }

    fn plan(&mut self, phase: Phase, tick: Tick) {
        let Session {
            phases,
            commands,
            work,
            ahead_for,
            ahead_tick,
            stats,
            ..
        } = self;
        for (index, entry) in phases.entries(phase).iter().enumerate() {
            let Entry::Work(item) = entry else {
                continue;
            };
            let entry = EntryId {
                phase,
                index: index as u32,
            };
            if item.schedule == Schedule::Ahead {
                if *ahead_tick == tick && ahead_for.contains(&entry) {
                    continue;
                }
                stats.fallbacks += 1;
            }
            work.push(WorkOrder {
                entry,
                name: item.name,
                schedule: Schedule::InStep,
                readback: item.readback,
                tick,
                request: commands.reserve_request(),
            });
        }
    }

    fn plan_ahead(&mut self, tick: Tick) {
        let Session {
            phases,
            bulk,
            flight,
            commands,
            work,
            ahead_for,
            ahead_tick,
            ..
        } = self;
        ahead_for.clear();
        *ahead_tick = tick;
        for phase in [Phase::Dispatch, Phase::Simulation] {
            for (index, entry) in phases.entries(phase).iter().enumerate() {
                let Entry::Work(item) = entry else {
                    continue;
                };
                if item.schedule != Schedule::Ahead {
                    continue;
                }
                let entry = EntryId {
                    phase,
                    index: index as u32,
                };
                let ready = item
                    .read_set()
                    .iter()
                    .all(|id| bulk.is_live(*id) && !flight.writes_pending(*id));
                if !ready {
                    continue;
                }
                ahead_for.push(entry);
                work.push(WorkOrder {
                    entry,
                    name: item.name,
                    schedule: Schedule::Ahead,
                    readback: item.readback,
                    tick,
                    request: commands.reserve_request(),
                });
            }
        }
    }

    /// Stamps every record buffer with the tick and a sequence that advances while paused.
    pub fn publish(&mut self, into: &mut Publication<A>) -> Result<(), DomainError> {
        self.plan_ahead(self.tick);
        self.plan(Phase::Publication, self.tick);
        self.plan(Phase::Presentation, self.tick);
        self.sequence += 1;
        let stamp = Stamp {
            tick: self.tick,
            sequence: self.sequence,
        };
        self.app.publish(&mut into.app, stamp);
        let root = self.views.root();
        let library = Library {
            geometry: &self.prepared,
            materials: &self.materials,
        };
        let mut count = 0;
        for domain in self.domains.iter() {
            for &target in domain.views().iter().filter(|target| target.image == root) {
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
                domain.publish(target.view, library, &mut into.views[count].records, stamp)?;
                count += 1;
            }
        }
        into.views.truncate(count);
        into.stamp = stamp;
        Ok(())
    }

    /// Call after the frame's last tick; systems see an empty input until the next boundary.
    pub fn take_input(&mut self) -> Input {
        std::mem::take(&mut self.input)
    }

    pub fn results(&self) -> &[CommandResult] {
        &self.results
    }

    pub fn pick(&self, ndc: [f32; 2]) -> Option<Pick> {
        self.domains.pick(&self.views, &self.prepared, ndc)
    }

    /// `Pending` while a deferred command or a reservation is outstanding, `Readback` while a required readback is unlanded, and `CheckpointTick` when an authoritative checkpoint is from another tick.
    pub fn snapshot(&self) -> Result<SessionSnapshot<A>, RestoreError> {
        if !self.commands.is_empty() || self.entities().has_reservations() {
            return Err(RestoreError::Pending);
        }
        if let Some(order) = self.flight.any_required() {
            return Err(RestoreError::Readback(order.name));
        }
        for (id, spec) in self.bulk.iter() {
            if spec.snapshot != SnapshotPolicy::Authoritative {
                continue;
            }
            let checkpoint = self.checkpoints.get(id.index()).and_then(Option::as_ref);
            if checkpoint.is_some_and(|checkpoint| checkpoint.tick != self.tick) {
                return Err(RestoreError::CheckpointTick(spec.name));
            }
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
            bulk: self.bulk.snapshot(&self.checkpoints),
        })
    }

    /// Refuses `NoCheckpoint` before touching anything; then cancels pending commands, reservations, and in-flight work, advances the epoch so every earlier external handle fails, and queues the bulk plan for `apply_restore`.
    pub fn restore(&mut self, from: &SessionSnapshot<A>) -> Result<(), RestoreError> {
        if from.domains.len() != self.domains.len() {
            let first = from.domains.len().min(self.domains.len());
            return Err(RestoreError::Domain(DomainId::new(first)));
        }
        for (index, spec) in from.bulk.specs.iter().enumerate() {
            let authoritative = spec.snapshot == SnapshotPolicy::Authoritative;
            if !from.bulk.live[index] || !authoritative {
                continue;
            }
            if from
                .bulk
                .checkpoints
                .get(index)
                .and_then(Option::as_ref)
                .is_none()
            {
                return Err(RestoreError::NoCheckpoint(spec.name));
            }
        }
        self.flight.cancel_all();
        self.bulk.restore(&from.bulk);
        self.checkpoints.clear();
        self.checkpoints.extend_from_slice(&from.bulk.checkpoints);
        self.restore_plan.clear();
        for (index, spec) in from.bulk.specs.iter().enumerate() {
            if !from.bulk.live[index] {
                continue;
            }
            let action = match spec.snapshot {
                SnapshotPolicy::Authoritative => BulkAction::Replace,
                SnapshotPolicy::Reinitializable => BulkAction::Reinitialize,
                SnapshotPolicy::Derived => continue,
            };
            self.restore_plan.push((BulkId::new(index), action));
        }
        self.wait = None;
        self.resume = None;
        self.work.clear();
        self.work_head = 0;
        self.ahead_for.clear();
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

    fn run_entry(&mut self, phase: Phase, index: usize, step: Step) -> Result<bool, DomainError> {
        let Session {
            app,
            domains,
            views,
            phases,
            commands,
            results,
            input,
            prepared,
            flight,
            wait,
            ..
        } = self;
        let Some(Entry::System(system)) = phases.entries_mut(phase).get_mut(index) else {
            return Ok(true);
        };
        if let Some(work) = system.access().awaited() {
            if let Some(order) = flight.outstanding(work) {
                *wait = Some(Wait {
                    entry: EntryId {
                        phase,
                        index: index as u32,
                    },
                    work,
                    request: order.request,
                    tick: order.tick,
                });
                return Ok(false);
            }
        }
        system.run(Ctx {
            app,
            domains,
            views,
            commands,
            results: results.as_slice(),
            input,
            prepared: prepared.as_slice(),
            step,
        })?;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use loam_math::{EuclideanR4, Iso4Flat, Space};

    use super::*;
    use crate::command::SpawnBundle;
    use crate::domain::{Instance, Pose};
    use crate::entity::Entity;
    use crate::phase::Readback;
    use crate::store::tests::alloc_probe::bytes_allocated_by;
    use crate::store::{LogCapacity, Store};
    use crate::view::{DepthEnvelope, DomainRay, ImageRay, ViewMapping, ViewSpec};

    crate::stores! {
        #[derive(Default)]
        pub struct Quiet {}
    }

    #[test]
    fn a_warmed_tick_allocates_while_it_orders_its_work_items() {
        let mut session = Session::new(Quiet::default(), SimConfig::default());
        let grid = session.register_bulk(BulkSpec {
            name: "grid",
            element_size: 4,
            count: 64,
            readback: Readback::Optional,
            snapshot: SnapshotPolicy::Derived,
            schedule: Schedule::InStep,
        });
        session.work(
            Phase::Simulation,
            WorkItem::new("step", Schedule::InStep, Readback::Optional).writes(grid),
        );
        session.work(
            Phase::Simulation,
            WorkItem::new("blur", Schedule::Ahead, Readback::Optional).reads(grid),
        );
        let rows = [0u8; 256];
        let mut publication = Publication::default();
        let mut requests: Vec<RequestId> = Vec::with_capacity(8);
        let cycle = |session: &mut Session<Quiet>,
                     publication: &mut Publication<Quiet>,
                     requests: &mut Vec<RequestId>| {
            session.boundary(Input::default()).unwrap();
            session.tick().unwrap();
            let settle = |session: &mut Session<Quiet>, requests: &mut Vec<RequestId>| {
                requests.clear();
                session.issue_work(|order| requests.push(order.request));
                for request in requests.iter() {
                    session.land_readback(*request, Some(&rows));
                    session.release_readback(*request);
                }
            };
            settle(session, requests);
            session.publish(publication).unwrap();
            settle(session, requests);
        };
        for _ in 0..8 {
            cycle(&mut session, &mut publication, &mut requests);
        }

        let bytes = bytes_allocated_by(|| {
            for _ in 0..16 {
                cycle(&mut session, &mut publication, &mut requests);
            }
        });
        assert_eq!(
            bytes, 0,
            "16 warmed ticks of work ordering asked the allocator for {bytes} bytes"
        );
        assert_eq!(session.work_stats().delayed, 0);
        assert_eq!(session.work_stats().discarded, 0);
    }

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
                        bulk_elements: 0,
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
