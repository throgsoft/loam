use loam_shape::polytope::Polytope4Topology;

use crate::bridge::{Bridge, BridgeError, BridgeSpec, Drag, DragError, DragRelease};
use crate::command::{
    Command, CommandResult, Commands, Dispatch, Outcome, Rejection, Request, RequestId,
};
use crate::domain::{
    ChartCommand, ChartPoint, DomainBuilder, DomainError, DomainHandle, DomainId, DomainSnapshot,
    DomainSpace, Domains,
};
use crate::entity::{Entities, EntitiesSnapshot, Epoch, RuntimeId, SceneId};
use crate::input::Input;
use crate::phase::{Ctx, Order, Phase, PhaseError, Phases, Step, System, SystemEntry, Tick};
use crate::relation::{LinkId, Relation, RelationSnapshot};
use crate::store::{Owner, SchemaId, StoreField};
use crate::stores::Stores;
use crate::view::{ImageRay, Pick, Rigid, ViewRecords, ViewTarget, Views, ViewsSnapshot};

const RELEASE_STALE_SECONDS: f64 = 0.12;

/// Fixed steps on both hosts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SimConfig {
    pub fixed_hz: u32,
    pub max_ticks_per_frame: u32,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PaletteId(u32);

impl PaletteId {
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
    /// Edges for the wireframe and cells for the section, at canonical coordinates times `scale`.
    Polytope4 {
        polytope: loam_shape::polytope::Polytope4,
        scale: f32,
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
            Self::Polytope4 { polytope, scale } => {
                polytope
                    .topology()
                    .vertices
                    .iter()
                    .map(|vertex| vertex.length())
                    .fold(0.0, f32::max)
                    * scale
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
    pub(crate) geometry: &'a [PreparedGeometry],
    pub(crate) materials: &'a [Material],
    pub(crate) palettes: &'a [Vec<[f32; 4]>],
}

impl Library<'_> {
    pub(crate) fn line_style(&self, material: MaterialId) -> ([f32; 4], f32) {
        match self.materials.get(material.index()) {
            Some(Material::Lines { color, width_px }) => (*color, *width_px),
            Some(Material::Flat { color }) => (*color, 1.0),
            None => ([1.0; 4], 1.0),
        }
    }

    pub(crate) fn palette(&self, palette: PaletteId) -> &[[f32; 4]] {
        self.palettes
            .get(palette.index())
            .map_or(&[][..], Vec::as_slice)
    }
}

#[derive(Default)]
pub(crate) struct Assets {
    prepared: Vec<PreparedGeometry>,
    materials: Vec<Material>,
    palettes: Vec<Vec<[f32; 4]>>,
}

impl Assets {
    pub(crate) fn prepare(&mut self, geometry: PreparedGeometry) -> PreparedId {
        self.prepared.push(geometry);
        PreparedId((self.prepared.len() - 1) as u32)
    }

    pub(crate) fn prepared(&self, id: PreparedId) -> Option<&PreparedGeometry> {
        self.prepared.get(id.index())
    }

    pub(crate) fn add_palette(&mut self, colors: Vec<[f32; 4]>) -> PaletteId {
        self.palettes.push(colors);
        PaletteId((self.palettes.len() - 1) as u32)
    }

    pub(crate) fn palette(&self, id: PaletteId) -> Option<&[[f32; 4]]> {
        self.palettes.get(id.index()).map(Vec::as_slice)
    }

    pub(crate) fn add_material(&mut self, material: Material) -> MaterialId {
        self.materials.push(material);
        MaterialId((self.materials.len() - 1) as u32)
    }

    pub(crate) fn material(&self, id: MaterialId) -> Option<&Material> {
        self.materials.get(id.index())
    }

    pub(crate) fn geometry(&self) -> &[PreparedGeometry] {
        &self.prepared
    }

    pub(crate) fn library(&self) -> Library<'_> {
        Library {
            geometry: &self.prepared,
            materials: &self.materials,
            palettes: &self.palettes,
        }
    }
}

pub struct PublishedView {
    pub domain: DomainId,
    pub target: ViewTarget,
    pub placement: Rigid,
    pub records: ViewRecords,
}

/// What a frame presents; publication writes it and the host reads it.
#[derive(Default)]
pub struct Publication {
    pub views: Vec<PublishedView>,
    pub stamp: Stamp,
    source: Option<SceneId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PublishError {
    Borrowed,
    Phase(PhaseError),
}

impl From<PhaseError> for PublishError {
    fn from(error: PhaseError) -> Self {
        Self::Phase(error)
    }
}

/// The one record buffer a session publishes into.
pub struct Records {
    idle: Option<Publication>,
}

impl Default for Records {
    fn default() -> Self {
        Self {
            idle: Some(Publication::default()),
        }
    }
}

impl Records {
    pub fn publish<A: Stores>(&mut self, session: &mut Session<A>) -> Result<Stamp, PublishError> {
        let buffer = self.idle.as_mut().ok_or(PublishError::Borrowed)?;
        session.publish(buffer)?;
        Ok(buffer.stamp)
    }

    /// Until the buffer is released, [`Self::publish`] returns [`PublishError::Borrowed`].
    pub fn lend(&mut self) -> Option<Publication> {
        self.idle.take()
    }

    pub fn release(&mut self, publication: Publication) {
        self.idle = Some(publication);
    }
}

pub struct SessionSnapshot<A: Stores> {
    runtime: RuntimeId,
    app: A::Snapshot,
    entities: EntitiesSnapshot,
    domains: Vec<DomainSnapshot>,
    views: ViewsSnapshot,
    bridges: RelationSnapshot<Bridge>,
    tick: Tick,
    config: SimConfig,
    next_request: RequestId,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RestoreError {
    Pending,
    Unfinished(Phase),
    NoInitial,
    ForeignRuntime,
    Schema(SchemaId),
    Domain(DomainId),
    /// A domain's world refused its snapshot before anything was touched.
    #[cfg(feature = "physics")]
    Edit(loam_physics::EditError),
}

impl std::fmt::Display for RestoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pending => f.write_str("deferred mutation is still pending"),
            Self::Unfinished(phase) => write!(f, "the {phase:?} phase is unfinished"),
            Self::NoInitial => f.write_str("no initial snapshot was captured"),
            Self::ForeignRuntime => f.write_str("the snapshot belongs to another runtime"),
            Self::Schema(schema) => write!(f, "no stored snapshot for {}", schema.name()),
            Self::Domain(domain) => write!(f, "{domain} refused the snapshot"),
            #[cfg(feature = "physics")]
            Self::Edit(error) => std::fmt::Display::fmt(error, f),
        }
    }
}

#[derive(Default)]
pub(crate) struct Manipulation {
    drag: Option<Drag>,
}

impl Manipulation {
    pub(crate) fn dragging(&self) -> Option<Drag> {
        self.drag
    }

    pub(crate) fn grab<A: Stores>(
        &mut self,
        domains: &Domains,
        views: &Views,
        prepared: &[PreparedGeometry],
        commands: &mut Commands<A>,
        ndc: [f32; 2],
        time: f64,
    ) -> Result<Pick, DragError> {
        let pick = domains
            .pick_lifted(views, prepared, ndc)
            .ok_or(DragError::NoPick)?;
        let domain = domains
            .get(pick.domain)
            .ok_or(DomainError::UnknownDomain(pick.domain))?;
        let into = views
            .to_root(pick.image)
            .and_then(|to| to.rigid())
            .ok_or(DomainError::Unsupported("image space"))?
            .inverse();
        let center = domain
            .image_of(pick.view, pick.entity)
            .ok_or(DomainError::Stale(pick.entity))?;
        let forward = views
            .get(views.root())
            .ok_or(DomainError::Unsupported("image space"))?
            .eye
            .forward;
        let plane = into.apply(pick.image_point);
        let hit = pick.hit.ok_or(DomainError::Unsupported("view ray lift"))?;
        if let Some(held) = self.drag.take() {
            commands.submit(Command::Chart(
                held.domain,
                ChartCommand::Release {
                    entity: held.entity,
                },
            ));
        }
        self.drag = Some(Drag {
            entity: pick.entity,
            domain: pick.domain,
            view: pick.view,
            image: pick.image,
            plane,
            normal: into.direction(forward),
            center,
            at: plane,
            time,
            velocity: [0.0; 3],
        });
        commands.submit(Command::Chart(
            pick.domain,
            ChartCommand::Grab {
                entity: pick.entity,
                point: hit,
            },
        ));
        Ok(pick)
    }

    pub(crate) fn drag<A: Stores>(
        &mut self,
        domains: &Domains,
        views: &Views,
        commands: &mut Commands<A>,
        ndc: [f32; 2],
        time: f64,
    ) -> Result<ChartPoint, DragError> {
        let mut drag = self.drag.ok_or(DragError::NotGrabbed)?;
        let domain = domains
            .get(drag.domain)
            .ok_or(DomainError::UnknownDomain(drag.domain))?;
        let name = domain
            .view(drag.view)
            .ok_or(DomainError::UnknownView(drag.view))?
            .name;
        let ray = views
            .ray(drag.image, ndc)
            .ok_or(DomainError::Unsupported("image space"))?;
        let at = drag.meet(&ray).ok_or(DragError::Ambiguous(name))?;
        let point = domain.lift_origin(
            drag.view,
            &ImageRay {
                origin: drag.moved(at),
                direction: drag.normal,
            },
        )?;
        drag.sample(at, time);
        self.drag = Some(drag);
        commands.submit(Command::Chart(
            drag.domain,
            ChartCommand::Move {
                entity: drag.entity,
                point,
            },
        ));
        Ok(point)
    }

    fn finish<A: Stores>(commands: &mut Commands<A>, drag: Drag, throw: bool) -> DragRelease {
        let mut release = drag.released();
        if !throw {
            release.velocity = [0.0; 3];
        }
        commands.submit(Command::Chart(
            drag.domain,
            ChartCommand::Release {
                entity: drag.entity,
            },
        ));
        release
    }

    pub(crate) fn release<A: Stores>(&mut self, commands: &mut Commands<A>) -> Option<DragRelease> {
        let drag = self.drag.take()?;
        Some(Self::finish(commands, drag, true))
    }

    pub(crate) fn release_at<A: Stores>(
        &mut self,
        commands: &mut Commands<A>,
        time: f64,
    ) -> Option<DragRelease> {
        let drag = self.drag.take()?;
        let age = time - drag.time;
        Some(Self::finish(
            commands,
            drag,
            (0.0..=RELEASE_STALE_SECONDS).contains(&age),
        ))
    }

    pub(crate) fn cancel<A: Stores>(&mut self, commands: &mut Commands<A>) -> Option<DragRelease> {
        let drag = self.drag.take()?;
        Some(Self::finish(commands, drag, false))
    }

    pub(crate) fn clear(&mut self) {
        self.drag = None;
    }
}

/// The simulation entry `Session::new` registers; `system_at` places app entries around it.
pub const DOMAIN_STEP: &str = "domain step";

/// A CPU value with no GPU or window object; `Send`, and it builds for wasm32.
pub struct Session<A: Stores> {
    pub app: A,
    domains: Domains,
    views: Views,
    bridges: Relation<Bridge>,
    manipulation: Manipulation,
    phases: Phases<A>,
    commands: Commands<A>,
    batch: Vec<Request<A>>,
    results: Vec<CommandResult>,
    input: Input,
    assets: Assets,
    config: SimConfig,
    tick: Tick,
    sequence: u64,
    initial: Option<SessionSnapshot<A>>,
    restored: bool,
    unfinished: Option<Phase>,
    phase_error: Option<PhaseError>,
}

impl<A: Stores> Session<A> {
    pub fn new(mut app: A, config: SimConfig) -> Self {
        let scene = SceneId {
            runtime: RuntimeId::allocate(),
            epoch: Epoch::default(),
        };
        app.bind(scene, Owner::new());
        let mut phases = Phases::new();
        phases.push(
            Phase::Simulation,
            SystemEntry::new(DOMAIN_STEP, |ctx: Ctx<'_, A>| {
                for domain in ctx.domains.iter_mut() {
                    domain.step(ctx.step)?;
                }
                Ok(())
            }),
        );
        let mut bridges = Relation::new();
        StoreField::bind(&mut bridges, scene, Owner::new());
        Self {
            app,
            domains: Domains::new(scene.runtime),
            views: Views::new(),
            bridges,
            manipulation: Manipulation::default(),
            phases,
            commands: Commands::new(scene),
            batch: Vec::new(),
            results: Vec::new(),
            input: Input::default(),
            assets: Assets::default(),
            config,
            tick: Tick::default(),
            sequence: 0,
            initial: None,
            restored: false,
            unfinished: None,
            phase_error: None,
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

    pub fn faulted_phase(&self) -> Option<Phase> {
        self.unfinished
    }

    pub fn phase_error(&self) -> Option<PhaseError> {
        self.phase_error
    }

    fn unfinished_error(&self) -> Option<PhaseError> {
        self.unfinished.map(|phase| {
            self.phase_error.unwrap_or_else(|| {
                PhaseError::unnamed(
                    phase,
                    DomainError::Unsupported("unfinished phase requires restore"),
                )
            })
        })
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

    pub fn submit(&mut self, command: Command<A>) -> RequestId {
        self.commands.submit(command)
    }

    pub fn prepare(&mut self, geometry: PreparedGeometry) -> PreparedId {
        self.assets.prepare(geometry)
    }

    pub fn prepared(&self, id: PreparedId) -> Option<&PreparedGeometry> {
        self.assets.prepared(id)
    }

    /// Two colors per prepared segment, start then end, read in the prepared geometry's own order; a segment past the palette's end keeps the material color.
    pub fn add_palette(&mut self, colors: Vec<[f32; 4]>) -> PaletteId {
        self.assets.add_palette(colors)
    }

    pub fn palette(&self, id: PaletteId) -> Option<&[[f32; 4]]> {
        self.assets.palette(id)
    }

    pub fn add_material(&mut self, material: Material) -> MaterialId {
        self.assets.add_material(material)
    }

    pub fn material(&self, id: MaterialId) -> Option<&Material> {
        self.assets.material(id)
    }

    pub fn system(&mut self, phase: Phase, name: &'static str, system: impl System<A>) {
        self.phases.push(phase, SystemEntry::new(name, system));
    }

    /// `None` when no entry of that phase has the named anchor.
    pub fn system_at(
        &mut self,
        phase: Phase,
        order: Order,
        name: &'static str,
        system: impl System<A>,
    ) -> Option<()> {
        self.phases
            .insert(phase, order, SystemEntry::new(name, system))
    }

    pub fn entries(&self, phase: Phase) -> &[SystemEntry<A>] {
        self.phases.entries(phase)
    }

    pub fn dispatch<R>(&mut self, f: impl FnOnce(&mut Dispatch<'_, A>) -> R) -> R {
        let result = {
            let mut dispatch = Dispatch::new(
                &mut self.app,
                &mut self.domains,
                &mut self.views,
                self.commands.entities_mut(),
                &mut self.bridges,
                &mut self.assets,
            );
            f(&mut dispatch)
        };
        self.domains.synchronize();
        result
    }

    pub fn bridges(&self) -> &Relation<Bridge> {
        &self.bridges
    }

    /// Checks the anchor, domain, view, and placement kind before it links the view.
    pub fn bridge(&mut self, spec: BridgeSpec) -> Result<LinkId, BridgeError> {
        if self.entities().resolve(spec.anchor).is_none() {
            return Err(BridgeError::Domain(DomainError::Stale(spec.anchor)));
        }
        let summary = {
            let source = self
                .domains
                .get(spec.source)
                .ok_or(BridgeError::UnknownDomain(spec.source))?;
            source
                .view(spec.view)
                .ok_or(BridgeError::UnknownView(spec.view))?
        };
        if spec.placement.rigid().is_none() {
            return Err(BridgeError::Nonlinear(summary.name));
        }
        if self.entities().resolve(summary.eye).is_none() {
            return Err(BridgeError::Domain(DomainError::Stale(summary.eye)));
        }
        let image = self
            .views
            .place(spec.into, spec.placement)
            .ok_or(BridgeError::UnknownImage(spec.into))?;
        let link = self.bridges.link(
            self.commands.entities(),
            spec.anchor,
            summary.eye,
            Bridge {
                source: spec.source,
                view: spec.view,
                image,
            },
        );
        let link = match link {
            Ok(link) => link,
            Err(error) => {
                self.views.unplace(image);
                return Err(BridgeError::Link(error));
            }
        };
        let source = self
            .domains
            .facade(spec.source)
            .ok_or(BridgeError::UnknownDomain(spec.source))?;
        source.retarget(spec.view, image)?;
        Ok(link)
    }

    pub fn dragging(&self) -> Option<Drag> {
        self.manipulation.dragging()
    }

    /// Picks among the views with a ray lift and records a drag plane through the hit facing the root eye.
    pub fn grab(&mut self, ndc: [f32; 2], time: f64) -> Result<Pick, DragError> {
        self.manipulation.grab(
            &self.domains,
            &self.views,
            self.assets.geometry(),
            &mut self.commands,
            ndc,
            time,
        )
    }

    /// Meets the pointer ray with the drag plane, refusing a parallel ray as ambiguous, moves the entity's image point by the pointer delta, lifts it through the view's own map, and submits a `Move`.
    pub fn drag(&mut self, ndc: [f32; 2], time: f64) -> Result<ChartPoint, DragError> {
        self.manipulation
            .drag(&self.domains, &self.views, &mut self.commands, ndc, time)
    }

    pub fn release(&mut self) -> Option<DragRelease> {
        self.manipulation.release(&mut self.commands)
    }

    pub fn release_at(&mut self, time: f64) -> Option<DragRelease> {
        self.manipulation.release_at(&mut self.commands, time)
    }

    pub fn cancel_drag(&mut self) -> Option<DragRelease> {
        self.manipulation.cancel(&mut self.commands)
    }

    /// Commits deferred commands, runs each dispatch entry and then its commands, counts what grew, and on the first boundary that finishes stores the session's initial snapshot if none was set.
    pub fn boundary(&mut self, input: Input) -> Result<Growth, PhaseError> {
        self.input = input;
        if let Some(error) = self.unfinished_error() {
            return Err(error);
        }
        self.unfinished = Some(Phase::Dispatch);
        self.results.clear();
        let mut growth = Growth::default();
        self.commit(&mut growth);
        let step = Step {
            tick: self.tick,
            dt: self.config.dt().unwrap_or(0.0),
        };
        let mut dispatched = Ok(());
        for index in 0..self.phases.entries(Phase::Dispatch).len() {
            if let Err(error) = self.run_entry(Phase::Dispatch, index, step) {
                dispatched = Err(error);
                break;
            }
            self.commit(&mut growth);
        }
        self.restored = false;
        dispatched?;
        self.app.boundary(Owner::new());
        for domain in self.domains.iter_mut() {
            domain.boundary();
        }
        self.unfinished = None;
        if self.initial.is_none() {
            match self.snapshot() {
                Ok(initial) => self.initial = Some(initial),
                Err(cause) => {
                    let error =
                        PhaseError::system(Phase::Dispatch, "initial", DomainError::Restore(cause));
                    self.unfinished = Some(Phase::Dispatch);
                    self.phase_error = Some(error);
                    return Err(error);
                }
            }
        }
        Ok(growth)
    }

    /// One fixed step with the domain step among the simulation entries.
    pub fn tick(&mut self) -> Result<(), PhaseError> {
        if let Some(error) = self.unfinished_error() {
            return Err(error);
        }
        let Some(dt) = self.config.dt() else {
            return Ok(());
        };
        self.unfinished = Some(Phase::Simulation);
        let step = Step {
            tick: self.tick,
            dt,
        };
        for index in 0..self.phases.entries(Phase::Simulation).len() {
            self.run_entry(Phase::Simulation, index, step)?;
        }
        self.tick = Tick(self.tick.0 + 1);
        self.unfinished = None;
        Ok(())
    }

    pub fn publish(&mut self, into: &mut Publication) -> Result<(), PhaseError> {
        if let Some(error) = self.unfinished_error() {
            return Err(error);
        }
        let step = Step {
            tick: self.tick,
            dt: self.config.dt().unwrap_or(0.0),
        };
        self.unfinished = Some(Phase::Publication);
        self.run_phase(Phase::Publication, step)?;
        let scene = self.scene();
        if into.source != Some(scene) {
            *into = Publication::default();
        }
        let sequence = self.sequence.wrapping_add(1);
        let stamp = Stamp {
            tick: self.tick,
            sequence,
        };
        let extracted = (|| {
            let library = self.assets.library();
            let mut count = 0;
            for domain in self.domains.owned() {
                for &target in domain.views() {
                    let Some(placement) =
                        self.views.to_root(target.image).and_then(|to| to.rigid())
                    else {
                        continue;
                    };
                    let current = into
                        .views
                        .get(count)
                        .is_some_and(|view| view.domain == domain.id() && view.target == target);
                    if !current {
                        into.views.truncate(count);
                        into.views.push(PublishedView {
                            domain: domain.id(),
                            target,
                            placement,
                            records: ViewRecords::default(),
                        });
                    }
                    into.views[count].placement = placement;
                    domain.publish(target.view, library, &mut into.views[count].records, stamp)?;
                    count += 1;
                }
            }
            into.views.truncate(count);
            into.stamp = stamp;
            Ok::<(), DomainError>(())
        })();
        if let Err(error) = extracted {
            *into = Publication::default();
            let error = PhaseError::unnamed(Phase::Publication, error);
            self.phase_error = Some(error);
            return Err(error);
        }
        into.source = Some(scene);
        self.sequence = sequence;
        self.unfinished = None;
        Ok(())
    }

    fn run_phase(&mut self, phase: Phase, step: Step) -> Result<(), PhaseError> {
        for index in 0..self.phases.entries(phase).len() {
            self.run_entry(phase, index, step)?;
        }
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
        self.domains.pick(&self.views, self.assets.geometry(), ndc)
    }

    /// `Unfinished` while a phase is incomplete and `Pending` while deferred mutation remains.
    pub fn snapshot(&self) -> Result<SessionSnapshot<A>, RestoreError> {
        if let Some(phase) = self.unfinished {
            return Err(RestoreError::Unfinished(phase));
        }
        if !self.commands.is_empty() || self.entities().has_reservations() {
            return Err(RestoreError::Pending);
        }
        Ok(SessionSnapshot {
            runtime: self.scene().runtime,
            app: self.app.snapshot(),
            entities: self.entities().snapshot(),
            domains: self
                .domains
                .owned()
                .map(|domain| domain.snapshot())
                .collect(),
            views: self.views.snapshot(),
            bridges: StoreField::snapshot(&self.bridges),
            tick: self.tick,
            config: self.config,
            next_request: self.commands.next_request(),
        })
    }

    /// Validates ownership and every domain before it cancels pending commands and advances the epoch; a failure after that faults the session until a later restore succeeds.
    pub fn restore(&mut self, from: &SessionSnapshot<A>) -> Result<(), RestoreError> {
        if from.runtime != self.scene().runtime {
            return Err(RestoreError::ForeignRuntime);
        }
        if from.domains.len() != self.domains.len() {
            let first = from.domains.len().min(self.domains.len());
            return Err(RestoreError::Domain(DomainId::new(first)));
        }
        for (domain, snapshot) in self.domains.owned().zip(&from.domains) {
            domain.check_restore(snapshot)?;
        }
        self.restored = true;
        self.commands.cancel_into(&mut self.results);
        self.commands.restore(&from.entities, from.next_request);
        let scene = self.scene();
        for (domain, snapshot) in self.domains.iter_mut().zip(&from.domains) {
            if let Err(error) = domain.restore(snapshot, scene) {
                self.unfinished = Some(Phase::Dispatch);
                self.phase_error = Some(PhaseError::system(
                    Phase::Dispatch,
                    "restore",
                    DomainError::Restore(error),
                ));
                return Err(error);
            }
        }
        self.app.restore(&from.app, scene, Owner::new());
        self.views.restore(&from.views);
        StoreField::restore(&mut self.bridges, &from.bridges, scene, Owner::new());
        self.manipulation.clear();
        self.tick = from.tick;
        self.config = from.config;
        self.unfinished = None;
        self.phase_error = None;
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
        for request in batch.drain(..) {
            let despawn = matches!(request.command, Command::Despawn(_));
            let name = request.command.name();
            let outcome = match request.command {
                Command::Reset if self.restored => Err(Rejection::Cancelled),
                Command::Reset => {
                    let unfinished = self.unfinished;
                    let phase_error = self.phase_error;
                    let outcome = self
                        .reset()
                        .map(|()| Outcome::Done)
                        .map_err(Rejection::Restore);
                    if outcome.is_ok() {
                        self.unfinished = unfinished;
                        self.phase_error = phase_error;
                    }
                    outcome
                }
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
                name,
                outcome,
            });
        }
        self.batch = batch;
    }

    fn run_entry(&mut self, phase: Phase, index: usize, step: Step) -> Result<(), PhaseError> {
        let Session {
            app,
            domains,
            views,
            phases,
            commands,
            results,
            input,
            assets,
            manipulation,
            phase_error,
            ..
        } = self;
        let Some(system) = phases.entries_mut(phase).get_mut(index) else {
            return Ok(());
        };
        let name = system.name();
        let result = system.run(Ctx {
            app,
            domains,
            views,
            commands,
            results: results.as_slice(),
            input,
            prepared: assets.geometry(),
            step,
            manipulation,
        });
        domains.synchronize();
        result.map_err(|cause| {
            let error = PhaseError::system(phase, name, cause);
            *phase_error = Some(error);
            error
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::view::Vec4;
    use loam_math::{EuclideanR4, Space};
    use loam_time::alloc::bytes_allocated_by;

    use super::*;
    use crate::command::SpawnBundle;
    use crate::domain::{Instance, Pose};
    use crate::entity::Entity;
    use crate::store::{LogCapacity, Store};
    use crate::view::{DepthEnvelope, DomainRay, ImageRay, ViewMapping, ViewSpec};

    crate::stores! {
        #[derive(Default)]
        pub struct Quiet {}
    }

    #[test]
    fn a_caught_system_unwind_keeps_the_unfinished_phase_blocked() {
        let mut session = Session::new(Quiet::default(), SimConfig::default());
        session.system(Phase::Simulation, "panic", |_ctx: Ctx<'_, Quiet>| {
            panic!("system panic")
        });
        let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| session.tick()));
        assert!(unwind.is_err());

        let mut publication = Publication::default();
        assert_eq!(
            session.publish(&mut publication),
            Err(PhaseError {
                phase: Phase::Simulation,
                system: None,
                cause: DomainError::Unsupported("unfinished phase requires restore"),
            })
        );
        assert_eq!(session.faulted_phase(), Some(Phase::Simulation));
    }

    fn shaded_segments(
        shading: crate::domain::EdgeShading,
        at_w: f32,
        line_style: Option<(f32, f32)>,
        sectioned: bool,
        prepare: impl FnOnce(&mut Session<Quiet>) -> (PreparedId, MaterialId),
    ) -> Vec<crate::view::SegmentRecord> {
        use crate::view::{Eye, Section4, Vec4};

        let mut session = Session::new(Quiet::default(), SimConfig::default());
        let r4 = session
            .register_domain(DomainBuilder::new("r4", EuclideanR4).tracked(LogCapacity::default()));
        let (geometry, material) = prepare(&mut session);
        let root = session.views().root();
        session
            .dispatch(|d| -> Result<(), Rejection> {
                let eye = d.spawn(SpawnBundle::new().at(r4, Pose::at(Vec4::ZERO)))?;
                let mut instance = Instance::new(geometry, material).shaded(shading);
                if let Some((width_px, opacity)) = line_style {
                    instance = instance.line_style(width_px, opacity);
                }
                if sectioned {
                    instance = instance.sectioned(material);
                }
                d.spawn(
                    SpawnBundle::new()
                        .at(r4, Pose::at(Vec4::new(0.0, 0.0, -4.0, at_w)))
                        .instance(instance),
                )?;
                d.domains
                    .typed(r4)?
                    .add_view(ViewSpec::new(root, eye, Section4 { w: at_w }))?;
                Ok(())
            })
            .expect("the view registered");
        session.views_mut().root_mut().eye = Eye::default();
        let mut records = Records::default();
        records.publish(&mut session).expect("published");
        let publication = records.lend().expect("the buffer is free");
        publication.views[0].records.segments().to_vec()
    }

    #[test]
    fn the_tesseract_cut_at_w_zero_publishes_the_square_caps_of_its_six_straddling_cells() {
        use crate::view::{Eye, Section4, Vec4, ViewId};

        const SIZE: f32 = 0.7;
        const LIFT: f32 = 0.2;
        let mut session = Session::new(Quiet::default(), SimConfig::default());
        let r4 = session
            .register_domain(DomainBuilder::new("r4", EuclideanR4).tracked(LogCapacity::default()));
        let geometry = session.prepare(PreparedGeometry::Polytope4 {
            polytope: loam_shape::polytope::Polytope4::Tesseract,
            scale: SIZE,
        });
        let body = session.add_material(Material::lines([1.0; 4], 1.0));
        let cut = session.add_material(Material::lines([1.0, 0.85, 0.35, 1.0], 2.0));
        let root = session.views().root();
        let view = session
            .dispatch(|d| -> Result<ViewId, Rejection> {
                let eye = d.spawn(SpawnBundle::new().at(r4, Pose::at(Vec4::ZERO)))?;
                d.spawn(
                    SpawnBundle::new()
                        .at(r4, Pose::at(Vec4::new(0.0, 0.0, -4.0, 0.0)))
                        .instance(Instance::new(geometry, body).sectioned(cut)),
                )?;
                d.domains
                    .typed(r4)?
                    .add_view(ViewSpec::new(root, eye, Section4 { w: 0.0 }))
                    .map_err(Rejection::from)
            })
            .expect("the view registered");
        session.views_mut().root_mut().eye = Eye::default();

        let mut records = Records::default();
        let read = |session: &mut Session<Quiet>, records: &mut Records| {
            records.publish(session).expect("published");
            let publication = records.lend().expect("the buffer is free");
            let view = &publication.views[0];
            let read = (
                view.records.triangles().len(),
                view.records.segments().len(),
                view.records.built(),
            );
            records.release(publication);
            read
        };

        let (triangles, segments, built) = read(&mut session, &mut records);
        assert_eq!(
            triangles, 24,
            "the six cells that straddle w = 0 each cut to a square, and each square fans around its centroid into four triangles, not {triangles}"
        );
        assert_eq!(
            segments,
            32 + 24,
            "the wireframe's 32 edges and the cut cube's 24 perimeter segments came to {segments}"
        );
        assert_eq!(
            read(&mut session, &mut records).2,
            built,
            "an unchanged source and view rebuilt the section"
        );

        session
            .dispatch(|d| -> Result<(), Rejection> {
                d.domains
                    .typed(r4)?
                    .view_mut(view)
                    .ok_or(Rejection::Unsupported("view"))?
                    .set_mapping(Section4 { w: LIFT });
                Ok(())
            })
            .expect("the slice moved");
        let (lifted, _, moved) = read(&mut session, &mut records);
        assert_ne!(
            moved, built,
            "a new slice left the section at its old build"
        );
        assert_eq!(
            lifted, 24,
            "the cut at w = {LIFT} still meets six cells, not {lifted}"
        );
    }

    #[test]
    fn a_depth_shaded_segment_reads_its_color_from_the_endpoints_own_w_not_the_bodys() {
        use crate::domain::EdgeShading;

        const EXTENT: f32 = 0.5;
        const BACK: [f32; 4] = [0.0, 0.0, 1.0, 1.0];
        const FRONT: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
        let shading = EdgeShading::Depth {
            back: BACK,
            front: FRONT,
            extent: EXTENT,
        };
        let prepare = |session: &mut Session<Quiet>| {
            (
                session.prepare(PreparedGeometry::Lines4 {
                    segments: vec![[[0.0; 4], [0.0, 0.0, 0.0, EXTENT]]],
                }),
                session.add_material(Material::lines([1.0; 4], 1.0)),
            )
        };

        for lifted in [0.0, 3.0] {
            let segments = shaded_segments(shading, lifted, None, false, prepare);
            assert_eq!(
                segments[0].start_color,
                [0.5, 0.0, 0.5, 1.0],
                "the midpoint of the depth ramp is wrong for a body at w {lifted}"
            );
            assert_eq!(
                segments[0].end_color, FRONT,
                "the far endpoint of the depth ramp is wrong for a body at w {lifted}"
            );
        }
    }

    #[test]
    fn a_palette_colors_each_segment_by_its_prepared_index_and_no_shading_keeps_the_material() {
        use crate::domain::EdgeShading;

        const COLORS: [[f32; 4]; 4] = [
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
        ];
        const MATERIAL: [f32; 4] = [0.25, 0.5, 0.75, 1.0];
        let two_edges = vec![
            [[0.0; 4], [0.5, 0.0, 0.0, 0.0]],
            [[0.0; 4], [0.0, 0.5, 0.0, 0.0]],
        ];

        let painted = shaded_segments(EdgeShading::Material, 0.0, None, false, |session| {
            (
                session.prepare(PreparedGeometry::Lines4 {
                    segments: two_edges.clone(),
                }),
                session.add_material(Material::lines(MATERIAL, 1.0)),
            )
        });
        assert_eq!(
            painted.iter().map(|s| s.start_color).collect::<Vec<_>>(),
            [MATERIAL, MATERIAL],
            "an instance with no shading lost the material's line color"
        );

        let mut id = None;
        let painted = shaded_segments(
            EdgeShading::Palette(PaletteId(0)),
            0.0,
            None,
            false,
            |session: &mut Session<Quiet>| {
                id = Some(session.add_palette(COLORS.to_vec()));
                (
                    session.prepare(PreparedGeometry::Lines4 {
                        segments: two_edges,
                    }),
                    session.add_material(Material::lines(MATERIAL, 1.0)),
                )
            },
        );
        assert_eq!(id, Some(PaletteId(0)));
        assert_eq!(
            painted
                .iter()
                .flat_map(|s| [s.start_color, s.end_color])
                .collect::<Vec<_>>(),
            COLORS,
            "the palette did not follow the prepared segment endpoints in order"
        );
    }

    #[test]
    fn an_instance_line_style_replaces_the_material_width_and_opacity() {
        let [segment] = shaded_segments(
            crate::domain::EdgeShading::Material,
            0.0,
            Some((2.5, 0.4)),
            false,
            |session| {
                (
                    session.prepare(PreparedGeometry::Lines4 {
                        segments: vec![[[0.0; 4], [0.5, 0.0, 0.0, 0.0]]],
                    }),
                    session.add_material(Material::lines([0.2, 0.4, 0.6, 0.8], 1.0)),
                )
            },
        )[..] else {
            panic!("the line did not publish as one segment");
        };
        assert_eq!(segment.width_px, 2.5);
        assert_eq!(segment.start_color, [0.2, 0.4, 0.6, 0.4]);
        assert_eq!(segment.end_color, [0.2, 0.4, 0.6, 0.4]);

        let sectioned = shaded_segments(
            crate::domain::EdgeShading::Material,
            0.0,
            Some((2.5, 0.4)),
            true,
            |session| {
                (
                    session.prepare(PreparedGeometry::Polytope4 {
                        polytope: loam_shape::polytope::Polytope4::Tesseract,
                        scale: 0.7,
                    }),
                    session.add_material(Material::lines([0.2, 0.4, 0.6, 0.8], 1.0)),
                )
            },
        );
        assert!(sectioned.len() > 32, "the section edge did not publish");
        assert!(sectioned[..sectioned.len() - 32].iter().all(|segment| {
            segment.width_px == 2.5
                && segment.start_color == [0.2, 0.4, 0.6, 0.4]
                && segment.end_color == [0.2, 0.4, 0.6, 0.4]
        }));
    }

    #[test]
    fn a_grab_takes_the_view_it_can_lift_while_a_plain_pick_keeps_the_nearer_one() {
        use crate::view::{Eye, Projection4, Section4, Vec4, ViewId};
        use loam_math::EuclideanR4;

        const DEPTH: f32 = 4.0;
        const FOCAL: f32 = 2.5;
        const AT_W: f32 = -2.5;

        let mut session = Session::new(Quiet::default(), SimConfig::default());
        let r4 = session
            .register_domain(DomainBuilder::new("r4", EuclideanR4).tracked(LogCapacity::default()));
        let edges = session.prepare(PreparedGeometry::Lines4 {
            segments: vec![[[0.5, 0.0, 0.0, 0.0], [-0.5, 0.0, 0.0, 0.0]]],
        });
        let white = session.add_material(Material::lines([1.0; 4], 1.0));
        let root = session.views().root();
        let (section, projection) = session
            .dispatch(|d| -> Result<(ViewId, ViewId), Rejection> {
                let eye = d.spawn(SpawnBundle::new().at(r4, Pose::at(Vec4::ZERO)))?;
                d.spawn(
                    SpawnBundle::new()
                        .at(r4, Pose::at(Vec4::new(0.0, 0.0, -DEPTH, AT_W)))
                        .instance(Instance::new(edges, white)),
                )?;
                let domain = d.domains.typed(r4)?;
                let section = domain.add_view(ViewSpec::new(root, eye, Section4 { w: AT_W }))?;
                let projection =
                    domain.add_view(ViewSpec::new(root, eye, Projection4 { focal: FOCAL }))?;
                Ok((section, projection))
            })
            .expect("the two views registered");
        session.views_mut().root_mut().eye = Eye::default();

        let picked = session
            .pick([0.0, 0.0])
            .expect("both views cover the origin");
        assert_eq!(
            picked.view, projection,
            "the plain pick lost the nearer projection layer"
        );
        let grabbed = session.grab([0.0, 0.0], 0.0).expect("the section lifts");
        assert_eq!(
            grabbed.view, section,
            "the grab resolved to a view it cannot lift, so a drag is a coin flip"
        );
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
                    d.spawn(SpawnBundle::new().at(r4, Pose::at(Vec4::ZERO)).row(value))
                        .unwrap()
                })
                .collect()
        });
        session.app.pool.set(pool);
        session.system(Phase::Simulation, "churn", |ctx: Ctx<'_, Churn>| {
            for (_, counter) in ctx.app.counters.iter_mut() {
                *counter += 1;
            }
            let pool = ctx.app.pool.get_mut();
            let retired = pool.swap_remove(0);
            ctx.commands.submit(Command::Despawn(retired));
            let fresh = ctx.commands.spawn(SpawnBundle::new()).unwrap();
            pool.push(fresh.entity);
            Ok(())
        });
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
        #[derive(Default)]
        pub struct Tagged {
            tags: Store<u32>,
        }
    }

    #[test]
    fn a_warmed_restore_allocates_beyond_the_rows_it_copies() {
        use crate::domain::{Field, FieldKind};
        use crate::field::FieldOp;
        use crate::view::Vec4;

        let mut session = Session::new(Tagged::default(), SimConfig::default());
        let r4 = session.register_domain(
            DomainBuilder::new("r4", EuclideanR4)
                .tracked(LogCapacity::default())
                .fields(),
        );
        session.dispatch(|d| {
            let at = |x: f32| SpawnBundle::new().at(r4, Pose::at(Vec4::new(x, 0.0, 0.0, 0.0)));
            let left = d.spawn(at(-1.0).row(1u32)).unwrap();
            let right = d.spawn(at(1.0).row(2u32)).unwrap();
            let union = d.spawn(at(0.0).row(3u32)).unwrap();
            for operand in [left, right] {
                d.attach_field(
                    r4,
                    operand,
                    Field {
                        kind: FieldKind::ExactDistance,
                        op: FieldOp::HyperSphere { radius: 1.0 },
                        operands: Vec::new(),
                    },
                )
                .unwrap();
            }
            d.attach_field(
                r4,
                union,
                Field {
                    kind: FieldKind::ExactDistance,
                    op: FieldOp::Union,
                    operands: vec![left, right],
                },
            )
            .unwrap();
        });
        let snapshot = session.snapshot().unwrap();
        for _ in 0..8 {
            session.restore(&snapshot).unwrap();
        }

        let bytes = bytes_allocated_by(|| {
            session.restore(&snapshot).unwrap();
        });
        let copied = (size_of::<Entity>() * 2) as u64;
        assert_eq!(
            bytes, copied,
            "a warmed restore of three rows asked the allocator for {bytes} bytes, not the {copied} its operand list copies"
        );
        session
            .domains_mut()
            .facade(r4.id())
            .unwrap()
            .compile_fields()
            .expect("the restored operands still name live rows");
    }

    crate::stores! {
        pub struct Shown {
            scores: Store<u32>,
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
            let relative = point - eye.point;
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
    fn a_foreign_publication_buffer_does_not_restamp_the_previous_sessions_records() {
        let build = |score: u32, offset: f32| {
            let shown = Shown {
                scores: Store::tracked(LogCapacity::default()),
            };
            let mut session = Session::new(shown, SimConfig::default());
            let r4 = session.register_domain(
                DomainBuilder::new("r4", EuclideanR4).tracked(LogCapacity::default()),
            );
            let geometry = session.prepare(PreparedGeometry::Lines4 {
                segments: vec![[[offset, 0.0, 0.0, 0.0], [offset + 1.0, 0.0, 0.0, 0.0]]],
            });
            let material = session.add_material(Material::flat([1.0; 4]));
            let root = session.views().root();
            session.dispatch(|dispatch| {
                let eye = dispatch
                    .spawn(SpawnBundle::new().at(r4, Pose::at(Vec4::ZERO)))
                    .unwrap();
                dispatch
                    .spawn(
                        SpawnBundle::new()
                            .at(r4, Pose::at(Vec4::ZERO))
                            .instance(Instance::new(geometry, material))
                            .row(score),
                    )
                    .unwrap();
                dispatch
                    .domains
                    .typed(r4)
                    .unwrap()
                    .add_view(ViewSpec::new(root, eye, Flat))
                    .unwrap();
            });
            session
        };
        let mut first = build(11, 0.0);
        let mut second = build(22, 4.0);
        let mut publication = Publication::default();

        first.publish(&mut publication).unwrap();
        assert_eq!(publication.views[0].records.segments()[0].start[0], 0.0);
        second.publish(&mut publication).unwrap();
        assert_eq!(publication.views[0].records.segments()[0].start[0], 4.0);
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
                .spawn(SpawnBundle::new().at(r4, Pose::at(Vec4::ZERO)))
                .unwrap();
            d.domains
                .typed(r4)
                .unwrap()
                .add_view(ViewSpec::new(root, eye, Flat))
                .unwrap();
            for value in 0..8 {
                d.spawn(
                    SpawnBundle::new()
                        .at(r4, Pose::at(Vec4::ZERO))
                        .instance(Instance::new(geometry, material))
                        .row(value),
                )
                .unwrap();
            }
        });
        session.system(Phase::Simulation, "churn", move |ctx: Ctx<'_, Shown>| {
            let domain = ctx.domains.typed(r4).unwrap();
            for (entity, score) in ctx.app.scores.iter_mut() {
                *score += 1;
                let mut point = domain.poses().get(entity).unwrap().point;
                point.x += ctx.step.dt;
                domain.set_point(entity, point).unwrap();
            }
            Ok(())
        });
        let mut publication = Publication::default();
        let cycle = |session: &mut Session<Shown>, publication: &mut Publication| {
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
        assert_eq!(publication.views[0].records.instances.rows().len(), 8);
    }
}
