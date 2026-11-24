use std::sync::Arc;

use loam_time::par::{Executor, Sequential};

use crate::assets::{Assets, Material, MaterialId, PaletteId, PreparedGeometry, PreparedId};
use crate::camera::{self, Rig, CAMERA_RIG, CAMERA_TRAVEL};
use crate::command::{
    Command, CommandResult, Commands, Dispatch, Outcome, Rejection, Request, RequestId,
};
use crate::domain::{
    presented_alpha, ChartPoint, DomainBuilder, DomainError, DomainHandle, DomainId,
    DomainSnapshot, DomainSpace, Domains,
};
use crate::drag::{Drag, DragError, DragRelease, Manipulation};
use crate::entity::{Entities, EntitiesSnapshot, Epoch, RuntimeId, SceneId};
use crate::input::{ActionId, Input};
use crate::phase::{Ctx, Order, Phase, PhaseError, Phases, Step, SystemEntry, Tick};
use crate::publication::{Publication, Stamp};
use crate::store::{Owner, SchemaId};
use crate::stores::Stores;
use crate::value::Setting;
use crate::view::{Overlay, Pick, Views, ViewsSnapshot};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SimConfig {
    pub fixed_hz: u32,
    pub max_ticks_per_frame: u32,
}

impl SimConfig {
    pub fn new(fixed_hz: u32, max_ticks_per_frame: u32) -> Result<Self, SimConfigError> {
        let config = Self {
            fixed_hz,
            max_ticks_per_frame,
        };
        config.check()?;
        Ok(config)
    }

    pub fn check(&self) -> Result<(), SimConfigError> {
        if self.max_ticks_per_frame == 0 {
            return Err(SimConfigError::NoCatchUp);
        }
        Ok(())
    }

    /// `None` when `fixed_hz` is zero, which stops the simulation.
    pub fn dt(&self) -> Option<f32> {
        (self.fixed_hz > 0).then(|| 1.0 / self.fixed_hz as f32)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimConfigError {
    NoCatchUp,
}

impl std::fmt::Display for SimConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoCatchUp => f.write_str("max_ticks_per_frame must be at least one"),
        }
    }
}

impl std::error::Error for SimConfigError {}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            fixed_hz: 60,
            max_ticks_per_frame: 4,
        }
    }
}

/// What one boundary applied.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Growth {
    pub commands: usize,
}

pub struct SessionSnapshot<A: Stores> {
    runtime: RuntimeId,
    app: A::Snapshot,
    entities: EntitiesSnapshot,
    domains: Vec<DomainSnapshot>,
    views: ViewsSnapshot,
    tick: Tick,
    config: SimConfig,
    paused: bool,
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

/// The simulation entry `Session::new` registers; `system_at` places app entries around it.
pub const DOMAIN_STEP: &str = "domain step";

type ValueSystem<A> =
    Box<dyn FnMut(Ctx<'_, A>, &mut <A as Stores>::Published) -> Result<(), DomainError> + Send>;

struct ValueEntry<A: Stores> {
    name: &'static str,
    system: ValueSystem<A>,
}

#[derive(Clone, Copy)]
enum Progress {
    Idle,
    Running(Phase),
    Faulted(PhaseError),
}

/// A CPU value with no GPU or window object; `Send`, and it builds for wasm32.
pub struct Session<A: Stores> {
    pub app: A,
    domains: Domains,
    views: Views,
    manipulation: Manipulation,
    phases: Phases<A>,
    value: Option<ValueEntry<A>>,
    commands: Commands<A>,
    batch: Vec<Request<A>>,
    results: Vec<CommandResult>,
    input: Input,
    assets: Assets,
    overlay: Overlay,
    config: SimConfig,
    paused: bool,
    tick: Tick,
    sequence: u64,
    initial: Option<SessionSnapshot<A>>,
    restored: bool,
    progress: Progress,
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
            Phase::Dispatch,
            SystemEntry::new(CAMERA_RIG, |ctx: Ctx<'_, A>| {
                camera::steer(ctx.views, ctx.input);
                Ok(())
            }),
        );
        phases.push(
            Phase::Simulation,
            SystemEntry::new(DOMAIN_STEP, |ctx: Ctx<'_, A>| {
                for domain in ctx.domains.iter_mut() {
                    domain.step(ctx.step, ctx.executor)?;
                }
                Ok(())
            }),
        );
        phases.push(
            Phase::Simulation,
            SystemEntry::new(CAMERA_TRAVEL, |ctx: Ctx<'_, A>| {
                camera::travel(ctx.views, ctx.input, ctx.step.dt);
                Ok(())
            }),
        );
        Self {
            app,
            domains: Domains::new(scene.runtime),
            views: Views::new(),
            manipulation: Manipulation::default(),
            phases,
            value: None,
            commands: Commands::new(scene),
            batch: Vec::new(),
            results: Vec::new(),
            input: Input::default(),
            assets: Assets::default(),
            overlay: Overlay::default(),
            config,
            paused: false,
            tick: Tick::default(),
            sequence: 0,
            initial: None,
            restored: false,
            progress: Progress::Idle,
        }
    }

    pub fn config(&self) -> SimConfig {
        self.config
    }

    pub fn paused(&self) -> bool {
        self.paused
    }

    /// While paused a tick moves only the free camera and leaves the tick count alone, and boundaries and publishes go on as usual.
    pub fn set_paused(&mut self, paused: bool) {
        self.paused = paused;
    }

    pub fn scene(&self) -> SceneId {
        self.commands.entities().scene()
    }

    pub fn current_tick(&self) -> Tick {
        self.tick
    }

    pub fn faulted_phase(&self) -> Option<Phase> {
        match self.progress {
            Progress::Idle => None,
            Progress::Running(phase) => Some(phase),
            Progress::Faulted(error) => Some(error.phase),
        }
    }

    pub fn phase_error(&self) -> Option<PhaseError> {
        match self.progress {
            Progress::Faulted(error) => Some(error),
            Progress::Idle | Progress::Running(_) => None,
        }
    }

    fn unfinished_error(&self) -> Option<PhaseError> {
        match self.progress {
            Progress::Idle => None,
            Progress::Running(phase) => Some(PhaseError::unnamed(
                phase,
                DomainError::Unsupported("unfinished phase requires restore"),
            )),
            Progress::Faulted(error) => Some(error),
        }
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

    pub fn set_rig(&mut self, rig: Rig) {
        self.views.root_mut().rig = rig;
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

    pub fn add_material(&mut self, material: Material) -> MaterialId {
        self.assets.add_material(material)
    }

    pub fn system(
        &mut self,
        phase: Phase,
        name: &'static str,
        system: impl FnMut(Ctx<'_, A>) -> Result<(), DomainError> + Send + 'static,
    ) {
        self.phases.push(phase, SystemEntry::new(name, system));
    }

    /// Runs `system` at its first chance, after each restore, and whenever `watch` returns a key unlike the one it returned right after the system's last `Ok` run. An error leaves the key unstored, so the system runs again at its next chance.
    pub fn system_on_change<K: PartialEq + Send + 'static>(
        &mut self,
        phase: Phase,
        name: &'static str,
        watch: impl Fn(&A) -> K + Send + 'static,
        system: impl FnMut(Ctx<'_, A>) -> Result<(), DomainError> + Send + 'static,
    ) {
        self.phases
            .push(phase, SystemEntry::on_change(name, watch, system));
    }

    /// Runs `system` at each publish after the publication systems, writing the publication's value in place; a later call replaces it.
    pub fn publish_value(
        &mut self,
        name: &'static str,
        system: impl FnMut(Ctx<'_, A>, &mut A::Published) -> Result<(), DomainError> + Send + 'static,
    ) {
        self.value = Some(ValueEntry {
            name,
            system: Box::new(system),
        });
    }

    /// `None` when no entry of that phase has the named anchor.
    pub fn system_at(
        &mut self,
        phase: Phase,
        order: Order,
        name: &'static str,
        system: impl FnMut(Ctx<'_, A>) -> Result<(), DomainError> + Send + 'static,
    ) -> Option<()> {
        self.phases
            .insert(phase, order, SystemEntry::new(name, system))
    }

    /// Adds a dispatch system after those already registered that runs `apply` as a command named after `action` at each boundary where it is pressed. The closure cannot keep state between presses, so such state belongs in the app's stores.
    pub fn on_action(
        &mut self,
        action: ActionId,
        apply: impl Fn(&mut Dispatch<'_, A>) -> Result<(), Rejection> + Send + Sync + 'static,
    ) {
        let apply = Arc::new(apply);
        self.submit_on(action, move || {
            let apply = Arc::clone(&apply);
            Command::try_app_fn(action.0, move |dispatch| {
                apply(dispatch).map(|()| Outcome::Done)
            })
        });
    }

    /// Adds a dispatch system after those already registered that submits [`Command::Reset`] at each boundary where `action` is pressed. The systems that [`Self::on_action`], [`Self::reset_on`], [`Self::pause_on`] and [`Self::toggle_on`] add take the action's name, so a system and an action should not share a name.
    pub fn reset_on(&mut self, action: ActionId) {
        self.submit_on(action, || Command::Reset);
    }

    /// Adds a dispatch system after those already registered that submits [`Command::TogglePause`] at each boundary where `action` is pressed.
    pub fn pause_on(&mut self, action: ActionId) {
        self.submit_on(action, || Command::TogglePause);
    }

    /// Adds a dispatch system after those already registered that flips `setting` as a command named after `action` at each boundary where it is pressed.
    pub fn toggle_on(&mut self, action: ActionId, setting: Setting<A>) {
        self.submit_on(action, move || Command::toggle(action.0, setting));
    }

    fn submit_on(&mut self, action: ActionId, command: impl Fn() -> Command<A> + Send + 'static) {
        self.system(Phase::Dispatch, action.0, move |ctx| {
            if ctx.input.pressed(action) {
                ctx.commands.submit(command());
            }
            Ok(())
        });
    }

    pub fn dispatch<R>(&mut self, f: impl FnOnce(&mut Dispatch<'_, A>) -> R) -> R {
        let result = {
            let mut dispatch = Dispatch::new(
                &mut self.app,
                &mut self.domains,
                &mut self.views,
                self.commands.entities_mut(),
                &mut self.assets,
            );
            f(&mut dispatch)
        };
        self.domains.synchronize();
        result
    }

    pub fn dragging(&self) -> Option<Drag> {
        self.manipulation.dragging()
    }

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

    pub fn drag(&mut self, ndc: [f32; 2], time: f64) -> Result<ChartPoint, DragError> {
        self.manipulation
            .drag(&self.domains, &self.views, &mut self.commands, ndc, time)
    }

    pub fn release(&mut self) -> Option<DragRelease> {
        self.manipulation.release(&self.domains, &mut self.commands)
    }

    pub fn release_at(&mut self, time: f64) -> Option<DragRelease> {
        self.manipulation
            .release_at(&self.domains, &mut self.commands, time)
    }

    pub fn cancel_drag(&mut self) -> Option<DragRelease> {
        self.manipulation.cancel(&mut self.commands)
    }

    pub fn boundary(&mut self, input: Input) -> Result<Growth, PhaseError> {
        self.input = input;
        if let Some(error) = self.unfinished_error() {
            return Err(error);
        }
        self.progress = Progress::Running(Phase::Dispatch);
        self.results.clear();
        let mut growth = Growth::default();
        self.commit(&mut growth);
        let step = Step {
            tick: self.tick,
            dt: self.config.dt().unwrap_or(0.0),
        };
        let mut dispatched = Ok(());
        for index in 0..self.phases.entries(Phase::Dispatch).len() {
            // A Reset whose restore failed leaves the session half restored, so no later system runs on it.
            if matches!(self.progress, Progress::Faulted(_)) {
                break;
            }
            if let Err(error) = self.run_entry(Phase::Dispatch, index, step, 1.0, &Sequential) {
                dispatched = Err(error);
                break;
            }
            self.commit(&mut growth);
        }
        self.restored = false;
        dispatched?;
        if let Progress::Faulted(error) = self.progress {
            return Err(error);
        }
        self.app.boundary(Owner::new());
        for domain in self.domains.iter_mut() {
            domain.boundary();
        }
        self.progress = Progress::Idle;
        if self.initial.is_none() {
            match self.snapshot() {
                Ok(initial) => self.initial = Some(initial),
                Err(cause) => {
                    let error =
                        PhaseError::system(Phase::Dispatch, "initial", DomainError::Restore(cause));
                    self.progress = Progress::Faulted(error);
                    return Err(error);
                }
            }
        }
        Ok(growth)
    }

    /// The result is the same for any `executor`.
    pub fn tick(&mut self, executor: &dyn Executor) -> Result<(), PhaseError> {
        if let Some(error) = self.unfinished_error() {
            return Err(error);
        }
        let Some(dt) = self.config.dt() else {
            return Ok(());
        };
        if self.paused {
            // An empty tick drops the blend, so a paused publish draws each pose where it stands.
            for domain in self.domains.iter_mut() {
                domain.begin_tick();
                domain.end_tick();
            }
            camera::travel(&mut self.views, &self.input, dt);
            return Ok(());
        }
        self.progress = Progress::Running(Phase::Simulation);
        let step = Step {
            tick: self.tick,
            dt,
        };
        for domain in self.domains.iter_mut() {
            domain.begin_tick();
        }
        let mut stepped = Ok(());
        for index in 0..self.phases.entries(Phase::Simulation).len() {
            stepped = self.run_entry(Phase::Simulation, index, step, 1.0, executor);
            if stepped.is_err() {
                break;
            }
        }
        for domain in self.domains.iter_mut() {
            domain.end_tick();
        }
        stepped?;
        self.tick = Tick(self.tick.0 + 1);
        self.progress = Progress::Idle;
        Ok(())
    }

    /// Draws each pose `alpha` of the way from where it stood before the last tick to where it stands now, with `alpha` clamped to [0, 1] and NaN taken as 1.
    pub fn publish(
        &mut self,
        into: &mut Publication<A::Published>,
        alpha: f32,
    ) -> Result<(), PhaseError> {
        if let Some(error) = self.unfinished_error() {
            return Err(error);
        }
        let alpha = presented_alpha(alpha);
        let step = Step {
            tick: self.tick,
            dt: self.config.dt().unwrap_or(0.0),
        };
        self.progress = Progress::Running(Phase::Publication);
        self.overlay.clear();
        for index in 0..self.phases.entries(Phase::Publication).len() {
            self.run_entry(Phase::Publication, index, step, alpha, &Sequential)?;
        }
        let scene = self.scene();
        into.claim(scene);
        self.run_value(&mut into.value, step, alpha)?;
        self.domains.synchronize();
        for domain in self.domains.iter_mut() {
            domain.present(alpha);
        }
        let sequence = self.sequence.wrapping_add(1);
        into.stamp = Stamp {
            tick: self.tick,
            sequence,
        };
        into.alpha = alpha;
        let reconciled = into.reconcile(
            scene,
            &self.domains,
            &self.views,
            self.assets.library(),
            &self.overlay,
        );
        if let Err(error) = reconciled {
            let error = PhaseError::unnamed(Phase::Publication, error);
            self.progress = Progress::Faulted(error);
            return Err(error);
        }
        self.sequence = sequence;
        self.progress = Progress::Idle;
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
        if let Some(phase) = self.faulted_phase() {
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
            tick: self.tick,
            config: self.config,
            paused: self.paused,
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
        self.commands.cancel_into(&mut self.results);
        self.commands.restore(&from.entities, from.next_request);
        let scene = self.scene();
        for (domain, snapshot) in self.domains.iter_mut().zip(&from.domains) {
            if let Err(error) = domain.restore(snapshot, scene) {
                self.progress = Progress::Faulted(PhaseError::system(
                    Phase::Dispatch,
                    "restore",
                    DomainError::Restore(error),
                ));
                return Err(error);
            }
        }
        self.app.restore(&from.app, scene, Owner::new());
        self.views.restore(&from.views);
        self.manipulation.clear();
        self.tick = from.tick;
        self.config = from.config;
        self.paused = from.paused;
        self.progress = Progress::Idle;
        Ok(())
    }

    /// Captures the initial snapshot when none exists and nothing is pending, and otherwise leaves it to the first boundary.
    pub fn capture_initial(&mut self) {
        if self.initial.is_none() {
            self.initial = self.snapshot().ok();
        }
    }

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
        self.restored = true;
        result
    }

    fn commit(&mut self, growth: &mut Growth) {
        let mut batch = std::mem::take(&mut self.batch);
        self.commands.drain_into(&mut batch);
        for request in batch.drain(..) {
            let name = request.command.name();
            let outcome = match request.command {
                Command::Reset if self.restored => Ok(Outcome::Done),
                Command::Reset => {
                    let progress = self.progress;
                    let outcome = self
                        .reset()
                        .map(|()| Outcome::Done)
                        .map_err(Rejection::Restore);
                    if outcome.is_ok() {
                        self.progress = progress;
                    }
                    outcome
                }
                Command::TogglePause => {
                    self.paused = !self.paused;
                    Ok(Outcome::Done)
                }
                command => self.dispatch(|dispatch| dispatch.apply(command)),
            };
            growth.commands += 1;
            self.results.push(CommandResult {
                request: request.id,
                name,
                outcome,
            });
        }
        self.batch = batch;
    }

    fn run_value(
        &mut self,
        value: &mut A::Published,
        step: Step,
        alpha: f32,
    ) -> Result<(), PhaseError> {
        let Session {
            app,
            domains,
            views,
            value: entry,
            commands,
            results,
            input,
            assets,
            overlay,
            manipulation,
            progress,
            ..
        } = self;
        let Some(entry) = entry else {
            return Ok(());
        };
        let result = (entry.system)(
            Ctx {
                app,
                domains,
                views,
                commands,
                results: results.as_slice(),
                input,
                prepared: assets.geometry(),
                step,
                alpha,
                overlay,
                executor: &Sequential,
                manipulation,
            },
            value,
        );
        domains.synchronize();
        result.map_err(|cause| {
            let error = PhaseError::system(Phase::Publication, entry.name, cause);
            *progress = Progress::Faulted(error);
            error
        })
    }

    fn run_entry(
        &mut self,
        phase: Phase,
        index: usize,
        step: Step,
        alpha: f32,
        executor: &dyn Executor,
    ) -> Result<(), PhaseError> {
        let Session {
            app,
            domains,
            views,
            phases,
            commands,
            results,
            input,
            assets,
            overlay,
            manipulation,
            progress,
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
            alpha,
            overlay,
            executor,
            manipulation,
        });
        domains.synchronize();
        result.map_err(|cause| {
            let error = PhaseError::system(phase, name, cause);
            *progress = Progress::Faulted(error);
            error
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::view::Vec4;
    use loam_math::EuclideanR4;

    use super::*;
    use crate::command::SpawnBundle;
    use crate::domain::{Instance, Pose};
    use crate::entity::Entity;
    use crate::store::LogCapacity;
    use crate::view::ViewSpec;

    crate::stores! {
        #[derive(Default)]
        pub struct Quiet {}
    }

    #[test]
    fn grab_and_plain_pick() {
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
        let root = session.views().root_id();
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

    #[test]
    fn section_pick_past_missed_body() {
        use crate::view::{Eye, Section4};
        use loam_shape::polytope::Polytope4;

        const SCALE: f32 = 0.45;

        let mut session = Session::new(Quiet::default(), SimConfig::default());
        let r4 = session
            .register_domain(DomainBuilder::new("r4", EuclideanR4).tracked(LogCapacity::default()));
        let tesseract = session.prepare(PreparedGeometry::Polytope4 {
            polytope: Polytope4::Tesseract,
            scale: SCALE,
        });
        let white = session.add_material(Material::lines([1.0; 4], 1.0));
        let root = session.views().root_id();
        let cut = session
            .dispatch(|d| -> Result<Entity, Rejection> {
                let eye = d.spawn(SpawnBundle::new().at(r4, Pose::at(Vec4::ZERO)))?;
                d.spawn(
                    SpawnBundle::new()
                        .at(r4, Pose::at(Vec4::new(0.0, 0.0, -3.0, 0.3)))
                        .instance(Instance::new(tesseract, white)),
                )?;
                let cut = d.spawn(
                    SpawnBundle::new()
                        .at(r4, Pose::at(Vec4::new(0.0, 0.0, -5.0, 0.0)))
                        .instance(Instance::new(tesseract, white)),
                )?;
                d.domains
                    .typed(r4)?
                    .add_view(ViewSpec::new(root, eye, Section4 { w: 0.0 }))?;
                Ok(cut)
            })
            .expect("the scene built");
        session.views_mut().root_mut().eye = Eye::default();

        let picked = session
            .pick([0.0, 0.0])
            .expect("the slice cuts the far tesseract");
        assert_eq!(
            picked.entity, cut,
            "the pick took the near tesseract, whose section is empty"
        );
        let face = picked.hit.expect("a section pick lifts").coordinates[2];
        assert!(
            (face - (-5.0 + 0.5 * SCALE)).abs() < 1e-4,
            "the hit is at z = {face}, off the tesseract's face"
        );
    }
}
