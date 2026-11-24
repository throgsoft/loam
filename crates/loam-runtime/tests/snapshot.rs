use std::any::Any;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use loam_math::{EuclideanR4, Space};
use loam_runtime::{
    ActionEvent, ActionId, AppCommand, Backdrop, Command, Ctx, DepthEnvelope, Dispatch,
    DomainBuilder, DomainError, DomainHandle, DomainRay, Entity, Eye, Facility, Ground, Growth,
    ImageRay, Input, Instance, LogCapacity, Material, Outcome, Owner, Phase, PhaseError, Pose,
    PreparedGeometry, PreparedId, Projection4, Publication, Rejection, Reservation, RestoreError,
    SchemaId, Section4, Session, SimConfig, Sky, SpawnBundle, Step, Store, Tick, ViewMapping,
    ViewSpec, ViewTarget, DOMAIN_STEP,
};
use loam_time::par::{Executor, Sequential};

type Vec4 = <EuclideanR4 as Space>::Point;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Tag(u32);

loam_runtime::stores! {
    #[derive(Default)]
    pub struct Probe {
        tags: Store<Tag>,
        pairs: Relation<u8>,
        log: Value<Vec<u32>>,
        reserved: Value<Vec<Reservation>>,
        prepared: Value<Option<PreparedId>>,
    }
}

fn session() -> (Session<Probe>, DomainHandle<EuclideanR4>) {
    let mut session = Session::new(Probe::default(), SimConfig::default());
    let r4 = session.register_domain(DomainBuilder::new("r4", EuclideanR4).untracked());
    (session, r4)
}

fn placed(r4: DomainHandle<EuclideanR4>, tag: Tag) -> SpawnBundle<Probe> {
    SpawnBundle::new().at(r4, Pose::at(Vec4::ZERO)).row(tag)
}

fn at(xyzw: [f32; 4]) -> Pose<EuclideanR4> {
    Pose::at(xyzw.into())
}

struct Mark(u32);

impl AppCommand<Probe> for Mark {
    fn name(&self) -> &'static str {
        "mark"
    }

    fn apply(&mut self, dispatch: &mut Dispatch<'_, Probe>) -> Result<Outcome, Rejection> {
        dispatch.app.log.get_mut().push(self.0);
        Ok(Outcome::Done)
    }
}

struct Drift {
    steps: u32,
}

impl Facility<EuclideanR4> for Drift {
    fn name(&self) -> &'static str {
        "drift"
    }

    fn step(
        &mut self,
        poses: &mut Store<Pose<EuclideanR4>>,
        step: Step,
        _executor: &dyn Executor,
        _owner: Owner,
    ) -> Result<(), DomainError> {
        self.steps += 1;
        for (_, pose) in poses.iter_mut() {
            pose.point.x += step.dt;
            pose.point.y = self.steps as f32;
        }
        Ok(())
    }

    fn snapshot(&self, _owner: Owner) -> Box<dyn Any + Send> {
        Box::new(self.steps)
    }

    fn check_restore(&self, from: &(dyn Any + Send), _owner: Owner) -> Result<(), RestoreError> {
        from.downcast_ref::<u32>()
            .map(|_| ())
            .ok_or(RestoreError::Schema(SchemaId::of::<u32>()))
    }

    fn restore(&mut self, from: &(dyn Any + Send), owner: Owner) -> Result<(), RestoreError> {
        self.check_restore(from, owner)?;
        self.steps = *from.downcast_ref::<u32>().unwrap();
        Ok(())
    }
}

struct DropW;

impl ViewMapping<EuclideanR4> for DropW {
    fn name(&self) -> &'static str {
        "drop w"
    }

    fn image_point(
        &self,
        eye: &Pose<EuclideanR4>,
        point: <EuclideanR4 as Space>::Point,
    ) -> Option<[f32; 3]> {
        let relative = point - eye.point;
        Some([relative.x, relative.y, relative.z])
    }

    fn lift(&self, _eye: &Pose<EuclideanR4>, _ray: &ImageRay) -> Option<DomainRay<EuclideanR4>> {
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
fn reset_with_pending_requests() {
    let (mut session, r4) = session();
    session.set_initial().unwrap();
    session.system(Phase::Simulation, "queue", move |ctx: Ctx<'_, Probe>| {
        if ctx.step.tick.0 == 0 {
            let reservation = ctx.commands.spawn(placed(r4, Tag(1))).unwrap();
            ctx.app.reserved.get_mut().push(reservation);
            ctx.commands.app(Mark(1));
        }
        Ok(())
    });
    session.tick(&Sequential).unwrap();
    let reservation = session.app.reserved.get()[0];
    assert!(session.entities().is_reserved(reservation.entity));

    session.reset().unwrap();
    assert_eq!(session.results().len(), 2);
    assert!(session
        .results()
        .iter()
        .all(|result| result.outcome == Err(Rejection::Cancelled)));
    assert_eq!(session.results()[0].request, reservation.request);
    assert!(!session.entities().is_reserved(reservation.entity));
    assert!(session.entities().is_empty());
    let cancelled = session.results()[1].request;

    let fresh = session.submit(Command::App(Box::new(Mark(2))));
    assert!(fresh > cancelled, "a cancelled request id was reused");
    assert_eq!(
        session.boundary(Input::default()).unwrap(),
        Growth { commands: 1 }
    );
    assert_eq!(*session.app.log.get(), [2]);
    assert!(session.app.tags.is_empty());
}

#[test]
fn reset_and_restore() {
    let (mut session, r4) = session();
    let kept = session.dispatch(|d| d.spawn(placed(r4, Tag(1)))).unwrap();
    session.set_initial().unwrap();
    let later = session.dispatch(|d| d.spawn(placed(r4, Tag(2)))).unwrap();
    session.dispatch(|d| d.despawn(kept)).unwrap();
    let epoch = session.scene().epoch();

    session.reset().unwrap();
    assert_eq!(session.scene().epoch(), epoch.advance());
    assert_eq!(session.entities().resolve(kept), None);
    assert_eq!(session.entities().resolve(later), None);
    assert_eq!(session.app.tags.get(kept), None);
    let rows: Vec<(Entity, Tag)> = session
        .app
        .tags
        .iter()
        .map(|(entity, &tag)| (entity, tag))
        .collect();
    let [(live, tag)] = rows[..] else {
        panic!("{} rows after reset", rows.len());
    };
    assert_eq!(
        (live.scene(), live.key(), tag),
        (session.scene(), kept.key(), Tag(1))
    );
    assert_eq!(session.entities().resolve(live), Some(kept.key()));
    assert!(session.domains().read(r4).unwrap().poses().contains(live));

    session.reset().unwrap();
    assert_eq!(session.scene().epoch(), epoch.advance().advance());
    assert_eq!(session.entities().resolve(live), None);

    let mut session = Session::new(Probe::default(), SimConfig::default());
    let r4 = session.register_domain(
        DomainBuilder::new("r4", EuclideanR4)
            .untracked()
            .facility(Drift { steps: 0 }),
    );
    let (a, b) = session.dispatch(|d| {
        let a = d.spawn(placed(r4, Tag(1))).unwrap();
        let b = d.spawn(placed(r4, Tag(2))).unwrap();
        d.link(a, b, 7).unwrap();
        (a, b)
    });
    let dt = session.config().dt().unwrap();
    for _ in 0..3 {
        session.tick(&Sequential).unwrap();
    }
    let snapshot = session.snapshot().unwrap();

    for _ in 0..2 {
        session.tick(&Sequential).unwrap();
    }
    let ab = session.app.pairs.outgoing(a).next().unwrap();
    session.app.pairs.unlink(ab).unwrap();
    session.dispatch(|d| d.link(b, a, 9)).unwrap();
    session.restore(&snapshot).unwrap();

    assert_eq!(session.current_tick(), Tick(3));
    let [link] = session.app.pairs.links() else {
        panic!("{} links after restore", session.app.pairs.len());
    };
    let (from, to) = (link.from(), link.to());
    assert_eq!((from.key(), to.key(), link.data), (a.key(), b.key(), 7));
    assert_eq!(from.scene(), session.scene());
    assert_eq!(session.app.pairs.outgoing(from).count(), 1);
    assert_eq!(session.app.pairs.incoming(to).count(), 1);
    session.tick(&Sequential).unwrap();
    let poses = session.domains().read(r4).unwrap().poses();
    let position = |entity| poses.get(entity).map(|pose| (pose.point.x, pose.point.y));
    assert_eq!(position(from), Some((dt + dt + dt + dt, 4.0)));
    assert_eq!(position(a), None);
}

#[test]
fn reset_views() {
    let (mut session, r4) = session();
    let root = session.views().root_id();
    let root_eye = Eye::looking_at([2.0, 3.0, 4.0], [0.0; 3], [0.0, 1.0, 0.0]);
    session.views_mut().root_mut().eye = root_eye;
    let (eye, view) = session.dispatch(|d| {
        let eye = d.spawn(SpawnBundle::new().at(r4, at([0.0; 4]))).unwrap();
        let domain = d.domains.typed(r4).unwrap();
        let view = domain
            .add_view(ViewSpec::new(root, eye, DropW).subject(eye))
            .unwrap();
        let spec = domain.view_mut(view).unwrap();
        spec.enabled = false;
        spec.edges = false;
        spec.section_edges = false;
        spec.section_faces = false;
        (eye, view)
    });
    session.set_initial().unwrap();

    let later_view = session.dispatch(|d| {
        let later_eye = d
            .spawn(SpawnBundle::new().at(r4, at([5.0, 0.0, 0.0, 0.0])))
            .unwrap();
        let domain = d.domains.typed(r4).unwrap();
        domain.set_view_subject(view, None).unwrap();
        let spec = domain.view_mut(view).unwrap();
        spec.enabled = true;
        spec.edges = true;
        spec.section_edges = true;
        spec.section_faces = true;
        spec.set_mapping(Section4 { w: 1.0 });
        domain
            .add_view(ViewSpec::new(root, later_eye, Section4 { w: 2.0 }))
            .unwrap()
    });
    session.views_mut().root_mut().eye = Eye::default();

    session.reset().unwrap();
    assert_eq!(session.views().root().eye, root_eye);
    let scene = session.scene();
    let domain = session.domains().read(r4).unwrap();
    let restored = domain.view(view).unwrap();
    assert_eq!(restored.mapping().name(), "drop w");
    assert_eq!(domain.view_eye(view).map(Entity::key), Some(eye.key()));
    assert_eq!(domain.view_eye(view).map(Entity::scene), Some(scene));
    assert!(!restored.enabled);
    assert!(!restored.edges);
    assert!(!restored.section_edges);
    assert!(!restored.section_faces);
    assert_eq!(domain.view_subject(view).map(Entity::key), Some(eye.key()));
    assert_eq!(domain.view_subject(view).map(Entity::scene), Some(scene));
    assert!(domain.view(later_view).is_none());

    let mut publication = Publication::default();
    session.publish(&mut publication, 1.0).unwrap();
    assert_eq!(publication.views.len(), 1);
    assert!(publication.views[0].records.instances.rows().is_empty());
}

struct RejectStep;

impl Facility<EuclideanR4> for RejectStep {
    fn name(&self) -> &'static str {
        "reject step"
    }

    fn step(
        &mut self,
        _poses: &mut Store<Pose<EuclideanR4>>,
        _step: Step,
        _executor: &dyn Executor,
        _owner: Owner,
    ) -> Result<(), DomainError> {
        Err(DomainError::Unsupported("injected"))
    }

    fn snapshot(&self, _owner: Owner) -> Box<dyn Any + Send> {
        Box::new(())
    }

    fn check_restore(&self, from: &(dyn Any + Send), _owner: Owner) -> Result<(), RestoreError> {
        from.downcast_ref::<()>()
            .map(|_| ())
            .ok_or(RestoreError::Schema(SchemaId::of::<()>()))
    }

    fn restore(&mut self, from: &(dyn Any + Send), owner: Owner) -> Result<(), RestoreError> {
        self.check_restore(from, owner)
    }
}

#[test]
fn failed_cpu_phase() {
    let mut session = Session::new(Probe::default(), SimConfig::default());
    session.register_domain(
        DomainBuilder::new("r4", EuclideanR4)
            .untracked()
            .facility(RejectStep),
    );
    let captured = session.snapshot().unwrap();
    session.set_initial().unwrap();

    let failure = PhaseError {
        phase: Phase::Simulation,
        system: Some(DOMAIN_STEP),
        cause: DomainError::Unsupported("injected"),
    };
    assert_eq!(session.tick(&Sequential), Err(failure));
    assert_eq!(session.current_tick(), Tick(0));
    assert_eq!(
        session.snapshot().err(),
        Some(RestoreError::Unfinished(Phase::Simulation))
    );
    let blocked_input = Input {
        scroll: [1.0, -2.0],
        ..Input::default()
    };
    assert_eq!(session.boundary(blocked_input), Err(failure));
    assert_eq!(session.take_input().scroll, [1.0, -2.0]);
    assert_eq!(
        session.snapshot().err(),
        Some(RestoreError::Unfinished(Phase::Simulation))
    );
    let mut publication = Publication::default();
    assert_eq!(session.publish(&mut publication, 1.0), Err(failure));
    assert_eq!(publication.stamp, Default::default());

    session.reset().unwrap();
    assert!(session.snapshot().is_ok());
    assert_eq!(session.tick(&Sequential), Err(failure));
    session.restore(&captured).unwrap();
    assert!(session.snapshot().is_ok());
}

struct RejectRestore {
    armed: Arc<AtomicBool>,
}

impl Facility<EuclideanR4> for RejectRestore {
    fn name(&self) -> &'static str {
        "reject restore"
    }

    fn step(
        &mut self,
        _poses: &mut Store<Pose<EuclideanR4>>,
        _step: Step,
        _executor: &dyn Executor,
        _owner: Owner,
    ) -> Result<(), DomainError> {
        Ok(())
    }

    fn snapshot(&self, _owner: Owner) -> Box<dyn Any + Send> {
        Box::new(())
    }

    fn check_restore(&self, _from: &(dyn Any + Send), _owner: Owner) -> Result<(), RestoreError> {
        Ok(())
    }

    fn restore(&mut self, _from: &(dyn Any + Send), _owner: Owner) -> Result<(), RestoreError> {
        if self.armed.load(Ordering::Relaxed) {
            return Err(RestoreError::Schema(SchemaId::of::<()>()));
        }
        Ok(())
    }
}

#[test]
fn failed_restore_in_a_reset_command() {
    let armed = Arc::new(AtomicBool::new(false));
    let mut session = Session::new(Probe::default(), SimConfig::default());
    session.register_domain(DomainBuilder::new("r4", EuclideanR4).untracked().facility(
        RejectRestore {
            armed: armed.clone(),
        },
    ));
    let captured = session.snapshot().unwrap();
    session.set_initial().unwrap();

    armed.store(true, Ordering::Relaxed);
    session.submit(Command::Reset);
    let failure = session
        .boundary(Input::default())
        .expect_err("a boundary whose reset stopped half way ended clean");
    assert_eq!(failure.phase, Phase::Dispatch);
    assert_eq!(session.phase_error(), Some(failure));
    assert_eq!(session.faulted_phase(), Some(failure.phase));
    assert_eq!(session.tick(&Sequential), Err(failure));

    armed.store(false, Ordering::Relaxed);
    session.restore(&captured).unwrap();
    assert_eq!(
        (session.faulted_phase(), session.phase_error()),
        (None, None)
    );
    session.tick(&Sequential).unwrap();
}

#[test]
fn pause_through_reset_and_restore() {
    let (mut session, _) = session();
    session.set_initial().unwrap();
    session.tick(&Sequential).unwrap();
    session.submit(Command::TogglePause);
    session.boundary(Input::default()).unwrap();
    let paused = session.snapshot().unwrap();
    session.tick(&Sequential).unwrap();
    assert_eq!(session.current_tick(), Tick(1));

    session.submit(Command::Reset);
    session.boundary(Input::default()).unwrap();
    session.tick(&Sequential).unwrap();
    assert_eq!(
        session.current_tick(),
        Tick(1),
        "a reset kept a pause its initial state never had"
    );

    session.restore(&paused).unwrap();
    session.tick(&Sequential).unwrap();
    assert_eq!(
        session.current_tick(),
        Tick(1),
        "a restore dropped the pause its snapshot was taken with"
    );
}

#[test]
fn unchanged_view_and_pose_edit() {
    let mut session = Session::new(Probe::default(), SimConfig::default());
    let r4 = session
        .register_domain(DomainBuilder::new("r4", EuclideanR4).tracked(LogCapacity::default()));
    let geometry = session.prepare(PreparedGeometry::Lines4 {
        segments: Vec::new(),
    });
    let material = session.add_material(Material::flat([1.0; 4]));
    let root = session.views().root_id();
    let (view, drawn) = session.dispatch(|d| {
        let eye = d.spawn(SpawnBundle::new().at(r4, at([0.0; 4]))).unwrap();
        let view = d
            .domains
            .typed(r4)
            .unwrap()
            .add_view(ViewSpec::new(root, eye, DropW))
            .unwrap();
        let drawn = d
            .spawn(
                SpawnBundle::new()
                    .at(r4, at([1.0, 2.0, 3.0, 4.0]))
                    .instance(Instance::new(geometry, material)),
            )
            .unwrap();
        (view, drawn)
    });

    let mut publication = Publication::default();
    session.publish(&mut publication, 1.0).unwrap();
    let first = publication.views[0].records.built();
    session.publish(&mut publication, 1.0).unwrap();
    assert_eq!(publication.views[0].records.built(), first);

    for step in 0..24 {
        session.boundary(Input::default()).unwrap();
        session.publish(&mut publication, 1.0).unwrap();
        assert_eq!(
            publication.views[0].records.built(),
            first,
            "boundary {step} rebuilt an idle view"
        );
    }

    session
        .domains_mut()
        .typed(r4)
        .unwrap()
        .view_mut(view)
        .expect("the view exists");
    session.publish(&mut publication, 1.0).unwrap();
    assert_eq!(
        publication.views[0].records.built(),
        first,
        "an unchanged view borrow rebuilt the records"
    );

    for alpha in [0.2, 0.9] {
        session.tick(&Sequential).unwrap();
        session.publish(&mut publication, alpha).unwrap();
        assert_eq!(
            publication.views[0].records.built(),
            first,
            "a still world was rebuilt between ticks"
        );
    }

    session
        .domains_mut()
        .typed(r4)
        .unwrap()
        .set_pose(drawn, at([5.0, 6.0, 7.0, 8.0]))
        .unwrap();
    session.publish(&mut publication, 1.0).unwrap();
    let records = &publication.views[0].records;
    assert!(records.built().sequence > first.sequence);
    assert_eq!(records.instances.rows()[0].image_point, [5.0, 6.0, 7.0]);
}

#[test]
fn mapped_segment_endpoints() {
    let mut session = Session::new(Probe::default(), SimConfig::default());
    let r4 = session
        .register_domain(DomainBuilder::new("r4", EuclideanR4).tracked(LogCapacity::default()));
    let geometry = session.prepare(PreparedGeometry::Lines4 {
        segments: vec![[[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 1.0]]],
    });
    let material = session.add_material(Material::lines([0.2, 0.4, 0.6, 1.0], 3.0));
    let root = session.views().root_id();
    session.dispatch(|d| {
        let eye = d
            .spawn(SpawnBundle::new().at(r4, at([0.0, 0.0, 0.0, 2.0])))
            .unwrap();
        d.domains
            .typed(r4)
            .unwrap()
            .add_view(ViewSpec::new(root, eye, Projection4 { focal: 2.0 }))
            .unwrap();
        d.spawn(
            SpawnBundle::new()
                .at(r4, at([0.0, 0.0, -3.0, 0.0]))
                .instance(Instance::new(geometry, material)),
        )
        .unwrap();
    });

    let mut publication = Publication::default();
    session.publish(&mut publication, 1.0).unwrap();
    let [segment] = publication.views[0].records.segments() else {
        panic!(
            "{} segments published",
            publication.views[0].records.segments().len()
        );
    };
    let close = |got: [f32; 3], want: [f32; 3]| {
        assert!(
            got.iter().zip(want).all(|(g, w)| (g - w).abs() <= 1e-6),
            "{got:?} is not {want:?}"
        );
    };
    close(segment.start, [0.5, 0.0, -1.5]);
    close(segment.end, [0.0, 4.0 / 3.0, -2.0]);
    assert_eq!(segment.start_color, [0.2, 0.4, 0.6, 1.0]);
    assert_eq!(segment.width_px, 3.0);
}

struct PrepareSegment;

impl AppCommand<Probe> for PrepareSegment {
    fn name(&self) -> &'static str {
        "prepare"
    }

    fn apply(&mut self, dispatch: &mut Dispatch<'_, Probe>) -> Result<Outcome, Rejection> {
        let id = dispatch.prepare(PreparedGeometry::Lines4 {
            segments: vec![[[0.0; 4], [1.0, 0.0, 0.0, 0.0]]],
        });
        dispatch.app.prepared.set(Some(id));
        Ok(Outcome::Done)
    }
}

#[test]
fn prepared_geometry_across_restore() {
    let (mut session, r4) = session();
    let before = session.snapshot().unwrap();
    session.submit(Command::App(Box::new(PrepareSegment)));
    session.boundary(Input::default()).unwrap();
    let id = session
        .app
        .prepared
        .get()
        .expect("the command prepared geometry");
    assert!(session.prepared(id).is_some());

    session.restore(&before).unwrap();
    assert_eq!(*session.app.prepared.get(), None);
    assert!(
        session.prepared(id).is_some(),
        "a restore invalidated a library id"
    );

    let material = session.add_material(Material::flat([1.0; 4]));
    let root = session.views().root_id();
    session.dispatch(|d| {
        let eye = d.spawn(SpawnBundle::new().at(r4, at([0.0; 4]))).unwrap();
        d.spawn(
            SpawnBundle::new()
                .at(r4, at([0.0, 0.0, -2.0, 0.0]))
                .instance(Instance::new(id, material)),
        )
        .unwrap();
        d.domains
            .typed(r4)
            .unwrap()
            .add_view(ViewSpec::new(root, eye, Projection4 { focal: 2.0 }))
            .unwrap();
    });
    let mut publication = Publication::default();
    session.publish(&mut publication, 1.0).unwrap();
    assert_eq!(publication.views[0].records.instances.rows().len(), 1);
}

#[test]
fn reset_in_a_batch() {
    let (mut session, r4) = session();
    session.dispatch(|d| d.spawn(placed(r4, Tag(7)))).unwrap();
    assert_eq!(session.reset(), Err(RestoreError::NoInitial));

    session.boundary(Input::default()).unwrap();
    session.submit(Command::App(Box::new(Mark(1))));
    session.boundary(Input::default()).unwrap();
    assert_eq!(*session.app.log.get(), [1]);
    let before = session.scene().epoch();

    session.reset().unwrap();
    let restored = session.scene().epoch();
    assert_eq!(restored, before.advance());
    assert!(session.app.log.get().is_empty());
    assert!(
        session.app.tags.iter().any(|(_, tag)| *tag == Tag(7)),
        "the state at the first boundary is the initial state"
    );

    session.submit(Command::Reset);
    session.boundary(Input::default()).unwrap();
    assert_eq!(
        session.scene().epoch(),
        restored,
        "a reset right after a restore restored again"
    );
    assert_eq!(session.results()[0].outcome, Ok(Outcome::Done));

    session.submit(Command::Reset);
    session.boundary(Input::default()).unwrap();
    assert_eq!(
        session.scene().epoch(),
        restored.advance(),
        "a reset in the next boundary was swallowed"
    );

    session.submit(Command::App(Box::new(Mark(1))));
    session.boundary(Input::default()).unwrap();
    let saved = session.snapshot().unwrap();
    session.submit(Command::App(Box::new(Mark(2))));
    session.boundary(Input::default()).unwrap();
    assert_eq!(*session.app.log.get(), [1, 2]);

    session.restore(&saved).unwrap();
    assert_eq!(*session.app.log.get(), [1]);
    let restored = session.scene().epoch();
    session.submit(Command::Reset);
    session.boundary(Input::default()).unwrap();
    assert_eq!(
        session.scene().epoch(),
        restored.advance(),
        "the reset after a restore to a saved snapshot was swallowed"
    );
    assert!(session.app.log.get().is_empty());

    let (mut session, _) = crate::session();
    session.set_initial().unwrap();
    session.system(Phase::Simulation, "reset", |ctx: Ctx<'_, Probe>| {
        if ctx.step.tick.0 == 0 {
            ctx.commands.app(Mark(1));
            ctx.commands.submit(Command::Reset);
            ctx.commands.app(Mark(2));
        }
        Ok(())
    });
    session.tick(&Sequential).unwrap();
    session.boundary(Input::default()).unwrap();
    let outcomes: Vec<_> = session
        .results()
        .iter()
        .map(|result| result.outcome)
        .collect();
    assert_eq!(
        outcomes,
        [Ok(Outcome::Done), Ok(Outcome::Done), Ok(Outcome::Done)]
    );
    assert_eq!(*session.app.log.get(), [2]);
    assert_eq!(session.current_tick(), Tick(0));
}

#[test]
fn reset_beside_another_handler() {
    const MARK: ActionId = ActionId("mark");
    const RESET: ActionId = ActionId("reset");
    let press = |actions: &[ActionId]| Input {
        actions: actions
            .iter()
            .map(|&action| ActionEvent {
                action,
                pressed: true,
            })
            .collect(),
        ..Input::default()
    };
    let (mut session, r4) = session();
    session.dispatch(|d| d.spawn(placed(r4, Tag(7)))).unwrap();
    session.on_action(MARK, |d| {
        d.app.log.get_mut().push(1);
        Ok(())
    });
    session.reset_on(RESET);
    session.set_initial().unwrap();

    session.boundary(press(&[MARK])).unwrap();
    session.tick(&Sequential).unwrap();
    assert_eq!(*session.app.log.get(), [1]);
    let epoch = session.scene().epoch();

    session.boundary(press(&[RESET, MARK])).unwrap();
    assert_eq!(session.scene().epoch(), epoch.advance());
    assert!(
        session.app.log.get().is_empty(),
        "the other handler's write outlived the reset"
    );
    assert_eq!(session.current_tick(), Tick(0));
    assert!(session.app.tags.iter().any(|(_, tag)| *tag == Tag(7)));
    assert!(session
        .results()
        .iter()
        .any(|result| result.name == "reset" && result.outcome == Ok(Outcome::Done)));
}

#[test]
fn dispatched_view() {
    let mut session = Session::new(Probe::default(), SimConfig::default());
    let r4 = session.register_domain(DomainBuilder::new("r4", EuclideanR4));
    let geometry = session.prepare(PreparedGeometry::Lines4 {
        segments: Vec::new(),
    });
    let material = session.add_material(Material::flat([1.0; 4]));
    let (view, drawn) = session.dispatch(|d| {
        let view = d.add_view(r4, at([1.0, 0.0, -2.0, 3.0]), DropW).unwrap();
        let drawn = d
            .spawn(
                SpawnBundle::new()
                    .at(r4, at([5.0, 6.0, 7.0, 8.0]))
                    .instance(Instance::new(geometry, material)),
            )
            .unwrap();
        (view, drawn)
    });

    let mut publication = Publication::default();
    session.publish(&mut publication, 1.0).unwrap();
    let [published] = &publication.views[..] else {
        panic!("{} views published", publication.views.len());
    };
    let image = session.views().root_id();
    assert_eq!(published.target, ViewTarget { view, image });
    let [row] = published.records.instances.rows() else {
        panic!("the eye or nothing was drawn");
    };
    assert_eq!((row.entity, row.image_point), (drawn, [4.0, 6.0, 9.0]));
}

#[test]
fn root_backdrop() {
    let (mut session, _) = session();
    let dusk = Backdrop {
        sky: Sky {
            below: [0.3, 0.1, 0.1],
            above: [0.1, 0.1, 0.4],
        },
        ground: Ground {
            y: -1.0,
            dark: [0.1; 3],
            light: [0.6; 3],
            fog_per_unit: 0.01,
            fog_start: 2.0,
            visible: true,
        },
    };
    session.views_mut().root_mut().backdrop = Some(dusk);
    let saved = session.snapshot().unwrap();
    let mut publication = Publication::default();

    session.publish(&mut publication, 1.0).unwrap();
    assert_eq!(publication.backdrop, Some(dusk));
    let drawn = publication.content();
    session.publish(&mut publication, 1.0).unwrap();
    assert_eq!(
        publication.content(),
        drawn,
        "an unchanged backdrop asked for a redraw"
    );

    let hidden = Backdrop {
        ground: Ground {
            visible: false,
            ..dusk.ground
        },
        ..dusk
    };
    session.views_mut().root_mut().backdrop = Some(hidden);
    session.publish(&mut publication, 1.0).unwrap();
    assert_eq!(publication.backdrop, Some(hidden));
    assert_ne!(
        publication.content(),
        drawn,
        "the hidden floor did not ask for a redraw"
    );

    session.restore(&saved).unwrap();
    session.publish(&mut publication, 1.0).unwrap();
    assert_eq!(publication.backdrop, Some(dusk));
}
