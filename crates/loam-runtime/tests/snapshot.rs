use std::any::Any;

use loam_math::{EuclideanR4, Space};
use loam_runtime::{
    AppCommand, Command, Ctx, DepthEnvelope, Dispatch, Domain, DomainBuilder, DomainError,
    DomainHandle, DomainRay, Entity, Eye, Facility, Field, FieldKind, FieldOp, Growth, ImageRay,
    Input, Instance, LogCapacity, Material, Outcome, Owner, Phase, PhaseError, Pose,
    PreparedGeometry, Projection4, Publication, PublishError, Records, Rejection, Reservation,
    RestoreError, SchemaId, Section4, Session, SimConfig, SpawnBundle, Step, Store, Tick,
    ViewMapping, ViewSpec, DOMAIN_STEP,
};

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
    }
}

fn session() -> (Session<Probe>, DomainHandle<EuclideanR4>) {
    let mut session = Session::new(Probe::default(), SimConfig::default());
    let r4 = session.register_domain(DomainBuilder::new("r4", EuclideanR4));
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
fn reset_rewinds_external_identity_or_resolves_an_old_handle() {
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
}

#[test]
fn pending_commands_or_reservations_survive_cancellation_into_the_restored_state() {
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
    session.tick().unwrap();
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
    assert_eq!(
        session.boundary(Input::default()).unwrap(),
        Growth::default()
    );
    assert!(session.app.log.get().is_empty());
    assert!(session.app.tags.is_empty());
}

#[test]
fn restored_relations_domain_poses_or_the_tick_differ_from_the_snapshot() {
    let mut session = Session::new(Probe::default(), SimConfig::default());
    let r4 =
        session.register_domain(DomainBuilder::new("r4", EuclideanR4).facility(Drift { steps: 0 }));
    let (a, b) = session.dispatch(|d| {
        let a = d.spawn(placed(r4, Tag(1))).unwrap();
        let b = d.spawn(placed(r4, Tag(2))).unwrap();
        d.link(a, b, 7).unwrap();
        (a, b)
    });
    let dt = session.config().dt().unwrap();
    for _ in 0..3 {
        session.tick().unwrap();
    }
    let snapshot = session.snapshot().unwrap();

    for _ in 0..2 {
        session.tick().unwrap();
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
    session.tick().unwrap();
    let poses = session.domains().read(r4).unwrap().poses();
    let position = |entity| poses.get(entity).map(|pose| (pose.point.x, pose.point.y));
    assert_eq!(position(from), Some((dt + dt + dt + dt, 4.0)));
    assert_eq!(position(a), None);
}

#[test]
fn reset_restores_the_captured_view_configuration_and_rebases_its_entities() {
    let (mut session, r4) = session();
    let root = session.views().root();
    let root_eye = Eye::looking_at([2.0, 3.0, 4.0], [0.0; 3], [0.0, 1.0, 0.0]);
    session.views_mut().root_mut().eye = root_eye;
    let (eye, view) = session.dispatch(|d| {
        let eye = d.spawn(SpawnBundle::new().at(r4, at([0.0; 4]))).unwrap();
        let domain = d.domains.typed(r4).unwrap();
        let view = domain.add_view(ViewSpec::new(root, eye, DropW).subject(eye));
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
        domain.set_view_eye(view, later_eye).unwrap();
        domain.set_view_subject(view, None).unwrap();
        let spec = domain.view_mut(view).unwrap();
        spec.enabled = true;
        spec.edges = true;
        spec.section_edges = true;
        spec.section_faces = true;
        spec.set_mapping(Section4 { w: 1.0 });
        domain.add_view(ViewSpec::new(root, later_eye, Section4 { w: 2.0 }))
    });
    session.views_mut().root_mut().eye = Eye::default();

    session.reset().unwrap();
    assert_eq!(session.views().get(root).unwrap().eye, root_eye);
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
    session.publish(&mut publication).unwrap();
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
        _owner: Owner,
    ) -> Result<(), DomainError> {
        Err(DomainError::ChartBoundary)
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
fn a_restored_composed_field_names_its_own_operands_as_stale() {
    let mut session = Session::new(Probe::default(), SimConfig::default());
    let r4 = session.register_domain(DomainBuilder::new("r4", EuclideanR4).fields());
    session.dispatch(|d| {
        let ball = |x| SpawnBundle::new().at(r4, at([x, 0.0, 0.0, 0.0]));
        let left = d.spawn(ball(-1.0)).unwrap();
        let right = d.spawn(ball(1.0)).unwrap();
        let union = d.spawn(ball(0.0)).unwrap();
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
    let compile = |session: &mut Session<Probe>| {
        session
            .domains_mut()
            .typed(r4)
            .unwrap()
            .compile_fields()
            .map(|_| ())
    };
    assert_eq!(compile(&mut session), Ok(()));

    let snapshot = session.snapshot().unwrap();
    session.restore(&snapshot).unwrap();
    assert_eq!(compile(&mut session), Ok(()));
}

#[test]
fn restore_does_not_launder_foreign_or_old_epoch_references_into_live_entities() {
    let mut owner = Session::new(Probe::default(), SimConfig::default());
    let r4 = owner.register_domain(DomainBuilder::new("r4", EuclideanR4).fields());
    let old = owner
        .dispatch(|dispatch| dispatch.spawn(SpawnBundle::new().at(r4, at([0.0; 4]))))
        .unwrap();
    let first = owner.snapshot().unwrap();
    owner.restore(&first).unwrap();
    let current = owner
        .domains()
        .read(r4)
        .unwrap()
        .poses()
        .iter()
        .next()
        .unwrap()
        .0;

    let mut other = Session::new(Probe::default(), SimConfig::default());
    let other_r4 = other.register_domain(DomainBuilder::new("r4", EuclideanR4));
    let foreign = other
        .dispatch(|dispatch| dispatch.spawn(SpawnBundle::new().at(other_r4, at([0.0; 4]))))
        .unwrap();
    assert_eq!(foreign.key(), current.key());

    let root = owner.views().root();
    owner.dispatch(|dispatch| {
        let operator = dispatch
            .spawn(SpawnBundle::new().at(r4, at([1.0, 0.0, 0.0, 0.0])))
            .unwrap();
        dispatch
            .attach_field(
                r4,
                current,
                Field {
                    kind: FieldKind::ExactDistance,
                    op: FieldOp::HyperSphere { radius: 1.0 },
                    operands: Vec::new(),
                },
            )
            .unwrap();
        dispatch
            .attach_field(
                r4,
                operator,
                Field {
                    kind: FieldKind::ExactDistance,
                    op: FieldOp::Union,
                    operands: vec![old, old],
                },
            )
            .unwrap();
        dispatch
            .domains
            .typed(r4)
            .unwrap()
            .add_view(ViewSpec::new(root, foreign, DropW));
    });
    let snapshot = owner.snapshot().unwrap();
    owner.restore(&snapshot).unwrap();

    assert_eq!(
        owner
            .domains_mut()
            .typed(r4)
            .unwrap()
            .compile_fields()
            .map(|_| ()),
        Err(DomainError::Stale(old))
    );
    let mut publication = Publication::default();
    assert_eq!(
        owner.publish(&mut publication),
        Err(PhaseError {
            phase: Phase::Publication,
            system: None,
            cause: DomainError::Stale(foreign),
        })
    );
}

#[test]
fn snapshot_is_taken_while_commands_are_pending_instead_of_being_refused() {
    let (mut session, _) = session();
    session.system(Phase::Simulation, "mark", |ctx: Ctx<'_, Probe>| {
        if ctx.step.tick.0 == 0 {
            ctx.commands.app(Mark(1));
        }
        Ok(())
    });
    session.tick().unwrap();
    assert_eq!(session.snapshot().err(), Some(RestoreError::Pending));
    session.boundary(Input::default()).unwrap();
    assert!(session.snapshot().is_ok());
}

#[test]
fn a_foreign_snapshot_cannot_cancel_pending_work_or_alias_library_ids() {
    let (mut owner, owner_r4) = session();
    let geometry = owner.prepare(PreparedGeometry::Lines4 {
        segments: Vec::new(),
    });
    let material_value = Material::flat([0.1, 0.2, 0.3, 1.0]);
    let material = owner.add_material(material_value);
    let entity = owner
        .dispatch(|d| d.spawn(placed(owner_r4, Tag(1)).instance(Instance::new(geometry, material))))
        .unwrap();
    owner.app.log.get_mut().push(1);
    let owned = owner.snapshot().unwrap();
    let scene = owner.scene();

    let (mut foreign, foreign_r4) = session();
    let foreign_geometry = foreign.prepare(PreparedGeometry::Lines3 {
        segments: Vec::new(),
    });
    let foreign_material = foreign.add_material(Material::flat([0.9, 0.8, 0.7, 1.0]));
    foreign
        .dispatch(|d| {
            d.spawn(
                placed(foreign_r4, Tag(9))
                    .instance(Instance::new(foreign_geometry, foreign_material)),
            )
        })
        .unwrap();
    foreign.app.log.get_mut().push(9);
    let foreign = foreign.snapshot().unwrap();

    owner.submit(Command::App(Box::new(Mark(2))));
    assert_eq!(owner.restore(&foreign), Err(RestoreError::ForeignRuntime));
    assert_eq!(owner.scene(), scene);
    assert_eq!(*owner.app.log.get(), [1]);
    assert_eq!(owner.app.tags.get(entity), Some(&Tag(1)));
    assert!(matches!(
        owner.prepared(geometry),
        Some(PreparedGeometry::Lines4 { segments }) if segments.is_empty()
    ));
    assert_eq!(owner.material(material), Some(&material_value));
    assert_eq!(owner.snapshot().err(), Some(RestoreError::Pending));

    owner.boundary(Input::default()).unwrap();
    assert_eq!(*owner.app.log.get(), [1, 2]);
    owner.restore(&owned).unwrap();
    assert_eq!(owner.scene().runtime(), scene.runtime());
    assert_eq!(owner.scene().epoch(), scene.epoch().advance());
    assert_eq!(*owner.app.log.get(), [1]);
    assert_eq!(owner.material(material), Some(&material_value));
}

#[test]
fn a_failed_cpu_phase_cannot_advance_or_publish_until_recovery() {
    let mut session = Session::new(Probe::default(), SimConfig::default());
    session.register_domain(DomainBuilder::new("r4", EuclideanR4).facility(RejectStep));
    let captured = session.snapshot().unwrap();
    session.set_initial().unwrap();

    let failure = PhaseError {
        phase: Phase::Simulation,
        system: Some(DOMAIN_STEP),
        cause: DomainError::ChartBoundary,
    };
    assert_eq!(session.tick(), Err(failure));
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
    assert_eq!(session.publish(&mut publication), Err(failure));
    assert_eq!(publication.stamp, Default::default());

    session.reset().unwrap();
    assert!(session.snapshot().is_ok());
    assert_eq!(session.tick(), Err(failure));
    session.restore(&captured).unwrap();
    assert!(session.snapshot().is_ok());
}

#[test]
fn failed_extraction_clears_partial_publication() {
    let (mut session, r4) = session();
    let geometry = session.prepare(PreparedGeometry::Lines4 {
        segments: Vec::new(),
    });
    let material = session.add_material(Material::flat([1.0; 4]));
    let root = session.views().root();
    let eye = session.dispatch(|d| {
        let eye = d.spawn(SpawnBundle::new().at(r4, at([0.0; 4]))).unwrap();
        d.spawn(
            SpawnBundle::new()
                .at(r4, at([1.0, 2.0, 3.0, 4.0]))
                .instance(Instance::new(geometry, material)),
        )
        .unwrap();
        d.domains
            .typed(r4)
            .unwrap()
            .add_view(ViewSpec::new(root, eye, DropW));
        eye
    });
    let mut publication = Publication::default();
    session.publish(&mut publication).unwrap();
    let complete = publication.stamp;
    assert_eq!(
        publication.views[0].records.instances.rows()[0].image_point,
        [1.0, 2.0, 3.0]
    );
    let snapshot = session.snapshot().unwrap();

    session.dispatch(|d| d.despawn(eye)).unwrap();
    let failure = PhaseError {
        phase: Phase::Publication,
        system: None,
        cause: DomainError::Stale(eye),
    };
    assert_eq!(session.publish(&mut publication), Err(failure));
    assert_eq!(publication.stamp, Default::default());
    assert!(publication.views.is_empty());
    assert_eq!(session.publish(&mut publication), Err(failure));

    session.restore(&snapshot).unwrap();
    session.publish(&mut publication).unwrap();
    assert_eq!(publication.stamp.sequence, complete.sequence + 1);
    assert_eq!(
        publication.views[0].records.instances.rows()[0].image_point,
        [1.0, 2.0, 3.0]
    );
}

#[test]
fn request_queued_after_a_reset_in_the_same_batch_applies_to_the_restored_state() {
    let (mut session, _) = session();
    session.set_initial().unwrap();
    session.system(Phase::Simulation, "reset", |ctx: Ctx<'_, Probe>| {
        if ctx.step.tick.0 == 0 {
            ctx.commands.app(Mark(1));
            ctx.commands.submit(Command::Reset);
            ctx.commands.app(Mark(2));
        }
        Ok(())
    });
    session.tick().unwrap();
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
fn domain_view_publishes_the_wrong_image_space_position_for_a_known_pose() {
    let mut session = Session::new(Probe::default(), SimConfig::default());
    let r4 = session
        .register_domain(DomainBuilder::new("r4", EuclideanR4).tracked(LogCapacity::default()));
    let geometry = session.prepare(PreparedGeometry::Lines4 {
        segments: Vec::new(),
    });
    let material = session.add_material(Material::flat([1.0; 4]));
    let root = session.views().root();
    let (a, b) = session.dispatch(|d| {
        let eye = d
            .spawn(SpawnBundle::new().at(r4, at([0.5, 0.0, 0.0, 9.0])))
            .unwrap();
        d.domains
            .typed(r4)
            .unwrap()
            .add_view(ViewSpec::new(root, eye, DropW));
        let instance = Instance::new(geometry, material);
        let a = d
            .spawn(
                SpawnBundle::new()
                    .at(r4, at([1.0, 2.0, 3.0, 4.0]))
                    .instance(instance),
            )
            .unwrap();
        let b = d
            .spawn(
                SpawnBundle::new()
                    .at(r4, at([-1.0, 0.0, 1.0, 2.0]))
                    .instance(instance),
            )
            .unwrap();
        (a, b)
    });

    let mut publication = Publication::default();
    session.publish(&mut publication).unwrap();
    let [view] = &publication.views[..] else {
        panic!("{} views published", publication.views.len());
    };
    assert_eq!(view.domain, r4.id());
    let mut records: Vec<(Entity, [f32; 3], [f32; 4])> = view
        .records
        .instances
        .rows()
        .iter()
        .map(|record| (record.entity, record.image_point, record.pose.coordinates))
        .collect();
    records.sort_by_key(|record| record.0);
    assert_eq!(
        records,
        [
            (a, [0.5, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]),
            (b, [-1.5, 0.0, 1.0], [-1.0, 0.0, 1.0, 2.0]),
        ]
    );

    let first = view.records.instances.stamp();
    session.publish(&mut publication).unwrap();
    let second = publication.views[0].records.instances.stamp();
    assert_eq!((first.tick, second.tick), (Tick(0), Tick(0)));
    assert!(second.sequence > first.sequence);
}

#[test]
fn cancelled_request_and_a_fresh_request_after_a_reset_share_an_id() {
    let (mut session, _) = session();
    session.set_initial().unwrap();
    session.system(Phase::Simulation, "deferred", |ctx: Ctx<'_, Probe>| {
        if ctx.step.tick.0 == 0 {
            ctx.commands.app(Mark(1));
        }
        Ok(())
    });
    session.system(Phase::Dispatch, "fresh", |ctx: Ctx<'_, Probe>| {
        ctx.commands.app(Mark(2));
        Ok(())
    });
    session.tick().unwrap();
    session.reset().unwrap();
    let [cancelled] = session.results() else {
        panic!("{} results after reset", session.results().len());
    };
    assert_eq!(cancelled.outcome, Err(Rejection::Cancelled));
    let cancelled = cancelled.request;

    session.boundary(Input::default()).unwrap();
    let [fresh] = session.results() else {
        panic!("{} results after the boundary", session.results().len());
    };
    assert_eq!(fresh.outcome, Ok(Outcome::Done));
    assert!(fresh.request > cancelled);
}

#[test]
fn publish_rebuilds_a_view_whose_rows_did_not_change_or_misses_one_that_did() {
    let mut session = Session::new(Probe::default(), SimConfig::default());
    let r4 = session
        .register_domain(DomainBuilder::new("r4", EuclideanR4).tracked(LogCapacity::default()));
    let geometry = session.prepare(PreparedGeometry::Lines4 {
        segments: Vec::new(),
    });
    let material = session.add_material(Material::flat([1.0; 4]));
    let root = session.views().root();
    let drawn = session.dispatch(|d| {
        let eye = d.spawn(SpawnBundle::new().at(r4, at([0.0; 4]))).unwrap();
        d.domains
            .typed(r4)
            .unwrap()
            .add_view(ViewSpec::new(root, eye, DropW));
        d.spawn(
            SpawnBundle::new()
                .at(r4, at([1.0, 2.0, 3.0, 4.0]))
                .instance(Instance::new(geometry, material)),
        )
        .unwrap()
    });

    let mut publication = Publication::default();
    session.publish(&mut publication).unwrap();
    let first = publication.views[0].records.built();
    session.publish(&mut publication).unwrap();
    assert_eq!(publication.views[0].records.built(), first);

    session
        .domains_mut()
        .typed(r4)
        .unwrap()
        .set_pose(drawn, at([5.0, 6.0, 7.0, 8.0]))
        .unwrap();
    session.publish(&mut publication).unwrap();
    let records = &publication.views[0].records;
    assert!(records.built().sequence > first.sequence);
    assert_eq!(records.instances.rows()[0].image_point, [5.0, 6.0, 7.0]);
}

#[test]
fn published_segments_miss_the_endpoints_the_mapping_sends_them_to() {
    let mut session = Session::new(Probe::default(), SimConfig::default());
    let r4 = session
        .register_domain(DomainBuilder::new("r4", EuclideanR4).tracked(LogCapacity::default()));
    let geometry = session.prepare(PreparedGeometry::Lines4 {
        segments: vec![[[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 1.0]]],
    });
    let material = session.add_material(Material::lines([0.2, 0.4, 0.6, 1.0], 3.0));
    let root = session.views().root();
    session.dispatch(|d| {
        let eye = d
            .spawn(SpawnBundle::new().at(r4, at([0.0, 0.0, 0.0, 2.0])))
            .unwrap();
        d.domains
            .typed(r4)
            .unwrap()
            .add_view(ViewSpec::new(root, eye, Projection4 { focal: 2.0 }));
        d.spawn(
            SpawnBundle::new()
                .at(r4, at([0.0, 0.0, -3.0, 0.0]))
                .instance(Instance::new(geometry, material)),
        )
        .unwrap();
    });

    let mut publication = Publication::default();
    session.publish(&mut publication).unwrap();
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

#[test]
fn publish_overwrites_a_record_buffer_the_renderer_still_holds() {
    let (mut session, _) = session();
    let mut records = Records::<Probe>::default();
    let first = records.publish(&mut session).unwrap();
    let held = records.lend().unwrap();
    assert_eq!(held.stamp, first);
    assert_eq!(records.publish(&mut session), Err(PublishError::Borrowed));
    records.release(held);
    let second = records.publish(&mut session).unwrap();
    assert!(second.sequence > first.sequence);
}

#[test]
fn a_view_change_alone_republishes_the_old_segments() {
    let mut session = Session::new(Probe::default(), SimConfig::default());
    let r4 = session
        .register_domain(DomainBuilder::new("r4", EuclideanR4).tracked(LogCapacity::default()));
    let geometry = session.prepare(PreparedGeometry::Lines4 {
        segments: vec![[[0.0; 4], [1.0, 0.0, 0.0, 0.0]]],
    });
    let material = session.add_material(Material::lines([1.0; 4], 1.0));
    let root = session.views().root();
    let (view, far) = session.dispatch(|d| {
        let near = d.spawn(SpawnBundle::new().at(r4, at([0.0; 4]))).unwrap();
        let far = d
            .spawn(SpawnBundle::new().at(r4, at([0.0, 4.0, 0.0, 0.0])))
            .unwrap();
        d.spawn(
            SpawnBundle::new()
                .at(r4, at([0.0; 4]))
                .instance(Instance::new(geometry, material)),
        )
        .unwrap();
        let view = d
            .domains
            .typed(r4)
            .unwrap()
            .add_view(ViewSpec::new(root, near, DropW));
        (view, far)
    });

    let mut publication = Publication::default();
    session.publish(&mut publication).unwrap();
    let [first] = publication.views[0].records.segments() else {
        panic!("one segment expected");
    };
    assert_eq!(first.start, [0.0; 3]);

    session
        .domains_mut()
        .typed(r4)
        .unwrap()
        .set_view_eye(view, far)
        .unwrap();
    session.publish(&mut publication).unwrap();
    let [second] = publication.views[0].records.segments() else {
        panic!("one segment expected");
    };
    assert_eq!(second.start, [0.0, -4.0, 0.0]);
}

#[test]
fn an_idle_view_rebuilds_when_its_cursor_expires_at_a_boundary() {
    let mut session = Session::new(Probe::default(), SimConfig::default());
    let r4 = session
        .register_domain(DomainBuilder::new("r4", EuclideanR4).tracked(LogCapacity::default()));
    let geometry = session.prepare(PreparedGeometry::Lines4 {
        segments: Vec::new(),
    });
    let material = session.add_material(Material::flat([1.0; 4]));
    let root = session.views().root();
    session.dispatch(|d| {
        let eye = d.spawn(SpawnBundle::new().at(r4, at([0.0; 4]))).unwrap();
        d.domains
            .typed(r4)
            .unwrap()
            .add_view(ViewSpec::new(root, eye, DropW));
        d.spawn(
            SpawnBundle::new()
                .at(r4, at([1.0, 2.0, 3.0, 4.0]))
                .instance(Instance::new(geometry, material)),
        )
        .unwrap();
    });

    let mut publication = Publication::default();
    session.publish(&mut publication).unwrap();
    let built = publication.views[0].records.built();
    for step in 0..24 {
        session.boundary(Input::default()).unwrap();
        session.publish(&mut publication).unwrap();
        assert_eq!(
            publication.views[0].records.built(),
            built,
            "boundary {step} rebuilt an idle view"
        );
    }
}

struct FailsRestoreOnce {
    failed: bool,
}

impl Facility<EuclideanR4> for FailsRestoreOnce {
    fn name(&self) -> &'static str {
        "fails restore once"
    }

    fn step(
        &mut self,
        _poses: &mut Store<Pose<EuclideanR4>>,
        _step: Step,
        _owner: Owner,
    ) -> Result<(), DomainError> {
        Ok(())
    }

    fn snapshot(&self, _owner: Owner) -> Box<dyn Any + Send> {
        Box::new(())
    }

    fn check_restore(&self, from: &(dyn Any + Send), _owner: Owner) -> Result<(), RestoreError> {
        from.downcast_ref::<()>()
            .map(|_| ())
            .ok_or(RestoreError::Schema(SchemaId::of::<()>()))
    }

    fn restore(&mut self, _from: &(dyn Any + Send), _owner: Owner) -> Result<(), RestoreError> {
        if self.failed {
            return Ok(());
        }
        self.failed = true;
        Err(RestoreError::Schema(SchemaId::of::<FailsRestoreOnce>()))
    }
}

#[test]
fn a_restore_that_fails_after_validation_blocks_the_session_until_one_succeeds() {
    let mut session = Session::new(Probe::default(), SimConfig::default());
    session.register_domain(
        DomainBuilder::new("r4", EuclideanR4).facility(FailsRestoreOnce { failed: false }),
    );
    session.set_initial().unwrap();
    session.tick().unwrap();

    assert!(session.reset().is_err());
    let fault = session
        .phase_error()
        .expect("a failed restore records a fault");
    assert_eq!(fault.phase, Phase::Dispatch);
    assert_eq!(fault.system, Some("restore"));
    assert!(session.tick().is_err());
    let mut publication = Publication::default();
    assert!(session.publish(&mut publication).is_err());

    session.reset().unwrap();
    assert!(session.phase_error().is_none());
    session.tick().unwrap();
}
