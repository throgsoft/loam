use std::any::Any;
use std::collections::BTreeMap;

use loam_math::{EuclideanR4, Iso4Flat, Space};
use loam_runtime::{
    Access, AppCommand, Command, Ctx, DepthEnvelope, Dispatch, DomainBuilder, DomainError,
    DomainHandle, DomainRay, Entity, Facility, Growth, ImageRay, Input, Instance, LogCapacity,
    Material, Outcome, Phase, Pose, PreparedGeometry, Publication, Publish, RecordBuffer,
    Rejection, Reservation, RestoreError, Session, SimConfig, SpawnBundle, Step, Store, Tick,
    ViewMapping, ViewSpec,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Tag(u32);

#[derive(Clone, Copy)]
struct Score(u32);

#[derive(Clone, Copy)]
struct Scored {
    entity: Entity,
    value: u32,
}

impl Publish for Score {
    type Record = Scored;

    fn record(&self, entity: Entity) -> Scored {
        Scored {
            entity,
            value: self.0,
        }
    }
}

loam_runtime::stores! {
    #[derive(Default)]
    pub struct Probe {
        tags: Store<Tag>,
        pairs: Relation<u8>,
        scores: Published<Score>,
        log: Value<Vec<u32>>,
        reserved: Value<Vec<Reservation>>,
    }
}

const SMALL: LogCapacity = LogCapacity {
    dirty: 4,
    removals: 2,
};

fn session() -> (Session<Probe>, DomainHandle<EuclideanR4>) {
    let mut session = Session::new(Probe::default(), SimConfig::default());
    let r4 = session.register_domain(DomainBuilder::new("r4", EuclideanR4));
    (session, r4)
}

fn placed(r4: DomainHandle<EuclideanR4>, tag: Tag) -> SpawnBundle<Probe> {
    SpawnBundle::new().at(r4, Pose(Iso4Flat::IDENTITY)).row(tag)
}

fn at(xyzw: [f32; 4]) -> Pose<EuclideanR4> {
    Pose(Iso4Flat::from_translation(xyzw.into()))
}

fn live(session: &Session<Probe>) -> BTreeMap<Entity, u32> {
    session
        .app
        .scores
        .iter()
        .map(|(entity, score)| (entity, score.0))
        .collect()
}

fn published(records: &RecordBuffer<Scored>) -> BTreeMap<Entity, u32> {
    records
        .rows()
        .iter()
        .map(|record| (record.entity, record.value))
        .collect()
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
    ) -> Result<(), DomainError> {
        self.steps += 1;
        for (_, pose) in poses.iter_mut() {
            pose.0.translation.x += step.dt;
            pose.0.translation.y = self.steps as f32;
        }
        Ok(())
    }

    fn snapshot(&self) -> Box<dyn Any + Send> {
        Box::new(self.steps)
    }

    fn restore(&mut self, from: &(dyn Any + Send)) -> Result<(), RestoreError> {
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
        let relative = point - eye.0.translation;
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
    session.set_initial();
    let later = session.dispatch(|d| d.spawn(placed(r4, Tag(2)))).unwrap();
    session.dispatch(|d| d.despawn(kept)).unwrap();
    let epoch = session.scene().epoch;

    session.reset().unwrap();
    assert_eq!(session.scene().epoch, epoch.advance());
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
    assert!(session
        .domains_mut()
        .typed(r4)
        .unwrap()
        .poses
        .contains(live));

    session.reset().unwrap();
    assert_eq!(session.scene().epoch, epoch.advance().advance());
    assert_eq!(session.entities().resolve(live), None);
}

#[test]
fn pending_commands_or_reservations_survive_cancellation_into_the_restored_state() {
    let (mut session, r4) = session();
    session.set_initial();
    session.system(
        Phase::Simulation,
        "queue",
        Access::new().commands(),
        move |ctx: Ctx<'_, Probe>| {
            if ctx.step.tick.0 == 0 {
                let reservation = ctx.commands.spawn(placed(r4, Tag(1))).unwrap();
                ctx.app.reserved.get_mut().push(reservation);
                ctx.commands.app(Mark(1));
            }
        },
    );
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
        d.app.pairs.link(a, b, 7).unwrap();
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
    session.app.pairs.link(b, a, 9).unwrap();
    session.restore(&snapshot).unwrap();

    assert_eq!(session.current_tick(), Tick(3));
    let [link] = session.app.pairs.links() else {
        panic!("{} links after restore", session.app.pairs.len());
    };
    let (from, to) = (link.from, link.to);
    assert_eq!((from.key(), to.key(), link.data), (a.key(), b.key(), 7));
    assert_eq!(from.scene(), session.scene());
    assert_eq!(session.app.pairs.outgoing(from).count(), 1);
    assert_eq!(session.app.pairs.incoming(to).count(), 1);
    session.tick().unwrap();
    let poses = &session.domains_mut().typed(r4).unwrap().poses;
    let position = |entity| {
        poses
            .get(entity)
            .map(|pose| (pose.0.translation.x, pose.0.translation.y))
    };
    assert_eq!(position(from), Some((dt + dt + dt + dt, 4.0)));
    assert_eq!(position(a), None);
}

#[test]
fn snapshot_is_taken_while_commands_are_pending_instead_of_being_refused() {
    let (mut session, _) = session();
    session.system(
        Phase::Simulation,
        "mark",
        Access::new().commands(),
        |ctx: Ctx<'_, Probe>| {
            if ctx.step.tick.0 == 0 {
                ctx.commands.app(Mark(1));
            }
        },
    );
    session.tick().unwrap();
    assert_eq!(session.snapshot().err(), Some(RestoreError::Pending));
    session.boundary(Input::default()).unwrap();
    assert!(session.snapshot().is_ok());
}

#[test]
fn request_queued_after_a_reset_in_the_same_batch_applies_to_the_restored_state() {
    let (mut session, _) = session();
    session.set_initial();
    session.system(
        Phase::Simulation,
        "reset",
        Access::new().commands(),
        |ctx: Ctx<'_, Probe>| {
            if ctx.step.tick.0 == 0 {
                ctx.commands.app(Mark(1));
                ctx.commands.submit(Command::Reset);
                ctx.commands.app(Mark(2));
            }
        },
    );
    session.tick().unwrap();
    session.boundary(Input::default()).unwrap();
    let outcomes: Vec<_> = session
        .results()
        .iter()
        .map(|result| result.outcome)
        .collect();
    assert_eq!(
        outcomes,
        [
            Ok(Outcome::Done),
            Ok(Outcome::Done),
            Err(Rejection::Cancelled)
        ]
    );
    assert!(session.app.log.get().is_empty());
    assert_eq!(session.current_tick(), Tick(0));
}

#[test]
fn publication_misses_a_dirty_row_or_a_removal_or_resumes_a_stale_buffer_without_a_resync() {
    let probe = Probe {
        scores: Store::tracked(SMALL),
        ..Probe::default()
    };
    let mut session = Session::new(probe, SimConfig::default());
    let spawn = |session: &mut Session<Probe>, value: u32| {
        session
            .dispatch(|d| d.spawn(SpawnBundle::new().row(Score(value))))
            .unwrap()
    };
    let e: Vec<Entity> = (0..3).map(|value| spawn(&mut session, value)).collect();
    let mut publication = Publication::default();
    session.publish(&mut publication).unwrap();
    assert_eq!(published(&publication.app.scores), live(&session));

    session.app.scores.get_mut(e[1]).unwrap().0 = 11;
    session.dispatch(|d| d.despawn(e[2])).unwrap();
    session.publish(&mut publication).unwrap();
    assert_eq!(published(&publication.app.scores), live(&session));
    assert_eq!(publication.app.scores.rows().len(), 2);

    session.app.scores.get_mut(e[0]).unwrap().0 = 10;
    for value in 20..25 {
        spawn(&mut session, value);
    }
    session.publish(&mut publication).unwrap();
    assert_eq!(published(&publication.app.scores), live(&session));
    assert_eq!(publication.app.scores.rows().len(), 7);
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
