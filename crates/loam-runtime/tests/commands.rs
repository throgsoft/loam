use std::any::Any;

use loam_math::{EuclideanR3, EuclideanR4, HyperbolicH3, Iso4Flat, IsometryGroup, Space};
use loam_runtime::{
    Access, ActionEvent, ActionId, AppCommand, ChartCommand, ChartId, ChartPose, ChartTangent,
    Command, CommandResult, Commands, Ctx, Dispatch, DomainBuilder, DomainError, DomainHandle,
    Domains, Entity, Facility, Input, Instance, Material, Order, Outcome, Phase, Pose,
    PreparedGeometry, Rejection, RequestId, Reservation, RestoreError, Session, SimConfig,
    SpawnBundle, Step, Store, StoreError, DOMAIN_STEP,
};

type Vec3 = <EuclideanR3 as Space>::Point;
type Vec4 = <EuclideanR4 as Space>::Point;

const IDENTITY_FRAME: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

const QUARTER_TURN: [[f32; 4]; 4] = [
    [0.0, 1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

fn chart(coordinates: [f32; 4], frame: [[f32; 4]; 4]) -> ChartPose {
    ChartPose {
        chart: ChartId(0),
        coordinates,
        frame,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Tag(u32);

loam_runtime::stores! {
    #[derive(Default)]
    pub struct Probe {
        tags: Store<Tag>,
        pairs: Relation<u8>,
        log: Value<Vec<u32>>,
        reserved: Value<Vec<Reservation>>,
        sample: Value<[Option<f32>; 2]>,
    }
}

fn session() -> (Session<Probe>, DomainHandle<EuclideanR4>) {
    let mut session = Session::new(Probe::default(), SimConfig::default());
    let r4 = session.register_domain(DomainBuilder::new("r4", EuclideanR4));
    (session, r4)
}

fn placed(r4: DomainHandle<EuclideanR4>, tag: Tag) -> SpawnBundle<Probe> {
    SpawnBundle::new().at(r4, Pose(Iso4Flat::IDENTITY)).row(tag)
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

struct Pair {
    from: Entity,
    to: Entity,
}

impl AppCommand<Probe> for Pair {
    fn name(&self) -> &'static str {
        "pair"
    }

    fn apply(&mut self, dispatch: &mut Dispatch<'_, Probe>) -> Result<Outcome, Rejection> {
        for entity in [self.from, self.to] {
            if dispatch.entities().resolve(entity).is_none() {
                return Err(Rejection::Stale(entity));
            }
        }
        dispatch.app.pairs.link(self.from, self.to, 0)?;
        Ok(Outcome::Done)
    }
}

struct Drift;

impl Facility<EuclideanR4> for Drift {
    fn name(&self) -> &'static str {
        "drift"
    }

    fn step(
        &mut self,
        poses: &mut Store<Pose<EuclideanR4>>,
        step: Step,
    ) -> Result<(), DomainError> {
        for (_, pose) in poses.iter_mut() {
            pose.0.translation.x += step.dt;
        }
        Ok(())
    }

    fn snapshot(&self) -> Box<dyn Any + Send> {
        Box::new(())
    }

    fn restore(&mut self, _from: &(dyn Any + Send)) -> Result<(), RestoreError> {
        Ok(())
    }
}

fn marks(values: &[u32]) -> Input {
    Input {
        actions: values
            .iter()
            .map(|&value| ActionEvent {
                action: ActionId(value),
                pressed: true,
            })
            .collect(),
        ..Input::default()
    }
}

#[test]
fn rejected_spawn_bundle_leaves_an_attachment_behind() {
    let (mut session, r4) = session();
    let doubled = move || placed(r4, Tag(1)).row(Tag(2));
    let rejection = session.dispatch(|d| d.spawn(doubled())).unwrap_err();
    assert!(matches!(
        rejection,
        Rejection::Store(StoreError::Occupied(_))
    ));
    assert!(session.app.tags.is_empty());
    assert!(session.entities().is_empty());
    assert!(session.domains_mut().typed(r4).unwrap().poses.is_empty());

    session.system(
        Phase::Simulation,
        "spawn",
        Access::new().commands(),
        move |ctx: Ctx<'_, Probe>| {
            if ctx.app.reserved.get().is_empty() {
                let reservation = ctx.commands.spawn(doubled()).unwrap();
                ctx.app.reserved.get_mut().push(reservation);
            }
        },
    );
    session.tick().unwrap();
    session.boundary(Input::default()).unwrap();
    let reservation = session.app.reserved.get()[0];
    assert!(matches!(
        session.results(),
        [CommandResult { request, outcome: Err(Rejection::Store(StoreError::Occupied(_))) }]
            if *request == reservation.request
    ));
    assert!(session.app.tags.is_empty());
    assert!(session.entities().is_empty());
    assert!(session.domains_mut().typed(r4).unwrap().poses.is_empty());
    assert_eq!(
        session.dispatch(|d| d.despawn(reservation.entity)),
        Err(Rejection::Stale(reservation.entity))
    );

    let fresh = session.dispatch(|d| d.spawn(placed(r4, Tag(3)))).unwrap();
    assert_eq!(fresh.key().slot(), reservation.entity.key().slot());
    assert_ne!(fresh, reservation.entity);
    assert_eq!(session.app.tags.get(fresh), Some(&Tag(3)));
}

#[test]
fn later_failed_command_rolls_back_an_earlier_successful_spawn() {
    let (mut session, r4) = session();
    session.system(
        Phase::Simulation,
        "spawn pair",
        Access::new().commands(),
        move |ctx: Ctx<'_, Probe>| {
            if !ctx.app.reserved.get().is_empty() {
                return;
            }
            let good = ctx.commands.spawn(placed(r4, Tag(1))).unwrap();
            let bad = ctx.commands.spawn(placed(r4, Tag(2)).row(Tag(3))).unwrap();
            ctx.commands.app(Pair {
                from: good.entity,
                to: bad.entity,
            });
            ctx.app.reserved.set(vec![good, bad]);
        },
    );
    session.tick().unwrap();
    session.boundary(Input::default()).unwrap();

    let [good, bad] = session.app.reserved.get()[..] else {
        panic!("two reservations expected");
    };
    let outcomes: Vec<_> = session.results().iter().map(|r| r.outcome).collect();
    assert_eq!(
        outcomes,
        [
            Ok(Outcome::Spawned(good.entity)),
            Err(Rejection::Store(StoreError::Occupied(bad.entity))),
            Err(Rejection::Stale(bad.entity)),
        ]
    );
    assert_eq!(session.entities().len(), 1);
    assert_eq!(
        session.entities().resolve(good.entity),
        Some(good.entity.key())
    );
    assert_eq!(session.app.tags.get(good.entity), Some(&Tag(1)));
    assert!(session
        .domains_mut()
        .typed(r4)
        .unwrap()
        .poses
        .contains(good.entity));
    assert!(session.app.pairs.is_empty());
}

#[test]
fn commands_reorder_repeat_or_vanish_around_pause_and_catch_up() {
    let (mut session, _) = session();
    session.system(
        Phase::Dispatch,
        "marks",
        Access::new().commands(),
        |input: &Input, commands: &mut Commands<Probe>| {
            for event in &input.actions {
                commands.app(Mark(event.action.0));
            }
        },
    );
    session.system(
        Phase::Simulation,
        "tick mark",
        Access::new().commands(),
        |ctx: Ctx<'_, Probe>| {
            ctx.commands.app(Mark(100 + ctx.step.tick.0 as u32));
        },
    );

    let schedule: [(&[u32], u32); 6] = [
        (&[1, 2], 2),
        (&[3], 0),
        (&[], 0),
        (&[4], 3),
        (&[5], 1),
        (&[], 0),
    ];
    let mut expected = Vec::new();
    let mut deferred = Vec::new();
    let mut delivered: Vec<RequestId> = Vec::new();
    let mut tick = 0;
    for (actions, ticks) in schedule {
        let growth = session.boundary(marks(actions)).unwrap();
        expected.append(&mut deferred);
        expected.extend_from_slice(actions);
        assert_eq!(growth.commands, session.results().len());
        assert!(session
            .results()
            .iter()
            .all(|result| result.outcome == Ok(Outcome::Done)));
        delivered.extend(session.results().iter().map(|result| result.request));
        for _ in 0..ticks {
            session.tick().unwrap();
            deferred.push(100 + tick);
            tick += 1;
        }
    }
    assert_eq!(expected, [1, 2, 100, 101, 3, 4, 102, 103, 104, 5, 105]);
    assert_eq!(*session.app.log.get(), expected);
    assert_eq!(delivered.len(), expected.len());
    assert!(delivered.windows(2).all(|pair| pair[0] < pair[1]));
}

#[test]
fn dispatch_spawn_returns_a_handle_that_is_not_live() {
    let (mut session, r4) = session();
    let direct = session.dispatch(|d| d.spawn(placed(r4, Tag(1)))).unwrap();
    assert_eq!(session.entities().resolve(direct), Some(direct.key()));
    assert_eq!(session.app.tags.get(direct), Some(&Tag(1)));

    session.system(
        Phase::Dispatch,
        "spawn",
        Access::new().commands(),
        move |ctx: Ctx<'_, Probe>| {
            let reservation = ctx.commands.spawn(placed(r4, Tag(2))).unwrap();
            ctx.app.reserved.get_mut().push(reservation);
        },
    );
    session.system(
        Phase::Dispatch,
        "observe",
        Access::new().reads::<Tag>(),
        |ctx: Ctx<'_, Probe>| {
            let reservation = ctx.app.reserved.get()[0];
            let live = ctx.app.tags.get(reservation.entity) == Some(&Tag(2));
            let reported = ctx.results.iter().any(|result| {
                result.request == reservation.request
                    && result.outcome == Ok(Outcome::Spawned(reservation.entity))
            });
            ctx.app.log.get_mut().push(u32::from(live));
            ctx.app.log.get_mut().push(u32::from(reported));
        },
    );
    session.boundary(Input::default()).unwrap();
    let reservation = session.app.reserved.get()[0];
    assert_eq!(*session.app.log.get(), [1, 1]);
    assert_eq!(
        session.entities().resolve(reservation.entity),
        Some(reservation.entity.key())
    );
    assert_eq!(session.results().len(), 1);
}

#[test]
fn reserved_handle_resolves_before_its_spawn_commits() {
    let (mut session, r4) = session();
    session.system(
        Phase::Simulation,
        "reserve",
        Access::new().reads::<Tag>().commands(),
        move |ctx: Ctx<'_, Probe>| {
            if ctx.step.tick.0 == 0 {
                let reservation = ctx.commands.spawn(placed(r4, Tag(1))).unwrap();
                ctx.app.reserved.get_mut().push(reservation);
            }
            let reservation = ctx.app.reserved.get()[0];
            let seen = ctx.app.tags.contains(reservation.entity);
            ctx.app.log.get_mut().push(u32::from(seen));
        },
    );
    session.tick().unwrap();
    let reservation = session.app.reserved.get()[0];
    assert_eq!(session.entities().resolve(reservation.entity), None);
    assert!(!session.app.tags.contains(reservation.entity));
    assert_eq!(
        session.dispatch(|d| d.despawn(reservation.entity)),
        Err(Rejection::Reserved(reservation.entity))
    );
    session.tick().unwrap();
    assert_eq!(*session.app.log.get(), [0, 0]);

    session.boundary(Input::default()).unwrap();
    assert_eq!(
        session.results(),
        [CommandResult {
            request: reservation.request,
            outcome: Ok(Outcome::Spawned(reservation.entity)),
        }]
    );
    assert_eq!(
        session.entities().resolve(reservation.entity),
        Some(reservation.entity.key())
    );
    assert_eq!(session.app.tags.get(reservation.entity), Some(&Tag(1)));
    session.tick().unwrap();
    assert_eq!(*session.app.log.get(), [0, 0, 1]);

    assert_eq!(
        session.dispatch(|d| d.apply(Command::Despawn(reservation.entity))),
        Ok(Outcome::Done)
    );
    assert_eq!(session.entities().resolve(reservation.entity), None);
    assert!(!session.app.tags.contains(reservation.entity));
}

#[test]
fn system_after_the_domain_step_does_not_see_the_steps_writes() {
    let mut session = Session::new(Probe::default(), SimConfig::default());
    let r4 = session.register_domain(DomainBuilder::new("r4", EuclideanR4).facility(Drift));
    let walker = session
        .dispatch(|d| d.spawn(SpawnBundle::new().at(r4, Pose(Iso4Flat::IDENTITY))))
        .unwrap();
    let sample = move |slot: usize| {
        move |app: &mut Probe, domains: &mut Domains| {
            let pose = domains.typed(r4).unwrap().poses.get(walker).unwrap();
            app.sample.get_mut()[slot] = Some(pose.0.translation.x);
        }
    };
    let access = || Access::new().domain(r4.id());
    session
        .system_at(
            Phase::Simulation,
            Order::After(DOMAIN_STEP),
            "after",
            access(),
            sample(1),
        )
        .unwrap();
    session
        .system_at(
            Phase::Simulation,
            Order::Before(DOMAIN_STEP),
            "before",
            access(),
            sample(0),
        )
        .unwrap();
    let dt = session.config().dt().unwrap();

    session.tick().unwrap();
    assert_eq!(*session.app.sample.get(), [Some(0.0), Some(dt)]);
    session.tick().unwrap();
    assert_eq!(*session.app.sample.get(), [Some(dt), Some(dt + dt)]);
}

#[test]
fn a_chart_spawn_lands_at_the_pose_and_instance_the_chart_names() {
    let mut session = Session::new(Probe::default(), SimConfig::default());
    let r4 = session.register_domain(DomainBuilder::new("r4", EuclideanR4));
    let r3 = session.register_domain(DomainBuilder::new("r3", EuclideanR3));
    let h3 = session.register_domain(DomainBuilder::new("h3", HyperbolicH3));
    let stub = session.prepare(PreparedGeometry::Lines4 {
        segments: vec![[[0.05, 0.0, 0.0, 0.0], [-0.05, 0.0, 0.0, 0.0]]],
    });
    let material = session.add_material(Material::flat([1.0; 4]));
    let instance = Instance::new(stub, material);

    let (flat4, flat3, curved) = session.dispatch(|d| {
        (
            d.spawn(
                SpawnBundle::new()
                    .at_chart(r4.id(), chart([2.0, 3.0, 5.0, 7.0], QUARTER_TURN))
                    .instance(instance),
            )
            .unwrap(),
            d.spawn(
                SpawnBundle::new().at_chart(r3.id(), chart([1.0, 2.0, 3.0, 0.0], QUARTER_TURN)),
            )
            .unwrap(),
            d.spawn(
                SpawnBundle::new().at_chart(h3.id(), chart([0.5, 0.0, 0.0, 0.0], IDENTITY_FRAME)),
            )
            .unwrap(),
        )
    });

    let domain4 = session.domains_mut().typed(r4).unwrap();
    let pose4 = domain4.poses.get(flat4).unwrap().0;
    let landed4 = EuclideanR4.iso_apply(pose4, Vec4::X);
    assert!(
        (landed4 - Vec4::new(2.0, 4.0, 5.0, 7.0)).length() < 1e-5,
        "the quarter turn put local x at {landed4}"
    );
    assert_eq!(domain4.instances.get(flat4), Some(&instance));

    let pose3 = session
        .domains_mut()
        .typed(r3)
        .unwrap()
        .poses
        .get(flat3)
        .unwrap()
        .0;
    let landed3 = EuclideanR3.iso_apply(pose3, Vec3::X);
    assert!(
        (landed3 - Vec3::new(1.0, 3.0, 3.0)).length() < 1e-5,
        "the quarter turn put local x at {landed3}"
    );

    let curved_pose = session
        .domains_mut()
        .typed(h3)
        .unwrap()
        .poses
        .get(curved)
        .unwrap()
        .0;
    let landed = HyperbolicH3.iso_apply(curved_pose, Vec3::ZERO);
    assert!(
        (landed - Vec3::new(0.5, 0.0, 0.0)).length() < 1e-5,
        "the transvection put the origin at {landed}"
    );
}

#[test]
fn a_non_orthonormal_chart_frame_is_refused_and_places_nothing() {
    let (mut session, r4) = session();
    let skewed = [
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let refusal = session
        .dispatch(|d| d.spawn(SpawnBundle::new().at_chart(r4.id(), chart([0.0; 4], skewed))))
        .unwrap_err();
    assert_eq!(refusal, Rejection::Domain(DomainError::InvalidFrame));
    assert_eq!(session.domains_mut().typed(r4).unwrap().poses.len(), 0);
}

#[test]
fn a_chart_walk_advances_the_pose_by_the_tangent_it_names() {
    let (mut session, r4) = session();
    let walker = session
        .dispatch(|d| {
            d.spawn(SpawnBundle::new().at_chart(r4.id(), chart([0.0; 4], IDENTITY_FRAME)))
        })
        .unwrap();
    session
        .dispatch(|d| {
            d.apply(Command::Chart(
                r4.id(),
                ChartCommand::Walk {
                    entity: walker,
                    tangent: ChartTangent {
                        chart: ChartId(0),
                        vector: [1.0, 0.0, 0.0, 2.0],
                    },
                    dt: 0.5,
                },
            ))
        })
        .unwrap();

    let walked = session
        .domains_mut()
        .typed(r4)
        .unwrap()
        .poses
        .get(walker)
        .unwrap()
        .0
        .translation;
    assert!(
        (walked - Vec4::new(0.5, 0.0, 0.0, 1.0)).length() < 1e-5,
        "half a second along (1, 0, 0, 2) ended at {walked}"
    );
}
