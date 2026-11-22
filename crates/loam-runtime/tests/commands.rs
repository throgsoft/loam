use std::any::Any;

use loam_math::{EuclideanR4, Space};
use loam_runtime::{
    ActionEvent, ActionId, AppCommand, Command, CommandResult, Ctx, Dispatch, DomainBuilder,
    DomainError, DomainHandle, Entity, Facility, Input, Instance, Material, Order, Outcome, Owner,
    Phase, Pose, PreparedGeometry, Rejection, RequestId, Reservation, RestoreError, SchemaId,
    Session, SimConfig, SpawnBundle, Step, Store, StoreError, DOMAIN_STEP,
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
        sample: Value<[Option<f32>; 2]>,
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
        dispatch.link(self.from, self.to, 0)?;
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
        _owner: Owner,
    ) -> Result<(), DomainError> {
        for (_, pose) in poses.iter_mut() {
            pose.point.x += step.dt;
        }
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

    fn restore(&mut self, from: &(dyn Any + Send), owner: Owner) -> Result<(), RestoreError> {
        self.check_restore(from, owner)
    }
}

const MARKS: [ActionId; 6] = [
    ActionId("0"),
    ActionId("1"),
    ActionId("2"),
    ActionId("3"),
    ActionId("4"),
    ActionId("5"),
];

fn marks(values: &[u32]) -> Input {
    Input {
        actions: values
            .iter()
            .map(|&value| ActionEvent {
                action: MARKS[value as usize],
                pressed: true,
            })
            .collect(),
        ..Input::default()
    }
}

#[test]
fn stale_attachments_do_not_claim_recycled_slots() {
    let mut session = Session::new(Probe::default(), SimConfig::default());
    let r4 = session.register_domain(DomainBuilder::new("r4", EuclideanR4).untracked());
    let geometry = session.prepare(PreparedGeometry::Lines4 {
        segments: Vec::new(),
    });
    let material = session.add_material(Material::flat([1.0; 4]));
    let (doomed, peer) = session.dispatch(|dispatch| {
        let doomed = dispatch.spawn(placed(r4, Tag(1))).unwrap();
        let peer = dispatch.spawn(placed(r4, Tag(2))).unwrap();
        (doomed, peer)
    });
    session
        .dispatch(|dispatch| dispatch.despawn(doomed))
        .unwrap();

    let instance = Instance::new(geometry, material);
    session.dispatch(|dispatch| {
        assert_eq!(
            dispatch.attach(doomed, Tag(5)),
            Err(Rejection::Domain(DomainError::Stale(doomed)))
        );
        assert_eq!(
            dispatch.link(doomed, peer, 1),
            Err(Rejection::Domain(DomainError::Stale(doomed)))
        );
        assert_eq!(
            dispatch.attach_instance(r4, doomed, instance),
            Err(Rejection::Domain(DomainError::Stale(doomed)))
        );

        let reused = dispatch.spawn(placed(r4, Tag(6))).unwrap();
        assert_eq!(reused.key().slot(), doomed.key().slot());
        dispatch.link(reused, peer, 2).unwrap();
        dispatch.attach_instance(r4, reused, instance).unwrap();
    });
}

#[test]
fn later_failed_command_rolls_back_an_earlier_successful_spawn() {
    let (mut session, r4) = session();
    session.system(
        Phase::Simulation,
        "spawn pair",
        move |ctx: Ctx<'_, Probe>| {
            if !ctx.app.reserved.get().is_empty() {
                return Ok(());
            }
            let good = ctx.commands.spawn(placed(r4, Tag(1))).unwrap();
            let bad = ctx.commands.spawn(placed(r4, Tag(2)).row(Tag(3))).unwrap();
            ctx.commands.app(Pair {
                from: good.entity,
                to: bad.entity,
            });
            ctx.app.reserved.set(vec![good, bad]);
            Ok(())
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
            Err(Rejection::Domain(DomainError::Stale(bad.entity))),
        ]
    );
    assert_eq!(session.entities().len(), 1);
    assert_eq!(
        session.entities().resolve(good.entity),
        Some(good.entity.key())
    );
    assert_eq!(session.app.tags.get(good.entity), Some(&Tag(1)));
    assert!(session
        .domains()
        .read(r4)
        .unwrap()
        .poses()
        .contains(good.entity));
    assert!(session.app.pairs.is_empty());
}

#[test]
fn commands_reorder_repeat_or_vanish_around_pause_and_catch_up() {
    let (mut session, _) = session();
    session.system(Phase::Dispatch, "marks", |ctx: Ctx<'_, Probe>| {
        for event in &ctx.input.actions {
            if let Some(value) = MARKS.iter().position(|&mark| mark == event.action) {
                ctx.commands.app(Mark(value as u32));
            }
        }
        Ok(())
    });
    session.system(Phase::Simulation, "tick mark", |ctx: Ctx<'_, Probe>| {
        ctx.commands.app(Mark(100 + ctx.step.tick.0 as u32));
        Ok(())
    });

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
fn a_spawn_is_live_at_once_through_dispatch_and_a_reserved_one_only_after_its_boundary() {
    let (mut session, r4) = session();
    let direct = session.dispatch(|d| d.spawn(placed(r4, Tag(1)))).unwrap();
    assert_eq!(session.entities().resolve(direct), Some(direct.key()));
    assert_eq!(session.app.tags.get(direct), Some(&Tag(1)));

    session.system(Phase::Dispatch, "spawn", move |ctx: Ctx<'_, Probe>| {
        let reservation = ctx.commands.spawn(placed(r4, Tag(2))).unwrap();
        ctx.app.reserved.get_mut().push(reservation);
        Ok(())
    });
    session.system(Phase::Dispatch, "observe", |ctx: Ctx<'_, Probe>| {
        let reservation = ctx.app.reserved.get()[0];
        let live = ctx.app.tags.get(reservation.entity) == Some(&Tag(2));
        let reported = ctx.results.iter().any(|result| {
            result.request == reservation.request
                && result.outcome == Ok(Outcome::Spawned(reservation.entity))
        });
        ctx.app.log.get_mut().push(u32::from(live));
        ctx.app.log.get_mut().push(u32::from(reported));
        Ok(())
    });
    session.boundary(Input::default()).unwrap();
    let reservation = session.app.reserved.get()[0];
    assert_eq!(*session.app.log.get(), [1, 1]);
    assert_eq!(
        session.entities().resolve(reservation.entity),
        Some(reservation.entity.key())
    );
    assert_eq!(session.results().len(), 1);

    let (mut session, r4) = crate::session();
    session.system(Phase::Simulation, "reserve", move |ctx: Ctx<'_, Probe>| {
        if ctx.step.tick.0 == 0 {
            let reservation = ctx.commands.spawn(placed(r4, Tag(1))).unwrap();
            ctx.app.reserved.get_mut().push(reservation);
        }
        let reservation = ctx.app.reserved.get()[0];
        let seen = ctx.app.tags.contains(reservation.entity);
        ctx.app.log.get_mut().push(u32::from(seen));
        Ok(())
    });
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
            name: "spawn",
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
    let r4 = session.register_domain(
        DomainBuilder::new("r4", EuclideanR4)
            .untracked()
            .facility(Drift),
    );
    let walker = session
        .dispatch(|d| d.spawn(SpawnBundle::new().at(r4, Pose::at(Vec4::ZERO))))
        .unwrap();
    let sample = move |slot: usize| {
        move |ctx: Ctx<'_, Probe>| {
            let pose = ctx.domains.read(r4).unwrap().poses().get(walker).unwrap();
            ctx.app.sample.get_mut()[slot] = Some(pose.point.x);
            Ok(())
        }
    };
    session
        .system_at(
            Phase::Simulation,
            Order::After(DOMAIN_STEP),
            "after",
            sample(1),
        )
        .unwrap();
    session
        .system_at(
            Phase::Simulation,
            Order::Before(DOMAIN_STEP),
            "before",
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
fn an_action_handler_runs_once_per_press_at_its_place_in_dispatch_under_the_actions_name() {
    const BUMP: ActionId = ActionId("bump");
    const REFUSE: ActionId = ActionId("refuse");
    let event = |action, pressed| Input {
        actions: vec![ActionEvent { action, pressed }],
        ..Input::default()
    };
    let observe = |base: u32| {
        move |ctx: Ctx<'_, Probe>| {
            let runs = ctx.app.log.get().iter().filter(|&&mark| mark == 1).count();
            ctx.app.log.get_mut().push(base + runs as u32);
            Ok(())
        }
    };

    // A stopped clock is a paused program, and its keys still have to work.
    let stopped = SimConfig::new(0, 1).unwrap();
    let mut session = Session::new(Probe::default(), stopped);
    session.system(Phase::Dispatch, "before", observe(10));
    session.on_action(BUMP, |d| {
        d.app.log.get_mut().push(1);
        Ok(())
    });
    session.system(Phase::Dispatch, "after", observe(20));
    session.on_action(REFUSE, |_| Err(Rejection::Unsupported("refused")));

    session.boundary(event(BUMP, true)).unwrap();
    assert_eq!(
        *session.app.log.get(),
        [10, 1, 21],
        "the handler ran out of place or not exactly once"
    );
    let [result] = session.results() else {
        panic!("{} results for one press", session.results().len());
    };
    assert_eq!((result.name, result.outcome), ("bump", Ok(Outcome::Done)));

    session.boundary(event(BUMP, false)).unwrap();
    assert_eq!(
        *session.app.log.get(),
        [10, 1, 21, 11, 21],
        "a release ran the handler"
    );
    assert!(session.results().is_empty());

    session.boundary(event(REFUSE, true)).unwrap();
    let [result] = session.results() else {
        panic!("{} results for one press", session.results().len());
    };
    assert_eq!(
        (result.name, result.outcome),
        ("refuse", Err(Rejection::Unsupported("refused")))
    );
}

#[test]
fn an_axis_is_the_held_direction_and_zero_when_both_or_neither_is_held() {
    const LEFT: ActionId = ActionId("left");
    const RIGHT: ActionId = ActionId("right");
    let axis = |held: &[ActionId]| {
        let input = Input {
            held: held.to_vec(),
            ..Input::default()
        };
        input.axis(LEFT, RIGHT)
    };
    assert_eq!(axis(&[RIGHT]), 1.0);
    assert_eq!(axis(&[LEFT]), -1.0);
    assert_eq!(axis(&[LEFT, RIGHT]), 0.0);
    assert_eq!(axis(&[]), 0.0);
}
