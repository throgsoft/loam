#![cfg(feature = "physics")]

use loam_math::{Bivector4, EuclideanR4, Rotor4, Space};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use loam_physics::euclidean_r4::{
    halfspace4_body_r4, register_default_narrowphase, sphere_body_r4,
};
use loam_physics::world::ISLANDS_PER_SOLVE_WORKER;
use loam_physics::{BodyDef, EditError};
use loam_runtime::{
    ActionEvent, ActionId, Backdrop, Change, ChartCommand, Command, Ctx, Cursor, DomainBuilder,
    DomainError, DomainHandle, DomainSpace, Entity, GrabConfig, Ground, Input, LogCapacity, Phase,
    PhysicsConfig, PhysicsConfigError, Pose, PreparedGeometry, Publication, Rejection, RestConfig,
    Session, SimConfig, Sky, SpawnBundle, Store,
};
use loam_shape::polytope::Polytope4;
use loam_time::par::{ChunkSource, Executor, Sequential};

loam_runtime::stores! {
    #[derive(Default)]
    pub struct Probe {
        tags: Store<u32>,
    }
}

type Vec4 = <EuclideanR4 as Space>::Point;

const GRAVITY: f32 = 10.0;
const DT: f32 = 1.0 / 60.0;
const RADIUS: f32 = 0.5;
const MASS: f32 = 1.0;

fn session() -> (Session<Probe>, DomainHandle<EuclideanR4>) {
    let mut session = Session::new(Probe::default(), SimConfig::default());
    let r4 = session.register_domain(
        DomainBuilder::new("r4", EuclideanR4)
            .tracked(LogCapacity::default())
            .physics(
                PhysicsConfig::new(register_default_narrowphase).gravity(Vec4::NEG_Y * GRAVITY),
            )
            .unwrap(),
    );
    (session, r4)
}

fn ball(session: &mut Session<Probe>, r4: DomainHandle<EuclideanR4>, at: Vec4) -> Entity {
    session
        .dispatch(|d| -> Result<Entity, Rejection> {
            let entity = d.spawn(SpawnBundle::new().at(r4, Pose::at(at)))?;
            d.domains.typed(r4)?.spawn_body(
                entity,
                sphere_body_r4(at, Vec4::ZERO, RADIUS, MASS).unwrap(),
            )?;
            Ok(entity)
        })
        .unwrap()
}

fn pose_of(session: &Session<Probe>, r4: DomainHandle<EuclideanR4>, entity: Entity) -> Vec4 {
    session
        .domains()
        .read(r4)
        .unwrap()
        .poses()
        .get(entity)
        .unwrap()
        .point
}

fn body_of(session: &Session<Probe>, r4: DomainHandle<EuclideanR4>, entity: Entity) -> Vec4 {
    let physics = session.domains().read(r4).unwrap().physics().unwrap();
    let id = physics.body(entity).unwrap();
    physics.world().body(id).unwrap().position
}

fn still() -> BodyDef<EuclideanR4> {
    sphere_body_r4(Vec4::ZERO, Vec4::ZERO, RADIUS, MASS).unwrap()
}

fn bundled(
    session: &mut Session<Probe>,
    r4: DomainHandle<EuclideanR4>,
    pose: Pose<EuclideanR4>,
    body: BodyDef<EuclideanR4>,
) -> Entity {
    session
        .dispatch(|d| d.spawn(SpawnBundle::new().at(r4, pose).body(body)))
        .unwrap()
}

// The window is half a second: 25 ticks at 60 Hz fall short of it and 35 pass it.
fn resting_session() -> (Session<Probe>, DomainHandle<EuclideanR4>) {
    let mut session = Session::new(Probe::default(), SimConfig::default());
    let r4 = session.register_domain(
        DomainBuilder::new("r4", EuclideanR4)
            .tracked(LogCapacity::default())
            .physics(
                PhysicsConfig::new(register_default_narrowphase)
                    .grab(GrabConfig::new(10.0, 10.0, 100.0))
                    .rest(RestConfig {
                        travel: 0.1,
                        window: 0.5,
                        speed: 0.1,
                        angular_speed: 0.1,
                    }),
            )
            .unwrap(),
    );
    (session, r4)
}

fn ticks(session: &mut Session<Probe>, count: usize) {
    for _ in 0..count {
        session.tick(&Sequential).unwrap();
    }
}

fn asleep(session: &Session<Probe>, r4: DomainHandle<EuclideanR4>, entity: Entity) -> bool {
    let physics = session.domains().read(r4).unwrap().physics().unwrap();
    let id = physics.body(entity).unwrap();
    physics.world().body(id).unwrap().is_sleeping()
}

fn only_pose(session: &Session<Probe>, r4: DomainHandle<EuclideanR4>) -> (Entity, Vec4) {
    let poses: &Store<Pose<EuclideanR4>> = session.domains().read(r4).unwrap().poses();
    let (entity, pose) = poses.iter().next().unwrap();
    (entity, pose.point)
}

#[test]
fn stepped_and_sleeping_body() {
    let (mut session, r4) = session();
    let entity = ball(&mut session, r4, Vec4::new(0.0, 10.0, 0.0, 0.0));
    let mut cursor = Cursor::default();
    session
        .domains()
        .read(r4)
        .unwrap()
        .poses()
        .catch_up(&mut cursor);

    session.tick(&Sequential).unwrap();

    let fallen = 10.0 - GRAVITY * DT * DT;
    let at = pose_of(&session, r4, entity);
    assert!(
        (at.y - fallen).abs() < 1e-6,
        "the mirror wrote {at}, not y = {fallen}"
    );
    let poses = session.domains().read(r4).unwrap().poses();
    let changes = poses.changes(&mut cursor);
    assert!(!changes.is_resync());
    let reported: Vec<Entity> = changes
        .filter_map(|change| match change {
            Change::Row(row, _) => Some(row),
            Change::Removed(_) => None,
        })
        .collect();
    assert_eq!(reported, [entity]);

    let physics = session
        .domains_mut()
        .typed(r4)
        .unwrap()
        .physics_mut()
        .unwrap();
    physics.sleep_body(entity).unwrap();
    session.tick(&Sequential).unwrap();

    let domain = session.domains().read(r4).unwrap();
    let before = domain.poses().version(entity).unwrap();
    let asleep = domain.poses().get(entity).unwrap().point;
    session.tick(&Sequential).unwrap();

    let domain = session.domains().read(r4).unwrap();
    assert_eq!(domain.poses().version(entity), Some(before));
    assert_eq!(domain.poses().get(entity).unwrap().point, asleep);
}

#[test]
fn released_entity() {
    let (mut session, r4) = session();
    let entity = ball(&mut session, r4, Vec4::new(0.0, 10.0, 0.0, 0.0));
    session.tick(&Sequential).unwrap();
    session.dispatch(|d| d.despawn(entity)).unwrap();
    session.tick(&Sequential).unwrap();

    let domain = session.domains().read(r4).unwrap();
    assert!(!domain.poses().contains(entity));
    let physics = domain.physics().unwrap();
    assert_eq!(physics.body(entity), None);
    assert_eq!(physics.world().bodies().len(), 0);
}

#[test]
fn restore_rewind() {
    let (mut session, r4) = session();
    let entity = ball(&mut session, r4, Vec4::new(0.0, 10.0, 0.0, 0.0));
    for _ in 0..5 {
        session.tick(&Sequential).unwrap();
    }
    let snapshot = session.snapshot().unwrap();
    let captured = (pose_of(&session, r4, entity), body_of(&session, r4, entity));
    let later = ball(&mut session, r4, Vec4::new(4.0, 10.0, 0.0, 0.0));
    for _ in 0..7 {
        session.tick(&Sequential).unwrap();
    }
    assert_ne!(pose_of(&session, r4, entity), captured.0);

    session.restore(&snapshot).unwrap();
    let (restored, at) = only_pose(&session, r4);
    assert_eq!(at, captured.0);
    assert_eq!(body_of(&session, r4, restored), captured.1);
    let physics = session.domains().read(r4).unwrap().physics().unwrap();
    assert_eq!(physics.body(later), None);
    assert_eq!(physics.world().bodies().len(), 1);
}

#[test]
fn checked_pose_edit() {
    let (mut session, r4) = session();
    let entity = ball(&mut session, r4, Vec4::new(0.0, 10.0, 0.0, 0.0));
    let body = session
        .domains()
        .read(r4)
        .unwrap()
        .physics()
        .unwrap()
        .body(entity)
        .unwrap();
    session
        .domains_mut()
        .typed(r4)
        .unwrap()
        .physics_mut()
        .unwrap()
        .sleep_body(entity)
        .unwrap();
    let moved = Vec4::new(-2.0, 3.0, 4.0, 5.0);
    session
        .domains_mut()
        .typed(r4)
        .unwrap()
        .set_point(entity, moved)
        .unwrap();
    assert_eq!(pose_of(&session, r4, entity), moved);
    assert_eq!(body_of(&session, r4, entity), moved);
    assert!(!session
        .domains()
        .read(r4)
        .unwrap()
        .physics()
        .unwrap()
        .world()
        .body(body)
        .unwrap()
        .is_sleeping());
    let paused = Vec4::new(6.0, 7.0, 8.0, 9.0);
    session.system(
        Phase::Dispatch,
        "domain edit",
        move |ctx: Ctx<'_, Probe>| {
            ctx.domains
                .typed(r4)
                .unwrap()
                .set_point(entity, paused)
                .unwrap();
            Ok(())
        },
    );
    session.system(
        Phase::Dispatch,
        "dependent pose query",
        move |ctx: Ctx<'_, Probe>| {
            assert_eq!(
                ctx.domains
                    .read(r4)
                    .unwrap()
                    .poses()
                    .get(entity)
                    .unwrap()
                    .point,
                paused
            );
            Ok(())
        },
    );
    session.boundary(Input::default()).unwrap();
    assert_eq!(pose_of(&session, r4, entity), paused);
    session.tick(&Sequential).unwrap();
    assert_eq!(pose_of(&session, r4, entity), body_of(&session, r4, entity));
}

#[test]
fn refused_bundle() {
    let (mut session, r4) = session();
    let plain = session.register_domain(DomainBuilder::new("plain", EuclideanR4).untracked());

    let refused = session.dispatch(|d| {
        d.spawn(
            SpawnBundle::new()
                .at(plain, Pose::at(Vec4::X))
                .body(still()),
        )
    });
    assert!(
        matches!(refused, Err(Rejection::Domain(DomainError::NoPhysics(_)))),
        "a domain without physics took a body: {refused:?}"
    );
    assert!(session.domains().read(plain).unwrap().poses().is_empty());

    let nowhere = Pose::at(Vec4::new(f32::NAN, 0.0, 0.0, 0.0));
    let refused = session.dispatch(|d| d.spawn(SpawnBundle::new().at(r4, nowhere).body(still())));
    assert!(refused.is_err());
    let domain = session.domains().read(r4).unwrap();
    assert!(domain.poses().is_empty());
    assert_eq!(domain.physics().unwrap().world().bodies().len(), 0);
}

#[test]
fn bundled_at_turned_pose() {
    let (mut session, r4) = session();
    let turned = |x: f32| Pose {
        point: Vec4::new(x, 1.0, -2.0, 0.5),
        frame: Rotor4::from_rotation_arc(Vec4::Y, Vec4::W),
    };
    let velocity = Vec4::new(1.0, 2.0, 0.0, -1.0);
    let spin = Bivector4 {
        xy: 0.5,
        zw: -0.25,
        ..Bivector4::ZERO
    };
    let moving = || still().velocity(velocity, spin, &EuclideanR4).unwrap();
    let row = |session: &Session<Probe>, entity: Entity| {
        let physics = session.domains().read(r4).unwrap().physics().unwrap();
        *physics.world().body(physics.body(entity).unwrap()).unwrap()
    };

    let thrown = bundled(&mut session, r4, turned(3.0), moving());
    let body = row(&session, thrown);
    assert_eq!(body.position, turned(3.0).point);
    assert_eq!((body.velocity, body.angular_velocity), (velocity, spin));
    assert!(!body.is_sleeping());

    let slept_last = bundled(&mut session, r4, turned(8.0), moving().asleep());
    let slept_first = still().asleep().velocity(velocity, spin, &EuclideanR4);
    let slept_first = bundled(&mut session, r4, turned(13.0), slept_first.unwrap());
    for entity in [slept_last, slept_first] {
        let body = row(&session, entity);
        assert!(body.is_sleeping(), "placing the pose woke the body");
        assert_eq!(
            (body.velocity, body.angular_velocity),
            (Vec4::ZERO, Bivector4::ZERO)
        );
    }
}

#[test]
fn rest_window() {
    let (mut session, r4) = resting_session();
    let in_bundle = bundled(&mut session, r4, Pose::at(Vec4::ZERO), still());
    let attached = ball(&mut session, r4, Vec4::X * 5.0);
    let held = ball(&mut session, r4, Vec4::X * 10.0);
    let grab = ChartCommand::Grab {
        entity: held,
        point: EuclideanR4.chart_point(Vec4::X * 10.0),
    };
    session.submit(Command::Chart(r4.id(), grab));
    session.boundary(Input::default()).unwrap();

    ticks(&mut session, 25);
    for entity in [in_bundle, attached, held] {
        assert!(!asleep(&session, r4, entity), "a body slept early");
    }
    ticks(&mut session, 10);
    assert!(asleep(&session, r4, in_bundle), "the bundled body is awake");
    assert!(asleep(&session, r4, attached), "the attached body is awake");
    assert!(!asleep(&session, r4, held), "a held body fell asleep");

    session.submit(Command::Chart(
        r4.id(),
        ChartCommand::Release { entity: held },
    ));
    session.boundary(Input::default()).unwrap();
    ticks(&mut session, 25);
    assert!(
        !asleep(&session, r4, held),
        "the time in the hand counted as rest"
    );
    ticks(&mut session, 10);
    assert!(asleep(&session, r4, held));
}

#[test]
fn rest_time_restore() {
    let (mut session, r4) = resting_session();
    let entity = ball(&mut session, r4, Vec4::ZERO);
    ticks(&mut session, 20);
    let snapshot = session.snapshot().unwrap();
    session
        .domains_mut()
        .physics(r4)
        .unwrap()
        .set_velocity(entity, Vec4::X, Bivector4::ZERO)
        .unwrap();
    ticks(&mut session, 5);

    session.restore(&snapshot).unwrap();
    let (restored, _) = only_pose(&session, r4);
    ticks(&mut session, 5);
    assert!(!asleep(&session, r4, restored));
    ticks(&mut session, 10);
    assert!(
        asleep(&session, r4, restored),
        "the restore restarted a rest window that had 20 ticks behind it"
    );
}

#[test]
fn touching_entities() {
    let (mut session, r4) = session();
    let floor = bundled(
        &mut session,
        r4,
        Pose::at(Vec4::ZERO),
        halfspace4_body_r4(Vec4::Y, 0.0).unwrap(),
    );
    let resting = ball(&mut session, r4, Vec4::Y * RADIUS);
    let falling = ball(&mut session, r4, Vec4::new(4.0, 20.0, 0.0, 0.0));
    let bodiless = session
        .dispatch(|d| d.spawn(SpawnBundle::new().at(r4, Pose::at(Vec4::X * 8.0))))
        .unwrap();
    ticks(&mut session, 30);

    let physics = session.domains().read(r4).unwrap().physics().unwrap();
    let touched = |entity: Entity| -> Vec<Entity> {
        physics.touching(entity).map(|(other, _)| other).collect()
    };
    assert_eq!(touched(resting), [floor]);
    assert_eq!(
        touched(floor),
        [resting],
        "the floor does not name the body that names it"
    );
    assert_eq!(touched(falling), [], "a body in the air touches something");
    assert_eq!(touched(bodiless), []);
}

#[test]
fn paused_ticks() {
    const PAUSE: ActionId = ActionId("pause");
    let press = || Input {
        actions: vec![ActionEvent {
            action: PAUSE,
            pressed: true,
        }],
        ..Input::default()
    };
    let (mut session, r4) = session();
    session.pause_on(PAUSE);
    let stepped = Arc::new(AtomicUsize::new(0));
    let steps = stepped.clone();
    session.system(Phase::Simulation, "count", move |_ctx: Ctx<'_, Probe>| {
        steps.fetch_add(1, Ordering::Relaxed);
        Ok(())
    });
    let entity = ball(&mut session, r4, Vec4::new(0.0, 10.0, 0.0, 0.0));
    let state = |session: &Session<Probe>| {
        let physics = session.domains().read(r4).unwrap().physics().unwrap();
        let body = *physics.world().body(physics.body(entity).unwrap()).unwrap();
        (
            pose_of(session, r4, entity),
            body.position,
            body.velocity,
            session.current_tick(),
            stepped.load(Ordering::Relaxed),
        )
    };
    let drawn = |session: &Session<Probe>, alpha: f32| {
        let domain = session.domains().read(r4).unwrap();
        domain.drawn_pose(entity, alpha).unwrap().point
    };
    ticks(&mut session, 3);
    assert_ne!(
        drawn(&session, 0.5),
        pose_of(&session, r4, entity),
        "the fixture draws no blend for a pause to drop"
    );

    session.boundary(press()).unwrap();
    let frozen = state(&session);
    let (pose, _, velocity, tick, steps) = frozen;
    ticks(&mut session, 5);
    assert_eq!(state(&session), frozen, "a paused tick moved the world on");
    let mut publication = Publication::default();
    session.publish(&mut publication, 0.5).unwrap();
    assert_eq!(publication.stamp.tick, tick);
    for alpha in [0.0, 0.5] {
        assert_eq!(
            drawn(&session, alpha),
            pose,
            "a paused publish at alpha {alpha} drew the body off its pose"
        );
    }

    session.boundary(press()).unwrap();
    session.tick(&Sequential).unwrap();
    let (_, _, resumed, later, ran) = state(&session);
    assert_eq!(
        (later.0, ran),
        (tick.0 + 1, steps + 1),
        "the press at a paused boundary did not resume the ticks"
    );
    let want = velocity.y - GRAVITY * DT;
    assert!(
        (resumed.y - want).abs() < 1e-5,
        "the body fell on at {}, not at {want} from the speed it was paused with",
        resumed.y
    );
}

struct Counting {
    drains: AtomicUsize,
}

impl Executor for Counting {
    fn parallelism(&self) -> usize {
        2
    }

    fn for_each_chunk(&self, _workers: usize, chunks: &(dyn ChunkSource + Sync)) {
        self.drains.fetch_add(1, Ordering::Relaxed);
        while chunks.run_next() {}
    }
}

#[test]
fn many_islands() {
    let (mut session, r4) = session();
    for pair in 0..2 * ISLANDS_PER_SOLVE_WORKER {
        let x = pair as f32 * 4.0;
        ball(&mut session, r4, Vec4::X * x);
        ball(&mut session, r4, Vec4::X * (x + 1.5 * RADIUS));
    }
    let executor = Counting {
        drains: AtomicUsize::new(0),
    };

    session.tick(&executor).unwrap();

    let physics = session.domains().read(r4).unwrap().physics().unwrap();
    let islands = physics.world().islands().len();
    assert!(
        islands >= 2 * ISLANDS_PER_SOLVE_WORKER,
        "the fixture made {islands} islands, too few for a second solve worker"
    );
    assert!(
        executor.drains.load(Ordering::Relaxed) > 0,
        "the world solved its islands without the executor the tick was given"
    );
}

fn backdrop(ground_y: f32) -> Backdrop {
    Backdrop {
        sky: Sky {
            below: [0.3, 0.1, 0.1],
            above: [0.1, 0.1, 0.4],
        },
        ground: Ground {
            y: ground_y,
            dark: [0.1; 3],
            light: [0.6; 3],
            fog_per_unit: 0.01,
            fog_start: 2.0,
            visible: true,
        },
    }
}

#[test]
fn prepared_polytope_on_the_floor() {
    const SCALE: f32 = 0.8;
    const GROUND: f32 = -1.0;
    let mut session = Session::new(Probe::default(), SimConfig::default());
    let r4 = session.register_domain(
        DomainBuilder::new("r4", EuclideanR4)
            .tracked(LogCapacity::default())
            .physics(PhysicsConfig::default().gravity(Vec4::NEG_Y * GRAVITY))
            .unwrap(),
    );
    let tesseract = session.prepare(PreparedGeometry::Polytope4 {
        polytope: Polytope4::Tesseract,
        scale: SCALE,
    });
    let dusk = backdrop(GROUND);
    let dropped = session
        .try_dispatch(|d| {
            d.floor(r4, dusk)?;
            let body = d.body_of(tesseract, MASS)?;
            let above = Pose::at(Vec4::new(0.0, GROUND + 1.0, 0.0, 0.0));
            d.spawn(SpawnBundle::new().at(r4, above).body(body))
        })
        .unwrap();
    assert_eq!(session.views().root().backdrop, Some(dusk));

    ticks(&mut session, 240);
    // A tesseract of circumradius SCALE has a half extent of SCALE / 2.
    let rest = GROUND + SCALE / 2.0;
    let height = pose_of(&session, r4, dropped).y;
    assert!(
        (height - rest).abs() < 0.02,
        "the body of the prepared tesseract came to rest at {height}, not {rest}"
    );
}

#[test]
fn named_refusals() {
    let (mut session, r4) = session();
    let plain = session.register_domain(DomainBuilder::new("plain", EuclideanR4).untracked());
    let lines = session.prepare(PreparedGeometry::Lines4 {
        segments: Vec::new(),
    });
    let tesseract = session.prepare(PreparedGeometry::Polytope4 {
        polytope: Polytope4::Tesseract,
        scale: 1.0,
    });

    session.dispatch(|d| {
        assert_eq!(
            d.body_of(lines, MASS).err(),
            Some(Rejection::NotPolytope(lines))
        );
        assert_eq!(
            d.body_of(tesseract, f32::NAN).err(),
            Some(Rejection::Domain(DomainError::Physics(
                EditError::InvalidMass
            ))),
            "the builder's cause did not reach the caller"
        );
        assert!(
            d.floor(plain, backdrop(0.0)).is_err(),
            "a domain without physics took a floor"
        );
        assert_eq!(
            d.spawn(SpawnBundle::new().body(still())).err(),
            Some(Rejection::BodyUnplaced)
        );
        assert_eq!(
            d.apply(Command::Reset).err(),
            Some(Rejection::BoundaryCommand)
        );
    });
    assert_eq!(
        session.views().root().backdrop,
        None,
        "a refused floor set the backdrop"
    );
    assert!(session.domains().read(plain).unwrap().poses().is_empty());
    assert!(session.domains().read(r4).unwrap().poses().is_empty());
}

#[test]
fn refused_config() {
    let config = PhysicsConfig::<EuclideanR4>::default;
    let rest = RestConfig {
        travel: -0.1,
        window: 0.5,
        speed: 0.1,
        angular_speed: 0.1,
    };
    for (config, cause) in [
        (
            config().gravity(Vec4::new(0.0, f32::NAN, 0.0, 0.0)),
            PhysicsConfigError::Gravity,
        ),
        (
            config().grab(GrabConfig::new(-1.0, 10.0, 100.0)),
            PhysicsConfigError::Grab,
        ),
        (config().rest(rest), PhysicsConfigError::Rest),
        (config().angular_damping(-1.0), PhysicsConfigError::Rest),
        (
            config().adaptive_substeps(1, 4, 0.0, |_| 0.0),
            PhysicsConfigError::StepTravel,
        ),
    ] {
        let refused = DomainBuilder::new("r4", EuclideanR4)
            .untracked()
            .physics(config)
            .err();
        assert_eq!(refused, Some(DomainError::PhysicsConfig(cause)));
    }
}

#[test]
fn refused_edit_cause() {
    let (mut session, r4) = session();
    let entity = ball(&mut session, r4, Vec4::new(0.0, 10.0, 0.0, 0.0));
    let bare = session
        .dispatch(|d| d.spawn(SpawnBundle::new().at(r4, Pose::at(Vec4::X))))
        .unwrap();
    let nowhere = Vec4::new(f32::NAN, 0.0, 0.0, 0.0);
    let not_finite = DomainError::Physics(EditError::NotFinite);

    let refused = session.try_dispatch(|d| {
        d.domains
            .physics(r4)?
            .set_velocity(entity, nowhere, Bivector4::ZERO)?;
        Ok(())
    });
    assert_eq!(refused, Err(Rejection::Domain(not_finite)));
    assert_eq!(
        session
            .domains_mut()
            .physics(r4)
            .unwrap()
            .set_restitution(bare, 0.5),
        Err(DomainError::NoBody(bare))
    );

    session.system(Phase::Dispatch, "bad edit", move |ctx: Ctx<'_, Probe>| {
        ctx.domains
            .physics(r4)?
            .set_velocity(entity, nowhere, Bivector4::ZERO)
    });
    let fault = session.boundary(Input::default()).unwrap_err();
    assert_eq!(
        fault.cause, not_finite,
        "a system's refused edit differs from a command's"
    );
}
