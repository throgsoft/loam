#![cfg(feature = "physics")]

use loam_math::{Bivector4, EuclideanR4, Rotor4, Space};
use loam_physics::euclidean_r4::{register_default_narrowphase, sphere_body_r4};
use loam_physics::BodyDef;
use loam_runtime::{
    Change, ChartCommand, Command, Ctx, Cursor, DomainBuilder, DomainError, DomainHandle,
    DomainSpace, Entity, GrabConfig, Input, LogCapacity, Phase, PhysicsConfig, Pose, Rejection,
    RestConfig, Session, SimConfig, SpawnBundle, Store,
};

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
        session.tick().unwrap();
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
fn a_stepped_body_moves_its_entity_pose_and_reports_the_row_and_a_sleeping_one_writes_nothing() {
    let (mut session, r4) = session();
    let entity = ball(&mut session, r4, Vec4::new(0.0, 10.0, 0.0, 0.0));
    let mut cursor = Cursor::default();
    session
        .domains()
        .read(r4)
        .unwrap()
        .poses()
        .catch_up(&mut cursor);

    session.tick().unwrap();

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
    session.tick().unwrap();

    let domain = session.domains().read(r4).unwrap();
    let before = domain.poses().version(entity).unwrap();
    let asleep = domain.poses().get(entity).unwrap().point;
    session.tick().unwrap();

    let domain = session.domains().read(r4).unwrap();
    assert_eq!(domain.poses().version(entity), Some(before));
    assert_eq!(domain.poses().get(entity).unwrap().point, asleep);
}

#[test]
fn a_released_entity_leaves_no_body_and_no_pose_row() {
    let (mut session, r4) = session();
    let entity = ball(&mut session, r4, Vec4::new(0.0, 10.0, 0.0, 0.0));
    session.tick().unwrap();
    session.dispatch(|d| d.despawn(entity)).unwrap();
    session.tick().unwrap();

    let domain = session.domains().read(r4).unwrap();
    assert!(!domain.poses().contains(entity));
    let physics = domain.physics().unwrap();
    assert_eq!(physics.body(entity), None);
    assert_eq!(physics.world().bodies().len(), 0);
}

#[test]
fn a_restore_rewinds_the_pose_store_and_the_world_and_drops_a_later_body() {
    let (mut session, r4) = session();
    let entity = ball(&mut session, r4, Vec4::new(0.0, 10.0, 0.0, 0.0));
    for _ in 0..5 {
        session.tick().unwrap();
    }
    let snapshot = session.snapshot().unwrap();
    let captured = (pose_of(&session, r4, entity), body_of(&session, r4, entity));
    let later = ball(&mut session, r4, Vec4::new(4.0, 10.0, 0.0, 0.0));
    for _ in 0..7 {
        session.tick().unwrap();
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
fn checked_pose_edits_update_the_body_and_mirror_before_step() {
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
    session.tick().unwrap();
    assert_eq!(pose_of(&session, r4, entity), body_of(&session, r4, entity));
}

#[test]
fn a_bundle_whose_body_is_refused_leaves_no_pose_and_no_body() {
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
        matches!(refused, Err(Rejection::Domain(DomainError::Unsupported(_)))),
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
fn a_bundled_body_keeps_its_velocity_or_its_sleep_at_a_moved_and_turned_pose() {
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
fn a_still_body_sleeps_once_its_rest_window_closes_and_a_held_body_only_after_its_release() {
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
fn a_restore_keeps_the_rest_time_a_body_had_at_the_snapshot() {
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
