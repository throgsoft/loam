#![cfg(feature = "physics")]

use loam_math::{EuclideanR4, Space};
use loam_physics::euclidean_r4::{register_default_narrowphase, sphere_body_r4};
use loam_runtime::{
    Change, Ctx, Cursor, DomainBuilder, DomainHandle, Entity, Input, LogCapacity, Phase,
    PhysicsConfig, Pose, Rejection, Session, SimConfig, SpawnBundle, Store,
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
