#![cfg(feature = "physics")]

use loam_math::{EuclideanR4, Iso4Flat, Space};
use loam_physics::euclidean_r4::{register_default_narrowphase, sphere_body_r4};
use loam_physics::{ColliderKind, EditError};
use loam_runtime::{
    Change, ChartCommand, ChartId, ChartPose, Command, Cursor, DomainBuilder, DomainHandle, Entity,
    LogCapacity, PhysicsConfig, Pose, Rejection, RestoreError, Session, SimConfig, SpawnBundle,
    Store,
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

const IDENTITY_FRAME: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

fn session() -> (Session<Probe>, DomainHandle<EuclideanR4>) {
    let mut session = Session::new(Probe::default(), SimConfig::default());
    let r4 = session.register_domain(
        DomainBuilder::new("r4", EuclideanR4)
            .tracked(LogCapacity::default())
            .physics(
                PhysicsConfig::new(register_default_narrowphase).gravity(Vec4::NEG_Y * GRAVITY),
            ),
    );
    (session, r4)
}

fn ball(session: &mut Session<Probe>, r4: DomainHandle<EuclideanR4>, at: Vec4) -> Entity {
    session
        .dispatch(|d| -> Result<Entity, Rejection> {
            let entity =
                d.spawn(SpawnBundle::new().at(r4, Pose(Iso4Flat::from_translation(at))))?;
            let physics = d.domains.typed(r4)?.physics_mut().unwrap();
            physics.spawn(
                entity,
                sphere_body_r4(at, Vec4::ZERO, RADIUS, MASS).unwrap(),
            );
            Ok(entity)
        })
        .unwrap()
}

fn pose_of(session: &mut Session<Probe>, r4: DomainHandle<EuclideanR4>, entity: Entity) -> Vec4 {
    session
        .domains_mut()
        .typed(r4)
        .unwrap()
        .poses
        .get(entity)
        .unwrap()
        .0
        .translation
}

fn body_of(session: &mut Session<Probe>, r4: DomainHandle<EuclideanR4>, entity: Entity) -> Vec4 {
    let physics = session.domains_mut().typed(r4).unwrap().physics().unwrap();
    let id = physics.body(entity).unwrap();
    physics.world().bodies.get(id).unwrap().position
}

fn only_pose(session: &mut Session<Probe>, r4: DomainHandle<EuclideanR4>) -> (Entity, Vec4) {
    let poses: &Store<Pose<EuclideanR4>> = &session.domains_mut().typed(r4).unwrap().poses;
    let (entity, pose) = poses.iter().next().unwrap();
    (entity, pose.0.translation)
}

#[test]
fn a_stepped_body_moves_its_entity_pose_and_the_dirty_log_reports_the_row() {
    let (mut session, r4) = session();
    let entity = ball(&mut session, r4, Vec4::new(0.0, 10.0, 0.0, 0.0));
    let mut cursor = Cursor::default();
    session
        .domains_mut()
        .typed(r4)
        .unwrap()
        .poses
        .catch_up(&mut cursor);

    session.tick().unwrap();

    let fallen = 10.0 - GRAVITY * DT * DT;
    let at = pose_of(&mut session, r4, entity);
    assert!(
        (at.y - fallen).abs() < 1e-6,
        "the mirror wrote {at}, not y = {fallen}"
    );
    let poses = &session.domains_mut().typed(r4).unwrap().poses;
    let changes = poses.changes(&mut cursor);
    assert!(!changes.is_resync());
    let reported: Vec<Entity> = changes
        .filter_map(|change| match change {
            Change::Row(row, _) => Some(row),
            Change::Removed(_) => None,
        })
        .collect();
    assert_eq!(reported, [entity]);
}

#[test]
fn a_sleeping_body_does_not_rewrite_its_entity_pose() {
    let (mut session, r4) = session();
    let entity = ball(&mut session, r4, Vec4::new(0.0, 10.0, 0.0, 0.0));
    session.tick().unwrap();

    let physics = session
        .domains_mut()
        .typed(r4)
        .unwrap()
        .physics_mut()
        .unwrap();
    let id = physics.body(entity).unwrap();
    physics.world_mut().sleep_body(id).unwrap();
    session.tick().unwrap();

    let domain = session.domains_mut().typed(r4).unwrap();
    let before = domain.poses.version(entity).unwrap();
    let asleep = domain.poses.get(entity).unwrap().0.translation;
    session.tick().unwrap();

    let domain = session.domains_mut().typed(r4).unwrap();
    assert_eq!(domain.poses.version(entity), Some(before));
    assert_eq!(domain.poses.get(entity).unwrap().0.translation, asleep);
}

#[test]
fn a_released_entity_leaves_no_body_and_no_pose_row() {
    let (mut session, r4) = session();
    let entity = ball(&mut session, r4, Vec4::new(0.0, 10.0, 0.0, 0.0));
    session.tick().unwrap();
    session.dispatch(|d| d.despawn(entity)).unwrap();
    session.tick().unwrap();

    let domain = session.domains_mut().typed(r4).unwrap();
    assert!(!domain.poses.contains(entity));
    let physics = domain.physics().unwrap();
    assert_eq!(physics.body(entity), None);
    assert_eq!(physics.world().bodies.len(), 0);
}

#[test]
fn a_restore_rewinds_the_pose_store_and_the_world_to_one_tick() {
    let (mut session, r4) = session();
    let entity = ball(&mut session, r4, Vec4::new(0.0, 10.0, 0.0, 0.0));
    for _ in 0..5 {
        session.tick().unwrap();
    }
    let snapshot = session.snapshot().unwrap();
    let captured = (
        pose_of(&mut session, r4, entity),
        body_of(&mut session, r4, entity),
    );
    for _ in 0..7 {
        session.tick().unwrap();
    }
    assert_ne!(pose_of(&mut session, r4, entity), captured.0);

    session.restore(&snapshot).unwrap();
    let (restored, at) = only_pose(&mut session, r4);
    assert_eq!(at, captured.0);
    assert_eq!(body_of(&mut session, r4, restored), captured.1);
}

#[test]
fn a_restore_drops_a_body_the_snapshot_never_had() {
    let (mut session, r4) = session();
    ball(&mut session, r4, Vec4::new(0.0, 10.0, 0.0, 0.0));
    session.tick().unwrap();
    let snapshot = session.snapshot().unwrap();
    let later = ball(&mut session, r4, Vec4::new(4.0, 10.0, 0.0, 0.0));
    session.tick().unwrap();

    session.restore(&snapshot).unwrap();
    let physics = session.domains_mut().typed(r4).unwrap().physics().unwrap();
    assert_eq!(physics.body(later), None);
    assert_eq!(physics.world().bodies.len(), 1);
}

#[test]
fn a_refused_world_restore_leaves_the_pose_store_and_the_world_untouched() {
    let (mut session, r4) = session();
    let entity = ball(&mut session, r4, Vec4::new(0.0, 10.0, 0.0, 0.0));
    session.tick().unwrap();
    let snapshot = session.snapshot().unwrap();
    session
        .domains_mut()
        .typed(r4)
        .unwrap()
        .physics_mut()
        .unwrap()
        .world_mut()
        .narrowphase
        .register(
            ColliderKind::HalfSpace4D,
            ColliderKind::HalfSpace4D,
            |_, _, _, _| None,
        );
    session.tick().unwrap();
    let kept = (
        pose_of(&mut session, r4, entity),
        body_of(&mut session, r4, entity),
    );

    assert_eq!(
        session.restore(&snapshot),
        Err(RestoreError::Edit(EditError::RegistrationMismatch))
    );
    assert_eq!(only_pose(&mut session, r4).1, kept.0);
    assert_eq!(body_of(&mut session, r4, entity), kept.1);
}

#[test]
fn a_place_goes_through_the_validated_edit_and_refuses_a_nan() {
    let (mut session, r4) = session();
    let entity = ball(&mut session, r4, Vec4::new(0.0, 10.0, 0.0, 0.0));
    let place = |coordinates| ChartCommand::Place {
        entity,
        pose: ChartPose {
            chart: ChartId(0),
            coordinates,
            frame: IDENTITY_FRAME,
        },
    };

    session
        .dispatch(|d| d.apply(Command::Chart(r4.id(), place([1.0, 2.0, 3.0, 4.0]))))
        .unwrap();
    assert_eq!(
        body_of(&mut session, r4, entity),
        Vec4::new(1.0, 2.0, 3.0, 4.0)
    );
    session.tick().unwrap();
    assert_eq!(
        pose_of(&mut session, r4, entity),
        body_of(&mut session, r4, entity)
    );

    let placed = body_of(&mut session, r4, entity);
    let refusal = session
        .dispatch(|d| d.apply(Command::Chart(r4.id(), place([f32::NAN, 0.0, 0.0, 0.0]))))
        .unwrap_err();
    let Rejection::Edit(edit) = refusal else {
        panic!("{refusal:?} did not come from the validated edit");
    };
    assert_eq!(edit.to_string(), "the value is not finite");
    assert_eq!(body_of(&mut session, r4, entity), placed);
}
