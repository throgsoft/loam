#![cfg(feature = "physics")]

use loam_math::{EuclideanR4, Rotor4, Space};
use loam_physics::euclidean_r4::{register_default_narrowphase, sphere_body_r4};
use loam_physics::{ColliderKind, EditError};
use loam_runtime::{
    Change, ChartCommand, ChartId, ChartPoint, ChartPose, ChartTangent, Command, Ctx, Cursor,
    DomainBuilder, DomainError, DomainHandle, Entity, Input, LogCapacity, Phase, PhysicsConfig,
    Pose, Rejection, RestoreError, Session, SimConfig, SpawnBundle, Store,
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
fn a_stepped_body_moves_its_entity_pose_and_the_dirty_log_reports_the_row() {
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
fn a_restore_rewinds_the_pose_store_and_the_world_to_one_tick() {
    let (mut session, r4) = session();
    let entity = ball(&mut session, r4, Vec4::new(0.0, 10.0, 0.0, 0.0));
    for _ in 0..5 {
        session.tick().unwrap();
    }
    let snapshot = session.snapshot().unwrap();
    let captured = (pose_of(&session, r4, entity), body_of(&session, r4, entity));
    for _ in 0..7 {
        session.tick().unwrap();
    }
    assert_ne!(pose_of(&session, r4, entity), captured.0);

    session.restore(&snapshot).unwrap();
    let (restored, at) = only_pose(&session, r4);
    assert_eq!(at, captured.0);
    assert_eq!(body_of(&session, r4, restored), captured.1);
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
    let physics = session.domains().read(r4).unwrap().physics().unwrap();
    assert_eq!(physics.body(later), None);
    assert_eq!(physics.world().bodies().len(), 1);
}

#[test]
fn stale_and_foreign_entities_cannot_reach_a_current_physics_body() {
    let (mut current_session, r4) = session();
    let stale = ball(&mut current_session, r4, Vec4::new(0.0, 10.0, 0.0, 0.0));
    let snapshot = current_session.snapshot().unwrap();
    current_session.restore(&snapshot).unwrap();
    let (current, before) = only_pose(&current_session, r4);
    let (mut foreign_session, foreign_r4) = session();
    let foreign = ball(
        &mut foreign_session,
        foreign_r4,
        Vec4::new(0.0, 10.0, 0.0, 0.0),
    );
    assert_eq!(stale.key(), current.key());
    assert_eq!(foreign.key(), current.key());

    for invalid in [stale, foreign] {
        let physics = current_session
            .domains_mut()
            .typed(r4)
            .unwrap()
            .physics_mut()
            .unwrap();
        assert_eq!(physics.body(invalid), None);
        assert_eq!(physics.throw(invalid, Vec4::X), Err(EditError::StaleHandle));
        assert_eq!(physics.despawn(invalid), None);
        assert_eq!(
            current_session.dispatch(|d| d.apply(Command::Chart(
                r4.id(),
                ChartCommand::Move {
                    entity: invalid,
                    point: ChartPoint {
                        chart: ChartId(0),
                        coordinates: Vec4::X.to_array(),
                    },
                },
            ))),
            Err(Rejection::Stale(invalid))
        );
    }

    assert_eq!(only_pose(&current_session, r4), (current, before));
    assert_eq!(body_of(&current_session, r4, current), before);
    assert_eq!(
        current_session
            .domains()
            .read(r4)
            .unwrap()
            .physics()
            .unwrap()
            .world()
            .bodies()
            .len(),
        1
    );
}

#[test]
fn physics_place_keeps_the_frame_used_by_local_walk() {
    let (mut session, r4) = session();
    let start = Vec4::new(2.0, 3.0, 5.0, 7.0);
    let frame = Rotor4::from_rotation_arc(Vec4::X, Vec4::Y);
    let pose: Pose<EuclideanR4> = Pose {
        point: start,
        frame,
    };
    let (free, physical) = session
        .dispatch(|d| -> Result<_, Rejection> {
            let free = d.spawn(SpawnBundle::new().at(r4, Pose::at(start)))?;
            let physical = d.spawn(SpawnBundle::new().at(r4, Pose::at(start)))?;
            d.domains.typed(r4)?.spawn_body(
                physical,
                sphere_body_r4(start, Vec4::ZERO, RADIUS, MASS).unwrap(),
            )?;
            Ok((free, physical))
        })
        .unwrap();
    let place = |entity| {
        Command::Chart(
            r4.id(),
            ChartCommand::Place {
                entity,
                pose: ChartPose {
                    chart: ChartId(0),
                    coordinates: pose.point.to_array(),
                    frame: pose.frame.to_mat4(),
                },
            },
        )
    };
    let walk = |entity| {
        Command::Chart(
            r4.id(),
            ChartCommand::Walk {
                entity,
                tangent: ChartTangent {
                    chart: ChartId(0),
                    vector: Vec4::X.to_array(),
                },
                dt: 1.0,
            },
        )
    };

    session.dispatch(|d| d.apply(place(free))).unwrap();
    session.dispatch(|d| d.apply(place(physical))).unwrap();
    session.dispatch(|d| d.apply(walk(free))).unwrap();
    session.dispatch(|d| d.apply(walk(physical))).unwrap();

    let domain = session.domains().read(r4).unwrap();
    let free_pose = *domain.poses().get(free).unwrap();
    let physical_pose = *domain.poses().get(physical).unwrap();
    assert!((free_pose.point - (start + Vec4::Y)).length() <= 1e-5);
    assert!((physical_pose.point - free_pose.point).length() <= 1e-5);
    assert_eq!(physical_pose.frame, free_pose.frame);
}

#[test]
fn a_refused_world_restore_leaves_the_pose_store_the_world_or_the_scene_changed() {
    let (mut session, r4) = session();
    let entity = ball(&mut session, r4, Vec4::new(0.0, 10.0, 0.0, 0.0));
    session.tick().unwrap();
    let snapshot = session.snapshot().unwrap();
    let epoch = session.scene().epoch;
    session
        .domains_mut()
        .typed(r4)
        .unwrap()
        .physics_mut()
        .unwrap()
        .narrowphase_mut()
        .register(
            ColliderKind::HalfSpace4D,
            ColliderKind::HalfSpace4D,
            |_, _, _, _| None,
        );
    session.tick().unwrap();
    let kept = (pose_of(&session, r4, entity), body_of(&session, r4, entity));

    assert_eq!(
        session.restore(&snapshot),
        Err(RestoreError::Edit(EditError::RegistrationMismatch))
    );
    assert_eq!(only_pose(&session, r4).1, kept.0);
    assert_eq!(body_of(&session, r4, entity), kept.1);
    assert_eq!(session.scene().epoch, epoch);
    assert_eq!(session.entities().resolve(entity), Some(entity.key()));
}

#[test]
fn checked_pose_edits_update_the_body_and_mirror_before_step() {
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
    assert_eq!(pose_of(&session, r4, entity), Vec4::new(1.0, 2.0, 3.0, 4.0));
    assert_eq!(body_of(&session, r4, entity), Vec4::new(1.0, 2.0, 3.0, 4.0));
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

    let placed = body_of(&session, r4, entity);
    let refusal = session
        .dispatch(|d| d.apply(Command::Chart(r4.id(), place([f32::NAN, 0.0, 0.0, 0.0]))))
        .unwrap_err();
    assert_eq!(
        refusal,
        Rejection::Domain(DomainError::InvalidCoordinate("x"))
    );
    assert_eq!(body_of(&session, r4, entity), placed);
}
