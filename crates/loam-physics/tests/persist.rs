#![cfg(all(feature = "persist", feature = "r3"))]

use glam::Vec3;
use loam_math::EuclideanR3;
use loam_physics::collider::ColliderKind;
use loam_physics::euclidean_r3::{halfspace_body_r3, register_default_narrowphase, sphere_body_r3};
use loam_physics::field_contact::register_field_contacts;
use loam_physics::{PersistError, RigidBody, World, PERSIST_VERSION};
use loam_shape::field::{DistanceField, FieldKind};

const DT: f32 = 1.0 / 240.0;
const RADIUS: f32 = 0.5;
const REPLAY_STEPS: usize = 100;

fn sample(body: &RigidBody<EuclideanR3>, words: &mut Vec<u32>) {
    for value in [
        body.position.x,
        body.position.y,
        body.position.z,
        body.velocity.x,
        body.velocity.y,
        body.velocity.z,
        body.orientation.rotation.x,
        body.orientation.rotation.y,
        body.orientation.rotation.z,
        body.orientation.rotation.w,
        body.angular_velocity.xy,
        body.angular_velocity.yz,
        body.angular_velocity.zx,
    ] {
        words.push(value.to_bits());
    }
}

fn stack() -> World<EuclideanR3> {
    let mut world = World::new(EuclideanR3);
    register_default_narrowphase(&mut world.narrowphase);
    world.gravity = Some(Vec3::new(0.0, -9.8, 0.0));
    let floor = world.push_body(halfspace_body_r3(Vec3::Y, 0.0).unwrap());
    world.bodies[floor].restitution = 0.0;
    for level in 0..3 {
        let y = RADIUS + level as f32 * 2.0 * RADIUS;
        let id = world.push_body(
            sphere_body_r3(
                Vec3::new(0.05 * level as f32, y, 0.0),
                Vec3::ZERO,
                RADIUS,
                1.0,
            )
            .unwrap(),
        );
        world.bodies[id].restitution = 0.0;
    }
    for _ in 0..120 {
        world.step(DT);
    }
    world
}

fn receiver() -> World<EuclideanR3> {
    let mut world = World::new(EuclideanR3);
    register_default_narrowphase(&mut world.narrowphase);
    world.gravity = Some(Vec3::new(0.0, -9.8, 0.0));
    world
}

#[test]
fn a_reloaded_world_replays_the_originals_trajectory() {
    let mut original = stack();
    let saved = original.save().unwrap();

    let mut reloaded = receiver();
    reloaded.load(&saved).unwrap();
    assert_eq!(reloaded.state_hash(sample), original.state_hash(sample));

    for step in 0..REPLAY_STEPS {
        original.step(DT);
        reloaded.step(DT);
        assert_eq!(
            reloaded.state_hash(sample),
            original.state_hash(sample),
            "the reloaded world diverged at step {step}"
        );
    }
}

#[test]
fn a_bumped_schema_version_is_refused_and_the_receiver_keeps_its_rows() {
    let saved = stack().save().unwrap();
    let bumped = saved.replacen(
        &format!("version:{PERSIST_VERSION}"),
        &format!("version:{}", PERSIST_VERSION + 1),
        1,
    );
    assert_ne!(bumped, saved, "the version field was not found in the save");

    let mut receiving = stack();
    let before = receiving.state_hash(sample);
    assert!(matches!(
        receiving.load(&bumped),
        Err(PersistError::Version { .. })
    ));
    assert_eq!(receiving.state_hash(sample), before);
    assert!(receiving.load(&saved).is_ok());
}

struct Ground;

impl DistanceField for Ground {
    fn field_kind(&self) -> FieldKind {
        FieldKind::ExactDistance
    }

    fn distance(&self, point: [f32; 4]) -> f32 {
        point[1]
    }

    fn error(&self) -> f32 {
        0.0
    }
}

fn field_stack() -> World<EuclideanR3> {
    let mut world = World::new(EuclideanR3);
    register_default_narrowphase(&mut world.narrowphase);
    register_field_contacts(&mut world.field_narrowphase);
    world.gravity = Some(Vec3::new(0.0, -9.8, 0.0));
    let anchor = world.push_body(halfspace_body_r3(Vec3::Y, 0.0).unwrap());
    world.bodies[anchor].restitution = 0.0;
    let field = world.insert_field(anchor, Box::new(Ground)).unwrap();
    let ball =
        world.push_body(sphere_body_r3(Vec3::new(0.0, 2.0, 0.0), Vec3::ZERO, RADIUS, 1.0).unwrap());
    world.bodies[ball].restitution = 0.0;
    world.bind_field(ball, field).unwrap();
    world
}

#[test]
fn a_saved_binding_reloads_into_a_world_whose_fields_were_inserted_alike() {
    let mut original = field_stack();
    for _ in 0..120 {
        original.step(DT);
    }
    let saved = original.save().unwrap();

    let mut reloaded = field_stack();
    reloaded.load(&saved).unwrap();
    assert_eq!(reloaded.field_bindings(), original.field_bindings());
    assert_eq!(reloaded.state_hash(sample), original.state_hash(sample));

    for step in 0..REPLAY_STEPS {
        original.step(DT);
        reloaded.step(DT);
        assert_eq!(
            reloaded.state_hash(sample),
            original.state_hash(sample),
            "the reloaded field world diverged at step {step}"
        );
    }
}

#[test]
fn a_load_into_a_world_without_the_saved_fields_is_refused_by_name() {
    let saved = field_stack().save().unwrap();

    let mut receiving = receiver();
    let before = receiving.state_hash(sample);
    assert!(matches!(receiving.load(&saved), Err(PersistError::Fields)));
    assert_eq!(receiving.state_hash(sample), before);
}

#[test]
fn a_save_from_other_registrations_is_refused_and_the_receiver_keeps_its_rows() {
    let saved = stack().save().unwrap();

    let mut receiving = stack();
    receiving
        .narrowphase
        .register(ColliderKind::Box3, ColliderKind::Box3, |_, _, _, _| None);
    let before = receiving.state_hash(sample);
    assert!(matches!(
        receiving.load(&saved),
        Err(PersistError::Registrations)
    ));
    assert_eq!(receiving.state_hash(sample), before);
}
