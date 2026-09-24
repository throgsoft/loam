use glam::Vec4;
use loam::math::{Bivector4, EuclideanR4, Rotor, Rotor4};
use loam::physics::euclidean_r4::{
    halfspace4_body_r4, polytope_body_r4, register_default_narrowphase, regular_polytope4_inertia,
};
use loam::physics::manifold::PENETRATION_SLOP;
use loam::physics::{Collider, EditError, RigidBody};
use loam::runtime::{
    Dispatch, DomainError, DomainHandle, Domains, Entity, GrabConfig, Instance, Physics,
    PhysicsConfig, Pose, Rejection, Session, SpawnBundle, Step,
};
use loam::shape::polytope::Polytope4;

use crate::catalog::{ShapeEntry, SHAPE_CATALOG};
use crate::consts::{FLOOR_Y, GRAVITY};
use crate::{Card, HiddenSlot, Playground, Slot, Toy, Wall};

pub(crate) const BODY_SIZE: f32 = 0.45;
pub(crate) const ARENA_HALF: f32 = 3.6;
const ARENA_TOP: f32 = FLOOR_Y + 2.0 * ARENA_HALF;

const BODY_MASS: f32 = 1.0;
const BODY_RESTITUTION: f32 = 0.05;
const WALL_RESTITUTION: f32 = 0.4;
const PHYSICS_FLOOR_Y: f32 = FLOOR_Y + 2.0 * PENETRATION_SLOP;
const SPAWN_SPACING: f32 = 1.4;
const SPAWN_CLEARANCE: f32 = 0.20;
const TICK_DT: f32 = 1.0 / 60.0;
const BASE_SUBSTEPS: u32 = 4;
const MAX_SUBSTEPS: u32 = 16;
const STEP_TRAVEL_BUDGET: f32 = 0.135;
const GRAB_STIFFNESS: f32 = 40.0;
const MAX_CARRY_SPEED: f32 = 35.0;
const MAX_GRAB_ACCELERATION: f32 = 800.0;
const RELEASE_GAIN: f32 = 0.3;
const MAX_RELEASE_SPEED: f32 = 0.5 * STEP_TRAVEL_BUDGET / (TICK_DT / MAX_SUBSTEPS as f32);
const RELEASE_SPIN_GAIN: f32 = 0.35;
const MAX_ANGULAR_SPEED: f32 =
    0.5 * STEP_TRAVEL_BUDGET / (BODY_SIZE * (TICK_DT / MAX_SUBSTEPS as f32));
const ANGULAR_DAMPING: f32 = 1.2;
const REST_TRAVEL: f32 = 0.02;
const REST_WINDOW: f32 = 30.0 * TICK_DT;
const REST_SPEED: f32 = 0.15;
const REST_ANGULAR_SPEED: f32 = 0.3;
const TOYS: [Polytope4; 5] = [
    Polytope4::Cell24,
    Polytope4::Tesseract,
    Polytope4::Pentatope,
    Polytope4::Cell16,
    Polytope4::Tesseract,
];

fn motion_speed(body: &RigidBody<EuclideanR4>) -> f32 {
    body.velocity.length() + body.angular_velocity.magnitude() * BODY_SIZE
}

pub(crate) fn physics_config() -> PhysicsConfig<EuclideanR4> {
    PhysicsConfig::new(register_default_narrowphase)
        .gravity(Vec4::NEG_Y * GRAVITY)
        .grab(
            GrabConfig::new(GRAB_STIFFNESS, MAX_CARRY_SPEED, MAX_GRAB_ACCELERATION)
                .anchor_point(grab_anchor)
                .constrain_target(clamp_target_to_arena),
        )
        .adaptive_substeps(
            BASE_SUBSTEPS,
            MAX_SUBSTEPS,
            STEP_TRAVEL_BUDGET,
            motion_speed,
        )
}

fn grab_anchor(body: Vec4, picked: Vec4) -> Vec4 {
    Vec4::new(picked.x, picked.y, picked.z, body.w)
}

fn clamp_target_to_arena(current: Vec4, target: Vec4) -> Vec4 {
    let reach = ARENA_HALF - BODY_SIZE;
    Vec4::new(
        target.x.clamp(-reach, reach),
        target
            .y
            .clamp(PHYSICS_FLOOR_Y + BODY_SIZE, ARENA_TOP - BODY_SIZE),
        target.z.clamp(-reach, reach),
        current.w,
    )
}

pub(crate) fn release(
    physics: &mut Physics<EuclideanR4>,
    entity: Entity,
    pointer: [f32; 3],
    rope: bool,
) -> Result<(), EditError> {
    let body = *physics
        .world()
        .body(physics.body(entity).ok_or(EditError::StaleHandle)?)
        .ok_or(EditError::StaleHandle)?;
    let mut linear = body.velocity;
    let mut angular = body.angular_velocity;
    if !rope {
        let flick = Vec4::new(pointer[0], pointer[1], pointer[2], 0.0);
        let throw = flick * RELEASE_GAIN;
        linear = throw * (MAX_RELEASE_SPEED / throw.length().max(MAX_RELEASE_SPEED))
            + Vec4::W * body.velocity.w;
        let lever = physics
            .released_anchor(entity)
            .and_then(|handle| (handle - body.position).truncate().try_normalize());
        if let Some(lever) = lever {
            angular = angular
                + Bivector4::wedge(lever.extend(0.0), flick) * (RELEASE_SPIN_GAIN / BODY_SIZE);
        }
    }
    let speed = angular.magnitude();
    if speed > MAX_ANGULAR_SPEED {
        angular = angular * (MAX_ANGULAR_SPEED / speed);
    }
    physics.set_velocity(entity, linear, angular)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct DepthBand {
    pub(crate) entity: Entity,
    pub(crate) index: usize,
    pub(crate) center: f32,
    pub(crate) min: f32,
    pub(crate) max: f32,
    pub(crate) color: [f32; 3],
    pub(crate) label: &'static str,
    pub(crate) asleep: bool,
}

fn depth_bounds(position: Vec4, rotation: Rotor4, vertices: &[Vec4]) -> (f32, f32) {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for vertex in vertices {
        let w = (position + rotation.apply(*vertex)).w;
        min = min.min(w);
        max = max.max(w);
    }
    (min, max)
}

pub(crate) fn fill_depth_bands(
    session: &Session<Playground>,
    domain: DomainHandle<EuclideanR4>,
    out: &mut Vec<DepthBand>,
) {
    out.clear();
    for (entity, _) in session.app.toys.iter() {
        let Some(slot) = session.app.slots.get(entity) else {
            continue;
        };
        out.push(DepthBand {
            entity,
            index: slot.index,
            center: 0.0,
            min: 0.0,
            max: 0.0,
            color: slot.entry.body_color,
            label: slot.entry.label,
            asleep: false,
        });
    }
    let Ok(r4) = session.domains().read(domain) else {
        out.clear();
        return;
    };
    let Some(physics) = r4.physics() else {
        out.clear();
        return;
    };
    out.retain_mut(|band| {
        let Some(id) = physics.body(band.entity) else {
            return false;
        };
        let Some(body) = physics.world().body(id) else {
            return false;
        };
        let Some(Collider::ConvexPolytope4D { vertices }) = physics.world().collider(body) else {
            return false;
        };
        band.center = body.position.w;
        (band.min, band.max) = depth_bounds(body.position, body.orientation.rotation, vertices);
        band.asleep = body.is_sleeping();
        true
    });
    out.sort_by_key(|band| band.index);
}

fn arena() -> [(Vec4, f32, f32); 8] {
    [
        (Vec4::Y, PHYSICS_FLOOR_Y, BODY_RESTITUTION),
        (-Vec4::Y, -ARENA_TOP, WALL_RESTITUTION),
        (Vec4::X, -ARENA_HALF, WALL_RESTITUTION),
        (-Vec4::X, -ARENA_HALF, WALL_RESTITUTION),
        (Vec4::Z, -ARENA_HALF, WALL_RESTITUTION),
        (-Vec4::Z, -ARENA_HALF, WALL_RESTITUTION),
        (Vec4::W, -ARENA_HALF, WALL_RESTITUTION),
        (-Vec4::W, -ARENA_HALF, WALL_RESTITUTION),
    ]
}

fn face_down(polytope: Polytope4) -> Rotor4 {
    let topology = polytope.topology();
    let mut centroid = Vec4::ZERO;
    for index in topology.cells[0] {
        centroid += topology.vertices[*index as usize];
    }
    Rotor4::from_rotation_arc(centroid.normalize(), -Vec4::Y)
}

pub(crate) fn pose_at(polytope: Polytope4, x: f32) -> Pose<EuclideanR4> {
    let frame = face_down(polytope);
    let lowest = polytope
        .topology()
        .vertices
        .iter()
        .map(|vertex| frame.apply(*vertex * BODY_SIZE).y)
        .fold(f32::INFINITY, f32::min);
    Pose {
        point: Vec4::new(x, SPAWN_CLEARANCE - lowest, 0.0, 0.0),
        frame,
    }
}

fn asset_of(polytope: Polytope4, cards: &[Card]) -> Option<(ShapeEntry, Instance)> {
    SHAPE_CATALOG
        .iter()
        .position(|entry| entry.collider_polytope() == Some(polytope))
        .and_then(|index| Some((SHAPE_CATALOG[index], cards.get(index)?.toy()?)))
}

pub(crate) fn add_body(
    dispatch: &mut Dispatch<'_, Playground>,
    domain: DomainHandle<EuclideanR4>,
    entity: Entity,
    polytope: Polytope4,
    pose: Pose<EuclideanR4>,
) -> Result<(), Rejection> {
    let vertices = polytope
        .topology()
        .vertices
        .iter()
        .map(|vertex| *vertex * BODY_SIZE)
        .collect();
    let body = polytope_body_r4(pose.point, Vec4::ZERO, vertices, BODY_MASS)
        .ok_or(Rejection::Unsupported("invalid toy body"))?;
    let domain = dispatch.domains.typed(domain)?;
    domain.spawn_body(entity, body)?;
    let physics = domain
        .physics_mut()
        .ok_or(Rejection::Unsupported("the domain has no physics"))?;
    physics.set_mass_properties(
        entity,
        BODY_MASS,
        regular_polytope4_inertia(polytope, BODY_MASS, BODY_SIZE),
    )?;
    physics.set_restitution(entity, BODY_RESTITUTION)?;
    Ok(())
}

fn spawn_toy(
    dispatch: &mut Dispatch<'_, Playground>,
    domain: DomainHandle<EuclideanR4>,
    polytope: Polytope4,
    entry: ShapeEntry,
    instance: Instance,
    at: usize,
    count: usize,
) -> Result<(), Rejection> {
    let centre = count.saturating_sub(1) as f32 * 0.5;
    let pose = pose_at(polytope, (at as f32 - centre) * SPAWN_SPACING);
    let entity = dispatch.spawn(
        SpawnBundle::new()
            .at(domain, pose)
            .instance(instance)
            .row(Slot {
                index: at,
                entry,
                rest: pose.point,
            })
            .row(Toy {
                rest_anchor: pose.point,
                rest_time: 0.0,
            }),
    )?;
    add_body(dispatch, domain, entity, polytope, pose)
}

fn spawn_walls(
    dispatch: &mut Dispatch<'_, Playground>,
    domain: DomainHandle<EuclideanR4>,
) -> Result<(), Rejection> {
    for (normal, offset, restitution) in arena() {
        let plane = halfspace4_body_r4(normal, offset)
            .ok_or(Rejection::Unsupported("invalid arena plane"))?;
        let entity = dispatch.spawn(
            SpawnBundle::new()
                .at(domain, Pose::at(Vec4::ZERO))
                .row(Wall),
        )?;
        let domain = dispatch.domains.typed(domain)?;
        domain.spawn_body(entity, plane)?;
        domain
            .physics_mut()
            .ok_or(Rejection::Unsupported("the domain has no physics"))?
            .set_restitution(entity, restitution)?;
    }
    Ok(())
}

pub(crate) fn populate(
    dispatch: &mut Dispatch<'_, Playground>,
    domain: DomainHandle<EuclideanR4>,
    cards: &[Card],
) -> Result<(), Rejection> {
    let visible: Vec<(Entity, Slot)> = dispatch
        .app
        .slots
        .iter()
        .map(|(entity, slot)| (entity, *slot))
        .collect();
    for (entity, slot) in visible {
        let instance = dispatch
            .domains
            .typed(domain)?
            .instances()
            .get(entity)
            .copied();
        if instance.is_some() {
            dispatch.domains.typed(domain)?.remove_instance(entity)?;
        }
        dispatch.app.slots.remove(entity)?;
        dispatch.attach(entity, HiddenSlot { slot, instance })?;
    }

    for (at, polytope) in TOYS.into_iter().enumerate() {
        let (entry, instance) = asset_of(polytope, cards)
            .ok_or(Rejection::Unsupported("the toy has no prepared asset"))?;
        spawn_toy(dispatch, domain, polytope, entry, instance, at, TOYS.len())?;
    }
    spawn_walls(dispatch, domain)
}

pub(crate) fn clear(
    dispatch: &mut Dispatch<'_, Playground>,
    domain: DomainHandle<EuclideanR4>,
) -> Result<(), Rejection> {
    let spawned: Vec<Entity> = dispatch
        .app
        .toys
        .iter()
        .map(|(entity, _)| entity)
        .chain(dispatch.app.walls.iter().map(|(entity, _)| entity))
        .collect();
    for entity in spawned {
        dispatch.despawn(entity)?;
    }

    let hidden: Vec<(Entity, HiddenSlot)> = dispatch
        .app
        .hidden
        .iter()
        .map(|(entity, row)| (entity, *row))
        .collect();
    for (entity, row) in hidden {
        dispatch.app.hidden.remove(entity)?;
        dispatch.attach(entity, row.slot)?;
        if let Some(instance) = row.instance {
            dispatch.attach_instance(domain, entity, instance)?;
        }
    }
    Ok(())
}

pub(crate) fn reset(
    dispatch: &mut Dispatch<'_, Playground>,
    domain: DomainHandle<EuclideanR4>,
    cards: &[Card],
) -> Result<(), Rejection> {
    clear(dispatch, domain)?;
    populate(dispatch, domain, cards)
}

pub(crate) fn settle(
    app: &mut Playground,
    domains: &mut Domains,
    domain: DomainHandle<EuclideanR4>,
    step: Step,
) -> Result<(), DomainError> {
    let r4 = domains.typed(domain)?;
    let physics = r4
        .physics_mut()
        .ok_or(DomainError::Unsupported("physics on r4"))?;
    let decay = (-ANGULAR_DAMPING * step.dt).exp();
    for (entity, toy) in app.toys.iter_mut() {
        let Some(id) = physics.body(entity) else {
            continue;
        };
        let held = physics.is_held(entity);
        let Some(mut body) = physics.world().body(id).copied() else {
            continue;
        };
        let angular = body.angular_velocity * decay;
        if angular != body.angular_velocity {
            physics.set_velocity(entity, body.velocity, angular)?;
            body.angular_velocity = angular;
        }
        if held {
            toy.rest_anchor = body.position;
            toy.rest_time = 0.0;
            continue;
        }
        if !body.is_sleeping() && toy.rest_time >= REST_WINDOW {
            toy.rest_anchor = body.position;
            toy.rest_time = 0.0;
        }
        let travelled = (body.position - toy.rest_anchor).length();
        let moving = body.velocity.length() > REST_SPEED
            || body.angular_velocity.magnitude() > REST_ANGULAR_SPEED;
        if travelled > REST_TRAVEL || moving {
            toy.rest_anchor = body.position;
            toy.rest_time = 0.0;
        } else {
            toy.rest_time += step.dt;
            if toy.rest_time >= REST_WINDOW {
                physics.sleep_body(entity)?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_drag_target_cannot_pull_the_toy_hull_through_the_arena() {
        let current = Vec4::new(0.0, 1.0, 0.0, 2.0);
        let rotation =
            Rotor4::from_rotation_arc(Vec4::X, Vec4::new(1.0, 1.0, 0.0, 1.0).normalize());
        for far in [
            Vec4::new(99.0, -99.0, -99.0, -8.0),
            Vec4::new(-99.0, 99.0, 99.0, 8.0),
        ] {
            let target = clamp_target_to_arena(current, far);
            for vertex in Polytope4::Tesseract.topology().vertices {
                let point = target + rotation.apply(*vertex * BODY_SIZE);
                assert!(point.x.abs() <= ARENA_HALF + 1e-6);
                assert!(point.z.abs() <= ARENA_HALF + 1e-6);
                assert!(point.y >= FLOOR_Y - 1e-6);
                assert!(point.y <= ARENA_TOP + 1e-6);
            }
            assert_eq!(target.w, current.w);
        }
    }

    #[test]
    fn a_depth_band_uses_the_oriented_hull_w_extent() {
        let position = Vec4::new(0.0, 0.0, 0.0, 0.7);
        let vertices = [Vec4::X * BODY_SIZE, -Vec4::X * BODY_SIZE];
        let flat = depth_bounds(position, Rotor4::IDENTITY, &vertices);
        let turned = depth_bounds(
            position,
            Rotor4::from_rotation_arc(Vec4::X, Vec4::W),
            &vertices,
        );
        assert_eq!(flat, (position.w, position.w));
        assert!((turned.0 - (position.w - BODY_SIZE)).abs() < 1e-6);
        assert!((turned.1 - (position.w + BODY_SIZE)).abs() < 1e-6);
    }
}
