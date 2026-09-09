use glam::Vec4;
use loam_math::{EuclideanR4, Iso4Flat};
use loam_physics::euclidean_r4::{halfspace4_body_r4, polytope_body_r4};
use loam_runtime::{Dispatch, DomainHandle, Entity, Pose, Rejection, SpawnBundle};

use crate::consts::{ARENA_HALF, BODY_SIZE, FLOOR_Y};
use crate::{Playground, Wall};

const BODY_MASS: f32 = 1.0;

/// Solid on the far side of each plane: the ground under y, then the four walls in x and z, then the two in w.
fn arena() -> [(Vec4, f32); 7] {
    [
        (Vec4::Y, FLOOR_Y),
        (Vec4::X, -ARENA_HALF),
        (-Vec4::X, -ARENA_HALF),
        (Vec4::Z, -ARENA_HALF),
        (-Vec4::Z, -ARENA_HALF),
        (Vec4::W, -ARENA_HALF),
        (-Vec4::W, -ARENA_HALF),
    ]
}

pub(crate) fn populate(
    dispatch: &mut Dispatch<'_, Playground>,
    domain: DomainHandle<EuclideanR4>,
) -> Result<(), Rejection> {
    let mut spawned: Vec<(Entity, Vec4, Vec<Vec4>)> = Vec::new();
    for (entity, slot) in dispatch.app.slots.iter() {
        let Some(polytope) = slot.entry.collider_polytope() else {
            continue;
        };
        let vertices = polytope
            .topology()
            .vertices
            .iter()
            .map(|vertex| *vertex * BODY_SIZE)
            .collect();
        spawned.push((entity, slot.rest, vertices));
    }
    let r4 = dispatch.domains.typed(domain)?;
    let physics = r4
        .physics_mut()
        .ok_or(Rejection::Unsupported("the domain has no physics"))?;
    for (entity, rest, vertices) in spawned {
        let body = polytope_body_r4(rest, Vec4::ZERO, vertices, BODY_MASS)
            .ok_or(Rejection::Unsupported("invalid toy body"))?;
        physics.spawn(entity, body);
    }
    for (normal, offset) in arena() {
        let plane = halfspace4_body_r4(normal, offset)
            .ok_or(Rejection::Unsupported("invalid arena plane"))?;
        let entity = dispatch.spawn(
            SpawnBundle::new()
                .at(domain, Pose(Iso4Flat::IDENTITY))
                .row(Wall),
        )?;
        dispatch
            .domains
            .typed(domain)?
            .physics_mut()
            .ok_or(Rejection::Unsupported("the domain has no physics"))?
            .spawn(entity, plane);
    }
    Ok(())
}

pub(crate) fn clear(
    dispatch: &mut Dispatch<'_, Playground>,
    domain: DomainHandle<EuclideanR4>,
) -> Result<(), Rejection> {
    let walls: Vec<Entity> = dispatch
        .app
        .walls
        .iter()
        .map(|(entity, _)| entity)
        .collect();
    for wall in walls {
        dispatch.despawn(wall)?;
    }
    let rests: Vec<(Entity, Vec4)> = dispatch
        .app
        .slots
        .iter()
        .map(|(entity, slot)| (entity, slot.rest))
        .collect();
    let r4 = dispatch.domains.typed(domain)?;
    for (entity, rest) in rests {
        if let Some(physics) = r4.physics_mut() {
            physics.despawn(entity);
        }
        if let Some(pose) = r4.poses.get_mut(entity) {
            pose.0 = Iso4Flat::from_translation(rest);
        }
    }
    Ok(())
}
