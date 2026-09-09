use std::any::Any;
use std::ops::Sub;

use loam_physics::collision::VectorOps;
use loam_physics::{BodyDef, BodyId, EditError, Narrowphase, PhysicsSpace, World, WorldState};

use crate::command::{Outcome, Rejection};
use crate::domain::{
    ChartCommand, ChartPoint, ChartPose, ChartTangent, DomainBuilder, DomainError, DomainSpace,
    Facility, Pose, TypedDomain,
};
use crate::entity::{Entity, EntityKey};
use crate::phase::Step;
use crate::session::RestoreError;
use crate::store::{SchemaId, Store};

const IDENTITY_FRAME: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

/// Gravity and the narrowphase registration a domain's world is built with.
pub struct PhysicsConfig<S: PhysicsSpace> {
    pub gravity: Option<S::Vector>,
    pub register: fn(&mut Narrowphase<S>),
}

impl<S: PhysicsSpace> PhysicsConfig<S> {
    pub fn new(register: fn(&mut Narrowphase<S>)) -> Self {
        Self {
            gravity: None,
            register,
        }
    }

    pub fn gravity(mut self, gravity: S::Vector) -> Self {
        self.gravity = Some(gravity);
        self
    }
}

/// A domain's world plus the map between its entities and bodies; each step writes every dirty body's pose into its entity's row, and it claims `Place` with an identity frame, `Move` keeping the body's orientation, and `Walk` for entities that have a body.
pub struct Physics<S: PhysicsSpace> {
    world: World<S>,
    to_body: Vec<Option<(u32, BodyId)>>,
    to_entity: Vec<Option<(u32, EntityKey)>>,
}

struct PhysicsSnapshot<S: PhysicsSpace> {
    world: WorldState<S>,
    to_entity: Vec<Option<(u32, EntityKey)>>,
}

fn body_at(to_body: &[Option<(u32, BodyId)>], key: EntityKey) -> Option<BodyId> {
    match to_body.get(key.slot() as usize)? {
        Some((generation, body)) if *generation == key.generation() => Some(*body),
        _ => None,
    }
}

fn entity_at(to_entity: &[Option<(u32, EntityKey)>], body: BodyId) -> Option<EntityKey> {
    match to_entity.get(body.slot() as usize)? {
        Some((generation, key)) if *generation == body.generation() => Some(*key),
        _ => None,
    }
}

fn put<T: Copy>(slots: &mut Vec<Option<T>>, index: usize, value: Option<T>) {
    if index >= slots.len() {
        if value.is_none() {
            return;
        }
        slots.resize(index + 1, None);
    }
    slots[index] = value;
}

impl<S> Physics<S>
where
    S: DomainSpace + PhysicsSpace + Copy,
    S::Vector: VectorOps + Default,
    S::Point: Sub<Output = S::Vector>,
{
    pub fn world(&self) -> &World<S> {
        &self.world
    }

    pub fn world_mut(&mut self) -> &mut World<S> {
        &mut self.world
    }

    pub fn body(&self, entity: Entity) -> Option<BodyId> {
        body_at(&self.to_body, entity.key())
    }

    pub fn entity(&self, body: BodyId) -> Option<EntityKey> {
        entity_at(&self.to_entity, body)
    }

    /// Replaces any body the entity already has.
    pub fn spawn(&mut self, entity: Entity, body: BodyDef<S>) -> BodyId {
        let _ = self.despawn(entity);
        let id = self.world.push_body(body);
        let key = entity.key();
        put(
            &mut self.to_body,
            key.slot() as usize,
            Some((key.generation(), id)),
        );
        put(
            &mut self.to_entity,
            id.slot() as usize,
            Some((id.generation(), key)),
        );
        id
    }

    pub fn despawn(&mut self, entity: Entity) -> Option<BodyId> {
        let id = self.body(entity)?;
        let _ = self.world.despawn_body(id);
        put(&mut self.to_body, entity.key().slot() as usize, None);
        put(&mut self.to_entity, id.slot() as usize, None);
        Some(id)
    }

    fn place(&mut self, id: BodyId, pose: &ChartPose) -> Result<Outcome, Rejection> {
        if pose.frame != IDENTITY_FRAME {
            return Err(DomainError::Unsupported("chart pose frame").into());
        }
        let orientation = self
            .world
            .bodies
            .get(id)
            .ok_or(Rejection::Edit(EditError::StaleHandle))?
            .orientation;
        let position = self.world.space.local_point(pose.coordinates);
        self.world
            .set_pose(id, position, orientation)
            .map_err(Rejection::Edit)?;
        Ok(Outcome::Done)
    }

    fn move_to(&mut self, id: BodyId, point: &ChartPoint) -> Result<Outcome, Rejection> {
        let orientation = self
            .world
            .bodies
            .get(id)
            .ok_or(Rejection::Edit(EditError::StaleHandle))?
            .orientation;
        let position = self.world.space.local_point(point.coordinates);
        self.world
            .set_pose(id, position, orientation)
            .map_err(Rejection::Edit)?;
        Ok(Outcome::Done)
    }

    fn walk(&mut self, id: BodyId, tangent: &ChartTangent, dt: f32) -> Result<Outcome, Rejection> {
        let space = self.world.space;
        let row = self
            .world
            .bodies
            .get(id)
            .ok_or(Rejection::Edit(EditError::StaleHandle))?;
        let (position, orientation) = (row.position, row.orientation);
        let velocity = space.log(space.origin(), space.local_point(tangent.vector));
        let to = space.exp(position, velocity * dt);
        self.world
            .set_pose(id, to, orientation)
            .map_err(Rejection::Edit)?;
        Ok(Outcome::Done)
    }
}

impl<S> Facility<S> for Physics<S>
where
    S: DomainSpace + PhysicsSpace + Copy,
    S::Vector: VectorOps + Default,
    S::Point: Sub<Output = S::Vector>,
{
    fn name(&self) -> &'static str {
        "physics"
    }

    fn step(&mut self, poses: &mut Store<Pose<S>>, step: Step) -> Result<(), DomainError> {
        let space = self.world.space;
        let scene = poses.scene();
        self.world.step(step.dt);
        let to_entity = &self.to_entity;
        for (id, row) in self.world.drain_dirty() {
            let Some(key) = entity_at(to_entity, id) else {
                continue;
            };
            let Some(pose) = poses.get_mut(Entity::new(scene, key)) else {
                continue;
            };
            pose.0 = space.iso_compose(space.transvection(row.position), row.orientation);
        }
        Ok(())
    }

    fn snapshot(&self) -> Box<dyn Any + Send> {
        Box::new(PhysicsSnapshot::<S> {
            world: self.world.snapshot(),
            to_entity: self.to_entity.clone(),
        })
    }

    fn restore(&mut self, from: &(dyn Any + Send)) -> Result<(), RestoreError> {
        let from = from
            .downcast_ref::<PhysicsSnapshot<S>>()
            .ok_or_else(|| RestoreError::Schema(SchemaId::of::<PhysicsSnapshot<S>>()))?;
        self.world
            .restore(&from.world)
            .map_err(RestoreError::Edit)?;
        self.to_entity.clear();
        self.to_entity.extend_from_slice(&from.to_entity);
        for slot in &mut self.to_body {
            *slot = None;
        }
        for dense in 0..self.world.bodies.len() {
            let id = self.world.bodies.id_at(dense);
            let Some(key) = entity_at(&self.to_entity, id) else {
                continue;
            };
            put(
                &mut self.to_body,
                key.slot() as usize,
                Some((key.generation(), id)),
            );
        }
        Ok(())
    }

    fn release(&mut self, entity: Entity) {
        let _ = self.despawn(entity);
    }

    fn apply(&mut self, command: &ChartCommand) -> Option<Result<Outcome, Rejection>> {
        match command {
            ChartCommand::Place { entity, pose } => {
                let id = self.body(*entity)?;
                Some(self.place(id, pose))
            }
            ChartCommand::Walk {
                entity,
                tangent,
                dt,
            } => {
                let id = self.body(*entity)?;
                Some(self.walk(id, tangent, *dt))
            }
            ChartCommand::Move { entity, point } => {
                let id = self.body(*entity)?;
                Some(self.move_to(id, point))
            }
            _ => None,
        }
    }
}

impl<S> DomainBuilder<S>
where
    S: DomainSpace + PhysicsSpace + Copy,
    S::Vector: VectorOps + Default,
    S::Point: Sub<Output = S::Vector>,
{
    /// Only a space that is also a `PhysicsSpace` can name this, so a domain over any other space never builds a world.
    pub fn physics(self, config: PhysicsConfig<S>) -> Self {
        let mut world = World::new(self.space);
        world.gravity = config.gravity;
        (config.register)(&mut world.narrowphase);
        self.facility(Physics {
            world,
            to_body: Vec::new(),
            to_entity: Vec::new(),
        })
    }
}

impl<S> TypedDomain<S>
where
    S: DomainSpace + PhysicsSpace + Copy,
    S::Vector: VectorOps + Default,
    S::Point: Sub<Output = S::Vector>,
{
    pub fn physics(&self) -> Option<&Physics<S>> {
        self.facilities
            .iter()
            .find_map(|facility| (facility.as_ref() as &dyn Any).downcast_ref::<Physics<S>>())
    }

    pub fn physics_mut(&mut self) -> Option<&mut Physics<S>> {
        self.facilities
            .iter_mut()
            .find_map(|facility| (facility.as_mut() as &mut dyn Any).downcast_mut::<Physics<S>>())
    }
}

#[cfg(test)]
mod tests {
    use loam_math::{EuclideanR4, Iso4Flat};
    use loam_physics::euclidean_r4::{register_default_narrowphase, sphere_body_r4};

    use super::*;
    use crate::domain::{Domain, DomainId};
    use crate::entity::{Entities, Epoch, RuntimeId, SceneId};
    use crate::phase::Tick;
    use crate::store::tests::alloc_probe::bytes_allocated_by;
    use crate::store::DEFAULT_LOG_CAPACITY;
    use crate::view::Vec4;

    const MOVING: usize = 64;
    const WARM_STEPS: u64 = 8;

    fn step_at(tick: u64) -> Step {
        Step {
            tick: Tick(tick),
            dt: 1.0 / 60.0,
        }
    }

    #[test]
    fn a_warmed_step_with_sixty_four_moving_bodies_allocates_nothing() {
        let mut entities = Entities::new(SceneId {
            runtime: RuntimeId::allocate(),
            epoch: Epoch::default(),
        });
        let mut domain = DomainBuilder::new("r4", EuclideanR4)
            .tracked(DEFAULT_LOG_CAPACITY)
            .physics(PhysicsConfig::new(register_default_narrowphase).gravity(Vec4::NEG_Y))
            .build(DomainId::new(0), entities.scene());
        for index in 0..MOVING {
            let at = Vec4::new(index as f32 * 4.0, 10.0, 0.0, 0.0);
            let entity = entities.spawn();
            domain
                .poses
                .insert(entity, Pose(Iso4Flat::from_translation(at)))
                .expect("pose row");
            domain.physics_mut().expect("physics").spawn(
                entity,
                sphere_body_r4(at, Vec4::ZERO, 0.5, 1.0).expect("body"),
            );
        }
        for tick in 0..WARM_STEPS {
            domain.step(step_at(tick)).expect("warm step");
        }

        let bytes = bytes_allocated_by(|| {
            domain.step(step_at(WARM_STEPS)).expect("step");
        });
        assert_eq!(bytes, 0, "a warmed physics step allocated {bytes} bytes");
    }
}
