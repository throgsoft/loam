use std::any::Any;

use loam_physics::{BodyDef, BodyId, PhysicsSpace};

use super::{DomainError, Homogeneous, TypedDomain};
use crate::entity::Entity;
use crate::physics::Physics;
use crate::store::StoreError;

impl<S> TypedDomain<S>
where
    S: Homogeneous + PhysicsSpace + Copy,
{
    pub fn physics(&self) -> Option<&Physics<S>> {
        self.facilities
            .iter()
            .find_map(|facility| (facility.as_ref() as &dyn Any).downcast_ref::<Physics<S>>())
    }

    pub fn physics_mut(&mut self) -> Result<&mut Physics<S>, DomainError> {
        self.facilities
            .iter_mut()
            .find_map(|facility| (facility.as_mut() as &mut dyn Any).downcast_mut::<Physics<S>>())
            .ok_or(DomainError::NoPhysics(self.id))
    }

    pub fn spawn_body(&mut self, entity: Entity, body: BodyDef<S>) -> Result<BodyId, DomainError> {
        let pose = *self.poses.get(entity).ok_or(StoreError::Missing(entity))?;
        let physics = self
            .facilities
            .iter_mut()
            .find_map(|facility| (facility.as_mut() as &mut dyn Any).downcast_mut::<Physics<S>>())
            .ok_or(DomainError::NoPhysics(self.id))?;
        physics.check_pose(pose)?;
        let id = physics.spawn(entity, body);
        let pushed = *physics
            .world()
            .body(id)
            .expect("a body that was just pushed is live");
        let placed = physics
            .replace_pose(id, pose, &mut self.poses)
            .map_err(DomainError::from)
            .and_then(|()| {
                if pushed.is_sleeping() {
                    physics.sleep_body(entity)
                } else {
                    physics.set_velocity(entity, pushed.velocity, pushed.angular_velocity)
                }
            });
        if let Err(error) = placed {
            let _ = physics.despawn(entity);
            return Err(error);
        }
        physics.restart_rest(id);
        Ok(id)
    }
}
