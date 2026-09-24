use std::any::Any;
use std::ops::Sub;

use loam_math::Bivector;
use loam_physics::collision::VectorOps;
use loam_physics::{
    BodyDef, BodyId, Collider, EditError, Narrowphase, PhysicsSpace, RigidBody, World, WorldState,
};

use crate::command::{Outcome, Rejection};
use crate::domain::{
    ChartCommand, ChartPose, DomainBuilder, DomainError, Facility, Homogeneous, Pose, TypedDomain,
};
use crate::entity::{Entity, EntityKey, SceneId};
use crate::phase::Step;
use crate::session::RestoreError;
use crate::store::{Owner, SchemaId, Store};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GrabHold {
    /// Pulls the body's center, so carrying adds no spin.
    #[default]
    Center,
    /// Pulls the grabbed point, so the body swings from it.
    Anchor,
}

#[derive(Clone, Copy)]
pub struct GrabConfig<S: PhysicsSpace> {
    stiffness: f32,
    max_carry_speed: f32,
    max_acceleration: f32,
    hold: GrabHold,
    anchor_point: fn(S::Point, S::Point) -> S::Point,
    constrain_target: fn(S::Point, S::Point) -> S::Point,
}

impl<S: PhysicsSpace> GrabConfig<S> {
    pub fn new(stiffness: f32, max_carry_speed: f32, max_acceleration: f32) -> Self {
        Self {
            stiffness,
            max_carry_speed,
            max_acceleration,
            hold: GrabHold::Center,
            anchor_point: |_, point| point,
            constrain_target: |_, point| point,
        }
    }

    pub fn hold(mut self, hold: GrabHold) -> Self {
        self.hold = hold;
        self
    }

    pub fn anchor_point(mut self, anchor_point: fn(S::Point, S::Point) -> S::Point) -> Self {
        self.anchor_point = anchor_point;
        self
    }

    pub fn constrain_target(
        mut self,
        constrain_target: fn(S::Point, S::Point) -> S::Point,
    ) -> Self {
        self.constrain_target = constrain_target;
        self
    }
}

pub struct PhysicsConfig<S: PhysicsSpace> {
    gravity: Option<S::Vector>,
    register: fn(&mut Narrowphase<S>),
    min_substeps: u32,
    max_substeps: u32,
    grab: Option<GrabConfig<S>>,
    max_step_travel: Option<f32>,
    motion_speed: fn(&RigidBody<S>) -> f32,
}

fn linear_speed<S>(body: &RigidBody<S>) -> f32
where
    S: PhysicsSpace,
    S::Vector: VectorOps,
{
    body.velocity.length()
}

impl<S> PhysicsConfig<S>
where
    S: PhysicsSpace,
    S::Vector: VectorOps,
{
    pub fn new(register: fn(&mut Narrowphase<S>)) -> Self {
        Self {
            gravity: None,
            register,
            min_substeps: 1,
            max_substeps: 1,
            grab: None,
            max_step_travel: None,
            motion_speed: linear_speed::<S>,
        }
    }

    pub fn gravity(mut self, gravity: S::Vector) -> Self {
        self.gravity = Some(gravity);
        self
    }

    pub fn substeps(mut self, substeps: u32) -> Self {
        self.min_substeps = substeps.max(1);
        self.max_substeps = self.min_substeps;
        self
    }

    pub fn adaptive_substeps(
        mut self,
        minimum: u32,
        maximum: u32,
        max_step_travel: f32,
        motion_speed: fn(&RigidBody<S>) -> f32,
    ) -> Self {
        self.min_substeps = minimum.max(1);
        self.max_substeps = maximum.max(self.min_substeps);
        self.max_step_travel = Some(max_step_travel);
        self.motion_speed = motion_speed;
        self
    }

    pub fn grab(mut self, grab: GrabConfig<S>) -> Self {
        self.grab = Some(grab);
        self
    }

    fn validate(&self, space: &S) -> Result<(), EditError> {
        if self
            .gravity
            .is_some_and(|gravity| !space.valid_vector(gravity))
        {
            return Err(EditError::InvalidGravity);
        }
        if let Some(grab) = &self.grab {
            if !grab.stiffness.is_finite()
                || grab.stiffness < 0.0
                || grab.max_carry_speed.is_nan()
                || grab.max_carry_speed < 0.0
                || grab.max_acceleration.is_nan()
                || grab.max_acceleration < 0.0
            {
                return Err(EditError::InvalidGrabConfig);
            }
        }
        if self
            .max_step_travel
            .is_some_and(|travel| travel.is_nan() || travel <= 0.0)
        {
            return Err(EditError::InvalidStepTravel);
        }
        Ok(())
    }
}

struct Held<S: PhysicsSpace> {
    body: BodyId,
    target: S::Point,
    anchor: S::Vector,
    offset: S::Vector,
}

struct Released<S: PhysicsSpace> {
    body: BodyId,
    anchor: S::Vector,
}

impl<S: PhysicsSpace> Clone for Held<S> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<S: PhysicsSpace> Copy for Held<S> {}

impl<S: PhysicsSpace> Clone for Released<S> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<S: PhysicsSpace> Copy for Released<S> {}

pub struct Physics<S: PhysicsSpace> {
    scene: SceneId,
    world: World<S>,
    to_body: Vec<Option<(u32, BodyId)>>,
    to_entity: Vec<Option<(u32, EntityKey)>>,
    held: Option<Held<S>>,
    released: Option<Released<S>>,
    min_substeps: u32,
    max_substeps: u32,
    grab: Option<GrabConfig<S>>,
    max_step_travel: Option<f32>,
    motion_speed: fn(&RigidBody<S>) -> f32,
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

fn clamp_length<V: VectorOps>(vector: V, maximum: f32) -> V {
    let length = vector.length();
    if length > maximum {
        vector * (maximum / length)
    } else {
        vector
    }
}

fn domain_edit(entity: Entity, error: EditError) -> DomainError {
    match error {
        EditError::StaleHandle => DomainError::Stale(entity),
        error => error.into(),
    }
}

fn domain_rejection(entity: Entity, rejection: Rejection) -> DomainError {
    match rejection {
        Rejection::Domain(error) => error,
        Rejection::Edit(error) => domain_edit(entity, error),
        _ => DomainError::Unsupported("physics pose edit"),
    }
}

impl<S> Physics<S>
where
    S: Homogeneous + PhysicsSpace + Copy,
    S::Vector: VectorOps + Default,
    S::Point: Sub<Output = S::Vector>,
{
    pub fn world(&self) -> &World<S> {
        &self.world
    }

    pub fn narrowphase_mut(&mut self) -> &mut Narrowphase<S> {
        &mut self.world.narrowphase
    }

    pub fn set_solver_iterations(&mut self, iterations: usize) {
        self.world.set_solver_iterations(iterations);
    }

    /// Has no effect on a domain built without a grab.
    pub fn set_grab_hold(&mut self, hold: GrabHold) {
        if let Some(grab) = &mut self.grab {
            grab.hold = hold;
        }
    }

    pub fn set_gravity(&mut self, gravity: Option<S::Vector>) -> Result<(), EditError> {
        self.world.set_gravity(gravity)
    }

    pub fn set_solver_tolerance(&mut self, tolerance: f32) -> Result<(), EditError> {
        self.world.set_solver_tolerance(tolerance)
    }

    pub fn body(&self, entity: Entity) -> Option<BodyId> {
        if entity.scene() != self.scene {
            return None;
        }
        body_at(&self.to_body, entity.key())
    }

    pub fn entity(&self, body: BodyId) -> Option<Entity> {
        entity_at(&self.to_entity, body).map(|key| Entity::new(self.scene, key))
    }

    pub fn set_velocity(
        &mut self,
        entity: Entity,
        velocity: S::Vector,
        angular_velocity: S::AngVel,
    ) -> Result<(), EditError> {
        let id = self.body(entity).ok_or(EditError::StaleHandle)?;
        self.world.set_velocity(id, velocity, angular_velocity)
    }

    pub fn set_restitution(&mut self, entity: Entity, restitution: f32) -> Result<(), EditError> {
        let id = self.body(entity).ok_or(EditError::StaleHandle)?;
        self.world.set_restitution(id, restitution)
    }

    pub fn set_collision_filter(
        &mut self,
        entity: Entity,
        group: u32,
        mask: u32,
    ) -> Result<(), EditError> {
        let id = self.body(entity).ok_or(EditError::StaleHandle)?;
        self.world.set_collision_filter(id, group, mask)
    }

    pub fn set_mass_properties(
        &mut self,
        entity: Entity,
        mass: f32,
        inertia: S::Inertia,
    ) -> Result<(), EditError> {
        let id = self.body(entity).ok_or(EditError::StaleHandle)?;
        self.world.set_mass_properties(id, mass, inertia)
    }

    pub fn set_collider(
        &mut self,
        entity: Entity,
        collider: Collider,
        inertia: S::Inertia,
    ) -> Result<(), EditError> {
        let id = self.body(entity).ok_or(EditError::StaleHandle)?;
        self.world.set_collider(id, collider, inertia)
    }

    pub fn sleep_body(&mut self, entity: Entity) -> Result<(), EditError> {
        let id = self.body(entity).ok_or(EditError::StaleHandle)?;
        self.world.sleep_body(id)
    }

    pub fn wake_body(&mut self, entity: Entity) -> Result<(), EditError> {
        let id = self.body(entity).ok_or(EditError::StaleHandle)?;
        self.world.wake_body(id)
    }

    pub fn is_held(&self, entity: Entity) -> bool {
        self.body(entity)
            .is_some_and(|body| self.held.is_some_and(|held| held.body == body))
    }

    pub fn held_anchor(&self, entity: Entity) -> Option<S::Point> {
        let body = self.body(entity)?;
        let held = self.held.filter(|held| held.body == body)?;
        let row = self.world.body(body)?;
        let lever = self
            .world
            .space()
            .iso_transport(row.orientation, row.position, held.anchor);
        let point = self.world.space().exp(row.position, lever);
        self.world.space().valid_point(point).then_some(point)
    }

    pub fn throw(&mut self, entity: Entity, velocity: S::Vector) -> Result<(), EditError> {
        let id = self.body(entity).ok_or(EditError::StaleHandle)?;
        let angular = self
            .world
            .body(id)
            .ok_or(EditError::StaleHandle)?
            .angular_velocity;
        self.world.set_velocity(id, velocity, angular)?;
        if self.released.is_some_and(|released| released.body == id) {
            self.released = None;
        }
        Ok(())
    }

    /// The grab point of the body released last, moved with the body since.
    pub fn released_anchor(&self, entity: Entity) -> Option<S::Point> {
        let body = self.body(entity)?;
        let released = self.released.filter(|released| released.body == body)?;
        let row = self.world.body(body)?;
        let lever =
            self.world
                .space()
                .iso_transport(row.orientation, row.position, released.anchor);
        let point = self.world.space().exp(row.position, lever);
        self.world.space().valid_point(point).then_some(point)
    }

    pub(crate) fn spawn(&mut self, entity: Entity, body: BodyDef<S>) -> BodyId {
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
        if self.held.is_some_and(|held| held.body == id) {
            self.held = None;
        }
        if self.released.is_some_and(|released| released.body == id) {
            self.released = None;
        }
        let _ = self.world.despawn_body(id);
        put(&mut self.to_body, entity.key().slot() as usize, None);
        put(&mut self.to_entity, id.slot() as usize, None);
        Some(id)
    }

    fn mirror_entity(&self, id: BodyId, poses: &Store<Pose<S>>) -> Result<Entity, EditError> {
        let entity = self.entity(id).ok_or(EditError::StaleHandle)?;
        if entity.scene() != poses.scene() {
            return Err(EditError::StaleHandle);
        }
        poses.get(entity).ok_or(EditError::StaleHandle)?;
        Ok(entity)
    }

    fn mirror_pose(&self, id: BodyId, poses: &mut Store<Pose<S>>) -> Result<(), EditError> {
        let entity = self.mirror_entity(id, poses)?;
        let row = self.world.body(id).ok_or(EditError::StaleHandle)?;
        let pose = poses.get_mut(entity).ok_or(EditError::StaleHandle)?;
        *pose = self.world.space().pose_of(self.world.space().iso_compose(
            self.world.space().transvection(row.position),
            row.orientation,
        ));
        Ok(())
    }

    pub(crate) fn check_pose(&self, pose: Pose<S>) -> Result<(), EditError> {
        let (position, orientation) = self.physics_pose(pose);
        if !self.world.space().valid_point(position)
            || !self.world.space().valid_orientation(orientation)
        {
            return Err(EditError::NotFinite);
        }
        Ok(())
    }

    fn physics_pose(&self, pose: Pose<S>) -> (S::Point, S::Iso) {
        let space = *self.world.space();
        let orientation = space.iso_compose(
            space.iso_inverse(space.transvection(pose.point)),
            space.iso_of(&pose),
        );
        (pose.point, orientation)
    }

    pub(crate) fn replace_pose(
        &mut self,
        id: BodyId,
        pose: Pose<S>,
        poses: &mut Store<Pose<S>>,
    ) -> Result<(), EditError> {
        self.mirror_entity(id, poses)?;
        let (position, orientation) = self.physics_pose(pose);
        self.world.set_pose(id, position, orientation)?;
        self.mirror_pose(id, poses)
    }

    fn place(
        &mut self,
        id: BodyId,
        pose: &ChartPose,
        poses: &mut Store<Pose<S>>,
    ) -> Result<Outcome, Rejection> {
        let pose = self.world.space().pose_from_chart(pose)?;
        self.replace_pose(id, pose, poses)
            .map_err(Rejection::Edit)?;
        Ok(Outcome::Done)
    }

    fn move_to(&mut self, id: BodyId, position: S::Point) -> Result<Outcome, Rejection> {
        let orientation = self
            .world
            .body(id)
            .ok_or(Rejection::Edit(EditError::StaleHandle))?
            .orientation;
        if !self.world.space().valid_point(position) {
            return Err(Rejection::Edit(EditError::NotFinite));
        }
        if let Some(held) = &mut self.held {
            if held.body == id {
                let target = self.grab.map_or(position, |grab| {
                    (grab.constrain_target)(held.target, position)
                });
                if !self.world.space().valid_point(target) {
                    return Err(Rejection::Edit(EditError::NotFinite));
                }
                held.target = target;
                return Ok(Outcome::Done);
            }
        }
        self.world
            .set_pose(id, position, orientation)
            .map_err(Rejection::Edit)?;
        Ok(Outcome::Done)
    }

    fn move_to_and_mirror(
        &mut self,
        id: BodyId,
        point: S::Point,
        poses: &mut Store<Pose<S>>,
    ) -> Result<Outcome, Rejection> {
        self.mirror_entity(id, poses).map_err(Rejection::Edit)?;
        let outcome = self.move_to(id, point)?;
        self.mirror_pose(id, poses).map_err(Rejection::Edit)?;
        Ok(outcome)
    }

    fn grab(&mut self, id: BodyId, point: S::Point) -> Result<Outcome, Rejection> {
        let Some(grab) = self.grab else {
            return Ok(Outcome::Done);
        };
        let row = self
            .world
            .body(id)
            .ok_or(Rejection::Edit(EditError::StaleHandle))?;
        let target = row.position;
        let anchor = (grab.anchor_point)(target, point);
        if !self.world.space().valid_point(anchor) {
            return Err(Rejection::Edit(EditError::NotFinite));
        }
        let lever = self.world.space().log(target, anchor);
        let local_anchor = self.world.space().iso_transport(
            self.world.space().iso_inverse(row.orientation),
            target,
            lever,
        );
        if !self.world.space().valid_vector(local_anchor) {
            return Err(Rejection::Edit(EditError::NotFinite));
        }
        self.world
            .set_velocity(id, S::Vector::default(), S::AngVel::zero())
            .map_err(Rejection::Edit)?;
        self.released = None;
        self.held = Some(Held {
            body: id,
            target,
            anchor: local_anchor,
            offset: lever,
        });
        Ok(Outcome::Done)
    }

    fn release_grab(&mut self, id: BodyId) -> Result<Outcome, Rejection> {
        if let Some(held) = self.held.filter(|held| held.body == id) {
            self.held = None;
            self.released = Some(Released {
                body: held.body,
                anchor: held.anchor,
            });
        }
        Ok(Outcome::Done)
    }

    fn drive_held(&mut self, dt: f32) -> Result<(), DomainError> {
        let Some(held) = self.held else {
            return Ok(());
        };
        let Some(grab) = self.grab else {
            return Ok(());
        };
        let space = *self.world.space();
        let row = *self.world.body(held.body).ok_or(EditError::StaleHandle)?;
        if grab.hold == GrabHold::Center {
            let desired = clamp_length(
                space.log(row.position, held.target) * grab.stiffness,
                grab.max_carry_speed,
            );
            let acceleration = clamp_length(desired - row.velocity, grab.max_acceleration * dt);
            return self
                .world
                .set_velocity(held.body, row.velocity + acceleration, row.angular_velocity)
                .map_err(DomainError::from);
        }
        let lever = space.iso_transport(row.orientation, row.position, held.anchor);
        let point = space.exp(row.position, lever);
        let goal = space.exp(held.target, held.offset);
        let moving = space.velocity_at_point(&row, point);
        let desired = clamp_length(
            space.log(point, goal) * grab.stiffness,
            grab.max_carry_speed,
        );
        let change = clamp_length(desired - moving, grab.max_acceleration * dt);
        let size = change.length();
        if size <= 0.0 {
            return Ok(());
        }
        let direction = change * (1.0 / size);
        let mut probe = row;
        probe.apply_impulse_at_point(&space, direction, point);
        let response = (space.velocity_at_point(&probe, point) - moving).dot(direction);
        if response <= 0.0 {
            return Ok(());
        }
        self.world
            .apply_impulse_at_point(held.body, direction * (size / response), point)
            .map_err(DomainError::from)
    }

    fn step_count(&self, dt: f32) -> u32 {
        if self.min_substeps == self.max_substeps {
            return self.min_substeps;
        }
        let Some(max_step_travel) = self.max_step_travel else {
            return self.min_substeps;
        };
        let fastest = self
            .world
            .bodies()
            .iter()
            .map(|body| (self.motion_speed)(body))
            .fold(0.0f32, f32::max);
        let needed = (fastest * dt / max_step_travel).ceil() as u32;
        needed.clamp(self.min_substeps, self.max_substeps)
    }

    fn sync_poses(&mut self, poses: &mut Store<Pose<S>>) {
        let space = *self.world.space();
        let scene = poses.scene();
        let to_entity = &self.to_entity;
        for (id, row) in self.world.drain_dirty() {
            let Some(key) = entity_at(to_entity, id) else {
                continue;
            };
            let Some(pose) = poses.get_mut(Entity::new(scene, key)) else {
                continue;
            };
            *pose =
                space.pose_of(space.iso_compose(space.transvection(row.position), row.orientation));
        }
    }
}

impl<S> Facility<S> for Physics<S>
where
    S: Homogeneous + PhysicsSpace + Copy,
    S::Vector: VectorOps + Default,
    S::AngVel: PartialEq,
    S::Point: Sub<Output = S::Vector>,
{
    fn name(&self) -> &'static str {
        "physics"
    }

    fn bind(&mut self, scene: SceneId, _owner: Owner) {
        self.scene = scene;
    }

    fn step(
        &mut self,
        poses: &mut Store<Pose<S>>,
        step: Step,
        _owner: Owner,
    ) -> Result<(), DomainError> {
        if !step.dt.is_finite() || step.dt < 0.0 {
            return Err(EditError::InvalidTimeStep.into());
        }
        self.drive_held(step.dt)?;
        let substeps = self.step_count(step.dt);
        let dt = step.dt / substeps as f32;
        for _ in 0..substeps {
            self.world.step(dt)?;
        }
        self.sync_poses(poses);
        self.released = None;
        Ok(())
    }

    fn synchronize(&mut self, poses: &mut Store<Pose<S>>, _owner: Owner) {
        self.sync_poses(poses);
    }

    fn set_pose(
        &mut self,
        entity: Entity,
        pose: Pose<S>,
        poses: &mut Store<Pose<S>>,
        _owner: Owner,
    ) -> Option<Result<(), DomainError>> {
        let id = self.body(entity)?;
        Some(
            self.replace_pose(id, pose, poses)
                .map_err(|error| domain_edit(entity, error)),
        )
    }

    fn move_to(
        &mut self,
        entity: Entity,
        point: S::Point,
        poses: &mut Store<Pose<S>>,
        _owner: Owner,
    ) -> Option<Result<(), DomainError>> {
        let id = self.body(entity)?;
        Some(
            self.move_to_and_mirror(id, point, poses)
                .map(|_| ())
                .map_err(|error| domain_rejection(entity, error)),
        )
    }

    fn snapshot(&self, _owner: Owner) -> Box<dyn Any + Send> {
        Box::new(PhysicsSnapshot::<S> {
            world: self.world.snapshot(),
            to_entity: self.to_entity.clone(),
        })
    }

    fn check_restore(&self, from: &(dyn Any + Send), _owner: Owner) -> Result<(), RestoreError> {
        let from = from
            .downcast_ref::<PhysicsSnapshot<S>>()
            .ok_or_else(|| RestoreError::Schema(SchemaId::of::<PhysicsSnapshot<S>>()))?;
        self.world
            .check_restore(&from.world)
            .map_err(RestoreError::Edit)
    }

    fn restore(&mut self, from: &(dyn Any + Send), _owner: Owner) -> Result<(), RestoreError> {
        let from = from
            .downcast_ref::<PhysicsSnapshot<S>>()
            .ok_or_else(|| RestoreError::Schema(SchemaId::of::<PhysicsSnapshot<S>>()))?;
        self.world
            .restore(&from.world)
            .map_err(RestoreError::Edit)?;
        self.to_entity.clear();
        self.to_entity.extend_from_slice(&from.to_entity);
        self.held = None;
        self.released = None;
        for slot in &mut self.to_body {
            *slot = None;
        }
        for dense in 0..self.world.bodies().len() {
            let id = self.world.bodies().id_at(dense);
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

    fn release(&mut self, entity: Entity, _owner: Owner) {
        let _ = self.despawn(entity);
    }

    fn apply(
        &mut self,
        command: &ChartCommand,
        poses: &mut Store<Pose<S>>,
        _owner: Owner,
    ) -> Option<Result<Outcome, Rejection>> {
        match command {
            ChartCommand::Place { entity, pose } => {
                let id = self.body(*entity)?;
                Some(self.place(id, pose, poses))
            }
            ChartCommand::Grab { entity, point } => {
                let id = self.body(*entity)?;
                let point = match self.world.space().point_from_chart(point) {
                    Ok(point) => point,
                    Err(error) => return Some(Err(error.into())),
                };
                Some(self.grab(id, point))
            }
            ChartCommand::Release { entity } => {
                let id = self.body(*entity)?;
                Some(self.release_grab(id))
            }
            _ => None,
        }
    }
}

impl<S> DomainBuilder<S>
where
    S: Homogeneous + PhysicsSpace + Copy,
    S::Vector: VectorOps + Default,
    S::AngVel: PartialEq,
    S::Point: Sub<Output = S::Vector>,
{
    pub fn physics(self, config: PhysicsConfig<S>) -> Result<Self, EditError> {
        config.validate(&self.space)?;
        let mut world = World::new(self.space);
        world.set_gravity(config.gravity)?;
        (config.register)(&mut world.narrowphase);
        Ok(self.facility(Physics {
            scene: SceneId::UNBOUND,
            world,
            to_body: Vec::new(),
            to_entity: Vec::new(),
            held: None,
            released: None,
            min_substeps: config.min_substeps,
            max_substeps: config.max_substeps,
            grab: config.grab,
            max_step_travel: config.max_step_travel,
            motion_speed: config.motion_speed,
        }))
    }
}

impl<S> TypedDomain<S>
where
    S: Homogeneous + PhysicsSpace + Copy,
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
    use loam_math::{EuclideanR4, Rotor4};
    use loam_physics::euclidean_r4::{register_default_narrowphase, sphere_body_r4};
    use loam_time::alloc::bytes_allocated_by;

    use super::*;
    use crate::domain::{DomainId, DomainOwner};
    use crate::entity::{Entities, Epoch, RuntimeId, SceneId};
    use crate::phase::Tick;
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
    fn a_constrained_grab_does_not_drop_its_release_lever() {
        let scene = SceneId {
            runtime: RuntimeId::allocate(),
            epoch: Epoch::default(),
        };
        let mut entities = Entities::new(scene);
        let mut domain = DomainBuilder::new("r4", EuclideanR4)
            .tracked(DEFAULT_LOG_CAPACITY)
            .physics(
                PhysicsConfig::new(register_default_narrowphase).grab(
                    GrabConfig::new(20.0, 20.0, 400.0)
                        .anchor_point(|body: Vec4, picked: Vec4| {
                            Vec4::new(picked.x, picked.y, picked.z, body.w)
                        })
                        .constrain_target(|current: Vec4, target: Vec4| {
                            Vec4::new(
                                target.x.clamp(-1.0, 1.0),
                                target.y.max(0.5),
                                target.z.clamp(-1.0, 1.0),
                                current.w,
                            )
                        }),
                ),
            )
            .unwrap()
            .build(DomainId::new(0), scene);
        let entity = entities.spawn();
        let start = Vec4::new(0.0, 2.0, 0.0, 0.75);
        domain.attach_pose(entity, Pose::at(start)).expect("pose");
        let physics = domain.physics_mut().expect("physics");
        let id = physics.spawn(
            entity,
            sphere_body_r4(start, Vec4::ZERO, 0.5, 1.0).expect("body"),
        );
        let mut orientation = physics.world.body(id).expect("body").orientation;
        orientation.rotation = Rotor4::from_rotation_arc(Vec4::Y, Vec4::W);
        physics
            .world
            .set_pose(id, start, orientation)
            .expect("pose");
        physics
            .grab(id, Vec4::new(0.0, 2.3, 0.0, -4.0))
            .expect("grab");
        physics
            .move_to(id, Vec4::new(9.0, -9.0, 0.0, 7.0))
            .expect("move");
        assert_eq!(
            physics.held.expect("held").target,
            Vec4::new(1.0, 0.5, 0.0, 0.75)
        );
        physics.release_grab(id).expect("release");
        let handle = physics.released_anchor(entity).expect("released");
        assert!(
            (handle - Vec4::new(0.0, 2.3, 0.0, 0.75)).length() < 1e-5,
            "the release lost the grabbed handle: {handle}"
        );
    }

    #[test]
    fn the_grab_hold_decides_whether_an_off_center_body_swings() {
        let lowest = |hold: GrabHold| {
            let scene = SceneId {
                runtime: RuntimeId::allocate(),
                epoch: Epoch::default(),
            };
            let mut entities = Entities::new(scene);
            let mut domain = DomainBuilder::new("r4", EuclideanR4)
                .tracked(DEFAULT_LOG_CAPACITY)
                .physics(
                    PhysicsConfig::new(register_default_narrowphase)
                        .gravity(Vec4::NEG_Y * 9.8)
                        .substeps(4)
                        .grab(GrabConfig::new(20.0, 20.0, 400.0).hold(hold)),
                )
                .unwrap()
                .build(DomainId::new(0), scene);
            let entity = entities.spawn();
            domain
                .attach_pose(entity, Pose::at(Vec4::ZERO))
                .expect("pose");
            let physics = domain.physics_mut().expect("physics");
            let id = physics.spawn(
                entity,
                sphere_body_r4(Vec4::ZERO, Vec4::ZERO, 0.5, 1.0).expect("body"),
            );
            physics.grab(id, Vec4::X * 0.4).expect("grab");
            let mut lowest = f32::INFINITY;
            for tick in 0..120 {
                domain.step(step_at(tick)).expect("step");
                let physics = domain.physics().expect("physics");
                lowest = lowest.min(physics.world.body(id).expect("body").position.y);
            }
            let grip = domain
                .physics()
                .expect("physics")
                .held_anchor(entity)
                .expect("held");
            assert!(
                (grip - Vec4::X * 0.4).length() < 0.1,
                "the {hold:?} hold let the grab point drift to {grip}"
            );
            lowest
        };
        let swung = lowest(GrabHold::Anchor);
        assert!(
            swung < -0.3,
            "the anchor hold never swung the body under its grab point; its center bottomed out at y {swung}"
        );
        let carried = lowest(GrabHold::Center);
        assert!(
            carried > -0.05,
            "the center hold let the body swing down to y {carried}"
        );
    }

    #[test]
    fn a_held_body_transfers_motion_through_sleeping_contacts() {
        let scene = SceneId {
            runtime: RuntimeId::allocate(),
            epoch: Epoch::default(),
        };
        let mut entities = Entities::new(scene);
        let mut domain = DomainBuilder::new("r4", EuclideanR4)
            .tracked(DEFAULT_LOG_CAPACITY)
            .physics(
                PhysicsConfig::new(register_default_narrowphase)
                    .substeps(4)
                    .grab(GrabConfig::new(20.0, 20.0, 400.0)),
            )
            .unwrap()
            .build(DomainId::new(0), scene);
        let held_at = Vec4::ZERO;
        let far_neighbor_at = Vec4::X * 2.0;
        let [held_body, neighbor_body, far_neighbor_body] = [0.0, 1.0, 2.0].map(|x| {
            let entity = entities.spawn();
            let at = Vec4::X * x;
            domain.attach_pose(entity, Pose::at(at)).expect("pose");
            domain.physics_mut().expect("physics").spawn(
                entity,
                sphere_body_r4(at, Vec4::ZERO, 0.5, 1.0).expect("body"),
            )
        });
        let physics = domain.physics_mut().expect("physics");
        physics.world.sleep_body(neighbor_body).expect("sleep");
        physics.world.sleep_body(far_neighbor_body).expect("sleep");
        physics.grab(held_body, held_at).expect("grab");
        physics.move_to(held_body, Vec4::X * 2.0).expect("move");

        for tick in 0..4 {
            domain.step(step_at(tick)).expect("step");
        }

        let physics = domain.physics().expect("physics");
        let neighbor = physics.world.body(neighbor_body).expect("neighbor");
        let far_neighbor = physics.world.body(far_neighbor_body).expect("far neighbor");
        assert!(
            !neighbor.is_sleeping() && !far_neighbor.is_sleeping(),
            "the struck chain stayed asleep"
        );
        assert!(
            far_neighbor.position.x > far_neighbor_at.x || far_neighbor.velocity.x > 0.1,
            "the held body transferred no motion through the chain: position {} velocity {}",
            far_neighbor.position.x,
            far_neighbor.velocity.x
        );
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
            .unwrap()
            .build(DomainId::new(0), entities.scene());
        for index in 0..MOVING {
            let at = Vec4::new(index as f32 * 4.0, 10.0, 0.0, 0.0);
            let entity = entities.spawn();
            domain.attach_pose(entity, Pose::at(at)).expect("pose row");
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
        })
        .expect("the counting allocator is installed");
        assert_eq!(bytes, 0, "a warmed physics step allocated {bytes} bytes");
    }

    #[test]
    fn invalid_physics_configuration_is_refused_before_installation() {
        let invalid_gravity =
            PhysicsConfig::new(register_default_narrowphase).gravity(Vec4::splat(f32::NAN));
        assert_eq!(
            DomainBuilder::new("r4", EuclideanR4)
                .physics(invalid_gravity)
                .err(),
            Some(EditError::InvalidGravity)
        );

        let invalid_grab = PhysicsConfig::new(register_default_narrowphase).grab(GrabConfig::new(
            f32::NAN,
            1.0,
            1.0,
        ));
        assert_eq!(
            DomainBuilder::new("r4", EuclideanR4)
                .physics(invalid_grab)
                .err(),
            Some(EditError::InvalidGrabConfig)
        );

        let invalid_travel = PhysicsConfig::new(register_default_narrowphase).adaptive_substeps(
            1,
            4,
            0.0,
            linear_speed::<EuclideanR4>,
        );
        assert_eq!(
            DomainBuilder::new("r4", EuclideanR4)
                .physics(invalid_travel)
                .err(),
            Some(EditError::InvalidStepTravel)
        );
    }
}
