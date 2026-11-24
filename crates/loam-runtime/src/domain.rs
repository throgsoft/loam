use std::any::Any;
use std::fmt;
use std::marker::PhantomData;

pub use loam_shape::field::FieldKind;

use loam_math::Space;
use loam_time::par::Executor;

use crate::assets::{Library, PreparedGeometry};
use crate::command::{Outcome, Rejection};
use crate::entity::{Entity, RuntimeId, SceneId};
use crate::field::FieldCost;
use crate::phase::Step;
use crate::publication::Stamp;
use crate::session::RestoreError;
use crate::store::{Owner, Store, StoreError, StoreField, StoreSnapshot};
use crate::view::{
    ImageRay, ImageSpaceId, Pick, Rigid, ViewId, ViewRecords, ViewSpec, ViewStyle, ViewSummary,
    ViewTarget, Views,
};

mod blend;
mod blended;
mod builder;
mod chart;
mod euclidean_r3;
mod euclidean_r4;
mod facility;
mod fields;
mod hyperbolic_h3;
mod instance;
#[cfg(feature = "physics")]
mod physics;
mod pick;
mod publish;
mod registry;
mod space;
mod views;

pub(crate) use blend::presented_alpha;
pub use builder::DomainBuilder;
pub use chart::{ChartCommand, ChartId, ChartPoint, ChartPose, ChartTangent};
pub use facility::Facility;
pub use fields::{Field, FieldProgram};
pub use instance::{EdgeShading, Instance};
pub use registry::Domains;
pub use space::{DomainSpace, Homogeneous};

use blend::Blend;
use fields::Fields;
use views::{ViewEntry, ViewTable};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DomainId(u32);

impl DomainId {
    pub(crate) fn new(index: usize) -> Self {
        Self(index as u32)
    }

    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for DomainId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "domain {}", self.0)
    }
}

pub struct DomainHandle<S> {
    id: DomainId,
    runtime: RuntimeId,
    space: PhantomData<fn() -> S>,
}

impl<S> Clone for DomainHandle<S> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<S> Copy for DomainHandle<S> {}

impl<S> fmt::Debug for DomainHandle<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DomainHandle")
            .field("id", &self.id)
            .field("runtime", &self.runtime)
            .finish()
    }
}

impl<S> DomainHandle<S> {
    pub fn id(self) -> DomainId {
        self.id
    }

    pub fn runtime(self) -> RuntimeId {
        self.runtime
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DomainError {
    ForeignRuntime,
    UnknownDomain(DomainId),
    UnknownDomainName(&'static str),
    AmbiguousDomainName(&'static str),
    UnknownView(ViewId),
    SpaceMismatch(DomainId),
    Stale(Entity),
    Store(StoreError),
    InvalidCoordinate(&'static str),
    InvalidFrame,
    ChartBoundary,
    NoConvergence,
    ErrorBudget,
    #[cfg(feature = "physics")]
    Physics(loam_physics::EditError),
    Unsupported(&'static str),
    FieldCycle(Entity),
    FieldArity(Entity),
    Restore(RestoreError),
}

impl fmt::Display for DomainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ForeignRuntime => f.write_str("the handle belongs to another runtime"),
            Self::UnknownDomain(domain) => write!(f, "no {domain} in this session"),
            Self::UnknownDomainName(name) => write!(f, "no domain named {name}"),
            Self::AmbiguousDomainName(name) => {
                write!(f, "more than one domain named {name}")
            }
            Self::UnknownView(view) => write!(f, "no {view} in this session"),
            Self::SpaceMismatch(domain) => write!(f, "{domain} has another space"),
            Self::Stale(entity) => write!(f, "{entity} is stale"),
            Self::Store(error) => fmt::Display::fmt(error, f),
            Self::InvalidCoordinate(coordinate) => write!(f, "invalid {coordinate}"),
            Self::InvalidFrame => f.write_str("the frame is not orthonormal"),
            Self::ChartBoundary => f.write_str("the step left the chart"),
            Self::NoConvergence => f.write_str("the transport did not converge"),
            Self::ErrorBudget => f.write_str("the metric error budget was exceeded"),
            #[cfg(feature = "physics")]
            Self::Physics(error) => fmt::Display::fmt(error, f),
            Self::Unsupported(what) => write!(f, "unsupported: {what}"),
            Self::FieldCycle(entity) => write!(f, "{entity} closes a field cycle"),
            Self::FieldArity(entity) => write!(f, "{entity} has the wrong operand count"),
            Self::Restore(error) => fmt::Display::fmt(error, f),
        }
    }
}

#[cfg(feature = "physics")]
impl From<loam_physics::EditError> for DomainError {
    fn from(error: loam_physics::EditError) -> Self {
        Self::Physics(error)
    }
}

impl From<StoreError> for DomainError {
    fn from(error: StoreError) -> Self {
        match error {
            StoreError::Stale(entity) => Self::Stale(entity),
            error => Self::Store(error),
        }
    }
}

fn put_row<T>(store: &mut Store<T>, entity: Entity, row: T) -> Result<(), Rejection> {
    match store.get_mut(entity) {
        Some(slot) => *slot = row,
        None => store.insert_raw(entity, row)?,
    }
    Ok(())
}

pub struct Pose<S: Space> {
    pub point: S::Point,
    pub frame: S::Frame,
}

impl<S: Space> Pose<S> {
    pub fn new(space: &S, point: S::Point) -> Self {
        Self {
            point,
            frame: space.frame_at(point),
        }
    }
}

impl<S: Space + Default> Pose<S> {
    pub fn at(point: S::Point) -> Self {
        Self::new(&S::default(), point)
    }
}

impl<S: Space> Clone for Pose<S> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<S: Space> Copy for Pose<S> {}

pub struct DomainSnapshot(pub(crate) Box<dyn Any + Send>);

struct TypedSnapshot<S: DomainSpace> {
    scene: SceneId,
    poses: StoreSnapshot<Pose<S>>,
    instances: StoreSnapshot<Instance>,
    fields: Option<StoreSnapshot<Field>>,
    facilities: Vec<Box<dyn Any + Send>>,
    views: Vec<ViewEntry<S>>,
    targets: Vec<ViewTarget>,
}

pub trait Domain: Send + 'static {
    fn id(&self) -> DomainId;

    fn name(&self) -> &'static str;

    fn tracks_changes(&self) -> bool;

    fn views(&self) -> &[ViewTarget];

    fn view(&self, id: ViewId) -> Option<ViewSummary>;

    fn pick(
        &self,
        view: ViewId,
        ray: &ImageRay,
        placement: Rigid,
        views: &Views,
        prepared: &[PreparedGeometry],
    ) -> Option<Pick>;

    fn image_of(&self, view: ViewId, entity: Entity) -> Option<[f32; 3]>;

    fn lift_origin(&self, view: ViewId, ray: &ImageRay) -> Result<ChartPoint, DomainError>;

    fn lift_tangent(
        &self,
        view: ViewId,
        at: [f32; 3],
        tangent: [f32; 3],
    ) -> Result<ChartTangent, DomainError>;

    fn compile_fields(&mut self) -> Result<FieldCost, DomainError>;

    fn field_program(&self) -> &FieldProgram;

    fn as_any(&self) -> &dyn Any;

    fn as_any_mut(&mut self) -> &mut dyn Any;
}

pub(crate) trait DomainOwner: Domain {
    fn retarget(&mut self, view: ViewId, image: ImageSpaceId) -> Result<(), DomainError>;

    fn publish(
        &self,
        view: ViewId,
        library: Library<'_>,
        into: &mut ViewRecords,
        stamp: Stamp,
        alpha: f32,
    ) -> Result<(), DomainError>;

    fn step(&mut self, step: Step, executor: &dyn Executor) -> Result<(), DomainError>;

    fn synchronize(&mut self);

    fn boundary(&mut self);

    fn release(&mut self, entity: Entity);

    fn snapshot(&self) -> DomainSnapshot;

    fn check_restore(&self, from: &DomainSnapshot) -> Result<(), RestoreError>;

    fn restore(&mut self, from: &DomainSnapshot, scene: SceneId) -> Result<(), RestoreError>;

    fn apply(&mut self, command: &ChartCommand) -> Result<Outcome, Rejection>;

    fn begin_tick(&mut self);

    fn end_tick(&mut self);

    fn present(&mut self, alpha: f32);
}

pub struct TypedDomain<S: DomainSpace> {
    id: DomainId,
    scene: SceneId,
    name: &'static str,
    space: S,
    poses: Store<Pose<S>>,
    instances: Store<Instance>,
    fields: Fields,
    views: ViewTable<S>,
    facilities: Vec<Box<dyn Facility<S>>>,
    blend: Blend<S>,
}

impl<S: DomainSpace> TypedDomain<S> {
    pub fn handle(&self) -> DomainHandle<S> {
        DomainHandle {
            id: self.id,
            runtime: self.scene.runtime,
            space: PhantomData,
        }
    }

    pub fn space(&self) -> &S {
        &self.space
    }

    pub fn poses(&self) -> &Store<Pose<S>> {
        &self.poses
    }

    pub fn instances(&self) -> &Store<Instance> {
        &self.instances
    }

    pub fn instance_mut(&mut self, entity: Entity) -> Option<&mut Instance> {
        self.instances.get_mut(entity)
    }

    pub fn remove_instance(&mut self, entity: Entity) -> Result<Instance, StoreError> {
        self.instances.remove(entity)
    }

    pub(crate) fn attach_pose(&mut self, entity: Entity, pose: Pose<S>) -> Result<(), Rejection> {
        self.check_pose(&pose)?;
        self.poses.insert_raw(entity, pose)?;
        Ok(())
    }

    pub(crate) fn attach_instance(
        &mut self,
        entity: Entity,
        instance: Instance,
    ) -> Result<(), Rejection> {
        if !self.poses.contains(entity) {
            return Err(StoreError::Missing(entity).into());
        }
        put_row(&mut self.instances, entity, instance)
    }

    pub(crate) fn attach_field(&mut self, entity: Entity, field: Field) -> Result<(), Rejection> {
        if !self.poses.contains(entity) {
            return Err(StoreError::Missing(entity).into());
        }
        self.fields.attach(entity, field)
    }

    /// Inside a tick the move is drawn blended across that tick; outside one it is drawn at once.
    pub fn set_pose(&mut self, entity: Entity, pose: Pose<S>) -> Result<(), DomainError> {
        self.check_pose(&pose)?;
        self.apply_pose(entity, pose)
    }

    /// Moves without drawing the poses in between, for a jump such as a respawn.
    pub fn teleport(&mut self, entity: Entity, pose: Pose<S>) -> Result<(), DomainError> {
        self.check_pose(&pose)?;
        self.apply_pose(entity, pose)?;
        self.blend.snap(entity, &self.poses);
        Ok(())
    }

    /// The pose a publish at `alpha` draws: blended across the last tick when only that tick wrote the pose, otherwise the current pose.
    pub fn drawn_pose(&self, entity: Entity, alpha: f32) -> Option<Pose<S>> {
        self.drawn(entity, presented_alpha(alpha))
    }

    #[inline]
    fn drawn(&self, entity: Entity, alpha: f32) -> Option<Pose<S>> {
        self.blend.drawn(&self.space, &self.poses, entity, alpha)
    }

    fn apply_pose(&mut self, entity: Entity, pose: Pose<S>) -> Result<(), DomainError> {
        if !self.poses.contains(entity) {
            return Err(StoreError::Missing(entity).into());
        }
        for facility in &mut self.facilities {
            if let Some(result) = facility.set_pose(entity, pose, &mut self.poses, Owner::new()) {
                return result;
            }
        }
        *self
            .poses
            .get_mut(entity)
            .ok_or(StoreError::Missing(entity))? = pose;
        Ok(())
    }

    pub fn set_point(&mut self, entity: Entity, point: S::Point) -> Result<(), DomainError> {
        let mut pose = *self.poses.get(entity).ok_or(StoreError::Missing(entity))?;
        pose.point = point;
        self.set_pose(entity, pose)
    }

    pub fn set_frame(&mut self, entity: Entity, frame: S::Frame) -> Result<(), DomainError> {
        let mut pose = *self.poses.get(entity).ok_or(StoreError::Missing(entity))?;
        pose.frame = frame;
        self.set_pose(entity, pose)
    }

    fn check_pose(&self, pose: &Pose<S>) -> Result<(), DomainError> {
        self.space
            .pose_from_chart(&self.space.chart_pose(pose))
            .map(|_| ())
    }

    pub fn add_view(&mut self, spec: ViewSpec<S>) -> Result<ViewId, DomainError> {
        self.views.add(spec, &self.poses)
    }

    pub fn view(&self, id: ViewId) -> Option<&ViewStyle<S>> {
        self.views.get(id).map(|view| &view.style)
    }

    pub fn view_eye(&self, id: ViewId) -> Option<Entity> {
        self.views.get(id).map(|view| view.eye)
    }

    pub fn view_subject(&self, id: ViewId) -> Option<Entity> {
        self.views.get(id)?.subject
    }

    pub fn view_mut(&mut self, id: ViewId) -> Option<&mut ViewStyle<S>> {
        self.views.style_mut(id)
    }

    pub fn set_view_eye(&mut self, id: ViewId, eye: Entity) -> Result<(), DomainError> {
        self.views.set_eye(id, eye, &self.poses)
    }

    pub fn set_view_subject(
        &mut self,
        id: ViewId,
        subject: Option<Entity>,
    ) -> Result<(), DomainError> {
        self.views.set_subject(id, subject, &self.poses)
    }

    pub fn fields(&self) -> Option<&Store<Field>> {
        self.fields.store()
    }

    pub fn field_mut(&mut self, entity: Entity) -> Option<&mut Field> {
        self.fields.get_mut(entity)
    }

    pub fn remove_field(&mut self, entity: Entity) -> Result<Field, StoreError> {
        self.fields.remove(entity)
    }

    pub fn walk(
        &mut self,
        entity: Entity,
        velocity: S::Vector,
        dt: f32,
    ) -> Result<(), DomainError> {
        let pose = self.poses.get(entity).ok_or(StoreError::Missing(entity))?;
        let next = self.space.walk(pose, velocity, dt)?;
        self.apply_pose(entity, next)
    }

    pub fn move_to(&mut self, entity: Entity, point: ChartPoint) -> Result<(), DomainError> {
        let target = self.space.point_from_chart(&point)?;
        if !self.poses.contains(entity) {
            return Err(StoreError::Missing(entity).into());
        }
        for facility in &mut self.facilities {
            if let Some(result) = facility.move_to(entity, target, &mut self.poses, Owner::new()) {
                return result;
            }
        }
        let space = &self.space;
        let pose = self
            .poses
            .get_mut(entity)
            .ok_or(StoreError::Missing(entity))?;
        let next = space.moved(pose, target)?;
        *pose = next;
        Ok(())
    }
}

impl<S: DomainSpace> Domain for TypedDomain<S> {
    fn id(&self) -> DomainId {
        self.id
    }

    fn name(&self) -> &'static str {
        self.name
    }

    fn tracks_changes(&self) -> bool {
        self.poses.is_tracked()
    }

    fn views(&self) -> &[ViewTarget] {
        self.views.targets()
    }

    fn view(&self, id: ViewId) -> Option<ViewSummary> {
        let spec = self.views.get(id)?;
        let target = self.views.target(id)?;
        Some(ViewSummary {
            name: spec.style.mapping.name(),
            eye: spec.eye,
            image: target.image,
            ray_lift: spec.style.mapping.ray_lift(),
        })
    }

    fn pick(
        &self,
        view: ViewId,
        ray: &ImageRay,
        placement: Rigid,
        views: &Views,
        prepared: &[PreparedGeometry],
    ) -> Option<Pick> {
        self.pick_view(view, ray, placement, views, prepared)
    }

    fn image_of(&self, view: ViewId, entity: Entity) -> Option<[f32; 3]> {
        self.image_in_view(view, entity)
    }

    fn lift_origin(&self, view: ViewId, ray: &ImageRay) -> Result<ChartPoint, DomainError> {
        self.lift_view_origin(view, ray)
    }

    fn lift_tangent(
        &self,
        view: ViewId,
        at: [f32; 3],
        tangent: [f32; 3],
    ) -> Result<ChartTangent, DomainError> {
        self.lift_view_tangent(view, at, tangent)
    }

    fn compile_fields(&mut self) -> Result<FieldCost, DomainError> {
        self.fields.compile(&self.space, &self.poses)
    }

    fn field_program(&self) -> &FieldProgram {
        self.fields.program()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl<S: DomainSpace> DomainOwner for TypedDomain<S> {
    fn retarget(&mut self, view: ViewId, image: ImageSpaceId) -> Result<(), DomainError> {
        self.views.retarget(view, image)
    }

    fn publish(
        &self,
        view: ViewId,
        library: Library<'_>,
        into: &mut ViewRecords,
        stamp: Stamp,
        alpha: f32,
    ) -> Result<(), DomainError> {
        self.publish_view(view, library, into, stamp, alpha)
    }

    fn step(&mut self, step: Step, executor: &dyn Executor) -> Result<(), DomainError> {
        for facility in &mut self.facilities {
            facility.step(&mut self.poses, step, executor, Owner::new())?;
        }
        Ok(())
    }

    fn boundary(&mut self) {
        self.synchronize();
        self.poses.boundary();
        self.instances.boundary();
        self.fields.boundary();
    }

    fn synchronize(&mut self) {
        for facility in &mut self.facilities {
            facility.synchronize(&mut self.poses, Owner::new());
        }
        self.views.synchronize();
    }

    fn release(&mut self, entity: Entity) {
        for facility in &mut self.facilities {
            facility.release(entity, Owner::new());
        }
        self.blend.release(entity, &self.poses);
        StoreField::release(&mut self.poses, entity, Owner::new());
        StoreField::release(&mut self.instances, entity, Owner::new());
        self.fields.release(entity);
    }

    fn snapshot(&self) -> DomainSnapshot {
        DomainSnapshot(Box::new(TypedSnapshot::<S> {
            scene: self.scene,
            poses: StoreField::snapshot(&self.poses),
            instances: StoreField::snapshot(&self.instances),
            fields: self.fields.snapshot(),
            facilities: self
                .facilities
                .iter()
                .map(|facility| facility.snapshot(Owner::new()))
                .collect(),
            views: self.views.entries().to_vec(),
            targets: self.views.targets().to_vec(),
        }))
    }

    fn check_restore(&self, from: &DomainSnapshot) -> Result<(), RestoreError> {
        let from = from
            .0
            .downcast_ref::<TypedSnapshot<S>>()
            .ok_or(RestoreError::Domain(self.id))?;
        if from.facilities.len() != self.facilities.len()
            || from.fields.is_some() != self.fields.enabled()
        {
            return Err(RestoreError::Domain(self.id));
        }
        for (facility, snapshot) in self.facilities.iter().zip(&from.facilities) {
            facility.check_restore(snapshot.as_ref(), Owner::new())?;
        }
        Ok(())
    }

    fn restore(&mut self, from: &DomainSnapshot, scene: SceneId) -> Result<(), RestoreError> {
        self.check_restore(from)?;
        let from = from
            .0
            .downcast_ref::<TypedSnapshot<S>>()
            .ok_or(RestoreError::Domain(self.id))?;
        for (facility, snapshot) in self.facilities.iter_mut().zip(&from.facilities) {
            facility.restore(snapshot.as_ref(), Owner::new())?;
            facility.bind(scene, Owner::new());
        }
        self.scene = scene;
        self.blend.clear();
        StoreField::restore(&mut self.poses, &from.poses, scene, Owner::new());
        StoreField::restore(&mut self.instances, &from.instances, scene, Owner::new());
        self.fields.restore(from.fields.as_ref(), from.scene, scene);
        self.views
            .restore(&from.views, &from.targets, from.scene, scene);
        Ok(())
    }

    fn begin_tick(&mut self) {
        self.blend.begin(&self.poses);
    }

    fn end_tick(&mut self) {
        self.blend.end(&self.poses);
    }

    fn present(&mut self, alpha: f32) {
        self.blend.present(alpha);
    }

    fn apply(&mut self, command: &ChartCommand) -> Result<Outcome, Rejection> {
        self.apply_command(command)
    }
}
