use std::any::Any;
use std::borrow::Cow;
use std::fmt;
use std::marker::PhantomData;

use loam_math::{EuclideanR3, EuclideanR4, HyperbolicH3, IsometryGroup, WgslSpace};

use crate::command::{Outcome, Rejection};
use crate::entity::{Entity, RuntimeId, SceneId};
use crate::phase::Step;
use crate::session::{MaterialId, PreparedId, RestoreError, Stamp};
use crate::store::{LogCapacity, Store, StoreField, StoreSnapshot};
use crate::view::{
    ImageRay, InstanceRecord, Pick, ViewId, ViewRecords, ViewSpec, ViewTarget, Views,
};

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

/// Names one domain of one session for typed access.
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChartId(pub u16);

/// Coordinates and a frame in one chart of the domain; unused entries are zero.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChartPose {
    pub chart: ChartId,
    pub coordinates: [f32; 4],
    /// Column-major, matching glam's `Mat4` and WGSL's `mat4x4<f32>`.
    pub frame: [[f32; 4]; 4],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChartTangent {
    pub chart: ChartId,
    pub vector: [f32; 4],
}

/// Heterogeneous commands; native systems use typed poses instead.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ChartCommand {
    Place {
        entity: Entity,
        pose: ChartPose,
    },
    Attach {
        entity: Entity,
        instance: Instance,
    },
    Walk {
        entity: Entity,
        tangent: ChartTangent,
        dt: f32,
    },
    Remove {
        entity: Entity,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Instance {
    pub geometry: PreparedId,
    pub material: MaterialId,
}

impl Instance {
    pub fn new(geometry: PreparedId, material: MaterialId) -> Self {
        Self { geometry, material }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DomainError {
    ForeignRuntime,
    UnknownDomain(DomainId),
    SpaceMismatch(DomainId),
    Stale(Entity),
    InvalidCoordinates,
    ChartBoundary,
    NoConvergence,
    ErrorBudget,
    Unsupported(&'static str),
}

/// A space a domain is built over; its poses cross the facade as chart data.
pub trait DomainSpace: IsometryGroup + WgslSpace + Send + Sync + 'static {
    fn origin(&self) -> Self::Point;

    fn chart_pose(&self, pose: &Self::Iso) -> ChartPose;

    fn pose_from_chart(&self, pose: &ChartPose) -> Result<Self::Iso, DomainError>;

    fn tangent_from_chart(&self, tangent: &ChartTangent) -> Result<Self::Vector, DomainError>;
}

impl DomainSpace for EuclideanR4 {
    fn origin(&self) -> Self::Point {
        [0.0; 4].into()
    }

    fn chart_pose(&self, pose: &Self::Iso) -> ChartPose {
        ChartPose {
            chart: ChartId(0),
            coordinates: pose.translation.to_array(),
            frame: pose.rotation.to_mat4(),
        }
    }

    fn pose_from_chart(&self, _pose: &ChartPose) -> Result<Self::Iso, DomainError> {
        todo!()
    }

    fn tangent_from_chart(&self, _tangent: &ChartTangent) -> Result<Self::Vector, DomainError> {
        todo!()
    }
}

impl DomainSpace for HyperbolicH3 {
    fn origin(&self) -> Self::Point {
        [0.0; 3].into()
    }

    fn chart_pose(&self, _pose: &Self::Iso) -> ChartPose {
        todo!()
    }

    fn pose_from_chart(&self, _pose: &ChartPose) -> Result<Self::Iso, DomainError> {
        todo!()
    }

    fn tangent_from_chart(&self, _tangent: &ChartTangent) -> Result<Self::Vector, DomainError> {
        todo!()
    }
}

impl DomainSpace for EuclideanR3 {
    fn origin(&self) -> Self::Point {
        [0.0; 3].into()
    }

    fn chart_pose(&self, _pose: &Self::Iso) -> ChartPose {
        todo!()
    }

    fn pose_from_chart(&self, _pose: &ChartPose) -> Result<Self::Iso, DomainError> {
        todo!()
    }

    fn tangent_from_chart(&self, _tangent: &ChartTangent) -> Result<Self::Vector, DomainError> {
        todo!()
    }
}

pub struct Pose<S: IsometryGroup>(pub S::Iso);

impl<S: IsometryGroup> Clone for Pose<S> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<S: IsometryGroup> Copy for Pose<S> {}

/// Engine-owned work inside one domain, run by the simulation phase's domain-step entry.
pub trait Facility<S: DomainSpace>: Send + 'static {
    fn name(&self) -> &'static str;

    fn step(&mut self, poses: &mut Store<Pose<S>>, step: Step) -> Result<(), DomainError>;

    fn snapshot(&self) -> Box<dyn Any + Send>;

    fn restore(&mut self, from: &(dyn Any + Send)) -> Result<(), RestoreError>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldKind {
    ExactDistance,
    ConservativeBound,
    Implicit,
}

/// A primitive, or an operator over the entities it lists.
#[derive(Clone, Debug)]
pub struct Field {
    pub kind: FieldKind,
    pub operands: Vec<Entity>,
}

/// Primitive buffer, postfix program, and stack requirement for the fixed traverser.
#[derive(Clone, Debug, Default)]
pub struct FieldProgram {
    pub primitives: Vec<f32>,
    pub program: Vec<u32>,
    pub stack: u32,
}

pub struct DomainSnapshot(pub Box<dyn Any + Send>);

struct TypedSnapshot<S: DomainSpace> {
    poses: StoreSnapshot<Pose<S>>,
    instances: StoreSnapshot<Instance>,
    fields: Option<StoreSnapshot<Field>>,
    facilities: Vec<Box<dyn Any + Send>>,
}

/// The facade the session holds; `S` never appears here.
pub trait Domain: Send + 'static {
    fn id(&self) -> DomainId;

    fn name(&self) -> &'static str;

    fn views(&self) -> &[ViewTarget];

    fn publish(
        &self,
        view: ViewId,
        into: &mut ViewRecords,
        stamp: Stamp,
    ) -> Result<(), DomainError>;

    fn pick(&self, view: ViewId, ray: &ImageRay) -> Option<Pick>;

    fn step(&mut self, step: Step) -> Result<(), DomainError>;

    fn boundary(&mut self);

    fn release(&mut self, entity: Entity);

    fn snapshot(&self) -> DomainSnapshot;

    fn restore(&mut self, from: &DomainSnapshot, scene: SceneId) -> Result<(), RestoreError>;

    fn apply(&mut self, command: &ChartCommand) -> Result<Outcome, Rejection>;

    fn compile_fields(&mut self) -> Result<FieldProgram, DomainError>;

    fn shader_prelude(&self) -> Cow<'static, str>;

    fn as_any_mut(&mut self) -> &mut dyn Any;
}

pub struct TypedDomain<S: DomainSpace> {
    id: DomainId,
    scene: SceneId,
    name: &'static str,
    pub space: S,
    pub poses: Store<Pose<S>>,
    pub instances: Store<Instance>,
    fields: Option<Store<Field>>,
    views: Vec<ViewSpec<S>>,
    targets: Vec<ViewTarget>,
    facilities: Vec<Box<dyn Facility<S>>>,
}

impl<S: DomainSpace> TypedDomain<S> {
    pub fn handle(&self) -> DomainHandle<S> {
        DomainHandle {
            id: self.id,
            runtime: self.scene.runtime,
            space: PhantomData,
        }
    }

    pub fn add_view(&mut self, spec: ViewSpec<S>) -> ViewId {
        let id = ViewId::new(self.views.len());
        self.targets.push(ViewTarget {
            view: id,
            image: spec.image,
        });
        self.views.push(spec);
        id
    }

    pub fn view(&self, id: ViewId) -> Option<&ViewSpec<S>> {
        self.views.get(id.index())
    }

    pub fn view_mut(&mut self, id: ViewId) -> Option<&mut ViewSpec<S>> {
        self.views.get_mut(id.index())
    }

    pub fn fields(&self) -> Option<&Store<Field>> {
        self.fields.as_ref()
    }

    pub fn fields_mut(&mut self) -> Option<&mut Store<Field>> {
        self.fields.as_mut()
    }

    /// Exponential along `velocity` for `dt`, with the frame transported.
    pub fn walk(
        &mut self,
        _entity: Entity,
        _velocity: S::Vector,
        _dt: f32,
    ) -> Result<(), DomainError> {
        todo!()
    }
}

impl<S: DomainSpace> Domain for TypedDomain<S> {
    fn id(&self) -> DomainId {
        self.id
    }

    fn name(&self) -> &'static str {
        self.name
    }

    fn views(&self) -> &[ViewTarget] {
        &self.targets
    }

    fn publish(
        &self,
        view: ViewId,
        into: &mut ViewRecords,
        stamp: Stamp,
    ) -> Result<(), DomainError> {
        let spec = self
            .views
            .get(view.index())
            .ok_or(DomainError::Unsupported("unknown view"))?;
        let eye = self
            .poses
            .get(spec.eye)
            .ok_or(DomainError::Stale(spec.eye))?;
        let origin = self.space.origin();
        let records = self.instances.iter().filter_map(|(entity, instance)| {
            let pose = self.poses.get(entity)?;
            let point = self.space.iso_apply(pose.0, origin);
            let image_point = spec.mapping.image_point(eye, point)?;
            Some((
                entity,
                InstanceRecord {
                    entity,
                    geometry: instance.geometry,
                    material: instance.material,
                    pose: self.space.chart_pose(&pose.0),
                    image_point,
                },
            ))
        });
        into.instances.replace(records, stamp);
        Ok(())
    }

    fn pick(&self, _view: ViewId, _ray: &ImageRay) -> Option<Pick> {
        todo!()
    }

    fn step(&mut self, step: Step) -> Result<(), DomainError> {
        for facility in &mut self.facilities {
            facility.step(&mut self.poses, step)?;
        }
        Ok(())
    }

    fn boundary(&mut self) {
        self.poses.boundary();
        self.instances.boundary();
        if let Some(fields) = &mut self.fields {
            fields.boundary();
        }
    }

    fn release(&mut self, entity: Entity) {
        self.poses.release(entity);
        self.instances.release(entity);
        if let Some(fields) = &mut self.fields {
            fields.release(entity);
        }
    }

    fn snapshot(&self) -> DomainSnapshot {
        DomainSnapshot(Box::new(TypedSnapshot::<S> {
            poses: StoreField::snapshot(&self.poses),
            instances: StoreField::snapshot(&self.instances),
            fields: self.fields.as_ref().map(StoreField::snapshot),
            facilities: self
                .facilities
                .iter()
                .map(|facility| facility.snapshot())
                .collect(),
        }))
    }

    fn restore(&mut self, from: &DomainSnapshot, scene: SceneId) -> Result<(), RestoreError> {
        let from = from
            .0
            .downcast_ref::<TypedSnapshot<S>>()
            .ok_or(RestoreError::Domain(self.id))?;
        if from.facilities.len() != self.facilities.len()
            || from.fields.is_some() != self.fields.is_some()
        {
            return Err(RestoreError::Domain(self.id));
        }
        for (facility, snapshot) in self.facilities.iter_mut().zip(&from.facilities) {
            facility.restore(snapshot.as_ref())?;
        }
        self.scene = scene;
        StoreField::restore(&mut self.poses, &from.poses, scene);
        StoreField::restore(&mut self.instances, &from.instances, scene);
        if let (Some(fields), Some(snapshot)) = (&mut self.fields, &from.fields) {
            StoreField::restore(fields, snapshot, scene);
        }
        for view in &mut self.views {
            view.eye = Entity::new(scene, view.eye.key());
        }
        Ok(())
    }

    fn apply(&mut self, _command: &ChartCommand) -> Result<Outcome, Rejection> {
        todo!()
    }

    fn compile_fields(&mut self) -> Result<FieldProgram, DomainError> {
        todo!()
    }

    fn shader_prelude(&self) -> Cow<'static, str> {
        self.space.wgsl_impl()
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Registers only the facilities the domain uses.
pub struct DomainBuilder<S: DomainSpace> {
    name: &'static str,
    space: S,
    tracking: Option<LogCapacity>,
    fields: bool,
    facilities: Vec<Box<dyn Facility<S>>>,
}

impl<S: DomainSpace> DomainBuilder<S> {
    pub fn new(name: &'static str, space: S) -> Self {
        Self {
            name,
            space,
            tracking: None,
            fields: false,
            facilities: Vec::new(),
        }
    }

    pub fn tracked(mut self, capacity: LogCapacity) -> Self {
        self.tracking = Some(capacity);
        self
    }

    pub fn fields(mut self) -> Self {
        self.fields = true;
        self
    }

    pub fn facility(mut self, facility: impl Facility<S>) -> Self {
        self.facilities.push(Box::new(facility));
        self
    }

    pub(crate) fn build(self, id: DomainId, scene: SceneId) -> TypedDomain<S> {
        fn store<T>(tracking: Option<LogCapacity>, scene: SceneId) -> Store<T> {
            let mut store = match tracking {
                Some(capacity) => Store::tracked(capacity),
                None => Store::untracked(),
            };
            store.bind(scene);
            store
        }
        TypedDomain {
            id,
            scene,
            name: self.name,
            space: self.space,
            poses: store(self.tracking, scene),
            instances: store(self.tracking, scene),
            fields: self.fields.then(|| store(self.tracking, scene)),
            views: Vec::new(),
            targets: Vec::new(),
            facilities: self.facilities,
        }
    }
}

pub struct Domains {
    runtime: RuntimeId,
    list: Vec<Box<dyn Domain>>,
}

impl Domains {
    pub(crate) fn new(runtime: RuntimeId) -> Self {
        Self {
            runtime,
            list: Vec::new(),
        }
    }

    pub(crate) fn next_id(&self) -> DomainId {
        DomainId::new(self.list.len())
    }

    pub(crate) fn push(&mut self, domain: Box<dyn Domain>) {
        self.list.push(domain);
    }

    pub fn runtime(&self) -> RuntimeId {
        self.runtime
    }

    pub fn len(&self) -> usize {
        self.list.len()
    }

    pub fn is_empty(&self) -> bool {
        self.list.is_empty()
    }

    pub fn facade(&mut self, id: DomainId) -> Option<&mut dyn Domain> {
        self.list.get_mut(id.index()).map(|domain| domain.as_mut())
    }

    pub fn iter(&self) -> impl Iterator<Item = &dyn Domain> {
        self.list.iter().map(|domain| domain.as_ref())
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut dyn Domain> {
        self.list.iter_mut().map(|domain| domain.as_mut())
    }

    /// Checks runtime, identity, and space once; the borrow scopes the typed traversal.
    pub fn typed<S: DomainSpace>(
        &mut self,
        handle: DomainHandle<S>,
    ) -> Result<&mut TypedDomain<S>, DomainError> {
        if handle.runtime != self.runtime {
            return Err(DomainError::ForeignRuntime);
        }
        let domain = self
            .list
            .get_mut(handle.id.index())
            .ok_or(DomainError::UnknownDomain(handle.id))?;
        domain
            .as_any_mut()
            .downcast_mut::<TypedDomain<S>>()
            .ok_or(DomainError::SpaceMismatch(handle.id))
    }

    /// The nearest hit by the root eye's projective depth across every view into `views`' root.
    pub fn pick(&self, _views: &Views, _ndc: [f32; 2]) -> Option<Pick> {
        todo!()
    }
}
