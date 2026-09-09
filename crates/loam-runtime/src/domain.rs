use std::any::Any;
use std::borrow::Cow;
use std::fmt;
use std::marker::PhantomData;
use std::ops::Mul;

pub use loam_shape::field::FieldKind;

use loam_shape::field::DistanceField;

use loam_math::hyperbolic::{in_poincare_ball, poincare_to_hyperboloid};
use loam_math::{
    EuclideanR3, EuclideanR4, HyperbolicH3, Iso3, Iso3H, Iso4Flat, IsometryGroup, Space, WgslSpace,
};

use crate::command::{Outcome, Rejection};
use crate::entity::{Entity, RuntimeId, SceneId};
use crate::field::{
    self, FieldCompiler, FieldCost, FieldError, FieldNode, FieldOp, FieldPrimitive,
};
use crate::phase::Step;
use crate::session::{Library, MaterialId, PreparedGeometry, PreparedId, RestoreError, Stamp};
use crate::store::{LogCapacity, Store, StoreField, StoreSnapshot};
use crate::view::{
    self, DomainRay, ImageRay, ImageSpaceId, InstanceRecord, Pick, Rigid, SegmentRecord, Vec3,
    Vec4, ViewId, ViewMapping, ViewRecords, ViewSpec, ViewSummary, ViewTarget, Views,
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChartPoint {
    pub chart: ChartId,
    pub coordinates: [f32; 4],
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
    Move {
        entity: Entity,
        point: ChartPoint,
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
    InvalidCoordinate(&'static str),
    InvalidFrame,
    ChartBoundary,
    NoConvergence,
    ErrorBudget,
    Unsupported(&'static str),
    FieldCycle(Entity),
    FieldArity(Entity),
}

/// A space a domain is built over; its poses cross the facade as chart data.
pub trait DomainSpace:
    IsometryGroup<Vector: Mul<f32, Output = <Self as Space>::Vector>>
    + WgslSpace
    + Send
    + Sync
    + Sized
    + 'static
{
    fn origin(&self) -> Self::Point;

    /// Names the first non-finite coordinate or reports the chart boundary; nothing is clamped.
    fn check(&self, point: Self::Point) -> Result<(), DomainError>;

    fn chart_point(&self, point: Self::Point) -> ChartPoint;

    /// Reads the leading coordinates the space has and ignores the rest.
    fn local_point(&self, coordinates: [f32; 4]) -> Self::Point;

    fn chart_pose(&self, pose: &Self::Iso) -> ChartPose;

    fn pose_from_chart(&self, pose: &ChartPose) -> Result<Self::Iso, DomainError>;

    fn tangent_from_chart(&self, tangent: &ChartTangent) -> Result<Self::Vector, DomainError>;

    /// Moves the origin to `to` along their geodesic, carrying the frame by parallel transport.
    fn transvection(&self, to: Self::Point) -> Self::Iso;

    /// Metric distance from the origin to a point at chart radius `radius`.
    fn chart_reach(&self, radius: f32) -> f32;

    /// Arc length along `ray` into the metric ball of `radius` around `center`: zero from inside, `None` on a miss.
    fn hit_ball(&self, ray: &DomainRay<Self>, center: Self::Point, radius: f32) -> Option<f32>;
}

const FRAME_TOLERANCE: f32 = 1e-3;

fn finite(values: [f32; 4]) -> Result<(), DomainError> {
    for (value, axis) in values.into_iter().zip(["x", "y", "z", "w"]) {
        if !value.is_finite() {
            return Err(DomainError::InvalidCoordinate(axis));
        }
    }
    Ok(())
}

fn frame_columns(frame: &[[f32; 4]; 4]) -> Result<[Vec3; 3], DomainError> {
    let columns = [0, 1, 2].map(|i| Vec3::new(frame[i][0], frame[i][1], frame[i][2]));
    let [x, y, z] = columns;
    let unit = |v: Vec3| v.is_finite() && (v.length_squared() - 1.0).abs() <= FRAME_TOLERANCE;
    let orthogonal = |a: Vec3, b: Vec3| a.dot(b).abs() <= FRAME_TOLERANCE;
    let valid = columns.iter().all(|column| unit(*column))
        && orthogonal(x, y)
        && orthogonal(y, z)
        && orthogonal(z, x)
        && x.cross(y).dot(z) > 0.0;
    valid.then_some(columns).ok_or(DomainError::InvalidFrame)
}

fn frame_of(columns: [Vec3; 3]) -> [[f32; 4]; 4] {
    let mut frame = [[0.0; 4]; 4];
    for (slot, column) in frame.iter_mut().zip(columns) {
        *slot = column.extend(0.0).to_array();
    }
    frame[3] = [0.0, 0.0, 0.0, 1.0];
    frame
}

// Shepperd, Quaternion from Rotation Matrix, J. Guidance and Control 1(3), 1978.
fn rotation_of([x, y, z]: [Vec3; 3]) -> Iso3 {
    let mut pose = Iso3::IDENTITY;
    let q = &mut pose.rotation;
    let trace = x.x + y.y + z.z;
    if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        (q.w, q.x, q.y, q.z) = (s * 0.25, (y.z - z.y) / s, (z.x - x.z) / s, (x.y - y.x) / s);
    } else if x.x > y.y && x.x > z.z {
        let s = (1.0 + x.x - y.y - z.z).sqrt() * 2.0;
        (q.w, q.x, q.y, q.z) = ((y.z - z.y) / s, s * 0.25, (y.x + x.y) / s, (z.x + x.z) / s);
    } else if y.y > z.z {
        let s = (1.0 + y.y - x.x - z.z).sqrt() * 2.0;
        (q.w, q.x, q.y, q.z) = ((z.x - x.z) / s, (y.x + x.y) / s, s * 0.25, (z.y + y.z) / s);
    } else {
        let s = (1.0 + z.z - x.x - y.y).sqrt() * 2.0;
        (q.w, q.x, q.y, q.z) = ((x.y - y.x) / s, (z.x + x.z) / s, (z.y + y.z) / s, s * 0.25);
    }
    pose.rotation = pose.rotation.normalize();
    pose
}

fn lorentz(a: Vec4, b: Vec4) -> f32 {
    a.x * b.x + a.y * b.y + a.z * b.z - a.w * b.w
}

impl DomainSpace for EuclideanR4 {
    fn origin(&self) -> Self::Point {
        Vec4::ZERO
    }

    fn check(&self, point: Self::Point) -> Result<(), DomainError> {
        finite(point.to_array())
    }

    fn chart_point(&self, point: Self::Point) -> ChartPoint {
        ChartPoint {
            chart: ChartId(0),
            coordinates: point.to_array(),
        }
    }

    fn local_point(&self, coordinates: [f32; 4]) -> Self::Point {
        Vec4::from_array(coordinates)
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

    fn transvection(&self, to: Self::Point) -> Self::Iso {
        Iso4Flat::from_translation(to)
    }

    fn chart_reach(&self, radius: f32) -> f32 {
        radius
    }

    fn hit_ball(&self, ray: &DomainRay<Self>, center: Self::Point, radius: f32) -> Option<f32> {
        let offset = ray.origin - center;
        view::ball_entry(
            offset.dot(ray.direction),
            offset.length_squared() - radius * radius,
        )
    }
}

impl DomainSpace for HyperbolicH3 {
    fn origin(&self) -> Self::Point {
        Vec3::ZERO
    }

    fn check(&self, point: Self::Point) -> Result<(), DomainError> {
        finite(point.extend(0.0).to_array())?;
        in_poincare_ball(point)
            .then_some(())
            .ok_or(DomainError::ChartBoundary)
    }

    fn chart_point(&self, point: Self::Point) -> ChartPoint {
        ChartPoint {
            chart: ChartId(0),
            coordinates: point.extend(0.0).to_array(),
        }
    }

    fn local_point(&self, coordinates: [f32; 4]) -> Self::Point {
        Vec3::from_slice(&coordinates[..3])
    }

    fn chart_pose(&self, pose: &Self::Iso) -> ChartPose {
        let position = self.iso_apply(*pose, Vec3::ZERO);
        let rotation = self.iso_compose(self.iso_inverse(Iso3H::from_translation(position)), *pose);
        let matrix = rotation.matrix;
        ChartPose {
            chart: ChartId(0),
            coordinates: position.extend(0.0).to_array(),
            frame: frame_of([
                matrix.x_axis.truncate(),
                matrix.y_axis.truncate(),
                matrix.z_axis.truncate(),
            ]),
        }
    }

    fn pose_from_chart(&self, pose: &ChartPose) -> Result<Self::Iso, DomainError> {
        finite(pose.coordinates)?;
        let position = Vec3::from_slice(&pose.coordinates[..3]);
        self.check(position)?;
        let [x, y, z] = frame_columns(&pose.frame)?;
        let mut rotation = Iso3H::IDENTITY;
        rotation.matrix.x_axis = x.extend(0.0);
        rotation.matrix.y_axis = y.extend(0.0);
        rotation.matrix.z_axis = z.extend(0.0);
        Ok(self.iso_compose(Iso3H::from_translation(position), rotation))
    }

    fn tangent_from_chart(&self, tangent: &ChartTangent) -> Result<Self::Vector, DomainError> {
        finite(tangent.vector)?;
        Ok(Vec3::from_slice(&tangent.vector[..3]))
    }

    fn transvection(&self, to: Self::Point) -> Self::Iso {
        Iso3H::from_translation(to)
    }

    fn chart_reach(&self, radius: f32) -> f32 {
        2.0 * radius.min(1.0).atanh()
    }

    fn hit_ball(&self, ray: &DomainRay<Self>, center: Self::Point, radius: f32) -> Option<f32> {
        // Ratcliffe, Foundations of Hyperbolic Manifolds, 2006, §3.2: cosh d(x, y) = -⟨x, y⟩ on the hyperboloid.
        let start = poincare_to_hyperboloid(ray.origin);
        let ahead = poincare_to_hyperboloid(self.exp(ray.origin, ray.direction));
        let tangent = (ahead - start * 1.0_f32.cosh()) / 1.0_f32.sinh();
        let target = poincare_to_hyperboloid(center);
        let reach = -lorentz(start, target);
        let along = -lorentz(tangent, target);
        let nearest = (reach * reach - along * along).max(1.0).sqrt();
        let foot = -(along / reach).atanh();
        let ratio = radius.cosh() / nearest;
        if ratio < 1.0 {
            return None;
        }
        let half = ratio.acosh();
        let entry = foot - half;
        if entry <= 0.0 {
            return (foot + half >= 0.0).then_some(0.0);
        }
        Some(entry)
    }
}

impl DomainSpace for EuclideanR3 {
    fn origin(&self) -> Self::Point {
        Vec3::ZERO
    }

    fn check(&self, point: Self::Point) -> Result<(), DomainError> {
        finite(point.extend(0.0).to_array())
    }

    fn chart_point(&self, point: Self::Point) -> ChartPoint {
        ChartPoint {
            chart: ChartId(0),
            coordinates: point.extend(0.0).to_array(),
        }
    }

    fn local_point(&self, coordinates: [f32; 4]) -> Self::Point {
        Vec3::from_slice(&coordinates[..3])
    }

    fn chart_pose(&self, pose: &Self::Iso) -> ChartPose {
        ChartPose {
            chart: ChartId(0),
            coordinates: pose.translation.extend(0.0).to_array(),
            frame: frame_of([Vec3::X, Vec3::Y, Vec3::Z].map(|axis| pose.rotation * axis)),
        }
    }

    fn pose_from_chart(&self, pose: &ChartPose) -> Result<Self::Iso, DomainError> {
        finite(pose.coordinates)?;
        let mut iso = rotation_of(frame_columns(&pose.frame)?);
        iso.translation = Vec3::from_slice(&pose.coordinates[..3]);
        Ok(iso)
    }

    fn tangent_from_chart(&self, tangent: &ChartTangent) -> Result<Self::Vector, DomainError> {
        finite(tangent.vector)?;
        Ok(Vec3::from_slice(&tangent.vector[..3]))
    }

    fn transvection(&self, to: Self::Point) -> Self::Iso {
        Iso3::from_translation(to)
    }

    fn chart_reach(&self, radius: f32) -> f32 {
        radius
    }

    fn hit_ball(&self, ray: &DomainRay<Self>, center: Self::Point, radius: f32) -> Option<f32> {
        let offset = ray.origin - center;
        view::ball_entry(
            offset.dot(ray.direction),
            offset.length_squared() - radius * radius,
        )
    }
}

fn image_of<S: DomainSpace>(
    space: &S,
    mapping: &dyn ViewMapping<S>,
    eye: &Pose<S>,
    pose: &Pose<S>,
    local: [f32; 4],
) -> Option<[f32; 3]> {
    let point = space.iso_apply(pose.0, space.local_point(local));
    space.check(point).ok()?;
    mapping.image_point(eye, point)
}

fn push_segments<S: DomainSpace>(
    space: &S,
    mapping: &dyn ViewMapping<S>,
    eye: &Pose<S>,
    pose: &Pose<S>,
    library: &Library<'_>,
    instance: &Instance,
    into: &mut Vec<SegmentRecord>,
) {
    let Some(geometry) = library.geometry.get(instance.geometry.index()) else {
        return;
    };
    let (color, width_px) = library.line_style(instance.material);
    let mut push = |a: [f32; 4], b: [f32; 4]| {
        let (Some(start), Some(end)) = (
            image_of(space, mapping, eye, pose, a),
            image_of(space, mapping, eye, pose, b),
        ) else {
            return;
        };
        into.push(SegmentRecord {
            start,
            _pad0: 0.0,
            end,
            _pad1: 0.0,
            start_color: color,
            end_color: color,
            width_px,
            _pad2: [0.0; 3],
        });
    };
    match geometry {
        PreparedGeometry::Lines4 { segments } => {
            for &[a, b] in segments {
                push(a, b);
            }
        }
        PreparedGeometry::Lines3 { segments } => {
            for &[a, b] in segments {
                push([a[0], a[1], a[2], 0.0], [b[0], b[1], b[2], 0.0]);
            }
        }
        PreparedGeometry::Mesh3 { .. } => {}
    }
}

pub struct Pose<S: IsometryGroup>(pub S::Iso);

impl<S: IsometryGroup> Clone for Pose<S> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<S: IsometryGroup> Copy for Pose<S> {}

/// Engine-owned work inside one domain: `step` runs in the simulation phase's domain-step entry, `release` at dispatch before the stores forget the entity, and `apply` gets first offer of a chart command.
pub trait Facility<S: DomainSpace>: Any + Send + 'static {
    fn name(&self) -> &'static str;

    fn step(&mut self, poses: &mut Store<Pose<S>>, step: Step) -> Result<(), DomainError>;

    fn snapshot(&self) -> Box<dyn Any + Send>;

    fn restore(&mut self, from: &(dyn Any + Send)) -> Result<(), RestoreError>;

    fn release(&mut self, _entity: Entity) {}

    /// `Some` claims the command with its outcome; `None` leaves it to the domain.
    fn apply(&mut self, _command: &ChartCommand) -> Option<Result<Outcome, Rejection>> {
        None
    }
}

/// A primitive, or an operator over the entities it lists.
#[derive(Clone, Debug)]
pub struct Field {
    pub kind: FieldKind,
    pub op: FieldOp,
    pub operands: Vec<Entity>,
}

/// Primitive buffer, postfix program, ball tree, and stack requirement for the fixed traverser.
#[derive(Clone, Debug, Default)]
pub struct FieldProgram {
    pub primitives: Vec<FieldPrimitive>,
    pub program: Vec<u32>,
    pub nodes: Vec<FieldNode>,
    pub stack: u32,
    pub kind: FieldKind,
}

impl FieldProgram {
    pub fn evaluate(&self, point: [f32; 4]) -> Result<(f32, FieldKind), FieldError> {
        field::evaluate(&self.program, &self.primitives, point).map(|value| (value, self.kind))
    }

    pub fn evaluate_bounded(
        &self,
        point: [f32; 4],
        tolerance: f32,
    ) -> Result<(f32, FieldKind), FieldError> {
        field::evaluate_bounded(
            &self.program,
            &self.primitives,
            &self.nodes,
            point,
            tolerance,
        )
        .map(|value| (value, self.kind))
    }
}

impl DistanceField for FieldProgram {
    fn field_kind(&self) -> FieldKind {
        self.kind
    }

    fn distance(&self, point: [f32; 4]) -> f32 {
        field::evaluate_bounded(&self.program, &self.primitives, &self.nodes, point, 0.0)
            .unwrap_or(field::FIELD_FAR)
    }

    fn error(&self) -> f32 {
        field::FIELD_PROGRAM_ERROR
    }
}

pub struct DomainSnapshot(pub Box<dyn Any + Send>);

struct TypedSnapshot<S: DomainSpace> {
    poses: StoreSnapshot<Pose<S>>,
    instances: StoreSnapshot<Instance>,
    fields: Option<StoreSnapshot<Field>>,
    facilities: Vec<Box<dyn Any + Send>>,
    targets: Vec<ViewTarget>,
}

/// The facade the session holds; `S` never appears here.
pub trait Domain: Send + 'static {
    fn id(&self) -> DomainId;

    fn name(&self) -> &'static str;

    fn views(&self) -> &[ViewTarget];

    fn view(&self, id: ViewId) -> Option<ViewSummary>;

    fn retarget(&mut self, view: ViewId, image: ImageSpaceId) -> Result<(), DomainError>;

    fn has_fields(&self) -> bool;

    fn publish(
        &self,
        view: ViewId,
        library: Library<'_>,
        into: &mut ViewRecords,
        stamp: Stamp,
    ) -> Result<(), DomainError>;

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

    fn step(&mut self, step: Step) -> Result<(), DomainError>;

    fn boundary(&mut self);

    fn release(&mut self, entity: Entity);

    fn snapshot(&self) -> DomainSnapshot;

    fn restore(&mut self, from: &DomainSnapshot, scene: SceneId) -> Result<(), RestoreError>;

    fn apply(&mut self, command: &ChartCommand) -> Result<Outcome, Rejection>;

    fn compile_fields(&mut self) -> Result<FieldCost, DomainError>;

    fn field_program(&self) -> &FieldProgram;

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
    pub(crate) facilities: Vec<Box<dyn Facility<S>>>,
    compiler: FieldCompiler,
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
        let spec = self.views.get_mut(id.index())?;
        spec.revision = spec.revision.wrapping_add(1);
        Some(spec)
    }

    pub fn fields(&self) -> Option<&Store<Field>> {
        self.fields.as_ref()
    }

    pub fn fields_mut(&mut self) -> Option<&mut Store<Field>> {
        self.fields.as_mut()
    }

    /// Exponential along `velocity`, a chart tangent in the entity's own frame, for `dt`, with the frame carried by parallel transport.
    pub fn walk(
        &mut self,
        entity: Entity,
        velocity: S::Vector,
        dt: f32,
    ) -> Result<(), DomainError> {
        let pose = self
            .poses
            .get_mut(entity)
            .ok_or(DomainError::Stale(entity))?;
        let origin = self.space.origin();
        let step = self.space.exp(origin, velocity * dt);
        self.space.check(step)?;
        // Helgason, Differential Geometry, Lie Groups, and Symmetric Spaces, 1978, Ch. IV §3: the transvection along a geodesic is parallel transport along it.
        let next = self
            .space
            .iso_compose(pose.0, self.space.transvection(step));
        self.space.check(self.space.iso_apply(next, origin))?;
        pose.0 = next;
        Ok(())
    }

    /// Moves the origin to `point` by transvection, keeping the frame.
    pub fn move_to(&mut self, entity: Entity, point: ChartPoint) -> Result<(), DomainError> {
        finite(point.coordinates)?;
        let target = self.space.local_point(point.coordinates);
        self.space.check(target)?;
        let origin = self.space.origin();
        let pose = self
            .poses
            .get_mut(entity)
            .ok_or(DomainError::Stale(entity))?;
        let here = self.space.iso_apply(pose.0, origin);
        let frame = self.space.iso_compose(
            self.space.iso_inverse(self.space.transvection(here)),
            pose.0,
        );
        pose.0 = self
            .space
            .iso_compose(self.space.transvection(target), frame);
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

    fn views(&self) -> &[ViewTarget] {
        &self.targets
    }

    fn view(&self, id: ViewId) -> Option<ViewSummary> {
        let spec = self.views.get(id.index())?;
        Some(ViewSummary {
            name: spec.mapping.name(),
            eye: spec.eye,
            image: spec.image,
            ray_lift: spec.mapping.ray_lift(),
        })
    }

    fn retarget(&mut self, view: ViewId, image: ImageSpaceId) -> Result<(), DomainError> {
        let spec = self
            .views
            .get_mut(view.index())
            .ok_or(DomainError::Unsupported("unknown view"))?;
        spec.image = image;
        spec.revision = spec.revision.wrapping_add(1);
        self.targets[view.index()].image = image;
        Ok(())
    }

    fn has_fields(&self) -> bool {
        self.fields.is_some()
    }

    fn publish(
        &self,
        view: ViewId,
        library: Library<'_>,
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
        let moved = self.poses.changed_since(&mut into.poses);
        let attached = self.instances.changed_since(&mut into.attachments);
        if !moved && !attached && into.revision == spec.revision {
            into.instances.restamp(stamp);
            return Ok(());
        }
        self.poses.catch_up(&mut into.poses);
        self.instances.catch_up(&mut into.attachments);
        into.revision = spec.revision;
        let origin = self.space.origin();
        let space = &self.space;
        let poses = &self.poses;
        let mapping = spec.mapping.as_ref();
        let segments = &mut into.segments;
        segments.clear();
        let records = self.instances.iter().filter_map(|(entity, instance)| {
            let pose = poses.get(entity)?;
            let point = space.iso_apply(pose.0, origin);
            let image_point = mapping.image_point(eye, point)?;
            push_segments(space, mapping, eye, pose, &library, instance, segments);
            Some((
                entity,
                InstanceRecord {
                    entity,
                    geometry: instance.geometry,
                    material: instance.material,
                    pose: space.chart_pose(&pose.0),
                    image_point,
                },
            ))
        });
        into.instances.replace(records, stamp);
        into.built = stamp;
        Ok(())
    }

    fn pick(
        &self,
        view: ViewId,
        ray: &ImageRay,
        placement: Rigid,
        views: &Views,
        prepared: &[PreparedGeometry],
    ) -> Option<Pick> {
        let spec = self.views.get(view.index())?;
        let eye = self.poses.get(spec.eye)?;
        let origin = self.space.origin();
        let lifted = spec.mapping.lift(eye, ray);
        let mut nearest: Option<Pick> = None;
        for (entity, instance) in self.instances.iter() {
            let Some(pose) = self.poses.get(entity) else {
                continue;
            };
            let Some(geometry) = prepared.get(instance.geometry.index()) else {
                continue;
            };
            let center = self.space.iso_apply(pose.0, origin);
            let radius = geometry.bounding_radius();
            let hit = match &lifted {
                Some(domain_ray) => {
                    let reach = self.space.chart_reach(radius);
                    self.space
                        .hit_ball(domain_ray, center, reach)
                        .map(|t| self.space.exp(domain_ray.origin, domain_ray.direction * t))
                        .and_then(|point| {
                            let image_point = spec.mapping.image_point(eye, point)?;
                            Some((image_point, Some(self.space.chart_point(point))))
                        })
                }
                None => spec
                    .mapping
                    .image_point(eye, center)
                    .and_then(|image_center| {
                        let image_radius = spec.mapping.image_radius(eye, center, radius);
                        let t = view::image_hit(ray, image_center, image_radius)?;
                        Some((ray.at(t), None))
                    }),
            };
            let Some((image_point, hit)) = hit else {
                continue;
            };
            let image_point = placement.apply(image_point);
            let Some(depth) = views.depth(image_point) else {
                continue;
            };
            if nearest.is_some_and(|best| best.depth >= depth) {
                continue;
            }
            nearest = Some(Pick {
                entity,
                domain: self.id,
                view,
                image: spec.image,
                image_point,
                depth,
                hit,
            });
        }
        nearest
    }

    fn image_of(&self, view: ViewId, entity: Entity) -> Option<[f32; 3]> {
        let spec = self.views.get(view.index())?;
        let eye = self.poses.get(spec.eye)?;
        let pose = self.poses.get(entity)?;
        let point = self.space.iso_apply(pose.0, self.space.origin());
        spec.mapping.image_point(eye, point)
    }

    fn lift_origin(&self, view: ViewId, ray: &ImageRay) -> Result<ChartPoint, DomainError> {
        let spec = self
            .views
            .get(view.index())
            .ok_or(DomainError::Unsupported("unknown view"))?;
        let eye = self
            .poses
            .get(spec.eye)
            .ok_or(DomainError::Stale(spec.eye))?;
        let lifted = spec
            .mapping
            .lift(eye, ray)
            .ok_or(DomainError::Unsupported(spec.mapping.name()))?;
        self.space.check(lifted.origin)?;
        Ok(self.space.chart_point(lifted.origin))
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
        for facility in &mut self.facilities {
            facility.release(entity);
        }
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
            targets: self.targets.clone(),
        }))
    }

    fn restore(&mut self, from: &DomainSnapshot, scene: SceneId) -> Result<(), RestoreError> {
        let from = from
            .0
            .downcast_ref::<TypedSnapshot<S>>()
            .ok_or(RestoreError::Domain(self.id))?;
        if from.facilities.len() != self.facilities.len()
            || from.fields.is_some() != self.fields.is_some()
            || from.targets.len() != self.targets.len()
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
        for (view, target) in self.views.iter_mut().zip(&from.targets) {
            view.eye = Entity::new(scene, view.eye.key());
            view.image = target.image;
            view.revision = view.revision.wrapping_add(1);
        }
        self.targets.clear();
        self.targets.extend_from_slice(&from.targets);
        self.compiler.invalidate();
        Ok(())
    }

    fn apply(&mut self, command: &ChartCommand) -> Result<Outcome, Rejection> {
        for facility in &mut self.facilities {
            if let Some(outcome) = facility.apply(command) {
                return outcome;
            }
        }
        match command {
            ChartCommand::Move { entity, point } => {
                self.move_to(*entity, *point)?;
                Ok(Outcome::Done)
            }
            _ => Err(Rejection::Unsupported("chart command")),
        }
    }

    fn compile_fields(&mut self) -> Result<FieldCost, DomainError> {
        let fields = self
            .fields
            .as_ref()
            .ok_or(DomainError::Unsupported("fields"))?;
        self.compiler.compile(&self.space, fields, &self.poses)
    }

    fn field_program(&self) -> &FieldProgram {
        self.compiler.program()
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
    pub(crate) space: S,
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
            compiler: FieldCompiler::new(),
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

    pub fn get(&self, id: DomainId) -> Option<&dyn Domain> {
        self.list.get(id.index()).map(|domain| domain.as_ref())
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

    /// The nearest hit by the root eye's projective depth across every view whose image space reaches the root, each cast with the ray pulled into its space.
    pub fn pick(
        &self,
        views: &Views,
        prepared: &[PreparedGeometry],
        ndc: [f32; 2],
    ) -> Option<Pick> {
        self.nearest(views, prepared, ndc, false)
    }

    pub fn pick_lifted(
        &self,
        views: &Views,
        prepared: &[PreparedGeometry],
        ndc: [f32; 2],
    ) -> Option<Pick> {
        self.nearest(views, prepared, ndc, true)
    }

    fn nearest(
        &self,
        views: &Views,
        prepared: &[PreparedGeometry],
        ndc: [f32; 2],
        lifted_only: bool,
    ) -> Option<Pick> {
        let mut nearest: Option<Pick> = None;
        for domain in self.iter() {
            for target in domain.views() {
                if lifted_only
                    && !domain
                        .view(target.view)
                        .is_some_and(|summary| summary.ray_lift)
                {
                    continue;
                }
                let Some(placement) = views.to_root(target.image).and_then(|to| to.rigid()) else {
                    continue;
                };
                let Some(ray) = views.ray(target.image, ndc) else {
                    continue;
                };
                let Some(pick) = domain.pick(target.view, &ray, placement, views, prepared) else {
                    continue;
                };
                if nearest.is_none_or(|best| pick.depth > best.depth) {
                    nearest = Some(pick);
                }
            }
        }
        nearest
    }
}
