use std::any::Any;
use std::borrow::Cow;
use std::fmt;
use std::marker::PhantomData;
use std::ops::Mul;

pub use loam_shape::field::FieldKind;

use loam_shape::field::DistanceField;

use loam_math::blended::{
    gauss_newton_log_checked, BlendedSpace, BlendingField, ConformallyFlat, GEODESIC_DEFAULT_STEPS,
    LOG_MAX_ITERS, LOG_RESIDUAL_TOL,
};
use loam_math::hyperbolic::{in_poincare_ball, poincare_to_hyperboloid};
use loam_math::{
    EuclideanR3, EuclideanR4, HyperbolicH3, Iso3, Iso3H, Iso4Flat, IsometryGroup, Mat3, Rotor4,
    Space, WPlane, WgslSpace,
};
use loam_shape::polytope::{polytope_section_faces_append, polytope_section_perimeter_append};

use crate::command::{Outcome, Rejection};
use crate::entity::{Entity, RuntimeId, SceneId};
use crate::field::{
    self, FieldCompiler, FieldCost, FieldError, FieldNode, FieldOp, FieldPrimitive,
};
use crate::phase::Step;
use crate::session::{
    Library, MaterialId, PaletteId, PreparedGeometry, PreparedId, RestoreError, Stamp,
};
use crate::store::{LogCapacity, Store, StoreField, StoreSnapshot};
use crate::view::{
    self, DomainRay, ImageRay, ImageSpaceId, InstanceRecord, Pick, Rigid, SegmentRecord,
    TriangleRecord, Vec3, Vec4, ViewId, ViewMapping, ViewRecords, ViewSpec, ViewSummary,
    ViewTarget, Views,
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Instance {
    pub geometry: PreparedId,
    pub material: MaterialId,
    pub shading: EdgeShading,
    /// The perimeter's line material; `None` publishes no section for a geometry that could be cut.
    pub section: Option<MaterialId>,
}

impl Instance {
    pub fn new(geometry: PreparedId, material: MaterialId) -> Self {
        Self {
            geometry,
            material,
            shading: EdgeShading::Material,
            section: None,
        }
    }

    pub fn shaded(mut self, shading: EdgeShading) -> Self {
        self.shading = shading;
        self
    }

    pub fn sectioned(mut self, perimeter: MaterialId) -> Self {
        self.section = Some(perimeter);
        self
    }
}

/// Where publication reads a segment's colour; the default is the material's line colour.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum EdgeShading {
    #[default]
    Material,
    /// Two colours per prepared segment, its start then its end, in the prepared geometry's own order.
    Palette(PaletteId),
    /// Reads `back` at `-extent` and `front` at `+extent` of an endpoint's last chart coordinate relative to the entity's origin, clamped between.
    Depth {
        back: [f32; 4],
        front: [f32; 4],
        extent: f32,
    },
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
    Space<Vector: Mul<f32, Output = <Self as Space>::Vector>> + Send + Sync + Sized + 'static
{
    type Placement: Copy + Send + Sync + 'static;

    type Relative: Copy + Send + Sync + 'static;

    fn origin(&self) -> Self::Point;

    /// Names the first non-finite coordinate or reports the chart boundary; nothing is clamped.
    fn check(&self, point: Self::Point) -> Result<(), DomainError>;

    /// The coordinates a chart point carries, 4 for R⁴ and 3 for R³ and H³.
    fn chart_dimension(&self) -> u32;

    fn chart_point(&self, point: Self::Point) -> ChartPoint;

    /// Reads the leading coordinates the space has and ignores the rest.
    fn local_point(&self, coordinates: [f32; 4]) -> Self::Point;

    fn chart_pose(&self, pose: &Pose<Self>) -> ChartPose;

    fn pose_from_chart(&self, pose: &ChartPose) -> Result<Pose<Self>, DomainError>;

    fn tangent_from_chart(&self, tangent: &ChartTangent) -> Result<Self::Vector, DomainError>;

    /// Metric distance from the origin to a point at chart radius `radius`.
    fn chart_reach(&self, at: Self::Point, radius: f32) -> f32;

    /// Arc length along `ray` into the metric ball of `radius` around `center`: zero from inside, `None` on a miss.
    fn hit_ball(&self, ray: &DomainRay<Self>, center: Self::Point, radius: f32) -> Option<f32>;

    fn prepare(&self, pose: &Pose<Self>) -> Self::Placement;

    fn place(&self, placement: &Self::Placement, local: Self::Point) -> Self::Point;

    fn local(&self, pose: &Pose<Self>, point: Self::Point) -> Result<Self::Point, DomainError>;

    fn relative(&self, eye: &Pose<Self>, pose: &Pose<Self>) -> Result<Self::Relative, DomainError>;

    fn place_relative(
        &self,
        relative: &Self::Relative,
        local: Self::Point,
    ) -> Result<Self::Point, DomainError>;

    fn carry(
        &self,
        pose: &Pose<Self>,
        local: Self::Point,
        tangent: Self::Vector,
    ) -> Result<Self::Vector, DomainError>;

    fn moved(&self, pose: &Pose<Self>, to: Self::Point) -> Result<Pose<Self>, DomainError>;

    fn walk(
        &self,
        pose: &Pose<Self>,
        tangent: Self::Vector,
        dt: f32,
    ) -> Result<Pose<Self>, DomainError> {
        let step = self.carry(pose, self.origin(), tangent * dt)?;
        let to = self.exp(pose.point, step);
        self.check(to)?;
        self.moved(pose, to)
    }
}

pub trait Homogeneous: DomainSpace + IsometryGroup {
    fn iso_of(&self, pose: &Pose<Self>) -> Self::Iso;

    fn pose_of(&self, iso: Self::Iso) -> Pose<Self>;

    /// Moves the origin to `to` along their geodesic, carrying the frame by parallel transport.
    fn transvection(&self, to: Self::Point) -> Self::Iso;
}

#[cfg(test)]
pub(crate) mod applied {
    use std::cell::Cell;

    thread_local! {
        static COUNT: Cell<u64> = const { Cell::new(0) };
    }

    pub(crate) fn bump() {
        COUNT.with(|count| count.set(count.get() + 1));
    }

    pub(crate) fn taken() -> u64 {
        COUNT.with(|count| count.replace(0))
    }
}

pub fn homogeneous_place<S: Homogeneous>(
    space: &S,
    placement: &S::Iso,
    local: S::Point,
) -> S::Point {
    #[cfg(test)]
    applied::bump();
    space.iso_apply(*placement, local)
}

pub fn homogeneous_local<S: Homogeneous>(
    space: &S,
    pose: &Pose<S>,
    point: S::Point,
) -> Result<S::Point, DomainError> {
    #[cfg(test)]
    applied::bump();
    Ok(space.iso_apply(space.iso_inverse(space.iso_of(pose)), point))
}

pub fn homogeneous_carry<S: Homogeneous>(
    space: &S,
    pose: &Pose<S>,
    local: S::Point,
    tangent: S::Vector,
) -> Result<S::Vector, DomainError> {
    #[cfg(test)]
    applied::bump();
    Ok(space.iso_transport(space.iso_of(pose), local, tangent))
}

pub fn homogeneous_relative<S: Homogeneous>(
    space: &S,
    eye: &Pose<S>,
    pose: &Pose<S>,
) -> Result<S::Iso, DomainError> {
    Ok(space.iso_compose(space.iso_inverse(space.iso_of(eye)), space.iso_of(pose)))
}

pub fn homogeneous_place_relative<S: Homogeneous>(
    space: &S,
    relative: &S::Iso,
    local: S::Point,
) -> Result<S::Point, DomainError> {
    #[cfg(test)]
    applied::bump();
    Ok(space.iso_apply(*relative, local))
}

pub fn homogeneous_moved<S: Homogeneous>(
    space: &S,
    pose: &Pose<S>,
    to: S::Point,
) -> Result<Pose<S>, DomainError> {
    space.check(to)?;
    let frame = space.iso_compose(
        space.iso_inverse(space.transvection(pose.point)),
        space.iso_of(pose),
    );
    Ok(space.pose_of(space.iso_compose(space.transvection(to), frame)))
}

pub fn homogeneous_walk<S: Homogeneous>(
    space: &S,
    pose: &Pose<S>,
    tangent: S::Vector,
    dt: f32,
) -> Result<Pose<S>, DomainError> {
    let step = space.exp(space.origin(), tangent * dt);
    space.check(step)?;
    // Helgason, Differential Geometry, Lie Groups, and Symmetric Spaces, 1978, Ch. IV §3: the transvection along a geodesic is parallel transport along it.
    let next = space.pose_of(space.iso_compose(space.iso_of(pose), space.transvection(step)));
    space.check(next.point)?;
    Ok(next)
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

fn put_row<T>(store: &mut Store<T>, entity: Entity, row: T) -> Result<(), Rejection> {
    match store.get_mut(entity) {
        Some(slot) => *slot = row,
        None => store.insert(entity, row)?,
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
    type Placement = Iso4Flat;

    type Relative = Iso4Flat;

    fn origin(&self) -> Self::Point {
        Vec4::ZERO
    }

    fn check(&self, point: Self::Point) -> Result<(), DomainError> {
        finite(point.to_array())
    }

    fn chart_dimension(&self) -> u32 {
        4
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

    fn chart_pose(&self, pose: &Pose<Self>) -> ChartPose {
        ChartPose {
            chart: ChartId(0),
            coordinates: pose.point.to_array(),
            frame: pose.frame.to_mat4(),
        }
    }

    fn pose_from_chart(&self, pose: &ChartPose) -> Result<Pose<Self>, DomainError> {
        finite(pose.coordinates)?;
        let frame = Rotor4::from_mat4(&pose.frame).ok_or(DomainError::InvalidFrame)?;
        Ok(Pose {
            point: Vec4::from_array(pose.coordinates),
            frame,
        })
    }

    fn tangent_from_chart(&self, tangent: &ChartTangent) -> Result<Self::Vector, DomainError> {
        finite(tangent.vector)?;
        Ok(Vec4::from_array(tangent.vector))
    }

    fn chart_reach(&self, _at: Self::Point, radius: f32) -> f32 {
        radius
    }

    fn hit_ball(&self, ray: &DomainRay<Self>, center: Self::Point, radius: f32) -> Option<f32> {
        let offset = ray.origin - center;
        view::ball_entry(
            offset.dot(ray.direction),
            offset.length_squared() - radius * radius,
        )
    }

    fn prepare(&self, pose: &Pose<Self>) -> Self::Placement {
        self.iso_of(pose)
    }

    fn place(&self, placement: &Self::Placement, local: Self::Point) -> Self::Point {
        homogeneous_place(self, placement, local)
    }

    fn local(&self, pose: &Pose<Self>, point: Self::Point) -> Result<Self::Point, DomainError> {
        homogeneous_local(self, pose, point)
    }

    fn relative(&self, eye: &Pose<Self>, pose: &Pose<Self>) -> Result<Self::Relative, DomainError> {
        homogeneous_relative(self, eye, pose)
    }

    fn place_relative(
        &self,
        relative: &Self::Relative,
        local: Self::Point,
    ) -> Result<Self::Point, DomainError> {
        homogeneous_place_relative(self, relative, local)
    }

    fn carry(
        &self,
        pose: &Pose<Self>,
        local: Self::Point,
        tangent: Self::Vector,
    ) -> Result<Self::Vector, DomainError> {
        homogeneous_carry(self, pose, local, tangent)
    }

    fn moved(&self, pose: &Pose<Self>, to: Self::Point) -> Result<Pose<Self>, DomainError> {
        homogeneous_moved(self, pose, to)
    }

    fn walk(
        &self,
        pose: &Pose<Self>,
        tangent: Self::Vector,
        dt: f32,
    ) -> Result<Pose<Self>, DomainError> {
        homogeneous_walk(self, pose, tangent, dt)
    }
}

impl Homogeneous for EuclideanR4 {
    fn iso_of(&self, pose: &Pose<Self>) -> Self::Iso {
        Iso4Flat {
            rotation: pose.frame,
            translation: pose.point,
        }
    }

    fn pose_of(&self, iso: Self::Iso) -> Pose<Self> {
        Pose {
            point: iso.translation,
            frame: iso.rotation,
        }
    }

    fn transvection(&self, to: Self::Point) -> Self::Iso {
        Iso4Flat::from_translation(to)
    }
}

impl From<Iso4Flat> for Pose<EuclideanR4> {
    fn from(iso: Iso4Flat) -> Self {
        EuclideanR4.pose_of(iso)
    }
}

impl DomainSpace for HyperbolicH3 {
    type Placement = Iso3H;

    type Relative = Iso3H;

    fn origin(&self) -> Self::Point {
        Vec3::ZERO
    }

    fn check(&self, point: Self::Point) -> Result<(), DomainError> {
        finite(point.extend(0.0).to_array())?;
        in_poincare_ball(point)
            .then_some(())
            .ok_or(DomainError::ChartBoundary)
    }

    fn chart_dimension(&self) -> u32 {
        3
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

    fn chart_pose(&self, pose: &Pose<Self>) -> ChartPose {
        let position = pose.point;
        let rotation = self.iso_compose(
            self.iso_inverse(Iso3H::from_translation(position)),
            pose.frame,
        );
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

    fn pose_from_chart(&self, pose: &ChartPose) -> Result<Pose<Self>, DomainError> {
        finite(pose.coordinates)?;
        let position = Vec3::from_slice(&pose.coordinates[..3]);
        self.check(position)?;
        let [x, y, z] = frame_columns(&pose.frame)?;
        let mut rotation = Iso3H::IDENTITY;
        rotation.matrix.x_axis = x.extend(0.0);
        rotation.matrix.y_axis = y.extend(0.0);
        rotation.matrix.z_axis = z.extend(0.0);
        Ok(self.pose_of(self.iso_compose(Iso3H::from_translation(position), rotation)))
    }

    fn tangent_from_chart(&self, tangent: &ChartTangent) -> Result<Self::Vector, DomainError> {
        finite(tangent.vector)?;
        Ok(Vec3::from_slice(&tangent.vector[..3]))
    }

    fn chart_reach(&self, _at: Self::Point, radius: f32) -> f32 {
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

    fn prepare(&self, pose: &Pose<Self>) -> Self::Placement {
        self.iso_of(pose)
    }

    fn place(&self, placement: &Self::Placement, local: Self::Point) -> Self::Point {
        homogeneous_place(self, placement, local)
    }

    fn local(&self, pose: &Pose<Self>, point: Self::Point) -> Result<Self::Point, DomainError> {
        homogeneous_local(self, pose, point)
    }

    fn relative(&self, eye: &Pose<Self>, pose: &Pose<Self>) -> Result<Self::Relative, DomainError> {
        homogeneous_relative(self, eye, pose)
    }

    fn place_relative(
        &self,
        relative: &Self::Relative,
        local: Self::Point,
    ) -> Result<Self::Point, DomainError> {
        homogeneous_place_relative(self, relative, local)
    }

    fn carry(
        &self,
        pose: &Pose<Self>,
        local: Self::Point,
        tangent: Self::Vector,
    ) -> Result<Self::Vector, DomainError> {
        homogeneous_carry(self, pose, local, tangent)
    }

    fn moved(&self, pose: &Pose<Self>, to: Self::Point) -> Result<Pose<Self>, DomainError> {
        homogeneous_moved(self, pose, to)
    }

    fn walk(
        &self,
        pose: &Pose<Self>,
        tangent: Self::Vector,
        dt: f32,
    ) -> Result<Pose<Self>, DomainError> {
        homogeneous_walk(self, pose, tangent, dt)
    }
}

impl Homogeneous for HyperbolicH3 {
    fn iso_of(&self, pose: &Pose<Self>) -> Self::Iso {
        pose.frame
    }

    fn pose_of(&self, iso: Self::Iso) -> Pose<Self> {
        Pose {
            point: self.iso_apply(iso, Vec3::ZERO),
            frame: iso,
        }
    }

    fn transvection(&self, to: Self::Point) -> Self::Iso {
        Iso3H::from_translation(to)
    }
}

impl From<Iso3H> for Pose<HyperbolicH3> {
    fn from(iso: Iso3H) -> Self {
        HyperbolicH3.pose_of(iso)
    }
}

impl DomainSpace for EuclideanR3 {
    type Placement = Iso3;

    type Relative = Iso3;

    fn origin(&self) -> Self::Point {
        Vec3::ZERO
    }

    fn check(&self, point: Self::Point) -> Result<(), DomainError> {
        finite(point.extend(0.0).to_array())
    }

    fn chart_dimension(&self) -> u32 {
        3
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

    fn chart_pose(&self, pose: &Pose<Self>) -> ChartPose {
        ChartPose {
            chart: ChartId(0),
            coordinates: pose.point.extend(0.0).to_array(),
            frame: frame_of([Vec3::X, Vec3::Y, Vec3::Z].map(|axis| pose.frame * axis)),
        }
    }

    fn pose_from_chart(&self, pose: &ChartPose) -> Result<Pose<Self>, DomainError> {
        finite(pose.coordinates)?;
        let frame = rotation_of(frame_columns(&pose.frame)?).rotation;
        Ok(Pose {
            point: Vec3::from_slice(&pose.coordinates[..3]),
            frame,
        })
    }

    fn tangent_from_chart(&self, tangent: &ChartTangent) -> Result<Self::Vector, DomainError> {
        finite(tangent.vector)?;
        Ok(Vec3::from_slice(&tangent.vector[..3]))
    }

    fn chart_reach(&self, _at: Self::Point, radius: f32) -> f32 {
        radius
    }

    fn hit_ball(&self, ray: &DomainRay<Self>, center: Self::Point, radius: f32) -> Option<f32> {
        let offset = ray.origin - center;
        view::ball_entry(
            offset.dot(ray.direction),
            offset.length_squared() - radius * radius,
        )
    }

    fn prepare(&self, pose: &Pose<Self>) -> Self::Placement {
        self.iso_of(pose)
    }

    fn place(&self, placement: &Self::Placement, local: Self::Point) -> Self::Point {
        homogeneous_place(self, placement, local)
    }

    fn local(&self, pose: &Pose<Self>, point: Self::Point) -> Result<Self::Point, DomainError> {
        homogeneous_local(self, pose, point)
    }

    fn relative(&self, eye: &Pose<Self>, pose: &Pose<Self>) -> Result<Self::Relative, DomainError> {
        homogeneous_relative(self, eye, pose)
    }

    fn place_relative(
        &self,
        relative: &Self::Relative,
        local: Self::Point,
    ) -> Result<Self::Point, DomainError> {
        homogeneous_place_relative(self, relative, local)
    }

    fn carry(
        &self,
        pose: &Pose<Self>,
        local: Self::Point,
        tangent: Self::Vector,
    ) -> Result<Self::Vector, DomainError> {
        homogeneous_carry(self, pose, local, tangent)
    }

    fn moved(&self, pose: &Pose<Self>, to: Self::Point) -> Result<Pose<Self>, DomainError> {
        homogeneous_moved(self, pose, to)
    }

    fn walk(
        &self,
        pose: &Pose<Self>,
        tangent: Self::Vector,
        dt: f32,
    ) -> Result<Pose<Self>, DomainError> {
        homogeneous_walk(self, pose, tangent, dt)
    }
}

impl Homogeneous for EuclideanR3 {
    fn iso_of(&self, pose: &Pose<Self>) -> Self::Iso {
        Iso3 {
            rotation: pose.frame,
            translation: pose.point,
        }
    }

    fn pose_of(&self, iso: Self::Iso) -> Pose<Self> {
        Pose {
            point: iso.translation,
            frame: iso.rotation,
        }
    }

    fn transvection(&self, to: Self::Point) -> Self::Iso {
        Iso3::from_translation(to)
    }
}

impl From<Iso3> for Pose<EuclideanR3> {
    fn from(iso: Iso3) -> Self {
        EuclideanR3.pose_of(iso)
    }
}

const LOCAL_ERROR_BUDGET: f32 = 1.0e-3;

impl<A, B, F> DomainSpace for BlendedSpace<A, B, F>
where
    A: Space<Point = Vec3, Vector = Vec3> + ConformallyFlat + Send + Sync + 'static,
    B: Space<Point = Vec3, Vector = Vec3> + ConformallyFlat + Send + Sync + 'static,
    F: BlendingField,
{
    type Placement = Pose<Self>;

    type Relative = (Pose<Self>, Pose<Self>);

    fn origin(&self) -> Self::Point {
        Vec3::ZERO
    }

    fn check(&self, point: Self::Point) -> Result<(), DomainError> {
        finite(point.extend(0.0).to_array())?;
        self.valid_point(point)
            .then_some(())
            .ok_or(DomainError::ChartBoundary)
    }

    fn chart_dimension(&self) -> u32 {
        3
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

    fn chart_pose(&self, pose: &Pose<Self>) -> ChartPose {
        let relative = transported_frame(self, pose.point).inverse() * pose.frame;
        ChartPose {
            chart: ChartId(0),
            coordinates: pose.point.extend(0.0).to_array(),
            frame: frame_of([relative.x_axis, relative.y_axis, relative.z_axis]),
        }
    }

    fn pose_from_chart(&self, pose: &ChartPose) -> Result<Pose<Self>, DomainError> {
        finite(pose.coordinates)?;
        let point = Vec3::from_slice(&pose.coordinates[..3]);
        self.check(point)?;
        let [x, y, z] = frame_columns(&pose.frame)?;
        let frame = transported_frame(self, point) * Mat3::from_cols(x, y, z);
        Ok(Pose { point, frame })
    }

    fn tangent_from_chart(&self, tangent: &ChartTangent) -> Result<Self::Vector, DomainError> {
        finite(tangent.vector)?;
        Ok(Vec3::from_slice(&tangent.vector[..3]))
    }

    fn chart_reach(&self, at: Self::Point, radius: f32) -> f32 {
        self.conformal_factor(at).sqrt() * radius
    }

    fn hit_ball(&self, ray: &DomainRay<Self>, center: Self::Point, radius: f32) -> Option<f32> {
        let unit = ray.direction.try_normalize()?;
        let chart_radius = radius / self.conformal_factor(center).sqrt();
        let offset = ray.origin - center;
        let entry = view::ball_entry(
            offset.dot(unit),
            offset.length_squared() - chart_radius * chart_radius,
        )?;
        Some(entry / ray.direction.length())
    }

    fn prepare(&self, pose: &Pose<Self>) -> Self::Placement {
        *pose
    }

    fn place(&self, placement: &Self::Placement, local: Self::Point) -> Self::Point {
        self.exp(placement.point, placement.frame * local)
    }

    fn local(&self, pose: &Pose<Self>, point: Self::Point) -> Result<Self::Point, DomainError> {
        self.check(point)?;
        if self.conformal_factor(point).sqrt() * LOG_RESIDUAL_TOL > LOCAL_ERROR_BUDGET {
            return Err(DomainError::ErrorBudget);
        }
        let (chart, failure) = gauss_newton_log_checked(
            self,
            pose.point,
            point,
            GEODESIC_DEFAULT_STEPS,
            LOG_MAX_ITERS,
        );
        if failure.is_some() {
            return Err(DomainError::NoConvergence);
        }
        let inverse = pose.frame.inverse();
        if !inverse.is_finite() {
            return Err(DomainError::InvalidFrame);
        }
        Ok(inverse * chart)
    }

    fn relative(&self, eye: &Pose<Self>, pose: &Pose<Self>) -> Result<Self::Relative, DomainError> {
        Ok((*eye, *pose))
    }

    fn place_relative(
        &self,
        relative: &Self::Relative,
        local: Self::Point,
    ) -> Result<Self::Point, DomainError> {
        let (eye, pose) = relative;
        self.local(eye, self.place(pose, local))
    }

    fn carry(
        &self,
        pose: &Pose<Self>,
        local: Self::Point,
        tangent: Self::Vector,
    ) -> Result<Self::Vector, DomainError> {
        let to = self.place(&self.prepare(pose), local);
        self.check(to)?;
        let carried = self.parallel_transport(pose.point, to, pose.frame * tangent);
        carried
            .is_finite()
            .then_some(carried)
            .ok_or(DomainError::NoConvergence)
    }

    fn moved(&self, pose: &Pose<Self>, to: Self::Point) -> Result<Pose<Self>, DomainError> {
        self.check(to)?;
        let frame = Mat3::from_cols(
            self.parallel_transport(pose.point, to, pose.frame.x_axis),
            self.parallel_transport(pose.point, to, pose.frame.y_axis),
            self.parallel_transport(pose.point, to, pose.frame.z_axis),
        );
        if !frame.is_finite() {
            return Err(DomainError::NoConvergence);
        }
        Ok(Pose { point: to, frame })
    }
}

fn transported_frame<S>(space: &S, to: S::Point) -> Mat3
where
    S: DomainSpace<Point = Vec3, Vector = Vec3, Frame = Mat3>,
{
    let origin = space.origin();
    let base = space.frame_at(origin);
    Mat3::from_cols(
        space.parallel_transport(origin, to, base.x_axis),
        space.parallel_transport(origin, to, base.y_axis),
        space.parallel_transport(origin, to, base.z_axis),
    )
}

fn image_of<S: DomainSpace>(
    space: &S,
    mapping: &dyn ViewMapping<S>,
    eye: &Pose<S>,
    pose: &Pose<S>,
    placement: &S::Placement,
    relative: &S::Relative,
    local: [f32; 4],
) -> Option<[f32; 3]> {
    let local = space.local_point(local);
    space.check(space.place(placement, local)).ok()?;
    mapping.image_local(space, eye, pose, relative, local)
}

fn lerp_color(back: [f32; 4], front: [f32; 4], t: f32) -> [f32; 4] {
    let mut mixed = [0.0; 4];
    for (channel, value) in mixed.iter_mut().enumerate() {
        *value = back[channel] + (front[channel] - back[channel]) * t;
    }
    mixed
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
    let palette = match instance.shading {
        EdgeShading::Palette(id) => library.palette(id),
        _ => &[],
    };
    let placement = space.prepare(pose);
    let Ok(relative) = space.relative(eye, pose) else {
        return;
    };
    let origin_depth = space
        .chart_point(space.place(&placement, space.origin()))
        .coordinates[3];
    let depth_color = |local: [f32; 4]| match instance.shading {
        EdgeShading::Depth {
            back,
            front,
            extent,
        } => {
            let depth = space
                .chart_point(space.place(&placement, space.local_point(local)))
                .coordinates[3]
                - origin_depth;
            Some(lerp_color(
                back,
                front,
                (depth / extent.max(1e-6) * 0.5 + 0.5).clamp(0.0, 1.0),
            ))
        }
        _ => None,
    };
    let mut push = |index: usize, a: [f32; 4], b: [f32; 4]| {
        let (Some(start), Some(end)) = (
            image_of(space, mapping, eye, pose, &placement, &relative, a),
            image_of(space, mapping, eye, pose, &placement, &relative, b),
        ) else {
            return;
        };
        let painted = |at: usize| palette.get(at).copied().unwrap_or(color);
        into.push(SegmentRecord {
            start,
            _pad0: 0.0,
            end,
            _pad1: 0.0,
            start_color: depth_color(a).unwrap_or_else(|| painted(index * 2)),
            end_color: depth_color(b).unwrap_or_else(|| painted(index * 2 + 1)),
            width_px,
            _pad2: [0.0; 3],
        });
    };
    match geometry {
        PreparedGeometry::Lines4 { segments } => {
            for (index, &[a, b]) in segments.iter().enumerate() {
                push(index, a, b);
            }
        }
        PreparedGeometry::Lines3 { segments } => {
            for (index, &[a, b]) in segments.iter().enumerate() {
                push(index, [a[0], a[1], a[2], 0.0], [b[0], b[1], b[2], 0.0]);
            }
        }
        PreparedGeometry::Polytope4 { polytope, scale } => {
            let topology = polytope.topology();
            for (index, &[a, b]) in topology.edges.iter().enumerate() {
                push(
                    index,
                    (topology.vertices[a as usize] * *scale).to_array(),
                    (topology.vertices[b as usize] * *scale).to_array(),
                );
            }
        }
        PreparedGeometry::Mesh3 { .. } => {}
    }
}

fn push_section<S: DomainSpace>(
    space: &S,
    mapping: &dyn ViewMapping<S>,
    eye: &Pose<S>,
    pose: &Pose<S>,
    library: &Library<'_>,
    instance: &Instance,
    into: &mut ViewRecords,
) {
    let Some(PreparedGeometry::Polytope4 { polytope, scale }) =
        library.geometry.get(instance.geometry.index())
    else {
        return;
    };
    let (Some(perimeter_material), Some(cut)) = (instance.section, mapping.section(eye, pose))
    else {
        return;
    };
    let Ok(relative) = space.relative(eye, pose) else {
        return;
    };
    let Some(place) = mapping.image_local(space, eye, pose, &relative, space.origin()) else {
        return;
    };
    let Ok(origin) = space.place_relative(&relative, space.origin()) else {
        return;
    };
    let base = Vec4::from(space.chart_point(origin).coordinates);
    let topology = polytope.topology();
    let scratch = &mut into.scratch;
    scratch.rotated.clear();
    for vertex in topology.vertices.iter() {
        let local = space.local_point((*vertex * *scale).to_array());
        let Ok(placed) = space.place_relative(&relative, local) else {
            scratch.rotated.clear();
            return;
        };
        scratch
            .rotated
            .push(Vec4::from(space.chart_point(placed).coordinates) - base);
    }
    let plane = WPlane::new(cut.offset);
    let placed = |point: [f32; 3]| {
        [
            point[0] * cut.scale + place[0],
            point[1] * cut.scale + place[1],
            point[2] * cut.scale + place[2],
        ]
    };

    let (edge_color, width_px) = library.line_style(perimeter_material);
    scratch.perimeter.segments.clear();
    scratch.perimeter.colors.clear();
    scratch.perimeter.widths.clear();
    polytope_section_perimeter_append(
        topology.edges,
        topology.cells,
        &scratch.rotated,
        plane,
        &mut scratch.cut,
        &mut scratch.perimeter,
    );
    for (start, end) in &scratch.perimeter.segments {
        into.segments.push(SegmentRecord {
            start: placed(*start),
            _pad0: 0.0,
            end: placed(*end),
            _pad1: 0.0,
            start_color: edge_color,
            end_color: edge_color,
            width_px,
            _pad2: [0.0; 3],
        });
    }

    let (fill_color, _) = library.line_style(instance.material);
    scratch.faces.vertices.clear();
    scratch.faces.colors.clear();
    scratch.faces.indices.clear();
    polytope_section_faces_append(
        topology.edges,
        topology.cells,
        &scratch.rotated,
        plane,
        fill_color,
        &mut scratch.cut,
        &mut scratch.faces,
    );
    for [a, b, c] in &scratch.faces.indices {
        into.triangles.push(TriangleRecord {
            vertices: [
                placed(scratch.faces.vertices[*a as usize]),
                placed(scratch.faces.vertices[*b as usize]),
                placed(scratch.faces.vertices[*c as usize]),
            ],
            color: fill_color,
        });
    }
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

/// Engine-owned work inside one domain: `step` runs in the simulation phase's domain-step entry, `release` at dispatch before the stores forget the entity, and `apply` gets first offer of a chart command.
pub trait Facility<S: DomainSpace>: Any + Send + 'static {
    fn name(&self) -> &'static str;

    fn step(&mut self, poses: &mut Store<Pose<S>>, step: Step) -> Result<(), DomainError>;

    fn snapshot(&self) -> Box<dyn Any + Send>;

    /// Refuses whatever `restore` would refuse, changing nothing; the default accepts, and the session calls it on every facility before the first one restores.
    fn check_restore(&self, _from: &(dyn Any + Send)) -> Result<(), RestoreError> {
        Ok(())
    }

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

    /// Refuses whatever `restore` would refuse, changing nothing.
    fn check_restore(&self, from: &DomainSnapshot) -> Result<(), RestoreError>;

    fn restore(&mut self, from: &DomainSnapshot, scene: SceneId) -> Result<(), RestoreError>;

    fn apply(&mut self, command: &ChartCommand) -> Result<Outcome, Rejection>;

    fn compile_fields(&mut self) -> Result<FieldCost, DomainError>;

    fn field_program(&self) -> &FieldProgram;

    fn shader_prelude(&self) -> Option<&str>;

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
    prelude: Option<Cow<'static, str>>,
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
        let space = &self.space;
        let pose = self
            .poses
            .get_mut(entity)
            .ok_or(DomainError::Stale(entity))?;
        let next = space.walk(pose, velocity, dt)?;
        *pose = next;
        Ok(())
    }

    /// Moves the origin to `point` by transvection, keeping the frame.
    pub fn move_to(&mut self, entity: Entity, point: ChartPoint) -> Result<(), DomainError> {
        finite(point.coordinates)?;
        let target = self.space.local_point(point.coordinates);
        self.space.check(target)?;
        let space = &self.space;
        let pose = self
            .poses
            .get_mut(entity)
            .ok_or(DomainError::Stale(entity))?;
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
        into.segments.clear();
        into.triangles.clear();
        for (entity, instance) in self.instances.iter() {
            let Some(pose) = poses.get(entity) else {
                continue;
            };
            push_section(space, mapping, eye, pose, &library, instance, into);
        }
        let segments = &mut into.segments;
        let records = self.instances.iter().filter_map(|(entity, instance)| {
            let pose = poses.get(entity)?;
            let relative = space.relative(eye, pose).ok()?;
            let image_point = mapping.image_local(space, eye, pose, &relative, origin)?;
            push_segments(space, mapping, eye, pose, &library, instance, segments);
            Some((
                entity,
                InstanceRecord {
                    entity,
                    geometry: instance.geometry,
                    material: instance.material,
                    pose: space.chart_pose(pose),
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
            let center = self.space.place(&self.space.prepare(pose), origin);
            let radius = geometry.bounding_radius();
            let hit = match &lifted {
                Some(domain_ray) => {
                    let reach = self.space.chart_reach(center, radius);
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
        let relative = self.space.relative(eye, pose).ok()?;
        spec.mapping
            .image_local(&self.space, eye, pose, &relative, self.space.origin())
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

    fn check_restore(&self, from: &DomainSnapshot) -> Result<(), RestoreError> {
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
        for (facility, snapshot) in self.facilities.iter().zip(&from.facilities) {
            facility.check_restore(snapshot.as_ref())?;
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
            facility.restore(snapshot.as_ref())?;
        }
        self.scene = scene;
        StoreField::restore(&mut self.poses, &from.poses, scene);
        StoreField::restore(&mut self.instances, &from.instances, scene);
        if let (Some(fields), Some(snapshot)) = (&mut self.fields, &from.fields) {
            StoreField::restore(fields, snapshot, scene);
            for field in fields.rows_mut_untracked() {
                for operand in &mut field.operands {
                    *operand = Entity::new(scene, operand.key());
                }
            }
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
            ChartCommand::Place { entity, pose } => {
                let placed = self.space.pose_from_chart(pose)?;
                put_row(&mut self.poses, *entity, placed)?;
                Ok(Outcome::Done)
            }
            ChartCommand::Attach { entity, instance } => {
                put_row(&mut self.instances, *entity, *instance)?;
                Ok(Outcome::Done)
            }
            ChartCommand::Walk {
                entity,
                tangent,
                dt,
            } => {
                let velocity = self.space.tangent_from_chart(tangent)?;
                self.walk(*entity, velocity, *dt)?;
                Ok(Outcome::Done)
            }
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

    fn shader_prelude(&self) -> Option<&str> {
        self.prelude.as_deref()
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
    prelude: Option<Cow<'static, str>>,
}

impl<S: DomainSpace> DomainBuilder<S> {
    pub fn new(name: &'static str, space: S) -> Self {
        Self {
            name,
            space,
            tracking: None,
            fields: false,
            facilities: Vec::new(),
            prelude: None,
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
            prelude: self.prelude,
        }
    }
}

impl<S: DomainSpace + WgslSpace> DomainBuilder<S> {
    pub fn marched(mut self) -> Self {
        self.prelude = Some(self.space.wgsl_impl());
        self
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

    /// The nearest hit by the root eye's projective depth across every view whose image space reaches the root, lifted or not, each cast with the ray pulled into its space.
    pub fn pick(
        &self,
        views: &Views,
        prepared: &[PreparedGeometry],
        ndc: [f32; 2],
    ) -> Option<Pick> {
        self.nearest(views, prepared, ndc, false)
    }

    /// The same search as `pick` over only the views with a ray lift, so a grab lands on a view it can drag.
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

#[cfg(test)]
mod tests {
    use loam_shape::polytope::Polytope4;

    use super::*;
    use crate::command::SpawnBundle;
    use crate::session::{Material, Publication, Session, SimConfig};
    use crate::store::LogCapacity;
    use crate::view::{DepthEnvelope, Projection4, Section4};

    crate::stores! {
        #[derive(Default)]
        pub struct Probe {
            tags: Store<u8>,
        }
    }

    const SEGMENTS: usize = 3;
    const EYE_AT: Vec4 = Vec4::new(0.0, 0.0, 0.0, 2.0);
    const OBJECT_AT: Vec4 = Vec4::new(0.0, 0.0, -4.0, 0.0);

    struct Nonlinear(Projection4);

    impl ViewMapping<EuclideanR4> for Nonlinear {
        fn name(&self) -> &'static str {
            "nonlinear"
        }

        fn image_point(&self, eye: &Pose<EuclideanR4>, point: Vec4) -> Option<[f32; 3]> {
            self.0.image_point(eye, point)
        }

        fn image_local(
            &self,
            space: &EuclideanR4,
            _eye: &Pose<EuclideanR4>,
            _pose: &Pose<EuclideanR4>,
            relative: &<EuclideanR4 as DomainSpace>::Relative,
            local: Vec4,
        ) -> Option<[f32; 3]> {
            let origin = space.place_relative(relative, Vec4::ZERO).ok()?;
            let point = space.place_relative(relative, local).ok()?;
            let placed = (point - origin).truncate() + origin.truncate();
            Some(placed.to_array())
        }

        fn lift(
            &self,
            _eye: &Pose<EuclideanR4>,
            _ray: &ImageRay,
        ) -> Option<DomainRay<EuclideanR4>> {
            None
        }

        fn ray_lift(&self) -> bool {
            false
        }

        fn depth_envelope(&self) -> DepthEnvelope {
            self.0.depth_envelope()
        }
    }

    fn stage(
        geometry: PreparedGeometry,
        mapping: impl ViewMapping<EuclideanR4>,
        section: bool,
    ) -> (Session<Probe>, u64) {
        let mut session = Session::new(Probe::default(), SimConfig::default());
        let r4 = session
            .register_domain(DomainBuilder::new("r4", EuclideanR4).tracked(LogCapacity::default()));
        let prepared = session.prepare(geometry);
        let material = session.add_material(Material::flat([1.0; 4]));
        let root = session.views().root();
        session.dispatch(|d| {
            let eye = d
                .spawn(SpawnBundle::new().at(r4, Pose::at(EYE_AT)))
                .expect("eye");
            let mut instance = Instance::new(prepared, material);
            if section {
                instance = instance.sectioned(material);
            }
            d.spawn(
                SpawnBundle::new()
                    .at(r4, Pose::at(OBJECT_AT))
                    .instance(instance),
            )
            .expect("object");
            d.domains
                .typed(r4)
                .expect("the r4 domain")
                .add_view(ViewSpec::new(root, eye, mapping));
        });
        let mut publication = Publication::default();
        let _ = applied::taken();
        session.publish(&mut publication).expect("publish");
        (session, applied::taken())
    }

    fn lines() -> PreparedGeometry {
        PreparedGeometry::Lines4 {
            segments: (0..SEGMENTS)
                .map(|i| {
                    let x = 0.1 + i as f32 * 0.1;
                    [[x, 0.0, 0.0, 0.0], [-x, 0.0, 0.0, 0.0]]
                })
                .collect(),
        }
    }

    #[test]
    fn a_published_segment_vertex_costs_more_isometry_applications_than_it_did() {
        let (_, applications) = stage(lines(), Section4 { w: 0.0 }, false);
        assert_eq!(applications, 3 + 6 * SEGMENTS as u64);
    }

    #[test]
    fn a_published_section_vertex_costs_more_isometry_applications_than_it_did() {
        let polytope = || PreparedGeometry::Polytope4 {
            polytope: Polytope4::Tesseract,
            scale: 0.25,
        };
        let (_, plain) = stage(polytope(), Section4 { w: 0.0 }, false);
        let (session, sectioned) = stage(polytope(), Section4 { w: 0.0 }, true);
        let vertices = Polytope4::Tesseract.topology().vertices.len() as u64;
        assert_eq!(session.domains().len(), 1);
        assert_eq!(sectioned - plain, 3 + vertices);
    }

    #[test]
    fn a_nonlinear_map_vertex_costs_more_isometry_applications_than_it_did() {
        let (_, applications) = stage(lines(), Nonlinear(Projection4 { focal: 2.0 }), false);
        assert_eq!(applications, 3 + 6 * SEGMENTS as u64);
    }
}
