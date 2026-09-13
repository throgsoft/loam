use std::any::Any;
use std::fmt;
use std::marker::PhantomData;
use std::ops::Mul;

pub use loam_shape::field::FieldKind;

use loam_shape::field::DistanceField;

use loam_math::blended::{
    geodesic_log_checked, integrate_geodesic_frame_checked, integrate_geodesic_frame_to_checked,
    BlendedSpace, BlendingField, ConformallyFlat, GeodesicError, GEODESIC_DEFAULT_STEPS,
    GEODESIC_ERROR_BUDGET, LOG_MAX_ITERS, LOG_RESIDUAL_TOL,
};
use loam_math::hyperbolic::{in_poincare_ball, poincare_to_hyperboloid};
use loam_math::{
    EuclideanR3, EuclideanR4, HyperbolicH3, Iso3, Iso3H, Iso4Flat, IsometryGroup, Mat3, Rotor4,
    Space, WPlane,
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
use crate::store::{Change, LogCapacity, Owner, Store, StoreError, StoreField, StoreSnapshot};
use crate::view::{
    self, DomainRay, EntityOutput, ImageRay, ImageSpaceId, InstanceRecord, Pick, RefusalSource,
    Rigid, SegmentRecord, TriangleRecord, Vec3, Vec4, ViewId, ViewMapping, ViewRecords,
    ViewRefusals, ViewSpec, ViewStyle, ViewSummary, ViewTarget, Views,
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
    Grab {
        entity: Entity,
        point: ChartPoint,
    },
    Release {
        entity: Entity,
    },
}

impl ChartCommand {
    pub(crate) fn entity(&self) -> Entity {
        match *self {
            Self::Place { entity, .. }
            | Self::Attach { entity, .. }
            | Self::Walk { entity, .. }
            | Self::Move { entity, .. }
            | Self::Grab { entity, .. }
            | Self::Release { entity } => entity,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Instance {
    pub geometry: PreparedId,
    pub material: MaterialId,
    pub shading: EdgeShading,
    pub line_width_px: Option<f32>,
    pub line_opacity: Option<f32>,
    /// The perimeter's line material; `None` publishes no section for a geometry that could be cut.
    pub section: Option<MaterialId>,
}

impl Instance {
    pub fn new(geometry: PreparedId, material: MaterialId) -> Self {
        Self {
            geometry,
            material,
            shading: EdgeShading::Material,
            line_width_px: None,
            line_opacity: None,
            section: None,
        }
    }

    pub fn shaded(mut self, shading: EdgeShading) -> Self {
        self.shading = shading;
        self
    }

    pub fn line_style(mut self, width_px: f32, opacity: f32) -> Self {
        self.line_width_px = Some(width_px);
        self.line_opacity = Some(opacity);
        self
    }

    pub fn sectioned(mut self, perimeter: MaterialId) -> Self {
        self.section = Some(perimeter);
        self
    }
}

/// Where publication reads a segment's color; the default is the material's line color.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum EdgeShading {
    #[default]
    Material,
    /// Two colors per prepared segment, its start then its end, in the prepared geometry's own order.
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

/// A space a domain is built over, naming no isometry group; its poses cross the facade as chart data, and a space with a group implements `Homogeneous` as well.
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

    fn point_from_chart(&self, point: &ChartPoint) -> Result<Self::Point, DomainError> {
        single_chart(point.chart)?;
        finite(point.coordinates)?;
        let point = self.local_point(point.coordinates);
        self.check(point)?;
        Ok(point)
    }

    fn chart_pose(&self, pose: &Pose<Self>) -> ChartPose;

    fn pose_from_chart(&self, pose: &ChartPose) -> Result<Pose<Self>, DomainError>;

    fn tangent_from_chart(&self, tangent: &ChartTangent) -> Result<Self::Vector, DomainError>;

    /// Metric distance across a chart radius `radius` around `at`; H³ converts at the chart origin whatever `at` is, so a displaced landmark's pick radius is the origin's.
    fn chart_reach(&self, at: Self::Point, radius: f32) -> f32;

    /// Arc length along `ray` into the metric ball of `radius` around `center`: zero from inside, `None` on a miss.
    fn hit_ball(&self, ray: &DomainRay<Self>, center: Self::Point, radius: f32) -> Option<f32>;

    /// What `place` needs from a pose, derived once per entity; a homogeneous space's isometry.
    fn prepare(&self, pose: &Pose<Self>) -> Self::Placement;

    /// The checked point at `local` in the entity's frame, one isometry application per vertex on a homogeneous space.
    fn place(
        &self,
        placement: &Self::Placement,
        local: Self::Point,
    ) -> Result<Self::Point, DomainError>;

    /// `point` in the entity's own frame; a blended chart refuses with `NoConvergence` when its log fails to converge and `ErrorBudget` when the metric error would exceed 1e-3, never returning a best guess.
    fn local(&self, pose: &Pose<Self>, point: Self::Point) -> Result<Self::Point, DomainError>;

    /// Homogeneous spaces cache the eye-relative isometry; blended spaces solve the local map per point.
    fn relative(&self, eye: &Pose<Self>, pose: &Pose<Self>) -> Result<Self::Relative, DomainError>;

    /// `place` into the eye's frame through a prepared `relative`.
    fn place_relative(
        &self,
        relative: &Self::Relative,
        local: Self::Point,
    ) -> Result<Self::Point, DomainError>;

    /// Transports `tangent`, given in the entity's frame, to the point at `local`, refusing a transport that does not converge with `NoConvergence`.
    fn carry(
        &self,
        pose: &Pose<Self>,
        local: Self::Point,
        tangent: Self::Vector,
    ) -> Result<Self::Vector, DomainError>;

    /// The pose at `to` with the frame carried from the pose's point: transvection on a homogeneous space, per-column transport on a conformally flat chart.
    fn moved(&self, pose: &Pose<Self>, to: Self::Point) -> Result<Pose<Self>, DomainError>;

    /// The pose after `dt` along `tangent`, read in the pose's frame at its point: `ChartBoundary` past the chart, `InvalidCoordinate` for a non-finite step, `NoConvergence` when a transport fails, and never a best guess.
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

/// A domain space with an isometry group: `iso_of` and `pose_of` convert a pose to and from a group element, the `homogeneous_` helpers implement the capabilities through it, and the physics facility requires it.
pub trait Homogeneous: DomainSpace + IsometryGroup {
    fn iso_of(&self, pose: &Pose<Self>) -> Self::Iso;

    fn pose_of(&self, iso: Self::Iso) -> Pose<Self>;

    /// Moves the origin to `to` along their geodesic, carrying the frame by parallel transport.
    fn transvection(&self, to: Self::Point) -> Self::Iso;
}

pub fn homogeneous_place<S: Homogeneous>(
    space: &S,
    placement: &S::Iso,
    local: S::Point,
) -> Result<S::Point, DomainError> {
    let point = space.iso_apply(*placement, local);
    space.check(point)?;
    Ok(point)
}

pub fn homogeneous_local<S: Homogeneous>(
    space: &S,
    pose: &Pose<S>,
    point: S::Point,
) -> Result<S::Point, DomainError> {
    Ok(space.iso_apply(space.iso_inverse(space.iso_of(pose)), point))
}

pub fn homogeneous_carry<S: Homogeneous>(
    space: &S,
    pose: &Pose<S>,
    local: S::Point,
    tangent: S::Vector,
) -> Result<S::Vector, DomainError> {
    Ok(space.iso_transport(space.iso_of(pose), local, tangent))
}

pub fn homogeneous_relative<S: Homogeneous>(
    space: &S,
    eye: &Pose<S>,
    pose: &Pose<S>,
) -> Result<S::Iso, DomainError> {
    Ok(space.iso_compose(space.iso_inverse(space.iso_of(eye)), space.iso_of(pose)))
}

macro_rules! homogeneous_capabilities {
    () => {
        type Placement = <Self as IsometryGroup>::Iso;

        type Relative = <Self as IsometryGroup>::Iso;

        fn prepare(&self, pose: &Pose<Self>) -> Self::Placement {
            self.iso_of(pose)
        }

        fn place(
            &self,
            placement: &Self::Placement,
            local: Self::Point,
        ) -> Result<Self::Point, DomainError> {
            homogeneous_place(self, placement, local)
        }

        fn local(&self, pose: &Pose<Self>, point: Self::Point) -> Result<Self::Point, DomainError> {
            homogeneous_local(self, pose, point)
        }

        fn relative(
            &self,
            eye: &Pose<Self>,
            pose: &Pose<Self>,
        ) -> Result<Self::Relative, DomainError> {
            homogeneous_relative(self, eye, pose)
        }

        fn place_relative(
            &self,
            relative: &Self::Relative,
            local: Self::Point,
        ) -> Result<Self::Point, DomainError> {
            homogeneous_place(self, relative, local)
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
    };
}

pub fn homogeneous_moved<S: Homogeneous>(
    space: &S,
    pose: &Pose<S>,
    to: S::Point,
) -> Result<Pose<S>, DomainError> {
    space.check(to)?;
    let pose_iso = space.iso_of(pose);
    let local = space.iso_apply(space.iso_inverse(pose_iso), to);
    let next = space.pose_of(space.iso_compose(pose_iso, space.transvection(local)));
    space.check(next.point)?;
    Ok(next)
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

fn single_chart(chart: ChartId) -> Result<(), DomainError> {
    (chart == ChartId(0))
        .then_some(())
        .ok_or(DomainError::Unsupported("chart id"))
}

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
        None => store.insert_raw(entity, row)?,
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
    homogeneous_capabilities!();

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
        single_chart(pose.chart)?;
        finite(pose.coordinates)?;
        let frame = Rotor4::from_mat4(&pose.frame).ok_or(DomainError::InvalidFrame)?;
        Ok(Pose {
            point: Vec4::from_array(pose.coordinates),
            frame,
        })
    }

    fn tangent_from_chart(&self, tangent: &ChartTangent) -> Result<Self::Vector, DomainError> {
        single_chart(tangent.chart)?;
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
    homogeneous_capabilities!();

    fn origin(&self) -> Self::Point {
        Vec3::ZERO
    }

    fn check(&self, point: Self::Point) -> Result<(), DomainError> {
        finite(point.extend(0.0).to_array())?;
        if !in_poincare_ball(point) {
            return Err(DomainError::ChartBoundary);
        }
        self.valid_point(point)
            .then_some(())
            .ok_or(DomainError::ErrorBudget)
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
        single_chart(pose.chart)?;
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
        single_chart(tangent.chart)?;
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
    homogeneous_capabilities!();

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
        single_chart(pose.chart)?;
        finite(pose.coordinates)?;
        let frame = rotation_of(frame_columns(&pose.frame)?).rotation;
        Ok(Pose {
            point: Vec3::from_slice(&pose.coordinates[..3]),
            frame,
        })
    }

    fn tangent_from_chart(&self, tangent: &ChartTangent) -> Result<Self::Vector, DomainError> {
        single_chart(tangent.chart)?;
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

fn geodesic_error(error: GeodesicError) -> DomainError {
    match error {
        GeodesicError::NonFinite => DomainError::InvalidCoordinate("geodesic"),
        GeodesicError::ChartBoundary => DomainError::ChartBoundary,
        GeodesicError::ErrorBudget { .. } => DomainError::ErrorBudget,
        GeodesicError::InvalidStepCount
        | GeodesicError::InvalidErrorBudget
        | GeodesicError::Singular
        | GeodesicError::NoConvergence => DomainError::NoConvergence,
    }
}

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
        let relative = self.frame_at(pose.point).inverse() * pose.frame;
        ChartPose {
            chart: ChartId(0),
            coordinates: pose.point.extend(0.0).to_array(),
            frame: frame_of([relative.x_axis, relative.y_axis, relative.z_axis]),
        }
    }

    fn pose_from_chart(&self, pose: &ChartPose) -> Result<Pose<Self>, DomainError> {
        single_chart(pose.chart)?;
        finite(pose.coordinates)?;
        let point = Vec3::from_slice(&pose.coordinates[..3]);
        self.check(point)?;
        let [x, y, z] = frame_columns(&pose.frame)?;
        let frame = self.frame_at(point) * Mat3::from_cols(x, y, z);
        Ok(Pose { point, frame })
    }

    fn tangent_from_chart(&self, tangent: &ChartTangent) -> Result<Self::Vector, DomainError> {
        single_chart(tangent.chart)?;
        finite(tangent.vector)?;
        Ok(Vec3::from_slice(&tangent.vector[..3]))
    }

    fn chart_reach(&self, at: Self::Point, radius: f32) -> f32 {
        self.conformal_factor(at).sqrt() * radius
    }

    // Chart-flat: a straight chart ray against a chart-radius ball; no blended map lifts a ray yet, so nothing reaches it.
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

    fn place(
        &self,
        placement: &Self::Placement,
        local: Self::Point,
    ) -> Result<Self::Point, DomainError> {
        self.walk(placement, local, 1.0).map(|pose| pose.point)
    }

    fn local(&self, pose: &Pose<Self>, point: Self::Point) -> Result<Self::Point, DomainError> {
        self.check(point)?;
        if self.conformal_factor(point).sqrt() * LOG_RESIDUAL_TOL > GEODESIC_ERROR_BUDGET {
            return Err(DomainError::ErrorBudget);
        }
        let chart = geodesic_log_checked(
            self,
            pose.point,
            point,
            GEODESIC_DEFAULT_STEPS,
            LOG_MAX_ITERS,
            GEODESIC_ERROR_BUDGET,
        )
        .map_err(geodesic_error)?
        .vector;
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
        self.local(eye, self.place(pose, local)?)
    }

    fn carry(
        &self,
        pose: &Pose<Self>,
        local: Self::Point,
        tangent: Self::Vector,
    ) -> Result<Self::Vector, DomainError> {
        let state = integrate_geodesic_frame_checked(
            self,
            pose.point,
            pose.frame * local,
            pose.frame,
            GEODESIC_DEFAULT_STEPS,
            GEODESIC_ERROR_BUDGET,
        )
        .map_err(geodesic_error)?;
        Ok(state.frame * tangent)
    }

    fn moved(&self, pose: &Pose<Self>, to: Self::Point) -> Result<Pose<Self>, DomainError> {
        self.check(to)?;
        let state = integrate_geodesic_frame_to_checked(
            self,
            pose.point,
            to,
            pose.frame,
            GEODESIC_DEFAULT_STEPS,
            LOG_MAX_ITERS,
            GEODESIC_ERROR_BUDGET,
        )
        .map_err(geodesic_error)?;
        Ok(Pose {
            point: state.point,
            frame: state.frame,
        })
    }

    fn walk(
        &self,
        pose: &Pose<Self>,
        tangent: Self::Vector,
        dt: f32,
    ) -> Result<Pose<Self>, DomainError> {
        let state = integrate_geodesic_frame_checked(
            self,
            pose.point,
            pose.frame * (tangent * dt),
            pose.frame,
            GEODESIC_DEFAULT_STEPS,
            GEODESIC_ERROR_BUDGET,
        )
        .map_err(geodesic_error)?;
        Ok(Pose {
            point: state.point,
            frame: state.frame,
        })
    }
}

fn lerp_color(back: [f32; 4], front: [f32; 4], t: f32) -> [f32; 4] {
    let mut mixed = [0.0; 4];
    for (channel, value) in mixed.iter_mut().enumerate() {
        *value = back[channel] + (front[channel] - back[channel]) * t;
    }
    mixed
}

struct SegmentProjection<'a, S: DomainSpace> {
    space: &'a S,
    mapping: &'a dyn ViewMapping<S>,
    eye: &'a Pose<S>,
    pose: &'a Pose<S>,
    relative: &'a S::Relative,
}

fn push_segments<S: DomainSpace>(
    projection: SegmentProjection<'_, S>,
    library: &Library<'_>,
    instance: &Instance,
    entity: Entity,
    segments: &mut Vec<SegmentRecord>,
    refusals: &mut ViewRefusals,
) {
    let SegmentProjection {
        space,
        mapping,
        eye,
        pose,
        relative,
    } = projection;
    let Some(geometry) = library.geometry.get(instance.geometry.index()) else {
        return;
    };
    let (color, material_width_px) = library.line_style(instance.material);
    let width_px = instance.line_width_px.unwrap_or(material_width_px);
    let opacity = instance.line_opacity;
    let palette = match instance.shading {
        EdgeShading::Palette(id) => library.palette(id),
        _ => &[],
    };
    let placement = space.prepare(pose);
    let origin = match space.place(&placement, space.origin()) {
        Ok(origin) => origin,
        Err(error) => {
            refusals.record(entity, error, RefusalSource::Segment);
            return;
        }
    };
    let origin_depth = space.chart_point(origin).coordinates[3];
    let depth_color = |point: S::Point| match instance.shading {
        EdgeShading::Depth {
            back,
            front,
            extent,
        } => {
            let depth = space.chart_point(point).coordinates[3] - origin_depth;
            Some(lerp_color(
                back,
                front,
                (depth / extent.max(1e-6) * 0.5 + 0.5).clamp(0.0, 1.0),
            ))
        }
        _ => None,
    };
    let mut push = |index: usize, a: [f32; 4], b: [f32; 4]| {
        let start = space.local_point(a);
        let end = space.local_point(b);
        let start_point = match space.place(&placement, start) {
            Ok(point) => point,
            Err(error) => {
                refusals.record(entity, error, RefusalSource::Segment);
                return;
            }
        };
        let end_point = match space.place(&placement, end) {
            Ok(point) => point,
            Err(error) => {
                refusals.record(entity, error, RefusalSource::Segment);
                return;
            }
        };
        let painted = |at: usize| palette.get(at).copied().unwrap_or(color);
        let with_opacity = |mut color: [f32; 4]| {
            if let Some(opacity) = opacity {
                color[3] = opacity;
            }
            color
        };
        let start_color =
            with_opacity(depth_color(start_point).unwrap_or_else(|| painted(index * 2)));
        let end_color =
            with_opacity(depth_color(end_point).unwrap_or_else(|| painted(index * 2 + 1)));
        let mut previous = None;
        mapping.image_segment(space, eye, pose, relative, [start, end], &mut |t, point| {
            let Some(point) = point else {
                previous = None;
                return;
            };
            if let Some((previous_t, previous_point)) = previous {
                segments.push(SegmentRecord {
                    start: previous_point,
                    _pad0: 0.0,
                    end: point,
                    _pad1: 0.0,
                    start_color: lerp_color(start_color, end_color, previous_t),
                    end_color: lerp_color(start_color, end_color, t),
                    width_px,
                    _pad2: [0.0; 3],
                });
            }
            previous = Some((t, point));
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
    }
}

struct ViewEntry<S: DomainSpace> {
    eye: Entity,
    subject: Option<Entity>,
    style: ViewStyle<S>,
}

impl<S: DomainSpace> Clone for ViewEntry<S> {
    fn clone(&self) -> Self {
        Self {
            eye: self.eye,
            subject: self.subject,
            style: self.style.clone(),
        }
    }
}

struct ViewProjection<'a, S: DomainSpace> {
    space: &'a S,
    spec: &'a ViewEntry<S>,
    eye: &'a Pose<S>,
}

impl<'a, S: DomainSpace> ViewProjection<'a, S> {
    fn push_section(
        &self,
        pose: &Pose<S>,
        library: &Library<'_>,
        instance: &Instance,
        scratch: &mut view::SectionScratchpad,
        segments: &mut Vec<SegmentRecord>,
        triangles: &mut Vec<TriangleRecord>,
    ) -> Result<(), DomainError> {
        let space = self.space;
        let spec = self.spec;
        let eye = self.eye;
        let mapping = spec.style.mapping.as_ref();
        let Some(PreparedGeometry::Polytope4 { polytope, scale }) =
            library.geometry.get(instance.geometry.index())
        else {
            return Ok(());
        };
        let (Some(perimeter_material), Some(cut)) = (instance.section, mapping.section(eye, pose))
        else {
            return Ok(());
        };
        let relative = space.relative(eye, pose)?;
        let Some(place) = mapping.image_local(space, eye, pose, &relative, space.origin()) else {
            return Ok(());
        };
        let origin = space.place_relative(&relative, space.origin())?;
        let base = Vec4::from(space.chart_point(origin).coordinates);
        let topology = polytope.topology();
        scratch.rotated.clear();
        for vertex in topology.vertices.iter() {
            let local = space.local_point((*vertex * *scale).to_array());
            let placed = match space.place_relative(&relative, local) {
                Ok(placed) => placed,
                Err(error) => {
                    scratch.rotated.clear();
                    return Err(error);
                }
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

        if spec.style.section_faces {
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
            if spec.style.section_edges {
                let (mut edge_color, material_width_px) = library.line_style(perimeter_material);
                let width_px = instance.line_width_px.unwrap_or(material_width_px);
                if let Some(opacity) = instance.line_opacity {
                    edge_color[3] = opacity;
                }
                for [_, start, end] in &scratch.faces.indices {
                    segments.push(SegmentRecord {
                        start: placed(scratch.faces.vertices[*start as usize]),
                        _pad0: 0.0,
                        end: placed(scratch.faces.vertices[*end as usize]),
                        _pad1: 0.0,
                        start_color: edge_color,
                        end_color: edge_color,
                        width_px,
                        _pad2: [0.0; 3],
                    });
                }
            }
            for [a, b, c] in &scratch.faces.indices {
                triangles.push(TriangleRecord {
                    vertices: [
                        placed(scratch.faces.vertices[*a as usize]),
                        placed(scratch.faces.vertices[*b as usize]),
                        placed(scratch.faces.vertices[*c as usize]),
                    ],
                    color: fill_color,
                });
            }
        } else if spec.style.section_edges {
            let (mut edge_color, material_width_px) = library.line_style(perimeter_material);
            let width_px = instance.line_width_px.unwrap_or(material_width_px);
            if let Some(opacity) = instance.line_opacity {
                edge_color[3] = opacity;
            }
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
                segments.push(SegmentRecord {
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
        }
        Ok(())
    }

    fn project_instance(
        &self,
        pose: &Pose<S>,
        library: &Library<'_>,
        instance: &Instance,
        entity: Entity,
        segments: &mut Vec<SegmentRecord>,
        refusals: &mut ViewRefusals,
    ) -> Option<InstanceRecord> {
        let space = self.space;
        let spec = self.spec;
        let eye = self.eye;
        let relative = match space.relative(eye, pose) {
            Ok(relative) => relative,
            Err(error) => {
                refusals.record(entity, error, RefusalSource::Instance);
                return None;
            }
        };
        let image_point =
            spec.style
                .mapping
                .image_local(space, eye, pose, &relative, space.origin())?;
        if spec.style.edges {
            push_segments(
                SegmentProjection {
                    space,
                    mapping: spec.style.mapping.as_ref(),
                    eye,
                    pose,
                    relative: &relative,
                },
                library,
                instance,
                entity,
                segments,
                refusals,
            );
        }
        Some(InstanceRecord {
            entity,
            geometry: instance.geometry,
            material: instance.material,
            pose: space.chart_pose(pose),
            image_point,
        })
    }
}

/// A point and a frame orthonormal in the metric at that point, implying no isometry; a homogeneous space converts through `Homogeneous::iso_of`.
pub struct Pose<S: Space> {
    pub point: S::Point,
    pub frame: S::Frame,
}

impl<S: Space> Pose<S> {
    /// The space-defined reference frame at `point`.
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

    fn bind(&mut self, _scene: SceneId, _owner: Owner) {}

    fn step(
        &mut self,
        poses: &mut Store<Pose<S>>,
        step: Step,
        owner: Owner,
    ) -> Result<(), DomainError>;

    fn set_pose(
        &mut self,
        _entity: Entity,
        _pose: Pose<S>,
        _poses: &mut Store<Pose<S>>,
        _owner: Owner,
    ) -> Option<Result<(), DomainError>> {
        None
    }

    fn move_to(
        &mut self,
        _entity: Entity,
        _point: S::Point,
        _poses: &mut Store<Pose<S>>,
        _owner: Owner,
    ) -> Option<Result<(), DomainError>> {
        None
    }

    fn synchronize(&mut self, _poses: &mut Store<Pose<S>>, _owner: Owner) {}

    fn snapshot(&self, owner: Owner) -> Box<dyn Any + Send>;

    /// Refuses whatever `restore` would refuse, changing nothing; the session calls it on every facility before the first one restores.
    fn check_restore(&self, from: &(dyn Any + Send), owner: Owner) -> Result<(), RestoreError>;

    fn restore(&mut self, from: &(dyn Any + Send), owner: Owner) -> Result<(), RestoreError>;

    fn release(&mut self, _entity: Entity, _owner: Owner) {}

    /// `Some` claims the command with its outcome; `None` leaves it to the domain.
    fn apply(
        &mut self,
        _command: &ChartCommand,
        _poses: &mut Store<Pose<S>>,
        _owner: Owner,
    ) -> Option<Result<Outcome, Rejection>> {
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
    pub(crate) primitives: Vec<FieldPrimitive>,
    pub(crate) program: Vec<u32>,
    pub(crate) nodes: Vec<FieldNode>,
    pub(crate) stack: u32,
    pub(crate) dimension: u32,
    pub(crate) kind: FieldKind,
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

    fn dimension(&self) -> u32 {
        self.dimension
    }

    fn distance(&self, point: [f32; 4]) -> f32 {
        field::evaluate_bounded(&self.program, &self.primitives, &self.nodes, point, 0.0)
            .unwrap_or(field::FIELD_FAR)
    }

    fn error_at(&self, point: [f32; 4]) -> f32 {
        field::evaluate_error(
            &self.program,
            &self.primitives,
            &self.nodes,
            point,
            self.kind,
        )
        .unwrap_or(f32::INFINITY)
    }
}

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

/// What a domain reads out; the session reaches the rest through `DomainOwner`.
pub trait Domain: Send + 'static {
    fn id(&self) -> DomainId;

    fn name(&self) -> &'static str;

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
    ) -> Result<(), DomainError>;

    fn step(&mut self, step: Step) -> Result<(), DomainError>;

    fn synchronize(&mut self);

    fn boundary(&mut self);

    fn release(&mut self, entity: Entity);

    fn snapshot(&self) -> DomainSnapshot;

    /// Refuses whatever `restore` would refuse, changing nothing.
    fn check_restore(&self, from: &DomainSnapshot) -> Result<(), RestoreError>;

    fn restore(&mut self, from: &DomainSnapshot, scene: SceneId) -> Result<(), RestoreError>;

    fn apply(&mut self, command: &ChartCommand) -> Result<Outcome, Rejection>;
}

pub struct TypedDomain<S: DomainSpace> {
    id: DomainId,
    scene: SceneId,
    name: &'static str,
    space: S,
    poses: Store<Pose<S>>,
    instances: Store<Instance>,
    fields: Option<Store<Field>>,
    views: Vec<ViewEntry<S>>,
    targets: Vec<ViewTarget>,
    view_revisions: Vec<u32>,
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
            return Err(Rejection::Domain(DomainError::Stale(entity)));
        }
        put_row(&mut self.instances, entity, instance)
    }

    pub(crate) fn attach_field(&mut self, entity: Entity, field: Field) -> Result<(), Rejection> {
        if !self.poses.contains(entity) {
            return Err(Rejection::Domain(DomainError::Stale(entity)));
        }
        let fields = self
            .fields
            .as_mut()
            .ok_or(Rejection::Unsupported("fields"))?;
        put_row(fields, entity, field)
    }

    pub fn set_pose(&mut self, entity: Entity, pose: Pose<S>) -> Result<(), DomainError> {
        self.check_pose(&pose)?;
        self.apply_pose(entity, pose)
    }

    fn apply_pose(&mut self, entity: Entity, pose: Pose<S>) -> Result<(), DomainError> {
        if !self.poses.contains(entity) {
            return Err(DomainError::Stale(entity));
        }
        for facility in &mut self.facilities {
            if let Some(result) = facility.set_pose(entity, pose, &mut self.poses, Owner::new()) {
                return result;
            }
        }
        *self
            .poses
            .get_mut(entity)
            .ok_or(DomainError::Stale(entity))? = pose;
        Ok(())
    }

    pub fn set_point(&mut self, entity: Entity, point: S::Point) -> Result<(), DomainError> {
        let mut pose = *self.poses.get(entity).ok_or(DomainError::Stale(entity))?;
        pose.point = point;
        self.set_pose(entity, pose)
    }

    pub fn set_frame(&mut self, entity: Entity, frame: S::Frame) -> Result<(), DomainError> {
        let mut pose = *self.poses.get(entity).ok_or(DomainError::Stale(entity))?;
        pose.frame = frame;
        self.set_pose(entity, pose)
    }

    fn check_pose(&self, pose: &Pose<S>) -> Result<(), DomainError> {
        self.space
            .pose_from_chart(&self.space.chart_pose(pose))
            .map(|_| ())
    }

    pub fn add_view(&mut self, spec: ViewSpec<S>) -> ViewId {
        let id = ViewId::new(self.views.len());
        self.targets.push(ViewTarget {
            view: id,
            image: spec.image,
        });
        self.view_revisions.push(0);
        self.views.push(ViewEntry {
            eye: spec.eye,
            subject: spec.subject,
            style: spec.style,
        });
        id
    }

    pub fn view(&self, id: ViewId) -> Option<&ViewStyle<S>> {
        self.views.get(id.index()).map(|view| &view.style)
    }

    pub fn view_eye(&self, id: ViewId) -> Option<Entity> {
        self.views.get(id.index()).map(|view| view.eye)
    }

    pub fn view_subject(&self, id: ViewId) -> Option<Entity> {
        self.views.get(id.index())?.subject
    }

    pub fn view_mut(&mut self, id: ViewId) -> Option<&mut ViewStyle<S>> {
        let revision = self.view_revisions.get_mut(id.index())?;
        *revision = revision.wrapping_add(1);
        let spec = self.views.get_mut(id.index())?;
        Some(&mut spec.style)
    }

    pub fn set_view_eye(&mut self, id: ViewId, eye: Entity) -> Result<(), DomainError> {
        let index = id.index();
        if self.views.get(index).is_none() {
            return Err(DomainError::UnknownView(id));
        }
        if !self.poses.contains(eye) {
            return Err(DomainError::Stale(eye));
        }
        if self.views[index].eye == eye {
            return Ok(());
        }
        self.views[index].eye = eye;
        self.view_revisions[index] = self.view_revisions[index].wrapping_add(1);
        Ok(())
    }

    pub fn set_view_subject(
        &mut self,
        id: ViewId,
        subject: Option<Entity>,
    ) -> Result<(), DomainError> {
        let index = id.index();
        if self.views.get(index).is_none() {
            return Err(DomainError::UnknownView(id));
        }
        if let Some(subject) = subject {
            if !self.poses.contains(subject) {
                return Err(DomainError::Stale(subject));
            }
        }
        if self.views[index].subject == subject {
            return Ok(());
        }
        self.views[index].subject = subject;
        self.view_revisions[index] = self.view_revisions[index].wrapping_add(1);
        Ok(())
    }

    fn rebuild_view(
        &self,
        spec: &ViewEntry<S>,
        library: Library<'_>,
        into: &mut ViewRecords,
        stamp: Stamp,
    ) -> Result<(), DomainError> {
        let ViewRecords {
            instances,
            segments,
            triangles,
            refusals,
            scratch,
            output,
            built,
            ..
        } = into;
        segments.clear();
        triangles.clear();
        *refusals = ViewRefusals::default();
        output.clear();
        if !spec.style.enabled {
            instances.replace(std::iter::empty(), stamp);
            *built = stamp;
            return Ok(());
        }
        let eye = self
            .poses
            .get(spec.eye)
            .ok_or(DomainError::Stale(spec.eye))?;
        let projection = ViewProjection {
            space: &self.space,
            spec,
            eye,
        };
        for (entity, instance) in self.instances.iter() {
            if spec.subject.is_some_and(|subject| subject != entity) {
                continue;
            }
            let section_start = segments.len();
            let triangle_start = triangles.len();
            if let Some(pose) = self.poses.get(entity) {
                if spec.style.section_edges || spec.style.section_faces {
                    if let Err(error) = projection
                        .push_section(pose, &library, instance, scratch, segments, triangles)
                    {
                        refusals.record(entity, error, RefusalSource::Section);
                    }
                }
            }
            output.push(EntityOutput {
                entity,
                section_segments: section_start..segments.len(),
                line_segments: 0..0,
                triangles: triangle_start..triangles.len(),
                instance: false,
            });
        }
        let records = self
            .instances
            .iter()
            .filter(|(entity, _)| spec.subject.is_none_or(|subject| subject == *entity))
            .zip(output.iter_mut())
            .filter_map(|((entity, instance), cached)| {
                let line_start = segments.len();
                let mut record_refusals = ViewRefusals::default();
                let record = self.poses.get(entity).and_then(|pose| {
                    projection.project_instance(
                        pose,
                        &library,
                        instance,
                        entity,
                        segments,
                        &mut record_refusals,
                    )
                });
                cached.line_segments = line_start..segments.len();
                cached.instance = record.is_some();
                if record_refusals.count != 0 {
                    refusals.count = refusals.count.saturating_add(record_refusals.count);
                    if refusals.first.is_none() {
                        refusals.first = record_refusals.first;
                    }
                    refusals.last = record_refusals.last;
                }
                record.map(|record| (entity, record))
            });
        instances.replace(records, stamp);
        *built = stamp;
        Ok(())
    }

    fn patch_view(
        &self,
        spec: &ViewEntry<S>,
        library: Library<'_>,
        into: &mut ViewRecords,
        stamp: Stamp,
    ) -> Result<bool, DomainError> {
        let eye = self
            .poses
            .get(spec.eye)
            .ok_or(DomainError::Stale(spec.eye))?;
        let projection = ViewProjection {
            space: &self.space,
            spec,
            eye,
        };
        let ViewRecords {
            instances,
            segments,
            triangles,
            scratch,
            output,
            patch_segments,
            patch_triangles,
            changed,
            built,
            ..
        } = into;
        let mut patched = false;
        for &entity in changed.iter() {
            if spec.subject.is_some_and(|subject| subject != entity) {
                continue;
            }
            let Some(instance) = self.instances.get(entity) else {
                continue;
            };
            let Some(pose) = self.poses.get(entity) else {
                return Ok(false);
            };
            let Some(cached) = output.get(entity).cloned() else {
                return Ok(false);
            };
            patch_segments.clear();
            patch_triangles.clear();
            let mut section_refusals = ViewRefusals::default();
            if spec.style.section_edges || spec.style.section_faces {
                if let Err(error) = projection.push_section(
                    pose,
                    &library,
                    instance,
                    scratch,
                    patch_segments,
                    patch_triangles,
                ) {
                    section_refusals.record(entity, error, RefusalSource::Section);
                }
            }
            if patch_segments.len() != cached.section_segments.end - cached.section_segments.start
                || patch_triangles.len() != cached.triangles.end - cached.triangles.start
                || section_refusals.count != 0
            {
                return Ok(false);
            }
            segments[cached.section_segments.clone()].copy_from_slice(patch_segments);
            triangles[cached.triangles.clone()].copy_from_slice(patch_triangles);

            patch_segments.clear();
            let mut record_refusals = ViewRefusals::default();
            let record = projection.project_instance(
                pose,
                &library,
                instance,
                entity,
                patch_segments,
                &mut record_refusals,
            );
            if patch_segments.len() != cached.line_segments.end - cached.line_segments.start
                || record.is_some() != cached.instance
                || record_refusals.count != 0
            {
                return Ok(false);
            }
            segments[cached.line_segments.clone()].copy_from_slice(patch_segments);
            if let Some(record) = record {
                if !instances.update(entity, record) {
                    return Ok(false);
                }
            }
            patched = true;
        }
        instances.restamp(stamp);
        if patched {
            *built = stamp;
        }
        Ok(true)
    }

    pub fn fields(&self) -> Option<&Store<Field>> {
        self.fields.as_ref()
    }

    pub fn field_mut(&mut self, entity: Entity) -> Option<&mut Field> {
        self.fields.as_mut()?.get_mut(entity)
    }

    pub fn remove_field(&mut self, entity: Entity) -> Result<Field, StoreError> {
        self.fields
            .as_mut()
            .ok_or(StoreError::Missing(entity))?
            .remove(entity)
    }

    /// `DomainSpace::walk` for `dt` along `velocity`, a tangent in the entity's frame at its point, with the frame carried by parallel transport.
    pub fn walk(
        &mut self,
        entity: Entity,
        velocity: S::Vector,
        dt: f32,
    ) -> Result<(), DomainError> {
        let pose = self.poses.get(entity).ok_or(DomainError::Stale(entity))?;
        let next = self.space.walk(pose, velocity, dt)?;
        self.apply_pose(entity, next)
    }

    /// `DomainSpace::moved` to `point`: transvection on a homogeneous space, per-column transport on a conformally flat chart.
    pub fn move_to(&mut self, entity: Entity, point: ChartPoint) -> Result<(), DomainError> {
        let target = self.space.point_from_chart(&point)?;
        if !self.poses.contains(entity) {
            return Err(DomainError::Stale(entity));
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
            .ok_or(DomainError::Stale(entity))?;
        let next = space.moved(pose, target)?;
        *pose = next;
        Ok(())
    }
}

#[cfg(feature = "physics")]
impl<S> TypedDomain<S>
where
    S: Homogeneous + loam_physics::PhysicsSpace + Copy,
    S::Vector: loam_physics::collision::VectorOps + Default,
    S::Point: std::ops::Sub<Output = S::Vector>,
{
    pub fn spawn_body(
        &mut self,
        entity: Entity,
        body: loam_physics::BodyDef<S>,
    ) -> Result<loam_physics::BodyId, DomainError> {
        let pose = *self.poses.get(entity).ok_or(DomainError::Stale(entity))?;
        let physics = self
            .facilities
            .iter_mut()
            .find_map(|facility| {
                (facility.as_mut() as &mut dyn Any).downcast_mut::<crate::physics::Physics<S>>()
            })
            .ok_or(DomainError::Unsupported("physics"))?;
        physics.check_pose(pose)?;
        let id = physics.spawn(entity, body);
        if let Err(error) = physics.replace_pose(id, pose, &mut self.poses) {
            let _ = physics.despawn(entity);
            return Err(error.into());
        }
        Ok(id)
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
        let target = self.targets.get(id.index())?;
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
        let spec = self.views.get(view.index())?;
        let target = self.targets.get(view.index())?;
        if !spec.style.enabled {
            return None;
        }
        let eye = self.poses.get(spec.eye)?;
        let origin = self.space.origin();
        let lifted = spec.style.mapping.lift(eye, ray);
        let mut nearest: Option<Pick> = None;
        for (entity, instance) in self.instances.iter() {
            if spec.subject.is_some_and(|subject| subject != entity) {
                continue;
            }
            let Some(pose) = self.poses.get(entity) else {
                continue;
            };
            let Some(geometry) = prepared.get(instance.geometry.index()) else {
                continue;
            };
            let Ok(center) = self.space.place(&self.space.prepare(pose), origin) else {
                continue;
            };
            let radius = geometry.bounding_radius();
            let hit = match &lifted {
                Some(domain_ray) => {
                    let reach = self.space.chart_reach(center, radius);
                    self.space
                        .hit_ball(domain_ray, center, reach)
                        .map(|t| self.space.exp(domain_ray.origin, domain_ray.direction * t))
                        .and_then(|point| {
                            let image_point = spec.style.mapping.image_point(eye, point)?;
                            Some((image_point, Some(self.space.chart_point(point))))
                        })
                }
                None => spec
                    .style
                    .mapping
                    .image_point(eye, center)
                    .and_then(|image_center| {
                        let image_radius = spec.style.mapping.image_radius(eye, center, radius);
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
                image: target.image,
                image_point,
                depth,
                hit,
            });
        }
        nearest
    }

    fn image_of(&self, view: ViewId, entity: Entity) -> Option<[f32; 3]> {
        let spec = self.views.get(view.index())?;
        if !spec.style.enabled || spec.subject.is_some_and(|subject| subject != entity) {
            return None;
        }
        let eye = self.poses.get(spec.eye)?;
        let pose = self.poses.get(entity)?;
        let relative = self.space.relative(eye, pose).ok()?;
        spec.style
            .mapping
            .image_local(&self.space, eye, pose, &relative, self.space.origin())
    }

    fn lift_origin(&self, view: ViewId, ray: &ImageRay) -> Result<ChartPoint, DomainError> {
        let spec = self
            .views
            .get(view.index())
            .ok_or(DomainError::UnknownView(view))?;
        let eye = self
            .poses
            .get(spec.eye)
            .ok_or(DomainError::Stale(spec.eye))?;
        let lifted = spec
            .style
            .mapping
            .lift(eye, ray)
            .ok_or(DomainError::Unsupported(spec.style.mapping.name()))?;
        self.space.check(lifted.origin)?;
        Ok(self.space.chart_point(lifted.origin))
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

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl<S: DomainSpace> DomainOwner for TypedDomain<S> {
    fn retarget(&mut self, view: ViewId, image: ImageSpaceId) -> Result<(), DomainError> {
        self.views
            .get(view.index())
            .ok_or(DomainError::UnknownView(view))?;
        self.targets[view.index()].image = image;
        self.view_revisions[view.index()] = self.view_revisions[view.index()].wrapping_add(1);
        Ok(())
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
            .ok_or(DomainError::UnknownView(view))?;
        let revision = *self
            .view_revisions
            .get(view.index())
            .ok_or(DomainError::UnknownView(view))?;
        let mut pose_cursor = into.poses;
        let pose_changes = self.poses.changes(&mut pose_cursor);
        let pose_resync = pose_changes.is_resync();
        into.changed.clear();
        let mut pose_removed = false;
        for change in pose_changes {
            match change {
                Change::Row(entity, _) => into.changed.push(entity),
                Change::Removed(_) => pose_removed = true,
            }
        }
        let mut attachment_cursor = into.attachments;
        let mut attachment_changes = self.instances.changes(&mut attachment_cursor);
        let attachment_resync = attachment_changes.is_resync();
        let attached = attachment_changes.next().is_some();
        if !pose_resync
            && !pose_removed
            && !attachment_resync
            && !attached
            && into.revision == revision
            && !into.changed.contains(&spec.eye)
        {
            let patched = if into.changed.is_empty() {
                into.instances.restamp(stamp);
                true
            } else if spec.style.enabled && into.refusals.count == 0 {
                self.patch_view(spec, library, into, stamp)?
            } else {
                false
            };
            if patched {
                into.poses = pose_cursor;
                into.attachments = attachment_cursor;
                return Ok(());
            }
        }
        self.poses.catch_up(&mut into.poses);
        self.instances.catch_up(&mut into.attachments);
        into.revision = revision;
        self.rebuild_view(spec, library, into, stamp)
    }

    fn step(&mut self, step: Step) -> Result<(), DomainError> {
        for facility in &mut self.facilities {
            facility.step(&mut self.poses, step, Owner::new())?;
        }
        Ok(())
    }

    fn boundary(&mut self) {
        self.synchronize();
        self.poses.boundary();
        self.instances.boundary();
        if let Some(fields) = &mut self.fields {
            fields.boundary();
        }
    }

    fn synchronize(&mut self) {
        for facility in &mut self.facilities {
            facility.synchronize(&mut self.poses, Owner::new());
        }
    }

    fn release(&mut self, entity: Entity) {
        for facility in &mut self.facilities {
            facility.release(entity, Owner::new());
        }
        StoreField::release(&mut self.poses, entity, Owner::new());
        StoreField::release(&mut self.instances, entity, Owner::new());
        if let Some(fields) = &mut self.fields {
            StoreField::release(fields, entity, Owner::new());
        }
    }

    fn snapshot(&self) -> DomainSnapshot {
        DomainSnapshot(Box::new(TypedSnapshot::<S> {
            scene: self.scene,
            poses: StoreField::snapshot(&self.poses),
            instances: StoreField::snapshot(&self.instances),
            fields: self.fields.as_ref().map(StoreField::snapshot),
            facilities: self
                .facilities
                .iter()
                .map(|facility| facility.snapshot(Owner::new()))
                .collect(),
            views: self.views.clone(),
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
        StoreField::restore(&mut self.poses, &from.poses, scene, Owner::new());
        StoreField::restore(&mut self.instances, &from.instances, scene, Owner::new());
        if let (Some(fields), Some(snapshot)) = (&mut self.fields, &from.fields) {
            StoreField::restore(fields, snapshot, scene, Owner::new());
            for field in fields.rows_mut_untracked() {
                for operand in &mut field.operands {
                    if operand.scene() == from.scene {
                        *operand = Entity::new(scene, operand.key());
                    }
                }
            }
        }
        let mut views = from.views.clone();
        let mut revisions = Vec::with_capacity(views.len());
        for (index, view) in views.iter_mut().enumerate() {
            if view.eye.scene() == from.scene {
                view.eye = Entity::new(scene, view.eye.key());
            }
            view.subject = view.subject.map(|subject| {
                if subject.scene() == from.scene {
                    Entity::new(scene, subject.key())
                } else {
                    subject
                }
            });
            revisions.push(
                self.view_revisions
                    .get(index)
                    .copied()
                    .unwrap_or(0)
                    .wrapping_add(1),
            );
        }
        self.views = views;
        self.targets.clone_from(&from.targets);
        self.view_revisions = revisions;
        self.compiler.invalidate();
        Ok(())
    }

    fn apply(&mut self, command: &ChartCommand) -> Result<Outcome, Rejection> {
        let owned = self.poses.contains(command.entity());
        if !owned && !matches!(command, ChartCommand::Place { .. }) {
            return Err(DomainError::Stale(command.entity()).into());
        }
        match command {
            ChartCommand::Place { pose, .. } => {
                self.space.pose_from_chart(pose)?;
            }
            ChartCommand::Walk { tangent, .. } => {
                self.space.tangent_from_chart(tangent)?;
            }
            ChartCommand::Move { point, .. } | ChartCommand::Grab { point, .. } => {
                self.space.point_from_chart(point)?;
            }
            _ => {}
        }
        if owned {
            for facility in &mut self.facilities {
                if let Some(outcome) = facility.apply(command, &mut self.poses, Owner::new()) {
                    return outcome;
                }
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
            ChartCommand::Grab { .. } | ChartCommand::Release { .. } => Ok(Outcome::Done),
        }
    }
}

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
        let mut facilities = self.facilities;
        for facility in &mut facilities {
            facility.bind(scene, Owner::new());
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
            view_revisions: Vec::new(),
            facilities,
            compiler: FieldCompiler::new(),
        }
    }
}

pub struct Domains {
    runtime: RuntimeId,
    list: Vec<Box<dyn DomainOwner>>,
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

    pub(crate) fn push(&mut self, domain: Box<dyn DomainOwner>) {
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
        self.list.get(id.index()).map(|domain| domain.as_ref() as _)
    }

    pub(crate) fn facade(&mut self, id: DomainId) -> Option<&mut dyn DomainOwner> {
        self.list.get_mut(id.index()).map(|domain| domain.as_mut())
    }

    pub fn iter(&self) -> impl Iterator<Item = &dyn Domain> {
        self.list.iter().map(|domain| domain.as_ref() as _)
    }

    pub(crate) fn owned(&self) -> impl Iterator<Item = &dyn DomainOwner> {
        self.list.iter().map(|domain| domain.as_ref())
    }

    pub(crate) fn iter_mut(&mut self) -> impl Iterator<Item = &mut dyn DomainOwner> {
        self.list.iter_mut().map(|domain| domain.as_mut())
    }

    pub(crate) fn synchronize(&mut self) {
        for domain in &mut self.list {
            domain.synchronize();
        }
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

    pub fn read<S: DomainSpace>(
        &self,
        handle: DomainHandle<S>,
    ) -> Result<&TypedDomain<S>, DomainError> {
        if handle.runtime != self.runtime {
            return Err(DomainError::ForeignRuntime);
        }
        let domain = self
            .list
            .get(handle.id.index())
            .ok_or(DomainError::UnknownDomain(handle.id))?;
        domain
            .as_any()
            .downcast_ref::<TypedDomain<S>>()
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
    use crate::view::{Projection4, Section4};

    crate::stores! {
        #[derive(Default)]
        pub struct Probe {
            tags: Store<u8>,
        }
    }

    const SEGMENTS: usize = 3;
    const EYE_AT: Vec4 = Vec4::new(0.0, 0.0, 0.0, 2.0);
    const OBJECT_AT: Vec4 = Vec4::new(0.0, 0.0, -4.0, 0.0);

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
    fn stale_view_identity_edits_leave_live_references_installed() {
        let mut session = Session::new(Probe::default(), SimConfig::default());
        let r4 = session.register_domain(DomainBuilder::new("r4", EuclideanR4));
        let root = session.views().root();
        let (eye, subject, stale, view) = session.dispatch(|dispatch| {
            let eye = dispatch
                .spawn(SpawnBundle::new().at(r4, Pose::at(Vec4::ZERO)))
                .unwrap();
            let subject = dispatch
                .spawn(SpawnBundle::new().at(r4, Pose::at(Vec4::ZERO)))
                .unwrap();
            let stale = dispatch
                .spawn(SpawnBundle::new().at(r4, Pose::at(Vec4::ZERO)))
                .unwrap();
            let view = dispatch
                .domains
                .typed(r4)
                .unwrap()
                .add_view(ViewSpec::new(root, eye, Section4 { w: 0.0 }).subject(subject));
            (eye, subject, stale, view)
        });
        session
            .dispatch(|dispatch| dispatch.despawn(stale))
            .unwrap();

        let domain = session.domains_mut().typed(r4).unwrap();
        let revision = domain.view_revisions[view.index()];
        assert_eq!(domain.set_view_eye(view, eye), Ok(()));
        assert_eq!(domain.set_view_subject(view, Some(subject)), Ok(()));
        assert_eq!(domain.view_revisions[view.index()], revision);
        assert_eq!(
            domain.set_view_eye(view, stale),
            Err(DomainError::Stale(stale))
        );
        assert_eq!(
            domain.set_view_subject(view, Some(stale)),
            Err(DomainError::Stale(stale))
        );
        assert_eq!(domain.view_eye(view), Some(eye));
        assert_eq!(domain.view_subject(view), Some(subject));
    }

    #[test]
    fn despawned_view_subject_stays_selected_and_publishes_an_empty_view() {
        let mut session = Session::new(Probe::default(), SimConfig::default());
        let r4 = session
            .register_domain(DomainBuilder::new("r4", EuclideanR4).tracked(LogCapacity::default()));
        let prepared = session.prepare(lines());
        let material = session.add_material(Material::flat([1.0; 4]));
        let root = session.views().root();
        let (subject, view) = session.dispatch(|dispatch| {
            let eye = dispatch
                .spawn(SpawnBundle::new().at(r4, Pose::at(EYE_AT)))
                .unwrap();
            let subject = dispatch
                .spawn(
                    SpawnBundle::new()
                        .at(r4, Pose::at(OBJECT_AT))
                        .instance(Instance::new(prepared, material)),
                )
                .unwrap();
            dispatch
                .spawn(
                    SpawnBundle::new()
                        .at(r4, Pose::at(Vec4::new(1.0, 0.0, -4.0, 0.0)))
                        .instance(Instance::new(prepared, material)),
                )
                .unwrap();
            let view = dispatch
                .domains
                .typed(r4)
                .unwrap()
                .add_view(ViewSpec::new(root, eye, Section4 { w: 0.0 }).subject(subject));
            (subject, view)
        });
        let mut publication = Publication::default();
        session.publish(&mut publication).unwrap();
        assert_eq!(publication.views[0].records.instances.rows().len(), 1);

        session
            .dispatch(|dispatch| dispatch.despawn(subject))
            .unwrap();
        session.publish(&mut publication).unwrap();
        assert!(publication.views[0].records.instances.rows().is_empty());
        assert_eq!(
            session
                .domains_mut()
                .typed(r4)
                .unwrap()
                .set_view_subject(view, Some(subject)),
            Err(DomainError::Stale(subject))
        );
        assert_eq!(
            session.domains().read(r4).unwrap().view_subject(view),
            Some(subject)
        );
    }

    #[test]
    fn cached_output_ranges_match_fresh_publication_across_view_lifecycle_changes() {
        let mut session = Session::new(Probe::default(), SimConfig::default());
        let r4 = session
            .register_domain(DomainBuilder::new("r4", EuclideanR4).tracked(LogCapacity::default()));
        let prepared = session.prepare(lines());
        let section_geometry = session.prepare(PreparedGeometry::Polytope4 {
            polytope: Polytope4::Tesseract,
            scale: 0.25,
        });
        let material = session.add_material(Material::flat([1.0; 4]));
        let root = session.views().root();
        let (edited, removed, alternate_eye, view, section_view) = session.dispatch(|dispatch| {
            let eye = dispatch
                .spawn(SpawnBundle::new().at(r4, Pose::at(Vec4::ZERO)))
                .unwrap();
            let alternate_eye = dispatch
                .spawn(SpawnBundle::new().at(r4, Pose::at(Vec4::new(2.0, 0.0, 0.0, 0.0))))
                .unwrap();
            let instance = || Instance::new(prepared, material);
            let edited = dispatch
                .spawn(
                    SpawnBundle::new()
                        .at(r4, Pose::at(OBJECT_AT))
                        .instance(Instance::new(section_geometry, material).sectioned(material)),
                )
                .unwrap();
            let removed = dispatch
                .spawn(
                    SpawnBundle::new()
                        .at(r4, Pose::at(Vec4::new(1.0, 0.0, -4.0, 0.0)))
                        .instance(instance()),
                )
                .unwrap();
            let domain = dispatch.domains.typed(r4).unwrap();
            let view = domain.add_view(ViewSpec::new(root, eye, Projection4 { focal: 2.0 }));
            let mut section_spec = ViewSpec::new(root, eye, Section4 { w: 0.0 }).subject(edited);
            section_spec.edges = false;
            let section_view = domain.add_view(section_spec);
            (edited, removed, alternate_eye, view, section_view)
        });
        let snapshot = session.snapshot().unwrap();
        let mut publication = Publication::default();
        let publish_and_compare =
            |session: &mut Session<Probe>, publication: &mut Publication<Probe>| {
                session.publish(publication).unwrap();
                let mut fresh = Publication::default();
                session.publish(&mut fresh).unwrap();
                assert_eq!(publication.views.len(), fresh.views.len());
                for (cached, fresh) in publication.views.iter().zip(&fresh.views) {
                    assert_eq!(cached.domain, fresh.domain);
                    assert_eq!(cached.target, fresh.target);
                    assert_eq!(cached.placement, fresh.placement);
                    assert_eq!(
                        cached.records.instances.rows(),
                        fresh.records.instances.rows()
                    );
                    assert_eq!(cached.records.segments(), fresh.records.segments());
                    assert_eq!(cached.records.triangles(), fresh.records.triangles());
                    assert_eq!(cached.records.refusals(), fresh.records.refusals());
                }
            };
        publish_and_compare(&mut session, &mut publication);
        let initial_segments = publication.views[1].records.segments().len();
        let initial_triangles = publication.views[1].records.triangles().len();
        assert!(initial_segments > 0);
        assert!(initial_triangles > 0);

        session
            .domains_mut()
            .typed(r4)
            .unwrap()
            .set_pose(edited, Pose::at(Vec4::new(0.5, 0.0, -4.0, 0.0)))
            .unwrap();
        publish_and_compare(&mut session, &mut publication);
        assert_eq!(
            publication.views[1].records.segments().len(),
            initial_segments
        );
        assert_eq!(
            publication.views[1].records.triangles().len(),
            initial_triangles
        );

        session
            .domains_mut()
            .typed(r4)
            .unwrap()
            .set_pose(edited, Pose::at(Vec4::new(0.5, 0.0, -4.0, 1.0)))
            .unwrap();
        publish_and_compare(&mut session, &mut publication);
        assert!(publication.views[1].records.segments().is_empty());
        assert!(publication.views[1].records.triangles().is_empty());
        session
            .domains_mut()
            .typed(r4)
            .unwrap()
            .set_pose(edited, Pose::at(Vec4::new(0.5, 0.0, -4.0, 0.0)))
            .unwrap();
        publish_and_compare(&mut session, &mut publication);
        assert_eq!(
            publication.views[1].records.segments().len(),
            initial_segments
        );
        assert_eq!(
            publication.views[1].records.triangles().len(),
            initial_triangles
        );

        session
            .domains_mut()
            .typed(r4)
            .unwrap()
            .view_mut(section_view)
            .unwrap()
            .set_mapping(Section4 { w: 0.5 });
        publish_and_compare(&mut session, &mut publication);
        assert!(publication.views[1].records.triangles().is_empty());
        assert_ne!(
            publication.views[1].records.triangles().len(),
            initial_triangles
        );

        session
            .domains_mut()
            .typed(r4)
            .unwrap()
            .set_pose(edited, Pose::at(Vec4::new(0.5, 0.0, -4.0, 3.0)))
            .unwrap();
        publish_and_compare(&mut session, &mut publication);
        session
            .domains_mut()
            .typed(r4)
            .unwrap()
            .set_view_eye(view, alternate_eye)
            .unwrap();
        publish_and_compare(&mut session, &mut publication);
        session
            .dispatch(|dispatch| dispatch.despawn(removed))
            .unwrap();
        publish_and_compare(&mut session, &mut publication);
        assert!(publication.views[0]
            .records
            .instances
            .rows()
            .iter()
            .all(|record| record.entity != removed));
        session.restore(&snapshot).unwrap();
        publish_and_compare(&mut session, &mut publication);
        assert_eq!(
            publication.views[1].records.triangles().len(),
            initial_triangles
        );
    }

    #[test]
    fn explicit_space_errors_are_reported_but_projection_clips_are_not() {
        let mut session = Session::new(Probe::default(), SimConfig::default());
        let r4 = session
            .register_domain(DomainBuilder::new("r4", EuclideanR4).tracked(LogCapacity::default()));
        let bad_x = session.prepare(PreparedGeometry::Lines4 {
            segments: vec![[[f32::NAN, 0.0, 0.0, 0.0], [0.0; 4]]],
        });
        let bad_y = session.prepare(PreparedGeometry::Lines4 {
            segments: vec![[[0.0, f32::MAX, 0.0, 0.0], [0.0; 4]]],
        });
        let empty = session.prepare(PreparedGeometry::Lines4 {
            segments: Vec::new(),
        });
        let material = session.add_material(Material::flat([1.0; 4]));
        let root = session.views().root();
        let (first, last, refusal_view, clipped_view) = session.dispatch(|dispatch| {
            let eye = dispatch
                .spawn(SpawnBundle::new().at(r4, Pose::at(Vec4::ZERO)))
                .unwrap();
            let first = dispatch
                .spawn(
                    SpawnBundle::new()
                        .at(r4, Pose::at(Vec4::new(0.0, 0.0, -3.0, 0.0)))
                        .instance(Instance::new(bad_x, material)),
                )
                .unwrap();
            let last = dispatch
                .spawn(
                    SpawnBundle::new()
                        .at(r4, Pose::at(Vec4::new(0.0, f32::MAX, -4.0, 0.0)))
                        .instance(Instance::new(bad_y, material)),
                )
                .unwrap();
            let clipped = dispatch
                .spawn(
                    SpawnBundle::new()
                        .at(r4, Pose::at(Vec4::new(0.0, 0.0, -3.0, 3.0)))
                        .instance(Instance::new(empty, material)),
                )
                .unwrap();
            let domain = dispatch.domains.typed(r4).unwrap();
            let refusal_view = domain.add_view(ViewSpec::new(root, eye, Section4 { w: 0.0 }));
            let mut clipped_spec =
                ViewSpec::new(root, eye, Projection4 { focal: 2.0 }).subject(clipped);
            clipped_spec.edges = false;
            clipped_spec.section_edges = false;
            clipped_spec.section_faces = false;
            let clipped_view = domain.add_view(clipped_spec);
            (first, last, refusal_view, clipped_view)
        });
        let mut publication = Publication::default();
        session.publish(&mut publication).unwrap();
        let refusal_records = &publication
            .views
            .iter()
            .find(|published| published.target.view == refusal_view)
            .unwrap()
            .records;
        assert_eq!(
            refusal_records.refusals(),
            crate::view::ViewRefusals {
                count: 2,
                first: Some(crate::view::ViewRefusal {
                    entity: first,
                    error: DomainError::InvalidCoordinate("x"),
                    source: RefusalSource::Segment,
                }),
                last: Some(crate::view::ViewRefusal {
                    entity: last,
                    error: DomainError::InvalidCoordinate("y"),
                    source: RefusalSource::Segment,
                }),
            }
        );
        let clipped_records = &publication
            .views
            .iter()
            .find(|published| published.target.view == clipped_view)
            .unwrap()
            .records;
        assert_eq!(clipped_records.refusals().count, 0);
        assert!(clipped_records.instances.rows().is_empty());
        let before = (
            refusal_records.built(),
            refusal_records.instances.rows().len(),
            refusal_records.segments().len(),
            refusal_records.triangles().len(),
            refusal_records.refusals(),
        );
        session.publish(&mut publication).unwrap();
        let refusal_records = &publication
            .views
            .iter()
            .find(|published| published.target.view == refusal_view)
            .unwrap()
            .records;
        assert_eq!(
            (
                refusal_records.built(),
                refusal_records.instances.rows().len(),
                refusal_records.segments().len(),
                refusal_records.triangles().len(),
                refusal_records.refusals(),
            ),
            before
        );
    }
}
