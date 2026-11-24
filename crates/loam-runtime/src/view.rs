use bytemuck::{Pod, Zeroable};
use loam_math::{EuclideanR3, EuclideanR4, Rotor, Rotor4, Space};
use loam_shape::polytope::SectionScratch;
use loam_shape::{FieldShape, LineMesh, TriangleMesh};
use std::fmt;
use std::ops::{Deref, DerefMut, Range};
use std::sync::Arc;

use crate::assets::{MaterialId, PreparedId};
use crate::camera::Rig;
use crate::domain::{ChartPoint, ChartPose, DomainError, DomainId, DomainSpace, Pose};
use crate::entity::Entity;
use crate::publication::Stamp;
use crate::store::RecordBuffer;

mod cache;

pub(crate) use cache::{
    EntityOutput, Motion, PatchCursors, PatchScratch, Planned, RecordCache, ViewState,
};

pub(crate) type Vec3 = <EuclideanR3 as Space>::Point;
pub(crate) type Vec4 = <EuclideanR4 as Space>::Point;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ViewId(u32);

impl ViewId {
    pub(crate) fn new(index: usize) -> Self {
        Self(index as u32)
    }

    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for ViewId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "view {}", self.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ImageSpaceId {
    slot: u32,
}

impl ImageSpaceId {
    pub fn index(self) -> usize {
        self.slot as usize
    }
}

impl fmt::Display for ImageSpaceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "image space {}", self.slot)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ViewTarget {
    pub view: ViewId,
    pub image: ImageSpaceId,
}

/// An orthonormal frame in an image space; there is no `Mat4` in the view contract.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Eye {
    pub position: [f32; 3],
    pub right: [f32; 3],
    pub up: [f32; 3],
    pub forward: [f32; 3],
    pub fov_y: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
}

impl Eye {
    pub fn looking_at(position: [f32; 3], target: [f32; 3], up: [f32; 3]) -> Self {
        let forward = (Vec3::from(target) - Vec3::from(position))
            .try_normalize()
            .unwrap_or(-Vec3::Z);
        let right = forward
            .cross(Vec3::from(up))
            .try_normalize()
            .unwrap_or(Vec3::X);
        Self {
            position,
            right: right.to_array(),
            up: right.cross(forward).to_array(),
            forward: forward.to_array(),
            ..Self::default()
        }
    }

    fn forward_distance(&self, point: [f32; 3]) -> (Vec3, f32) {
        let relative = Vec3::from(point) - Vec3::from(self.position);
        (relative, relative.dot(Vec3::from(self.forward)))
    }
}

impl Default for Eye {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            right: [1.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
            forward: [0.0, 0.0, -1.0],
            fov_y: 60.0_f32.to_radians(),
            aspect: 1.0,
            near: 0.05,
            far: 100.0,
        }
    }
}

/// Linear light; `below` is the color at ray y = -1, `above` at ray y = +1, mixed linearly in y.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Sky {
    pub below: [f32; 3],
    pub above: [f32; 3],
}

/// A checkered plane at height `y` whose color mixes into the sky with distance.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Ground {
    pub y: f32,
    pub dark: [f32; 3],
    pub light: [f32; 3],
    /// Half the sky is mixed in at `ln 2 / fog_per_unit` world units.
    pub fog_per_unit: f32,
    /// Distance from the eye before any fog mixes in.
    pub fog_start: f32,
    pub visible: bool,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Backdrop {
    pub sky: Sky,
    pub ground: Ground,
}

pub struct ImageSpace {
    pub eye: Eye,
    pub rig: Rig,
    /// Drawn behind every view of the root image; `None` leaves the host's background color.
    pub backdrop: Option<Backdrop>,
}

impl ImageSpace {
    /// Takes every field of `eye` except the aspect, which belongs to the surface.
    pub fn look(&mut self, eye: Eye) {
        self.eye = Eye {
            aspect: self.eye.aspect,
            ..eye
        };
    }

    /// Switches a `Rig::Pair` to its free camera or its orbit, starting the free camera from this eye, and leaves any other rig alone.
    pub fn set_flying(&mut self, flying: bool) {
        if let Rig::Pair(pair) = &mut self.rig {
            pair.set_flying(flying, self.eye);
        }
    }
}

#[derive(Clone, Default)]
pub struct ViewsSnapshot {
    root_eye: Eye,
    root_rig: Rig,
    root_backdrop: Option<Backdrop>,
}

pub struct Views {
    root: ImageSpace,
}

impl Views {
    pub(crate) fn new() -> Self {
        Self {
            root: ImageSpace {
                eye: Eye::default(),
                rig: Rig::Manual,
                backdrop: None,
            },
        }
    }

    pub fn root_id(&self) -> ImageSpaceId {
        ImageSpaceId { slot: 0 }
    }

    pub fn root(&self) -> &ImageSpace {
        &self.root
    }

    pub fn root_mut(&mut self) -> &mut ImageSpace {
        &mut self.root
    }

    pub fn ray(&self, image: ImageSpaceId, ndc: [f32; 2]) -> Option<ImageRay> {
        if image != self.root_id() {
            return None;
        }
        let eye = &self.root.eye;
        let half = (eye.fov_y * 0.5).tan();
        let direction = Vec3::from(eye.right) * (ndc[0] * half * eye.aspect)
            + Vec3::from(eye.up) * (ndc[1] * half)
            + Vec3::from(eye.forward);
        Some(ImageRay {
            origin: eye.position,
            direction: direction.try_normalize()?.to_array(),
        })
    }

    /// The root eye's projective depth of an image-space position; `None` nearer than its near plane.
    pub fn depth(&self, point: [f32; 3]) -> Option<f32> {
        let eye = &self.root.eye;
        let (_, forward) = eye.forward_distance(point);
        (forward >= eye.near).then(|| eye.near / forward)
    }

    pub fn ndc(&self, point: [f32; 3]) -> Option<[f32; 2]> {
        let eye = &self.root.eye;
        let (relative, forward) = eye.forward_distance(point);
        if forward < eye.near {
            return None;
        }
        let half = (eye.fov_y * 0.5).tan();
        Some([
            relative.dot(Vec3::from(eye.right)) / (forward * half * eye.aspect),
            relative.dot(Vec3::from(eye.up)) / (forward * half),
        ])
    }

    pub(crate) fn snapshot(&self) -> ViewsSnapshot {
        ViewsSnapshot {
            root_eye: self.root.eye,
            root_rig: self.root.rig,
            root_backdrop: self.root.backdrop,
        }
    }

    pub(crate) fn restore(&mut self, from: &ViewsSnapshot) {
        self.root.eye = from.root_eye;
        self.root.rig = from.root_rig;
        self.root.backdrop = from.root_backdrop;
        // A snapshot taken before the first boundary holds an eye the rig has not yet written.
        if let Some(eye) = self.root.rig.eye() {
            self.root.look(eye);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImageRay {
    pub origin: [f32; 3],
    pub direction: [f32; 3],
}

impl ImageRay {
    pub fn at(&self, t: f32) -> [f32; 3] {
        (Vec3::from(self.origin) + Vec3::from(self.direction) * t).to_array()
    }
}

/// `direction` has unit metric length, so `exp(origin, direction * t)` lies at distance `t`.
pub struct DomainRay<S: Space> {
    pub origin: S::Point,
    pub direction: S::Vector,
}

/// Distances from the viewer between which the map orders depth correctly.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthEnvelope {
    pub near: f32,
    pub far: f32,
}

const UNBOUNDED: DepthEnvelope = DepthEnvelope {
    near: 0.0,
    far: f32::INFINITY,
};

/// Results stay stable for the same inputs; configuration changes replace the map through `ViewStyle::set_mapping`.
pub trait ViewMapping<S: DomainSpace>: Send + Sync + 'static {
    fn name(&self) -> &'static str;

    fn image_point(&self, eye: &Pose<S>, point: S::Point) -> Option<[f32; 3]>;

    fn image_local(
        &self,
        space: &S,
        eye: &Pose<S>,
        pose: &Pose<S>,
        _relative: &S::Relative,
        local: S::Point,
    ) -> Option<[f32; 3]> {
        self.image_point(eye, space.place(&space.prepare(pose), local).ok()?)
    }

    /// Emits an ordered image polyline; `None` breaks the line.
    fn image_segment(
        &self,
        space: &S,
        eye: &Pose<S>,
        pose: &Pose<S>,
        relative: &S::Relative,
        segment: [S::Point; 2],
        emit: &mut dyn FnMut(f32, Option<[f32; 3]>),
    ) {
        emit(
            0.0,
            self.image_local(space, eye, pose, relative, segment[0]),
        );
        emit(
            1.0,
            self.image_local(space, eye, pose, relative, segment[1]),
        );
    }

    fn lift(&self, eye: &Pose<S>, ray: &ImageRay) -> Option<DomainRay<S>>;

    /// The domain tangent that the image vector `tangent` at the image point `at` shows, or `None` for a map without a tangent lift.
    fn lift_tangent(&self, _eye: &Pose<S>, _at: [f32; 3], _tangent: [f32; 3]) -> Option<S::Vector> {
        None
    }

    /// The image vector that shows the domain tangent `tangent` at `at`, or `None` for a map without a tangent lift.
    fn image_tangent(
        &self,
        _eye: &Pose<S>,
        _at: S::Point,
        _tangent: S::Vector,
    ) -> Option<[f32; 3]> {
        None
    }

    fn section(&self, _eye: &Pose<S>, _pose: &Pose<S>) -> Option<SectionCut> {
        None
    }

    /// The last eye-frame coordinate a field kernel cuts this image at, or `None` for a map that draws no fields.
    fn field_cut(&self) -> Option<f32> {
        None
    }

    /// True when `lift` recovers a domain ray from any image ray; a map without one refuses grabs by name.
    fn ray_lift(&self) -> bool {
        true
    }

    fn image_radius(&self, _eye: &Pose<S>, _point: S::Point, radius: f32) -> f32 {
        radius
    }

    fn depth_envelope(&self) -> DepthEnvelope;

    fn depth_envelope_at(&self, _eye: &Pose<S>) -> DepthEnvelope {
        self.depth_envelope()
    }
}

pub struct ViewStyle<S: DomainSpace> {
    pub enabled: bool,
    pub edges: bool,
    pub section_edges: bool,
    pub section_faces: bool,
    /// Draws each body that has a field shape as a field, except one whose section this view already fills.
    pub fields: bool,
    pub(crate) mapping: Arc<dyn ViewMapping<S>>,
}

impl<S: DomainSpace> ViewStyle<S> {
    pub fn set_mapping(&mut self, mapping: impl ViewMapping<S>) -> &mut Self {
        self.mapping = Arc::new(mapping);
        self
    }

    pub fn mapping(&self) -> &dyn ViewMapping<S> {
        self.mapping.as_ref()
    }
}

impl<S: DomainSpace> Clone for ViewStyle<S> {
    fn clone(&self) -> Self {
        Self {
            enabled: self.enabled,
            edges: self.edges,
            section_edges: self.section_edges,
            section_faces: self.section_faces,
            fields: self.fields,
            mapping: Arc::clone(&self.mapping),
        }
    }
}

pub struct ViewSpec<S: DomainSpace> {
    pub(crate) image: ImageSpaceId,
    pub(crate) eye: Entity,
    pub(crate) subject: Option<Entity>,
    pub(crate) style: ViewStyle<S>,
}

impl<S: DomainSpace> Clone for ViewSpec<S> {
    fn clone(&self) -> Self {
        Self {
            image: self.image,
            eye: self.eye,
            subject: self.subject,
            style: self.style.clone(),
        }
    }
}

impl<S: DomainSpace> ViewSpec<S> {
    pub fn new(image: ImageSpaceId, eye: Entity, mapping: impl ViewMapping<S>) -> Self {
        Self {
            image,
            eye,
            subject: None,
            style: ViewStyle {
                enabled: true,
                edges: true,
                section_edges: true,
                section_faces: true,
                fields: false,
                mapping: Arc::new(mapping),
            },
        }
    }

    pub fn subject(mut self, subject: Entity) -> Self {
        self.subject = Some(subject);
        self
    }
}

impl<S: DomainSpace> Deref for ViewSpec<S> {
    type Target = ViewStyle<S>;

    fn deref(&self) -> &Self::Target {
        &self.style
    }
}

impl<S: DomainSpace> DerefMut for ViewSpec<S> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.style
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SectionCut {
    /// The cut's last chart coordinate in the eye frame, relative to the entity's origin.
    pub offset: f32,
    /// Multiplies a cut point's leading chart coordinates before the entity's own image position is added.
    pub scale: f32,
}

pub struct Projection4 {
    pub focal: f32,
}

impl Projection4 {
    fn scale(&self, w: f32) -> Option<f32> {
        let depth = self.focal - w;
        (depth > 0.0).then(|| self.focal / depth)
    }
}

impl ViewMapping<EuclideanR4> for Projection4 {
    fn name(&self) -> &'static str {
        "projection4"
    }

    fn image_point(
        &self,
        eye: &Pose<EuclideanR4>,
        point: <EuclideanR4 as Space>::Point,
    ) -> Option<[f32; 3]> {
        let relative = eye_relative4(eye, point)?;
        let scale = self.scale(relative.w)?;
        Some((relative.truncate() * scale).to_array())
    }

    fn image_local(
        &self,
        space: &EuclideanR4,
        _eye: &Pose<EuclideanR4>,
        _pose: &Pose<EuclideanR4>,
        relative: &<EuclideanR4 as DomainSpace>::Relative,
        local: Vec4,
    ) -> Option<[f32; 3]> {
        let relative = space.place_relative(relative, local).ok()?;
        let scale = self.scale(relative.w)?;
        Some((relative.truncate() * scale).to_array())
    }

    fn lift(&self, _eye: &Pose<EuclideanR4>, _ray: &ImageRay) -> Option<DomainRay<EuclideanR4>> {
        None
    }

    fn ray_lift(&self) -> bool {
        false
    }

    fn image_radius(
        &self,
        eye: &Pose<EuclideanR4>,
        point: <EuclideanR4 as Space>::Point,
        radius: f32,
    ) -> f32 {
        eye_relative4(eye, point)
            .and_then(|relative| self.scale(relative.w))
            .map_or(radius, |scale| radius * scale)
    }

    fn depth_envelope(&self) -> DepthEnvelope {
        UNBOUNDED
    }
}

/// The hyperplane w = `w` of the eye's frame, with w dropped.
pub struct Section4 {
    pub w: f32,
}

impl ViewMapping<EuclideanR4> for Section4 {
    fn name(&self) -> &'static str {
        "section4"
    }

    fn image_point(
        &self,
        eye: &Pose<EuclideanR4>,
        point: <EuclideanR4 as Space>::Point,
    ) -> Option<[f32; 3]> {
        Some(eye_relative4(eye, point)?.truncate().to_array())
    }

    fn image_local(
        &self,
        space: &EuclideanR4,
        _eye: &Pose<EuclideanR4>,
        _pose: &Pose<EuclideanR4>,
        relative: &<EuclideanR4 as DomainSpace>::Relative,
        local: Vec4,
    ) -> Option<[f32; 3]> {
        Some(
            space
                .place_relative(relative, local)
                .ok()?
                .truncate()
                .to_array(),
        )
    }

    fn section(&self, eye: &Pose<EuclideanR4>, pose: &Pose<EuclideanR4>) -> Option<SectionCut> {
        Some(SectionCut {
            offset: self.w - eye_relative4(eye, pose.point)?.w,
            scale: 1.0,
        })
    }

    fn field_cut(&self) -> Option<f32> {
        Some(self.w)
    }

    fn lift(&self, eye: &Pose<EuclideanR4>, ray: &ImageRay) -> Option<DomainRay<EuclideanR4>> {
        let direction = Vec3::from(ray.direction).try_normalize()?;
        let origin = Vec3::from(ray.origin).extend(self.w);
        Some(DomainRay {
            origin: EuclideanR4.place(&EuclideanR4.prepare(eye), origin).ok()?,
            direction: EuclideanR4.carry(eye, origin, direction.extend(0.0)).ok()?,
        })
    }

    fn lift_tangent(
        &self,
        eye: &Pose<EuclideanR4>,
        at: [f32; 3],
        tangent: [f32; 3],
    ) -> Option<Vec4> {
        let origin = Vec3::from(at).extend(self.w);
        EuclideanR4
            .carry(eye, origin, Vec3::from(tangent).extend(0.0))
            .ok()
    }

    fn image_tangent(&self, eye: &Pose<EuclideanR4>, _at: Vec4, tangent: Vec4) -> Option<[f32; 3]> {
        Some(eye.frame.inverse().apply(tangent).truncate().to_array())
    }

    fn depth_envelope(&self) -> DepthEnvelope {
        UNBOUNDED
    }
}

fn eye_relative4(eye: &Pose<EuclideanR4>, point: Vec4) -> Option<Vec4> {
    EuclideanR4.local(eye, point).ok()
}

pub(crate) fn ball_entry(along: f32, outside: f32) -> Option<f32> {
    if outside <= 0.0 {
        return Some(0.0);
    }
    let discriminant = along * along - outside;
    if discriminant < 0.0 {
        return None;
    }
    let entry = -along - discriminant.sqrt();
    (entry >= 0.0).then_some(entry)
}

pub(crate) fn image_hit(ray: &ImageRay, center: [f32; 3], radius: f32) -> Option<f32> {
    let offset = Vec3::from(ray.origin) - Vec3::from(center);
    ball_entry(
        offset.dot(Vec3::from(ray.direction)),
        offset.length_squared() - radius * radius,
    )
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InstanceRecord {
    pub entity: Entity,
    pub geometry: PreparedId,
    pub material: MaterialId,
    pub pose: ChartPose,
    pub image_point: [f32; 3],
}

/// Instance vertex layout of `loam-render`'s `line_raster.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct SegmentRecord {
    pub start: [f32; 3],
    pub _pad0: f32,
    pub end: [f32; 3],
    pub _pad1: f32,
    pub start_color: [f32; 4],
    pub end_color: [f32; 4],
    pub width_px: f32,
    pub _pad2: [f32; 3],
}

/// One point of a view in its image space with a radius in pixels; `loam-render`'s `PointPass` builds its mesh from these.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct PointRecord {
    pub position: [f32; 3],
    pub radius_px: f32,
    pub color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct TriangleRecord {
    pub vertices: [[f32; 3]; 3],
    pub color: [f32; 4],
}

/// Screen text in logical points, with the origin measured from the top-left of the area the host's UI leaves free.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextRecord {
    pub origin_pt: [f32; 2],
    pub size_pt: f32,
    pub color: [f32; 4],
}

/// Lines and points in the root image space and screen text that publication systems draw each frame.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Overlay {
    pub lines: Vec<SegmentRecord>,
    pub points: Vec<PointRecord>,
    text: String,
    labels: Vec<(Range<usize>, TextRecord)>,
}

impl Overlay {
    /// Later text paints over earlier text.
    pub fn push_text(&mut self, text: &str, record: TextRecord) {
        let start = self.text.len();
        self.text.push_str(text);
        self.labels.push((start..self.text.len(), record));
    }

    pub fn text(&self) -> impl Iterator<Item = (&str, &TextRecord)> {
        self.labels
            .iter()
            .map(|(range, record)| (&self.text[range.clone()], record))
    }

    pub(crate) fn clear(&mut self) {
        self.lines.clear();
        self.points.clear();
        self.text.clear();
        self.labels.clear();
    }
}

/// A solid in its view's eye frame, which a field kernel cuts at [`ViewRecords::field_cut`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FieldRecord {
    pub entity: Entity,
    pub shape: FieldShape,
    pub size: f32,
    pub material: MaterialId,
    pub color: [f32; 4],
    pub position: [f32; 4],
    pub frame: Rotor4,
}

#[derive(Default)]
pub(crate) struct SectionScratchpad {
    pub(crate) rotated: Vec<Vec4>,
    pub(crate) cut: SectionScratch,
    pub(crate) faces: TriangleMesh<3>,
    pub(crate) perimeter: LineMesh<3>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefusalSource {
    Instance,
    Segment,
    Section,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ViewRefusal {
    pub entity: Entity,
    pub error: DomainError,
    pub source: RefusalSource,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ViewRefusals {
    pub count: u32,
    pub first: Option<ViewRefusal>,
    pub last: Option<ViewRefusal>,
}

impl ViewRefusals {
    pub(crate) fn record(&mut self, entity: Entity, error: DomainError, source: RefusalSource) {
        let refusal = ViewRefusal {
            entity,
            error,
            source,
        };
        self.count = self.count.saturating_add(1);
        self.first.get_or_insert(refusal);
        self.last = Some(refusal);
    }
}

#[derive(Default)]
pub struct ViewRecords {
    pub instances: RecordBuffer<InstanceRecord>,
    pub(crate) segments: Vec<SegmentRecord>,
    pub(crate) triangles: Vec<TriangleRecord>,
    pub(crate) fields: Vec<FieldRecord>,
    pub(crate) field_cut: Option<f32>,
    pub(crate) refusals: ViewRefusals,
    pub(crate) scratch: SectionScratchpad,
    pub(crate) cache: RecordCache,
}

impl ViewRecords {
    pub fn segments(&self) -> &[SegmentRecord] {
        &self.segments
    }

    pub fn triangles(&self) -> &[TriangleRecord] {
        &self.triangles
    }

    pub fn fields(&self) -> &[FieldRecord] {
        &self.fields
    }

    /// `None` when the view draws no fields.
    pub fn field_cut(&self) -> Option<f32> {
        self.field_cut
    }

    pub fn refusals(&self) -> ViewRefusals {
        self.refusals
    }

    /// The publication that last rebuilt these records; a skipped rebuild keeps it.
    pub fn built(&self) -> Stamp {
        self.cache.built()
    }
}

/// `hit` is the domain-space entry point when the view has a lift.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pick {
    pub entity: Entity,
    pub domain: DomainId,
    pub view: ViewId,
    pub image: ImageSpaceId,
    pub image_point: [f32; 3],
    pub depth: f32,
    pub hit: Option<ChartPoint>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ViewSummary {
    pub name: &'static str,
    pub eye: Entity,
    pub image: ImageSpaceId,
    pub ray_lift: bool,
}
