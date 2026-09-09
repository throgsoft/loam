use bytemuck::{Pod, Zeroable};
use loam_math::hyperbolic::{
    hyperboloid_to_klein, klein_to_poincare, poincare_to_hyperboloid, H3_DEPTH_ENVELOPE,
};
use loam_math::{EuclideanR3, EuclideanR4, HyperbolicH3, IsometryGroup, Space};

use crate::domain::{ChartPoint, ChartPose, DomainId, DomainSpace, Pose};
use crate::entity::Entity;
use crate::session::{MaterialId, PreparedId, Stamp};
use crate::store::{Cursor, RecordBuffer};

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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ImageSpaceId(u32);

impl ImageSpaceId {
    pub fn index(self) -> usize {
        self.0 as usize
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

/// Yaw and pitch around a target at a distance; pointer drag in NDC turns it.
pub struct Orbit {
    pub target: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
}

impl Orbit {
    pub fn around(target: [f32; 3], distance: f32) -> Self {
        Self {
            target,
            yaw: 0.0,
            pitch: 0.0,
            distance,
        }
    }

    pub fn drag(&mut self, delta: [f32; 2]) {
        self.yaw -= delta[0] * ORBIT_GAIN;
        self.pitch = (self.pitch + delta[1] * ORBIT_GAIN).clamp(-PITCH_LIMIT, PITCH_LIMIT);
    }

    pub fn eye(&self) -> Eye {
        let (yaw_sin, yaw_cos) = self.yaw.sin_cos();
        let (pitch_sin, pitch_cos) = self.pitch.sin_cos();
        let offset = Vec3::new(yaw_sin * pitch_cos, pitch_sin, yaw_cos * pitch_cos);
        Eye::looking_at(
            (Vec3::from(self.target) + offset * self.distance).to_array(),
            self.target,
            [0.0, 1.0, 0.0],
        )
    }
}

const ORBIT_GAIN: f32 = 2.0;
const PITCH_LIMIT: f32 = 1.5;

pub struct ImageSpace {
    pub eye: Eye,
}

/// R³ image spaces; the root one projects to the screen and every hit has a position in one.
pub struct Views {
    spaces: Vec<ImageSpace>,
}

impl Views {
    pub(crate) fn new() -> Self {
        Self {
            spaces: vec![ImageSpace {
                eye: Eye::default(),
            }],
        }
    }

    pub fn root(&self) -> ImageSpaceId {
        ImageSpaceId(0)
    }

    pub fn root_mut(&mut self) -> &mut ImageSpace {
        &mut self.spaces[0]
    }

    pub fn get(&self, id: ImageSpaceId) -> Option<&ImageSpace> {
        self.spaces.get(id.index())
    }

    pub fn get_mut(&mut self, id: ImageSpaceId) -> Option<&mut ImageSpace> {
        self.spaces.get_mut(id.index())
    }

    /// The ray from `image`'s eye through a y-up NDC point.
    pub fn ray(&self, image: ImageSpaceId, ndc: [f32; 2]) -> Option<ImageRay> {
        let eye = &self.get(image)?.eye;
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
        let eye = &self.spaces[0].eye;
        let (_, forward) = eye.forward_distance(point);
        (forward >= eye.near).then(|| eye.near / forward)
    }

    /// Inverse of [`Views::ray`] on the root image space.
    pub fn ndc(&self, point: [f32; 3]) -> Option<[f32; 2]> {
        let eye = &self.spaces[0].eye;
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

/// Eye-relative map from a domain into an image space, with its ray construction.
pub trait ViewMapping<S: DomainSpace>: Send + 'static {
    fn name(&self) -> &'static str;

    fn image_point(&self, eye: &Pose<S>, point: S::Point) -> Option<[f32; 3]>;

    fn lift(&self, eye: &Pose<S>, ray: &ImageRay) -> Option<DomainRay<S>>;

    /// Image-space radius of the metric ball of `radius` at `point`; isometric maps keep the default.
    fn image_radius(&self, _eye: &Pose<S>, _point: S::Point, radius: f32) -> f32 {
        radius
    }

    fn depth_envelope(&self) -> DepthEnvelope;
}

pub struct ViewSpec<S: DomainSpace> {
    pub image: ImageSpaceId,
    pub eye: Entity,
    pub mapping: Box<dyn ViewMapping<S>>,
    pub(crate) revision: u32,
}

impl<S: DomainSpace> ViewSpec<S> {
    pub fn new(image: ImageSpaceId, eye: Entity, mapping: impl ViewMapping<S>) -> Self {
        Self {
            image,
            eye,
            mapping: Box::new(mapping),
            revision: 0,
        }
    }
}

/// R⁴ to R³ from a center `focal` behind the eye along +W onto the eye's own hyperplane.
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
        let relative = eye_relative4(eye, point);
        let scale = self.scale(relative.w)?;
        Some((relative.truncate() * scale).to_array())
    }

    fn lift(&self, _eye: &Pose<EuclideanR4>, _ray: &ImageRay) -> Option<DomainRay<EuclideanR4>> {
        None
    }

    fn image_radius(
        &self,
        eye: &Pose<EuclideanR4>,
        point: <EuclideanR4 as Space>::Point,
        radius: f32,
    ) -> f32 {
        let relative = eye_relative4(eye, point);
        self.scale(relative.w)
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
        Some(eye_relative4(eye, point).truncate().to_array())
    }

    fn lift(&self, eye: &Pose<EuclideanR4>, ray: &ImageRay) -> Option<DomainRay<EuclideanR4>> {
        let direction = Vec3::from(ray.direction).try_normalize()?;
        let origin = Vec3::from(ray.origin).extend(self.w);
        Some(DomainRay {
            origin: EuclideanR4.iso_apply(eye.0, origin),
            direction: EuclideanR4.iso_transport(eye.0, origin, direction.extend(0.0)),
        })
    }

    fn depth_envelope(&self) -> DepthEnvelope {
        UNBOUNDED
    }
}

fn eye_relative4(eye: &Pose<EuclideanR4>, point: Vec4) -> Vec4 {
    EuclideanR4.iso_apply(EuclideanR4.iso_inverse(eye.0), point)
}

/// Klein model of H³ recentered at the eye; its geodesics are chords, so `lift` is exact.
pub struct Klein;

impl ViewMapping<HyperbolicH3> for Klein {
    fn name(&self) -> &'static str {
        "klein"
    }

    fn image_point(
        &self,
        eye: &Pose<HyperbolicH3>,
        point: <HyperbolicH3 as Space>::Point,
    ) -> Option<[f32; 3]> {
        let inverse = HyperbolicH3.iso_inverse(eye.0);
        // Cannon, Floyd, Kenyon, Parry, Hyperbolic Geometry, 1997, §7: Klein is the hyperboloid seen from the origin.
        Some(hyperboloid_to_klein(inverse.matrix * poincare_to_hyperboloid(point)).to_array())
    }

    fn lift(&self, eye: &Pose<HyperbolicH3>, ray: &ImageRay) -> Option<DomainRay<HyperbolicH3>> {
        let origin = Vec3::from(ray.origin);
        let direction = Vec3::from(ray.direction).try_normalize()?;
        let inside = 1.0 - origin.length_squared();
        if inside <= 0.0 {
            return None;
        }
        let along = origin.dot(direction);
        let exit = -along + (along * along + inside).sqrt();
        let from = klein_to_poincare(origin);
        let toward = klein_to_poincare(origin + direction * (exit * 0.5));
        let length = HyperbolicH3.distance(from, toward);
        if length <= 0.0 {
            return None;
        }
        let unit = HyperbolicH3.log(from, toward) * (1.0 / length);
        Some(DomainRay {
            origin: HyperbolicH3.iso_apply(eye.0, from),
            direction: HyperbolicH3.iso_transport(eye.0, from, unit),
        })
    }

    fn depth_envelope(&self) -> DepthEnvelope {
        DepthEnvelope {
            near: 0.0,
            far: H3_DEPTH_ENVELOPE,
        }
    }
}

pub struct Identity3;

impl ViewMapping<EuclideanR3> for Identity3 {
    fn name(&self) -> &'static str {
        "identity3"
    }

    fn image_point(
        &self,
        eye: &Pose<EuclideanR3>,
        point: <EuclideanR3 as Space>::Point,
    ) -> Option<[f32; 3]> {
        Some(
            EuclideanR3
                .iso_apply(EuclideanR3.iso_inverse(eye.0), point)
                .to_array(),
        )
    }

    fn lift(&self, eye: &Pose<EuclideanR3>, ray: &ImageRay) -> Option<DomainRay<EuclideanR3>> {
        let origin = Vec3::from(ray.origin);
        let direction = Vec3::from(ray.direction).try_normalize()?;
        Some(DomainRay {
            origin: EuclideanR3.iso_apply(eye.0, origin),
            direction: EuclideanR3.iso_transport(eye.0, origin, direction),
        })
    }

    fn depth_envelope(&self) -> DepthEnvelope {
        UNBOUNDED
    }
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

/// Records of one view map, in its image space.
#[derive(Default)]
pub struct ViewRecords {
    pub instances: RecordBuffer<InstanceRecord>,
    pub(crate) segments: Vec<SegmentRecord>,
    pub(crate) poses: Cursor,
    pub(crate) attachments: Cursor,
    pub(crate) revision: u32,
    pub(crate) built: Stamp,
}

impl ViewRecords {
    pub fn segments(&self) -> &[SegmentRecord] {
        &self.segments
    }

    /// The publication that last rebuilt these records; a skipped rebuild keeps it.
    pub fn built(&self) -> Stamp {
        self.built
    }
}

/// `hit` is the domain-space entry point when the view has a lift.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pick {
    pub entity: Entity,
    pub domain: DomainId,
    pub view: ViewId,
    pub image_point: [f32; 3],
    pub depth: f32,
    pub hit: Option<ChartPoint>,
}
