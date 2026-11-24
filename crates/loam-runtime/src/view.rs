use loam_math::{EuclideanR3, EuclideanR4, Rotor, Space};
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use crate::domain::{ChartPoint, DomainId, DomainSpace, Pose};
use crate::entity::Entity;
use crate::image::{ImageRay, ImageSpaceId};

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ViewTarget {
    pub view: ViewId,
    pub image: ImageSpaceId,
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
