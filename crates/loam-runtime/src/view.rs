use bytemuck::{Pod, Zeroable};
use loam_math::hyperbolic::{klein_to_poincare, poincare_to_klein, H3_DEPTH_ENVELOPE};
use loam_math::{EuclideanR3, EuclideanR4, HyperbolicH3, Iso3, IsometryGroup, Space};
use loam_shape::polytope::SectionScratch;
use loam_shape::{LineMesh, TriangleMesh};
use std::ops::{Deref, DerefMut, Range};
use std::sync::Arc;

use crate::domain::{ChartPoint, ChartPose, DomainError, DomainId, DomainSpace, Pose};
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

/// A slot and a generation; a later `place` that reuses the slot makes the earlier id stale.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ImageSpaceId {
    slot: u32,
    generation: u32,
}

impl ImageSpaceId {
    pub fn index(self) -> usize {
        self.slot as usize
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

/// Orbits a target using the delta convention in `Pointer`.
#[derive(Clone, Copy, Debug, PartialEq)]
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

    pub fn zoom(&mut self, lines: f32) {
        self.distance =
            (self.distance * (-lines * ZOOM_GAIN).exp()).clamp(MIN_DISTANCE, MAX_DISTANCE);
    }

    pub fn eye(&self) -> Eye {
        let (yaw_sin, yaw_cos) = self.yaw.sin_cos();
        let (pitch_sin, pitch_cos) = self.pitch.sin_cos();
        let offset = Vec3::new(yaw_sin * pitch_cos, -pitch_sin, yaw_cos * pitch_cos);
        Eye::looking_at(
            (Vec3::from(self.target) + offset * self.distance).to_array(),
            self.target,
            [0.0, 1.0, 0.0],
        )
    }
}

const ORBIT_GAIN: f32 = 0.006;
const PITCH_LIMIT: f32 = 1.5;
const ZOOM_GAIN: f32 = 0.12;
const MIN_DISTANCE: f32 = 1.5;
const MAX_DISTANCE: f32 = 20.0;

pub struct ImageSpace {
    pub eye: Eye,
}

/// A similarity of the root's R³: scale about the origin, then the pose.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rigid {
    pub pose: Iso3,
    pub scale: f32,
}

impl Rigid {
    pub const IDENTITY: Self = Self {
        pose: Iso3::IDENTITY,
        scale: 1.0,
    };

    pub fn apply(&self, point: [f32; 3]) -> [f32; 3] {
        EuclideanR3
            .iso_apply(self.pose, Vec3::from(point) * self.scale)
            .to_array()
    }

    /// `self` after `inner`; the inner translation is scaled by the outer scale, so two similarities compose exactly.
    pub fn compose(&self, inner: &Rigid) -> Rigid {
        Rigid {
            pose: EuclideanR3.iso_compose(
                self.pose,
                Iso3 {
                    rotation: inner.pose.rotation,
                    translation: inner.pose.translation * self.scale,
                },
            ),
            scale: self.scale * inner.scale,
        }
    }

    pub fn inverse(&self) -> Rigid {
        let inverse = EuclideanR3.iso_inverse(self.pose);
        Rigid {
            pose: Iso3 {
                rotation: inverse.rotation,
                translation: inverse.translation / self.scale,
            },
            scale: 1.0 / self.scale,
        }
    }

    pub fn direction(&self, direction: [f32; 3]) -> [f32; 3] {
        EuclideanR3
            .iso_transport(self.pose, Vec3::ZERO, Vec3::from(direction))
            .to_array()
    }

    pub fn ray(&self, ray: &ImageRay) -> ImageRay {
        ImageRay {
            origin: self.apply(ray.origin),
            direction: self.direction(ray.direction),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Placement {
    Rigid(Rigid),
    /// Declared by the kind of map that would need tessellation; refused for a raster view.
    Nonlinear(&'static str),
}

impl Placement {
    /// A nonlinear side wins.
    pub fn compose(&self, inner: &Placement) -> Placement {
        match (self, inner) {
            (Placement::Rigid(outer), Placement::Rigid(inner)) => {
                Placement::Rigid(outer.compose(inner))
            }
            (Placement::Nonlinear(kind), _) | (_, Placement::Nonlinear(kind)) => {
                Placement::Nonlinear(kind)
            }
        }
    }

    pub fn rigid(&self) -> Option<Rigid> {
        match self {
            Placement::Rigid(rigid) => Some(*rigid),
            Placement::Nonlinear(_) => None,
        }
    }
}

#[derive(Clone, Copy)]
struct Placed {
    parent: ImageSpaceId,
    placement: Placement,
    generation: u32,
    live: bool,
}

#[derive(Clone, Default)]
pub struct ViewsSnapshot {
    root_eye: Eye,
    placed: Vec<Placed>,
}

/// R³ image spaces; the root one projects to the screen and every hit has a position in one.
pub struct Views {
    root: ImageSpace,
    placed: Vec<Placed>,
}

impl Views {
    pub(crate) fn new() -> Self {
        Self {
            root: ImageSpace {
                eye: Eye::default(),
            },
            placed: Vec::new(),
        }
    }

    pub fn root(&self) -> ImageSpaceId {
        ImageSpaceId {
            slot: 0,
            generation: 0,
        }
    }

    pub fn root_mut(&mut self) -> &mut ImageSpace {
        &mut self.root
    }

    pub fn get(&self, id: ImageSpaceId) -> Option<&ImageSpace> {
        (id.slot == 0).then_some(&self.root)
    }

    pub fn get_mut(&mut self, id: ImageSpaceId) -> Option<&mut ImageSpace> {
        (id.slot == 0).then_some(&mut self.root)
    }

    pub(crate) fn place(
        &mut self,
        parent: ImageSpaceId,
        placement: Placement,
    ) -> Option<ImageSpaceId> {
        self.to_root(parent)?;
        let free = self
            .placed
            .iter()
            .position(|placed| !placed.live && placed.generation != u32::MAX);
        let (slot, generation) = match free {
            Some(slot) => {
                let placed = self.placed.get_mut(slot)?;
                placed.generation += 1;
                placed.parent = parent;
                placed.placement = placement;
                placed.live = true;
                (slot, placed.generation)
            }
            None => {
                self.placed.push(Placed {
                    parent,
                    placement,
                    generation: 0,
                    live: true,
                });
                (self.placed.len() - 1, 0)
            }
        };
        Some(ImageSpaceId {
            slot: (slot + 1) as u32,
            generation,
        })
    }

    pub(crate) fn unplace(&mut self, id: ImageSpaceId) {
        if let Some(placed) = self.slot_mut(id) {
            placed.live = false;
        }
    }

    pub fn placed(&self, id: ImageSpaceId) -> bool {
        self.slot_of(id).is_some()
    }

    fn slot_of(&self, id: ImageSpaceId) -> Option<&Placed> {
        let placed = self.placed.get(id.index().checked_sub(1)?)?;
        (placed.live && placed.generation == id.generation).then_some(placed)
    }

    fn slot_mut(&mut self, id: ImageSpaceId) -> Option<&mut Placed> {
        let placed = self.placed.get_mut(id.index().checked_sub(1)?)?;
        (placed.generation == id.generation).then_some(placed)
    }

    /// The composed placement from `image` into the root, `None` for a space never placed, unplaced, or whose slot a later `place` reused.
    pub fn to_root(&self, image: ImageSpaceId) -> Option<Placement> {
        let mut composed = Placement::Rigid(Rigid::IDENTITY);
        let mut current = image;
        while current.slot != 0 {
            let placed = self.slot_of(current)?;
            composed = placed.placement.compose(&composed);
            current = placed.parent;
        }
        Some(composed)
    }

    /// The root eye's ray through a y-up NDC point, pulled into `image` through the inverse of its composed placement; a placed space has no eye of its own.
    pub fn ray(&self, image: ImageSpaceId, ndc: [f32; 2]) -> Option<ImageRay> {
        let root = self.root_ray(ndc)?;
        Some(self.to_root(image)?.rigid()?.inverse().ray(&root))
    }

    fn root_ray(&self, ndc: [f32; 2]) -> Option<ImageRay> {
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

    /// Inverse of [`Views::ray`] on the root image space.
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
            placed: self.placed.clone(),
        }
    }

    pub(crate) fn restore(&mut self, from: &ViewsSnapshot) {
        self.root.eye = from.root_eye;
        self.placed.clear();
        self.placed.extend_from_slice(&from.placed);
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

/// Results stay stable for the same inputs; configuration changes replace the map through `ViewSpec::set_mapping`.
pub trait ViewMapping<S: DomainSpace>: Send + Sync + 'static {
    fn name(&self) -> &'static str;

    fn image_point(&self, eye: &Pose<S>, point: S::Point) -> Option<[f32; 3]>;

    /// Image of `local`, a point in the frame of an entity at `pose`; the default places it and maps the point, and a nonlinear map overrides it to place through `relative` and add the entity's image position after.
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

    /// Where this map cuts an entity at `pose`, for a prepared geometry that can be sectioned; a map that does not cut returns `None`.
    fn section(&self, _eye: &Pose<S>, _pose: &Pose<S>) -> Option<SectionCut> {
        None
    }

    /// True when `lift` recovers a domain ray from any image ray; a map without one refuses grabs and field bridges by name.
    fn ray_lift(&self) -> bool {
        true
    }

    /// Image-space radius of the metric ball of `radius` at `point`; isometric maps keep the default.
    fn image_radius(&self, _eye: &Pose<S>, _point: S::Point, radius: f32) -> f32 {
        radius
    }

    fn depth_envelope(&self) -> DepthEnvelope;
}

pub struct ViewSettings<S: DomainSpace> {
    pub(crate) eye: Entity,
    pub(crate) subject: Option<Entity>,
    pub(crate) style: ViewStyle<S>,
}

pub struct ViewStyle<S: DomainSpace> {
    pub enabled: bool,
    pub edges: bool,
    pub section_edges: bool,
    pub section_faces: bool,
    pub(crate) mapping: Arc<dyn ViewMapping<S>>,
}

impl<S: DomainSpace> ViewSettings<S> {
    pub(crate) fn new(eye: Entity, mapping: impl ViewMapping<S>) -> Self {
        Self {
            eye,
            subject: None,
            style: ViewStyle {
                enabled: true,
                edges: true,
                section_edges: true,
                section_faces: true,
                mapping: Arc::new(mapping),
            },
        }
    }

    pub fn eye(&self) -> Entity {
        self.eye
    }

    pub fn subject(&self) -> Option<Entity> {
        self.subject
    }
}

impl<S: DomainSpace> Clone for ViewSettings<S> {
    fn clone(&self) -> Self {
        Self {
            eye: self.eye,
            subject: self.subject,
            style: self.style.clone(),
        }
    }
}

impl<S: DomainSpace> Deref for ViewSettings<S> {
    type Target = ViewStyle<S>;

    fn deref(&self) -> &Self::Target {
        &self.style
    }
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
            mapping: Arc::clone(&self.mapping),
        }
    }
}

pub struct ViewSpec<S: DomainSpace> {
    pub(crate) image: ImageSpaceId,
    pub(crate) settings: ViewSettings<S>,
}

impl<S: DomainSpace> Clone for ViewSpec<S> {
    fn clone(&self) -> Self {
        Self {
            image: self.image,
            settings: self.settings.clone(),
        }
    }
}

impl<S: DomainSpace> ViewSpec<S> {
    pub fn new(image: ImageSpaceId, eye: Entity, mapping: impl ViewMapping<S>) -> Self {
        Self {
            image,
            settings: ViewSettings::new(eye, mapping),
        }
    }

    pub fn subject(mut self, subject: Entity) -> Self {
        self.settings.subject = Some(subject);
        self
    }
}

impl<S: DomainSpace> Deref for ViewSpec<S> {
    type Target = ViewStyle<S>;

    fn deref(&self) -> &Self::Target {
        &self.settings.style
    }
}

impl<S: DomainSpace> DerefMut for ViewSpec<S> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.settings.style
    }
}

/// Where a slicing map cuts an entity and how it places the cut in the image.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SectionCut {
    /// The cut's last chart coordinate in the eye frame, relative to the entity's origin.
    pub offset: f32,
    /// Multiplies a cut point's leading chart coordinates before the entity's own image position is added.
    pub scale: f32,
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

    fn lift(&self, eye: &Pose<EuclideanR4>, ray: &ImageRay) -> Option<DomainRay<EuclideanR4>> {
        let direction = Vec3::from(ray.direction).try_normalize()?;
        let origin = Vec3::from(ray.origin).extend(self.w);
        Some(DomainRay {
            origin: EuclideanR4.place(&EuclideanR4.prepare(eye), origin).ok()?,
            direction: EuclideanR4.carry(eye, origin, direction.extend(0.0)).ok()?,
        })
    }

    fn depth_envelope(&self) -> DepthEnvelope {
        UNBOUNDED
    }
}

fn eye_relative4(eye: &Pose<EuclideanR4>, point: Vec4) -> Option<Vec4> {
    EuclideanR4.local(eye, point).ok()
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
        // Cannon, Floyd, Kenyon, Parry, Hyperbolic Geometry, 1997, §7: Klein is the hyperboloid seen from the origin.
        Some(poincare_to_klein(HyperbolicH3.local(eye, point).ok()?).to_array())
    }

    fn image_local(
        &self,
        space: &HyperbolicH3,
        _eye: &Pose<HyperbolicH3>,
        _pose: &Pose<HyperbolicH3>,
        relative: &<HyperbolicH3 as DomainSpace>::Relative,
        local: Vec3,
    ) -> Option<[f32; 3]> {
        Some(poincare_to_klein(space.place_relative(relative, local).ok()?).to_array())
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
            origin: HyperbolicH3.place(&HyperbolicH3.prepare(eye), from).ok()?,
            direction: HyperbolicH3.carry(eye, from, unit).ok()?,
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
        Some(EuclideanR3.local(eye, point).ok()?.to_array())
    }

    fn image_local(
        &self,
        space: &EuclideanR3,
        _eye: &Pose<EuclideanR3>,
        _pose: &Pose<EuclideanR3>,
        relative: &<EuclideanR3 as DomainSpace>::Relative,
        local: Vec3,
    ) -> Option<[f32; 3]> {
        Some(space.place_relative(relative, local).ok()?.to_array())
    }

    fn lift(&self, eye: &Pose<EuclideanR3>, ray: &ImageRay) -> Option<DomainRay<EuclideanR3>> {
        let origin = Vec3::from(ray.origin);
        let direction = Vec3::from(ray.direction).try_normalize()?;
        Some(DomainRay {
            origin: EuclideanR3.place(&EuclideanR3.prepare(eye), origin).ok()?,
            direction: EuclideanR3.carry(eye, origin, direction).ok()?,
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

/// One point of a view in its image space with a radius in pixels; `loam-render`'s `PointPass` builds its mesh from these.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct PointRecord {
    pub position: [f32; 3],
    pub radius_px: f32,
    pub color: [f32; 4],
}

/// One triangle of a view in its image space, with one color for the face.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct TriangleRecord {
    pub vertices: [[f32; 3]; 3],
    pub color: [f32; 4],
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

const NO_OUTPUT: u32 = u32::MAX;

#[derive(Clone)]
pub(crate) struct EntityOutput {
    pub(crate) entity: Entity,
    pub(crate) section_segments: Range<usize>,
    pub(crate) line_segments: Range<usize>,
    pub(crate) triangles: Range<usize>,
    pub(crate) instance: bool,
}

#[derive(Default)]
pub(crate) struct ViewOutputCache {
    entries: Vec<EntityOutput>,
    positions: Vec<u32>,
}

impl ViewOutputCache {
    pub(crate) fn clear(&mut self) {
        self.entries.clear();
        self.positions.fill(NO_OUTPUT);
    }

    pub(crate) fn push(&mut self, output: EntityOutput) {
        let slot = output.entity.key().slot() as usize;
        if slot >= self.positions.len() {
            self.positions.resize(slot + 1, NO_OUTPUT);
        }
        self.positions[slot] = self.entries.len() as u32;
        self.entries.push(output);
    }

    pub(crate) fn get(&self, entity: Entity) -> Option<&EntityOutput> {
        let position = *self.positions.get(entity.key().slot() as usize)?;
        if position == NO_OUTPUT || self.entries[position as usize].entity != entity {
            return None;
        }
        Some(&self.entries[position as usize])
    }

    pub(crate) fn iter_mut(&mut self) -> impl Iterator<Item = &mut EntityOutput> {
        self.entries.iter_mut()
    }
}

/// Records of one view map, in its image space.
#[derive(Default)]
pub struct ViewRecords {
    pub instances: RecordBuffer<InstanceRecord>,
    pub(crate) segments: Vec<SegmentRecord>,
    pub(crate) triangles: Vec<TriangleRecord>,
    pub(crate) refusals: ViewRefusals,
    pub(crate) scratch: SectionScratchpad,
    pub(crate) output: ViewOutputCache,
    pub(crate) patch_segments: Vec<SegmentRecord>,
    pub(crate) patch_triangles: Vec<TriangleRecord>,
    pub(crate) changed: Vec<Entity>,
    pub(crate) poses: Cursor,
    pub(crate) attachments: Cursor,
    pub(crate) revision: u32,
    pub(crate) built: Stamp,
}

impl ViewRecords {
    pub fn segments(&self) -> &[SegmentRecord] {
        &self.segments
    }

    pub fn triangles(&self) -> &[TriangleRecord] {
        &self.triangles
    }

    pub fn refusals(&self) -> ViewRefusals {
        self.refusals
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
