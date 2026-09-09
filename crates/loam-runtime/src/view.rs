use loam_math::{EuclideanR3, EuclideanR4, HyperbolicH3, Space};

use crate::domain::{ChartPose, DomainId, DomainSpace, Pose};
use crate::entity::Entity;
use crate::session::{MaterialId, PreparedId};
use crate::store::RecordBuffer;

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
    pub near: f32,
    pub far: f32,
}

impl Eye {
    pub fn looking_at(_position: [f32; 3], _target: [f32; 3], _up: [f32; 3]) -> Self {
        todo!()
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

    pub fn drag(&mut self, _delta: [f32; 2]) {
        todo!()
    }

    pub fn eye(&self) -> Eye {
        todo!()
    }
}

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

    pub fn ray(&self, _image: ImageSpaceId, _ndc: [f32; 2]) -> Option<ImageRay> {
        todo!()
    }

    /// The root eye's projective depth of an image-space position.
    pub fn depth(&self, _point: [f32; 3]) -> Option<f32> {
        todo!()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImageRay {
    pub origin: [f32; 3],
    pub direction: [f32; 3],
}

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

/// Eye-relative map from a domain into an image space, with its ray construction.
pub trait ViewMapping<S: DomainSpace>: Send + 'static {
    fn name(&self) -> &'static str;

    fn image_point(&self, eye: &Pose<S>, point: S::Point) -> Option<[f32; 3]>;

    fn lift(&self, eye: &Pose<S>, ray: &ImageRay) -> Option<DomainRay<S>>;

    fn depth_envelope(&self) -> DepthEnvelope;
}

pub struct ViewSpec<S: DomainSpace> {
    pub image: ImageSpaceId,
    pub eye: Entity,
    pub mapping: Box<dyn ViewMapping<S>>,
}

impl<S: DomainSpace> ViewSpec<S> {
    pub fn new(image: ImageSpaceId, eye: Entity, mapping: impl ViewMapping<S>) -> Self {
        Self {
            image,
            eye,
            mapping: Box::new(mapping),
        }
    }
}

/// R⁴ to R³ by w-depth from the eye.
pub struct Projection4 {
    pub focal: f32,
}

impl ViewMapping<EuclideanR4> for Projection4 {
    fn name(&self) -> &'static str {
        "projection4"
    }

    fn image_point(
        &self,
        _eye: &Pose<EuclideanR4>,
        _point: <EuclideanR4 as Space>::Point,
    ) -> Option<[f32; 3]> {
        todo!()
    }

    fn lift(&self, _eye: &Pose<EuclideanR4>, _ray: &ImageRay) -> Option<DomainRay<EuclideanR4>> {
        todo!()
    }

    fn depth_envelope(&self) -> DepthEnvelope {
        todo!()
    }
}

/// The hyperplane w = `w` of R⁴.
pub struct Section4 {
    pub w: f32,
}

impl ViewMapping<EuclideanR4> for Section4 {
    fn name(&self) -> &'static str {
        "section4"
    }

    fn image_point(
        &self,
        _eye: &Pose<EuclideanR4>,
        _point: <EuclideanR4 as Space>::Point,
    ) -> Option<[f32; 3]> {
        todo!()
    }

    fn lift(&self, _eye: &Pose<EuclideanR4>, _ray: &ImageRay) -> Option<DomainRay<EuclideanR4>> {
        todo!()
    }

    fn depth_envelope(&self) -> DepthEnvelope {
        todo!()
    }
}

/// Klein model of H³ recentered at the eye.
pub struct Klein;

impl ViewMapping<HyperbolicH3> for Klein {
    fn name(&self) -> &'static str {
        "klein"
    }

    fn image_point(
        &self,
        _eye: &Pose<HyperbolicH3>,
        _point: <HyperbolicH3 as Space>::Point,
    ) -> Option<[f32; 3]> {
        todo!()
    }

    fn lift(&self, _eye: &Pose<HyperbolicH3>, _ray: &ImageRay) -> Option<DomainRay<HyperbolicH3>> {
        todo!()
    }

    fn depth_envelope(&self) -> DepthEnvelope {
        todo!()
    }
}

pub struct Identity3;

impl ViewMapping<EuclideanR3> for Identity3 {
    fn name(&self) -> &'static str {
        "identity3"
    }

    fn image_point(
        &self,
        _eye: &Pose<EuclideanR3>,
        _point: <EuclideanR3 as Space>::Point,
    ) -> Option<[f32; 3]> {
        todo!()
    }

    fn lift(&self, _eye: &Pose<EuclideanR3>, _ray: &ImageRay) -> Option<DomainRay<EuclideanR3>> {
        todo!()
    }

    fn depth_envelope(&self) -> DepthEnvelope {
        todo!()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InstanceRecord {
    pub entity: Entity,
    pub geometry: PreparedId,
    pub material: MaterialId,
    pub pose: ChartPose,
}

/// Records of one view map, in its image space.
#[derive(Default)]
pub struct ViewRecords {
    pub instances: RecordBuffer<InstanceRecord>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pick {
    pub entity: Entity,
    pub domain: DomainId,
    pub view: ViewId,
    pub image_point: [f32; 3],
    pub depth: f32,
}
