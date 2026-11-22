use crate::domain::{DomainError, DomainId};
use crate::entity::Entity;
use crate::store::StoreError;
use crate::view::{ImageRay, ImageSpaceId, Placement, Vec3, ViewId};

/// Linked from its anchor to the source view's eye; releasing either entity unlinks it and unplaces `image` inside that despawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bridge {
    pub source: DomainId,
    pub view: ViewId,
    pub image: ImageSpaceId,
}

pub struct BridgeSpec {
    pub anchor: Entity,
    pub into: ImageSpaceId,
    pub source: DomainId,
    pub view: ViewId,
    pub placement: Placement,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BridgeError {
    UnknownDomain(DomainId),
    UnknownView(ViewId),
    UnknownImage(ImageSpaceId),
    Nonlinear(&'static str),
    Domain(DomainError),
    Link(StoreError),
}

impl From<DomainError> for BridgeError {
    fn from(error: DomainError) -> Self {
        Self::Domain(error)
    }
}

impl From<StoreError> for BridgeError {
    fn from(error: StoreError) -> Self {
        Self::Link(error)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DragError {
    NoPick,
    NotGrabbed,
    /// The pointer ray runs parallel to the drag plane, so no point on it is nearer than another.
    Ambiguous(&'static str),
    Domain(DomainError),
}

impl From<DomainError> for DragError {
    fn from(error: DomainError) -> Self {
        Self::Domain(error)
    }
}

pub(crate) const VELOCITY_SAMPLES: usize = 12;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Drag {
    pub entity: Entity,
    pub domain: DomainId,
    pub view: ViewId,
    pub image: ImageSpaceId,
    pub(crate) plane: [f32; 3],
    pub(crate) normal: [f32; 3],
    pub(crate) depth: f32,
    pub(crate) center: [f32; 3],
    pub(crate) at: [f32; 3],
    pub(crate) samples: [([f32; 3], f64); VELOCITY_SAMPLES],
    pub(crate) sampled: usize,
    pub(crate) opened: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DragRelease {
    pub entity: Entity,
    pub domain: DomainId,
    pub image: ImageSpaceId,
    /// In the view's image space, per second.
    pub velocity: [f32; 3],
}

const PLANE_EPSILON: f32 = 1e-6;
const VELOCITY_WINDOW_SECONDS: f64 = 0.1;
// Keeps the ring from filling with one frame of high-rate pointer events.
const SAMPLE_SPACING_SECONDS: f64 = 0.01;

impl Drag {
    pub fn image_point(&self) -> [f32; 3] {
        self.at
    }

    pub(crate) fn plane_at(&self, eye: [f32; 3], forward: [f32; 3]) -> [f32; 3] {
        (Vec3::from(eye) + Vec3::from(forward) * self.depth).to_array()
    }

    pub(crate) fn meet(ray: &ImageRay, plane: [f32; 3], normal: [f32; 3]) -> Option<[f32; 3]> {
        let normal = Vec3::from(normal);
        let along = Vec3::from(ray.direction).dot(normal);
        if along.abs() <= PLANE_EPSILON {
            return None;
        }
        Some(ray.at((Vec3::from(plane) - Vec3::from(ray.origin)).dot(normal) / along))
    }

    pub(crate) fn moved(&self, at: [f32; 3]) -> [f32; 3] {
        (Vec3::from(self.center) + Vec3::from(at) - Vec3::from(self.plane)).to_array()
    }

    pub(crate) fn sample(&mut self, at: [f32; 3], time: f64) {
        if time < self.newest_time() {
            self.at = at;
            return;
        }
        if time - self.opened < SAMPLE_SPACING_SECONDS {
            self.samples[self.sampled.saturating_sub(1) % VELOCITY_SAMPLES] = (at, time);
        } else {
            self.samples[self.sampled % VELOCITY_SAMPLES] = (at, time);
            self.sampled += 1;
            self.opened = time;
        }
        self.at = at;
    }

    pub(crate) fn newest_time(&self) -> f64 {
        self.samples[self.sampled.saturating_sub(1) % VELOCITY_SAMPLES].1
    }

    pub(crate) fn velocity(&self) -> [f32; 3] {
        let held = self.sampled.min(VELOCITY_SAMPLES);
        if held < 2 {
            return [0.0; 3];
        }
        let newest = self.samples[(self.sampled - 1) % VELOCITY_SAMPLES];
        let mut oldest = newest;
        for back in 1..held {
            let sample = self.samples[(self.sampled - 1 - back) % VELOCITY_SAMPLES];
            if back > 1 && newest.1 - sample.1 > VELOCITY_WINDOW_SECONDS {
                break;
            }
            oldest = sample;
        }
        let elapsed = (newest.1 - oldest.1) as f32;
        if elapsed <= 0.0 {
            return [0.0; 3];
        }
        ((Vec3::from(newest.0) - Vec3::from(oldest.0)) / elapsed).to_array()
    }

    pub(crate) fn released(&self, velocity: [f32; 3]) -> DragRelease {
        DragRelease {
            entity: self.entity,
            domain: self.domain,
            image: self.image,
            velocity,
        }
    }
}
