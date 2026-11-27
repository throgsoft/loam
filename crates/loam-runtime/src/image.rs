use std::fmt;

use crate::camera::Rig;
use crate::view::Vec3;

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
