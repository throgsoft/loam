use glam::{Vec3, Vec4};
use loam_math::{Bivector, Plane4, Rotor4};

use crate::hypergimbal::Ring;

const MIN_SHAFT_INCIDENCE: f32 = 1e-2;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum Axis4 {
    X = 0,
    Y = 1,
    Z = 2,
    W = 3,
}

impl Axis4 {
    pub const ALL: [Self; 4] = [Self::X, Self::Y, Self::Z, Self::W];

    pub fn label(self) -> &'static str {
        match self {
            Self::X => "x",
            Self::Y => "y",
            Self::Z => "z",
            Self::W => "w",
        }
    }

    pub fn unit(self) -> Vec4 {
        match self {
            Self::X => Vec4::X,
            Self::Y => Vec4::Y,
            Self::Z => Vec4::Z,
            Self::W => Vec4::W,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Shaft {
    pub axis: Axis4,
    pub origin: Vec3,
    pub direction: Vec3,
    /// Drawn extent; neither the pick nor the drag is clipped to it.
    pub inner: f32,
    pub outer: f32,
    pub head: f32,
}

impl Shaft {
    pub fn point(&self, along: f32) -> Vec3 {
        self.origin + self.direction * along
    }

    pub fn head_start(&self) -> f32 {
        self.outer - self.head
    }

    /// Closest approach of the ray to the shaft's line (Ericson, *Real-Time Collision Detection*, 2005, §5.1.9); `None` within 0.6° of parallel.
    pub fn ray_parameter(&self, ray_origin: Vec3, ray_direction: Vec3) -> Option<f32> {
        let offset = self.origin - ray_origin;
        let alignment = self.direction.dot(ray_direction);
        let ray_length_squared = ray_direction.length_squared();
        let denominator = ray_length_squared - alignment * alignment;
        if denominator <= MIN_SHAFT_INCIDENCE * MIN_SHAFT_INCIDENCE * ray_length_squared {
            return None;
        }
        Some(
            (alignment * ray_direction.dot(offset)
                - self.direction.dot(offset) * ray_length_squared)
                / denominator,
        )
    }

    pub fn drag_translation(&self, grab: f32, cursor: f32) -> Vec4 {
        self.axis.unit() * (cursor - grab)
    }

}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum HandleId {
    Rotate(Plane4),
    Translate(Axis4),
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Handle {
    Rotate(Ring),
    Translate(Shaft),
}

impl Handle {
    pub fn id(self) -> HandleId {
        match self {
            Self::Rotate(ring) => HandleId::Rotate(ring.plane),
            Self::Translate(shaft) => HandleId::Translate(shaft.axis),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum TransformDelta {
    Rotate { plane: Plane4, angle: f32 },
    Translate { axis: Axis4, distance: f32 },
}

impl TransformDelta {
    pub fn rotor(self) -> Rotor4 {
        match self {
            Self::Rotate { plane, angle } => (plane.unit_bivector() * angle).exp(),
            Self::Translate { .. } => Rotor4::IDENTITY,
        }
    }

    pub fn translation(self) -> Vec4 {
        match self {
            Self::Rotate { .. } => Vec4::ZERO,
            Self::Translate { axis, distance } => axis.unit() * distance,
        }
    }
}

/// Anchored at the press, so a widget following the subject does not chase its own drag.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum HandleDrag {
    Rotate { ring: Ring, grab: Vec3 },
    Translate { shaft: Shaft, grab: f32 },
}

impl HandleDrag {
    pub fn press(handle: Handle, ray_origin: Vec3, ray_direction: Vec3) -> Option<Self> {
        match handle {
            Handle::Rotate(ring) => ring
                .ray_hit(ray_origin, ray_direction)
                .map(|grab| Self::Rotate { ring, grab }),
            Handle::Translate(shaft) => shaft
                .ray_parameter(ray_origin, ray_direction)
                .map(|grab| Self::Translate { shaft, grab }),
        }
    }

    pub fn id(&self) -> HandleId {
        match self {
            Self::Rotate { ring, .. } => HandleId::Rotate(ring.plane),
            Self::Translate { shaft, .. } => HandleId::Translate(shaft.axis),
        }
    }

    /// `None` when the ray cannot be read against the handle; callers hold the last delta.
    pub fn delta(&self, ray_origin: Vec3, ray_direction: Vec3) -> Option<TransformDelta> {
        match self {
            Self::Rotate { ring, grab } => {
                let cursor = ring.ray_hit(ray_origin, ray_direction)?;
                Some(TransformDelta::Rotate {
                    plane: ring.plane,
                    angle: ring.drag_angle(*grab, cursor),
                })
            }
            Self::Translate { shaft, grab } => {
                let cursor = shaft.ray_parameter(ray_origin, ray_direction)?;
                Some(TransformDelta::Translate {
                    axis: shaft.axis,
                    distance: cursor - grab,
                })
            }
        }
    }
}
