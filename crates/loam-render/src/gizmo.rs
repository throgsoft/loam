use glam::{Vec3, Vec4};
use loam_math::{Bivector, Plane4, Rotor4};
use loam_shape::LineMesh;

use crate::hypergimbal::{Hypergimbal, Ring, RingStyle};

const RING_REACH: f32 = 1.0 + std::f32::consts::SQRT_2;

const SHAFT_INNER: f32 = 0.45;

const SHAFT_HEAD: f32 = 0.28;

const SHAFT_CLEARANCE: f32 = 2.0 * SHAFT_HEAD;

const SHAFT_OUTER: f32 = RING_REACH + SHAFT_CLEARANCE;

const INV_SQRT_3: f32 = 0.577_350_26;

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

    /// `w` takes the direction equiangular to the three scene axes.
    pub fn shaft_direction(self) -> Vec3 {
        match self {
            Self::X => Vec3::X,
            Self::Y => Vec3::Y,
            Self::Z => Vec3::Z,
            Self::W => Vec3::new(-INV_SQRT_3, -INV_SQRT_3, -INV_SQRT_3),
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

    fn hit_depth(&self, ray_origin: Vec3, ray_direction: Vec3, tolerance: f32) -> Option<f32> {
        let along = self.ray_parameter(ray_origin, ray_direction)?;
        let nearest = self.point(along.clamp(self.head_start(), self.outer));
        let unit = ray_direction.normalize();
        let offset = nearest - ray_origin;
        let depth = offset.dot(unit);
        if depth <= 0.0 || (offset - unit * depth).length() > tolerance {
            return None;
        }
        Some(depth)
    }

    pub fn drag_translation(&self, grab: f32, cursor: f32) -> Vec4 {
        self.axis.unit() * (cursor - grab)
    }

    fn append_line_mesh(&self, style: &ShaftStyle, scale: f32, out: &mut LineMesh<3>) {
        let color = style.colors[self.axis as usize];
        let mut push = |from: Vec3, to: Vec3| {
            out.segments.push((from.to_array(), to.to_array()));
            out.colors.push((color, color));
            out.widths.push(style.width_px);
        };
        let tip = self.point(self.outer);
        push(self.point(self.inner), tip);

        let barb_root = self.point(self.head_start());
        let (across, up) = self.direction.any_orthonormal_pair();
        let radius = style.head_radius * scale;
        let step = std::f32::consts::TAU / style.head_barbs as f32;
        for k in 0..style.head_barbs {
            let angle = k as f32 * step;
            push(
                tip,
                barb_root + (across * angle.cos() + up * angle.sin()) * radius,
            );
        }
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

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct TransformGizmo {
    pub center: Vec3,
    pub scale: f32,
}

impl TransformGizmo {
    pub fn hypergimbal(&self) -> Hypergimbal {
        Hypergimbal {
            center: self.center,
            scale: self.scale,
        }
    }

    pub fn rings(&self) -> [Ring; 6] {
        self.hypergimbal().rings()
    }

    pub fn shafts(&self) -> [Shaft; 4] {
        Axis4::ALL.map(|axis| self.shaft(axis))
    }

    pub fn shaft(&self, axis: Axis4) -> Shaft {
        Shaft {
            axis,
            origin: self.center,
            direction: axis.shaft_direction(),
            inner: SHAFT_INNER * self.scale,
            outer: SHAFT_OUTER * self.scale,
            head: SHAFT_HEAD * self.scale,
        }
    }

    /// A shaft beats a ring wherever both are in range; the mesh draws shafts last.
    pub fn pick(&self, ray_origin: Vec3, ray_direction: Vec3, tolerance: f32) -> Option<Handle> {
        let mut nearest: Option<(f32, Shaft)> = None;
        for shaft in self.shafts() {
            let Some(depth) = shaft.hit_depth(ray_origin, ray_direction, tolerance) else {
                continue;
            };
            if nearest.is_none_or(|(best, _)| depth < best) {
                nearest = Some((depth, shaft));
            }
        }
        if let Some((_, shaft)) = nearest {
            return Some(Handle::Translate(shaft));
        }
        self.hypergimbal()
            .pick(ray_origin, ray_direction, tolerance)
            .map(Handle::Rotate)
    }

    pub fn append_line_mesh(&self, style: &GizmoStyle, out: &mut LineMesh<3>) {
        self.hypergimbal().append_line_mesh(&style.rings, out);
        for shaft in self.shafts() {
            shaft.append_line_mesh(&style.shafts, self.scale, out);
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct GizmoStyle {
    pub rings: RingStyle,
    pub shafts: ShaftStyle,
}

#[derive(Clone, Debug)]
pub struct ShaftStyle {
    pub width_px: f32,
    pub head_barbs: usize,
    pub head_radius: f32,
    /// RGBA per axis, in [`Axis4::ALL`] order.
    pub colors: [[f32; 4]; 4],
}

impl ShaftStyle {
    pub const AXIS_COLORS: [[f32; 4]; 4] = [
        [0.96, 0.36, 0.34, 1.0],
        [0.42, 0.90, 0.46, 1.0],
        [0.38, 0.58, 0.98, 1.0],
        [0.82, 0.44, 0.98, 1.0],
    ];
}

impl Default for ShaftStyle {
    fn default() -> Self {
        Self {
            width_px: 2.5,
            head_barbs: 4,
            head_radius: 0.09,
            colors: Self::AXIS_COLORS,
        }
    }
}
