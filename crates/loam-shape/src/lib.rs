//! Pose is extrinsic: shapes are defined in a local frame and positioned by
//! the caller's transform. [`Shape::Sphere`] and [`Shape::HyperSphere4D`] are
//! the exceptions, carrying a `center` that physics ignores.

pub mod field;
pub mod isovolume;
pub mod polytope;
pub mod polytope_geom;
pub mod projected_edges;
pub mod projection;
pub mod visualizable;

pub use field::{DistanceField, FieldKind};
pub use isovolume::Isovolume;
pub use visualizable::{LineMesh, NotVisualizable, PointMesh, TriangleMesh, Visualizable};

use glam::{Vec2, Vec3, Vec4};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Shape {
    Sphere {
        center: Vec3,
        radius: f32,
    },

    HalfSpace {
        /// Assumed unit: `dot(p, normal) - offset` is read as a signed distance.
        normal: Vec3,
        offset: f32,
    },

    /// Only meaningful on a static body (`inv_mass = 0`).
    HalfSpace4D {
        /// Assumed unit, as in [`Shape::HalfSpace`].
        normal: Vec4,
        offset: f32,
    },

    Box3 {
        /// The box spans `[-half_extents, half_extents]`.
        half_extents: Vec3,
    },

    Polygon2D {
        /// Counter-clockwise boundary loop in the local frame.
        vertices: Vec<Vec2>,
    },

    ConvexPolytope3D {
        /// Unordered point set in the shape frame; the collider is its hull.
        vertices: Vec<Vec3>,
    },

    ConvexPolytope4D {
        /// Unordered local points; the collider is their convex hull.
        vertices: Vec<Vec4>,
    },

    HyperSphere4D {
        center: Vec4,
        radius: f32,
    },
}

impl Shape {
    pub fn kind(&self) -> ShapeKind {
        match self {
            Shape::Sphere { .. } => ShapeKind::Sphere,
            Shape::HalfSpace { .. } => ShapeKind::HalfSpace,
            Shape::HalfSpace4D { .. } => ShapeKind::HalfSpace4D,
            Shape::Box3 { .. } => ShapeKind::Box3,
            Shape::Polygon2D { .. } => ShapeKind::Polygon2D,
            Shape::ConvexPolytope3D { .. } => ShapeKind::ConvexPolytope3D,
            Shape::ConvexPolytope4D { .. } => ShapeKind::ConvexPolytope4D,
            Shape::HyperSphere4D { .. } => ShapeKind::HyperSphere4D,
        }
    }

    pub fn sphere_at_origin(radius: f32) -> Self {
        Self::Sphere {
            center: Vec3::ZERO,
            radius,
        }
    }

    pub fn sphere_at(center: Vec3, radius: f32) -> Self {
        Self::Sphere { center, radius }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ShapeKind {
    Sphere,
    HalfSpace,
    HalfSpace4D,
    Box3,
    Polygon2D,
    ConvexPolytope3D,
    ConvexPolytope4D,
    HyperSphere4D,
}
