//! Each variant's hash, bounds, well-formedness, support and distance live in its own module.

pub mod box3;
pub mod convex_polytope3;
pub mod convex_polytope4;
pub mod half_space;
pub mod half_space4;
pub mod hyper_sphere4;
pub mod polygon2;
pub mod sphere;

use std::hash::Hasher;

use glam::{Vec2, Vec3, Vec4};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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

    /// Writes the kind index and then the bits of every field, so equal shapes hash equally.
    pub fn hash_into<H: Hasher>(&self, hasher: &mut H) {
        hasher.write_u8(self.kind().index());
        match self {
            Shape::Sphere { center, radius } => sphere::hash(hasher, *center, *radius),
            Shape::HalfSpace { normal, offset } => half_space::hash(hasher, *normal, *offset),
            Shape::HalfSpace4D { normal, offset } => half_space4::hash(hasher, *normal, *offset),
            Shape::Box3 { half_extents } => box3::hash(hasher, *half_extents),
            Shape::Polygon2D { vertices } => polygon2::hash(hasher, vertices),
            Shape::ConvexPolytope3D { vertices } => convex_polytope3::hash(hasher, vertices),
            Shape::ConvexPolytope4D { vertices } => convex_polytope4::hash(hasher, vertices),
            Shape::HyperSphere4D { center, radius } => {
                hyper_sphere4::hash(hasher, *center, *radius)
            }
        }
    }

    /// Distance from the local origin to the farthest point, infinite for a half space.
    pub fn bounding_radius(&self) -> f32 {
        match self {
            Shape::Sphere { radius, .. } => sphere::bounding_radius(*radius),
            Shape::HalfSpace { .. } => half_space::bounding_radius(),
            Shape::HalfSpace4D { .. } => half_space4::bounding_radius(),
            Shape::Box3 { half_extents } => box3::bounding_radius(*half_extents),
            Shape::Polygon2D { vertices } => polygon2::bounding_radius(vertices),
            Shape::ConvexPolytope3D { vertices } => convex_polytope3::bounding_radius(vertices),
            Shape::ConvexPolytope4D { vertices } => convex_polytope4::bounding_radius(vertices),
            Shape::HyperSphere4D { radius, .. } => hyper_sphere4::bounding_radius(*radius),
        }
    }

    pub fn is_well_formed(&self) -> bool {
        match self {
            Shape::Sphere { center, radius } => sphere::is_well_formed(*center, *radius),
            Shape::HalfSpace { normal, offset } => half_space::is_well_formed(*normal, *offset),
            Shape::HalfSpace4D { normal, offset } => half_space4::is_well_formed(*normal, *offset),
            Shape::Box3 { half_extents } => box3::is_well_formed(*half_extents),
            Shape::Polygon2D { vertices } => polygon2::is_well_formed(vertices),
            Shape::ConvexPolytope3D { vertices } => convex_polytope3::is_well_formed(vertices),
            Shape::ConvexPolytope4D { vertices } => convex_polytope4::is_well_formed(vertices),
            Shape::HyperSphere4D { center, radius } => {
                hyper_sphere4::is_well_formed(*center, *radius)
            }
        }
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

impl ShapeKind {
    /// Declaration order, which orders narrowphase registrations.
    pub fn index(self) -> u8 {
        self as u8
    }
}

fn hash_f32s<H: Hasher>(hasher: &mut H, values: &[f32]) {
    for value in values {
        hasher.write_u32(value.to_bits());
    }
}
