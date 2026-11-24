//! Boxes use chart coordinates; halfspaces require Space::is_chart_flat.
//! Only geodesic spheres have the same distance contract across curved spaces.

use glam::Vec3;
use loam_math::{Space, WgslSpace};
use loam_shape::shape::{box3, half_space, sphere};
use loam_shape::Shape;

use super::shapes;
use super::SENTINEL_DISTANCE;

pub trait Primitive {
    fn to_wgsl<S: WgslSpace>(&self, space: &S, name: &str) -> String;

    fn eval<S: Space<Point = Vec3, Vector = Vec3>>(&self, space: &S, p: Vec3) -> f32;
}

impl Primitive for Shape {
    fn to_wgsl<S: WgslSpace>(&self, space: &S, name: &str) -> String {
        match self {
            Shape::Sphere { center, radius } => shapes::sphere::wgsl(name, *center, *radius),
            Shape::Box3 { half_extents } => shapes::box3::wgsl(name, *half_extents),
            Shape::HalfSpace { normal, offset } if space.is_chart_flat() => {
                shapes::half_space::wgsl(name, *normal, *offset)
            }
            Shape::HalfSpace { .. }
            | Shape::HalfSpace4D { .. }
            | Shape::Polygon2D { .. }
            | Shape::ConvexPolytope3D { .. }
            | Shape::ConvexPolytope4D { .. }
            | Shape::HyperSphere4D { .. } => shapes::sentinel_wgsl(name),
        }
    }

    fn eval<S: Space<Point = Vec3, Vector = Vec3>>(&self, space: &S, p: Vec3) -> f32 {
        match self {
            Shape::Sphere { center, radius } => sphere::distance(space, p, *center, *radius),
            Shape::Box3 { half_extents } => box3::distance(p, *half_extents),
            Shape::HalfSpace { normal, offset } if space.is_chart_flat() => {
                half_space::distance(p, *normal, *offset)
            }
            Shape::HalfSpace { .. }
            | Shape::HalfSpace4D { .. }
            | Shape::Polygon2D { .. }
            | Shape::ConvexPolytope3D { .. }
            | Shape::ConvexPolytope4D { .. }
            | Shape::HyperSphere4D { .. } => SENTINEL_DISTANCE,
        }
    }
}
